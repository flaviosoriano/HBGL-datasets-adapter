"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import glob
import logging
import argparse
import math
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import random
import pickle

from s2s_ft.modeling_decoding import BertForSeq2SeqDecoder, BertConfig
from transformers.tokenization_bert import whitespace_tokenize
import s2s_ft.s2s_loader as seq2seq_loader
from s2s_ft.utils import load_and_cache_examples
from transformers import BertTokenizer
from hbgl_ranking import (
    build_document_label_scores,
    hierarchy_levels_from_training_file,
    label_ids_by_depth_from_taxonomy,
)

TOKENIZER_CLASSES = {
    'bert': BertTokenizer,
}


class WhitespaceTokenizer(object):
    def tokenize(self, text):
        return whitespace_tokenize(text)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def load_document_ids(path, expected_count):
    """Load prepared split IDs and verify their one-to-one JSONL alignment."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("could not load document ID sidecar: {}".format(path)) from error
    if payload.get("id_kind") not in {"idx", "text_idx"}:
        raise ValueError("document ID sidecar has invalid id_kind")
    if not isinstance(payload.get("ids"), list) or len(payload["ids"]) != expected_count:
        raise ValueError("document ID sidecar length does not match input JSONL")
    return payload


def hierarchy_token_ids_by_depth(tokenizer, label_map, taxonomy_path):
    """Translate canonical source-label depths to the tokenizer IDs used by HBGL."""
    levels = label_ids_by_depth_from_taxonomy(
        label_map, Path(taxonomy_path).read_text(encoding="utf-8")
    )
    source_to_token = {}
    for token in label_map.values():
        source_id = int(token[3:-1])
        token_id = tokenizer.convert_tokens_to_ids(token)
        if not isinstance(token_id, int) or token_id < 0:
            raise ValueError("tokenizer cannot resolve canonical label token {}".format(token))
        source_to_token[source_id] = token_id
    return (
        levels,
        [[source_to_token[label_id] for label_id in level] for level in levels],
        {token_id: source_id for source_id, token_id in source_to_token.items()},
    )


class HierarchyScoreCapture(object):
    """External test-time hook for Eq.-10 vectors; it never mutates HBGL."""

    def __init__(self, classifier, hier_labels):
        self._vectors = []
        self._hier_labels = hier_labels
        self._handle = classifier.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output):
        self._vectors.append(output[0].detach())

    def begin_batch(self):
        self._vectors = []

    def scores(self, expected_level_token_ids, source_id_by_token):
        if len(self._vectors) != len(self._hier_labels):
            raise RuntimeError(
                "HBGL emitted {} score vectors for {} hierarchy levels".format(
                    len(self._vectors), len(self._hier_labels)
                )
            )
        if len(expected_level_token_ids) != len(self._hier_labels):
            raise RuntimeError("prepared taxonomy and HBGL hierarchy have different depths")
        level_label_ids = []
        probabilities_by_level = []
        batch_size = None
        for level, raw_scores in enumerate(self._vectors):
            token_ids = torch.nonzero(self._hier_labels[level], as_tuple=True)[0].tolist()
            if set(token_ids) != set(expected_level_token_ids[level]):
                raise RuntimeError("prepared taxonomy does not match HBGL hierarchy mask at level {}".format(level))
            try:
                level_label_ids.append([source_id_by_token[token_id] for token_id in token_ids])
            except KeyError as error:
                raise RuntimeError("HBGL hierarchy mask includes a non-canonical token") from error
            if raw_scores.dim() != 3 or raw_scores.shape[1] != 1:
                raise RuntimeError("unexpected HBGL classifier score shape")
            if batch_size is None:
                batch_size = raw_scores.shape[0]
            elif batch_size != raw_scores.shape[0]:
                raise RuntimeError("HBGL score vectors have inconsistent batch sizes")
            probabilities_by_level.append(torch.sigmoid(raw_scores[:, 0, token_ids]).cpu().tolist())
        return [
            build_document_label_scores(
                level_label_ids,
                [probabilities[document_index] for probabilities in probabilities_by_level],
            )
            for document_index in range(batch_size)
        ]

    def close(self):
        self._handle.remove()


def ascii_print(text):
    text = text.encode("ascii", "ignore")
    print(text)


def main(flags=None):
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(TOKENIZER_CLASSES.keys()))
    parser.add_argument("--model_path", default=None, type=str, required=True,
                        help="Path to the model checkpoint.")
    parser.add_argument("--config_path", default=None, type=str,
                        help="Path to config.json for the model.")

    # tokenizer_name
    parser.add_argument("--tokenizer_name", default=None, type=str, required=True,
                        help="tokenizer name")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                            "Sequences longer than this will be truncated, and sequences shorter \n"
                            "than this will be padded.")

    # decoding parameters
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--no_cuda', action='store_true',
                        help="Whether to use CUDA for decoding")
    parser.add_argument("--input_file", type=str, help="Input file")
    parser.add_argument('--subset', type=int, default=0,
                        help="Decode a subset of the input dataset.")
    parser.add_argument("--output_file", type=str, help="output file")
    parser.add_argument("--split", type=str, default="",
                        help="Data split (train/val/test).")
    parser.add_argument('--tokenized_input', action='store_true',
                        help="Whether the input is tokenized.")
    parser.add_argument('--seed', type=int, default=123,
                        help="random seed for initialization")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for decoding.")
    parser.add_argument('--beam_size', type=int, default=1,
                        help="Beam size for searching")
    parser.add_argument('--length_penalty', type=float, default=0,
                        help="Length penalty for beam search")

    parser.add_argument('--forbid_duplicate_ngrams', action='store_true')
    parser.add_argument('--forbid_ignore_word', type=str, default=None,
                        help="Forbid the word during forbid_duplicate_ngrams")
    parser.add_argument("--min_len", default=1, type=int)
    parser.add_argument('--ngram_size', type=int, default=3)
    parser.add_argument('--mode', default="s2s",
                        choices=["s2s", "l2r", "both"])
    parser.add_argument('--max_tgt_length', type=int, default=128,
                        help="maximum length of target sequence")
    parser.add_argument('--s2s_special_token', action='store_true',
                        help="New special tokens ([S2S_SEP]/[S2S_CLS]) of S2S.")
    parser.add_argument('--s2s_add_segment', action='store_true',
                        help="Additional segmental for the encoder of S2S.")
    parser.add_argument('--s2s_share_segment', action='store_true',
                        help="Sharing segment embeddings for the encoder of S2S (used with --s2s_add_segment).")
    parser.add_argument('--pos_shift', action='store_true',
                        help="Using position shift for fine-tuning.")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--add_vocab_file", type=str, default=None)
    parser.add_argument("--cached_features_file", type=str, default=None)
    parser.add_argument('--softmax_label_only', action='store_true')
    parser.add_argument('--soft_label', action='store_true')
    parser.add_argument('--soft_label_hier_real_with_train_file', default=None, type=str)
    parser.add_argument("--ranking_output_file", type=str, default=None,
                        help="Optional HBGL-only dense ranking artifact using paper Eq.-10 probabilities.")
    parser.add_argument("--document_ids_file", type=str, default=None,
                        help="Prepared <split>_document_ids.json aligned with input_file.")
    parser.add_argument("--label_taxonomy_file", type=str, default=None,
                        help="Prepared label_taxonomy.tsv used to associate labels with decoder levels.")
    parser.add_argument("--ranking_thresholds", nargs="+", type=int, default=[1, 5, 10],
                        help="Positive HGCLR-compatible cutoffs recorded in ranking metadata.")

    if flags:
        print(flags)
        args = parser.parse_args(flags)
    else:
        args = parser.parse_args()


    if args.max_tgt_length >= args.max_seq_length - 2:
        raise ValueError("Maximum tgt length exceeds max seq length - 2.")
    ranking_enabled = bool(args.ranking_output_file or args.document_ids_file or args.label_taxonomy_file)
    if not (bool(args.ranking_output_file) == bool(args.document_ids_file) == bool(args.label_taxonomy_file)):
        raise ValueError("HBGL ranking export requires output, document IDs, and taxonomy files")
    if ranking_enabled:
        if args.beam_size != 1 or not args.soft_label or not args.soft_label_hier_real_with_train_file:
            raise ValueError("HBGL ranking export requires greedy hierarchical --soft_label decoding")
        if not args.add_vocab_file:
            raise ValueError("HBGL ranking export requires --add_vocab_file")
        if any(cutoff <= 0 for cutoff in args.ranking_thresholds):
            raise ValueError("ranking thresholds must be positive")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if ranking_enabled and n_gpu > 1:
        raise ValueError("HBGL ranking export currently requires one GPU so classifier hook order is unambiguous")

    if args.seed > 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)
    else:
        random_seed = random.randint(0, 10000)
        logger.info("Set random seed as: {}".format(random_seed))
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

    tokenizer = TOKENIZER_CLASSES[args.model_type].from_pretrained(
        args.tokenizer_name, do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None)

    if args.add_vocab_file:
        import pickle
        with open(args.add_vocab_file, 'rb') as f:
            label_map = pickle.load(f)
        labels_key = list(label_map.keys())
        # tokenizer.add_special_tokens({'additional_special_tokens': [label_map[label] for label in labels_key]})
        tokenizer.add_tokens([label_map[label] for label in labels_key])
        add_token_num = len(labels_key)
        ranking_level_token_ids = None
        ranking_source_id_by_token = None
        if ranking_enabled:
            _ranking_level_label_ids, ranking_level_token_ids, ranking_source_id_by_token = hierarchy_token_ids_by_depth(
                tokenizer, label_map, args.label_taxonomy_file
            )

    hierarchy_levels = None
    if args.soft_label_hier_real_with_train_file:
        hierarchy_levels = hierarchy_levels_from_training_file(
            args.soft_label_hier_real_with_train_file
        )
        # Training reserves one target position for EOS/SEP.  Hierarchical
        # decoding must stop after the actual label levels before the decoder
        # indexes ``model.hier_labels`` beyond its final level.
        args.max_tgt_length = len(hierarchy_levels)

    if args.model_type == "roberta":
        vocab = tokenizer.encoder
    elif args.model_type == "xlm-roberta":
        vocab = {}
        for tk_id in range(len(tokenizer)):
            tk = tokenizer._convert_id_to_token(tk_id)
            vocab[tk] = tk_id
    else:
        vocab = tokenizer.vocab

    if hasattr(tokenizer, 'model_max_length'):
        tokenizer.model_max_length = args.max_seq_length
    elif hasattr(tokenizer, 'max_len'):
        tokenizer.max_len = args.max_seq_length

    mask_word_id, eos_word_ids, sos_word_id = tokenizer.convert_tokens_to_ids(
        [tokenizer.mask_token, tokenizer.sep_token, tokenizer.sep_token])
    forbid_ignore_set = None
    if args.forbid_ignore_word:
        w_list = []
        for w in args.forbid_ignore_word.split('|'):
            if w.startswith('[') and w.endswith(']'):
                w_list.append(w.upper())
            else:
                w_list.append(w)
        forbid_ignore_set = set(tokenizer.convert_tokens_to_ids(w_list))
    print(args.model_path)
    found_checkpoint_flag = False
    for model_recover_path in glob.glob(args.model_path):
        if not os.path.isdir(model_recover_path):
            continue

        logger.info("***** Recover model: %s *****", model_recover_path)

        config_file = args.config_path if args.config_path else os.path.join(model_recover_path, "config.json")
        logger.info("Read decoding config from: %s" % config_file)
        config = BertConfig.from_json_file(config_file)

        bi_uni_pipeline = []
        bi_uni_pipeline.append(seq2seq_loader.Preprocess4Seq2seqDecoder(
            list(vocab.keys()), tokenizer.convert_tokens_to_ids, args.max_seq_length,
            max_tgt_length=args.max_tgt_length, pos_shift=args.pos_shift,
            source_type_id=config.source_type_id, target_type_id=config.target_type_id,
            cls_token=tokenizer.cls_token, sep_token=tokenizer.sep_token, pad_token=tokenizer.pad_token))

        found_checkpoint_flag = True
        model = BertForSeq2SeqDecoder.from_pretrained(
            model_recover_path, config=config, mask_word_id=mask_word_id, search_beam_size=args.beam_size,
            length_penalty=args.length_penalty, eos_id=eos_word_ids, sos_id=sos_word_id,
            forbid_duplicate_ngrams=args.forbid_duplicate_ngrams, forbid_ignore_set=forbid_ignore_set,
            ngram_size=args.ngram_size, min_len=args.min_len, mode=args.mode,
            max_position_embeddings=args.max_seq_length, pos_shift=args.pos_shift,
        )

        if args.softmax_label_only and args.add_vocab_file:
            label_tokens_start_index = model.bert.embeddings.word_embeddings.num_embeddings - add_token_num
            model.label_start_index = label_tokens_start_index

        if args.soft_label:
            model.soft_label = args.soft_label
            label_tokens_start_index = model.bert.embeddings.word_embeddings.num_embeddings - add_token_num
            model.label_start_index = label_tokens_start_index

            if args.soft_label_hier_real_with_train_file:
                assert hierarchy_levels is not None
                hier_labels = [
                    tokenizer.convert_tokens_to_ids([label.lower() for label in labels])
                    for labels in hierarchy_levels
                ]

                def to_multi_hot(label):
                    _label = torch.zeros(model.config.vocab_size)
                    for i in label:
                        _label[i] = 1
                    return _label.bool()

                model.hier_labels = [to_multi_hot(i) for i in hier_labels]
                model.soft_label_hier_real = True

        if args.fp16:
            model.half()
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        torch.cuda.empty_cache()
        model.eval()
        base_model = model.module if hasattr(model, 'module') else model
        score_capture = HierarchyScoreCapture(base_model.cls, base_model.hier_labels) if ranking_enabled else None
        next_i = 0
        max_src_length = args.max_seq_length - 2 - args.max_tgt_length
        if args.pos_shift:
            max_src_length += 1

        num_lines = sum(1 for line in open(args.input_file))
        if num_lines < 10000:
            to_pred = load_and_cache_examples(
                args.input_file, tokenizer, local_rank=-1,
                cached_features_file=args.cached_features_file, shuffle=False, eval_mode=True)
        else:
            from s2s_ft.utils import load_and_cache_examples_fast
            to_pred = load_and_cache_examples_fast(
                args.input_file, tokenizer, local_rank=-1,
                cached_features_file=args.cached_features_file, shuffle=False, eval_mode=True)

        document_ids = None
        if ranking_enabled:
            document_ids = load_document_ids(args.document_ids_file, num_lines)
        input_lines = []
        for line in to_pred:
            input_lines.append(tokenizer.convert_ids_to_tokens(line.source_ids)[:max_src_length])
        if args.subset > 0:
            logger.info("Decoding subset: %d", args.subset)
            input_lines = input_lines[:args.subset]
            if ranking_enabled:
                document_ids["ids"] = document_ids["ids"][:args.subset]
        if ranking_enabled and len(document_ids["ids"]) != len(input_lines):
            raise ValueError("document ID sidecar does not align with decoded input")

        input_lines = sorted(list(enumerate(input_lines)),
                             key=lambda x: -len(x[1]))
        output_lines = [""] * len(input_lines)
        ranking = {} if ranking_enabled else None
        total_batch = math.ceil(len(input_lines) / args.batch_size)

        with tqdm(total=total_batch) as pbar:
            batch_count = 0
            first_batch = True
            while next_i < len(input_lines):
                _chunk = input_lines[next_i:next_i + args.batch_size]
                buf_id = [x[0] for x in _chunk]
                buf = [x[1] for x in _chunk]
                next_i += args.batch_size
                batch_count += 1
                max_a_len = max([len(x) for x in buf])
                instances = []
                for instance in [(x, max_a_len) for x in buf]:
                    for proc in bi_uni_pipeline:
                        instances.append(proc(instance))
                with torch.no_grad():
                    batch = seq2seq_loader.batch_list_to_batch_tensors(
                        instances)
                    batch = [
                        t.to(device) if t is not None else None for t in batch]
                    input_ids, token_type_ids, position_ids, input_mask, mask_qkv, task_idx = batch
                    if ranking_enabled:
                        score_capture.begin_batch()
                    traces = model(
                        input_ids, token_type_ids, position_ids, input_mask,
                        task_idx=task_idx, mask_qkv=mask_qkv)
                    if ranking_enabled:
                        output_ids = traces.tolist()
                        batch_dense_scores = score_capture.scores(
                            ranking_level_token_ids, ranking_source_id_by_token
                        )
                    elif args.beam_size > 1:
                        traces = {k: v.tolist() for k, v in traces.items()}
                        output_ids = traces['pred_seq']
                    else:
                        output_ids = traces.tolist()
                    for i in range(len(buf)):
                        w_ids = output_ids[i]
                        output_buf = tokenizer.convert_ids_to_tokens(w_ids)
                        output_tokens = []
                        for t in output_buf:
                            if t in (tokenizer.sep_token, tokenizer.pad_token):
                                break
                            output_tokens.append(t)
                        if args.model_type == "roberta" or args.model_type == "xlm-roberta":
                            output_sequence = tokenizer.convert_tokens_to_string(output_tokens)
                        else:
                            output_sequence = ' '.join(detokenize(output_tokens))
                        if '\n' in output_sequence:
                            output_sequence = " [X_SEP] ".join(output_sequence.split('\n'))
                        output_lines[buf_id[i]] = output_sequence
                        if ranking_enabled:
                            document_key = "text_{}".format(document_ids["ids"][buf_id[i]])
                            if document_key in ranking:
                                raise ValueError("duplicate external document ID in ranking: {}".format(document_key))
                            ranking[document_key] = batch_dense_scores[i]
                        if first_batch or batch_count % 50 == 0:
                            logger.info("{} = {}".format(buf_id[i], output_sequence))
                pbar.update(1)
                first_batch = False
        if args.output_file:
            fn_out = args.output_file
        else:
            fn_out = model_recover_path+'.'+args.split
        with open(fn_out, "w", encoding="utf-8") as fout:
            for l in output_lines:
                fout.write(l)
                fout.write("\n")
        if ranking_enabled:
            if len(ranking) != len(document_ids["ids"]):
                raise RuntimeError("ranking coverage does not match decoded document IDs")
            ranking_path = Path(args.ranking_output_file)
            ranking_path.parent.mkdir(parents=True, exist_ok=True)
            with ranking_path.open("wb") as handle:
                pickle.dump(ranking, handle, protocol=4)
            metadata_path = ranking_path.with_suffix(ranking_path.suffix + ".metadata.json")
            metadata_path.write_text(json.dumps({
                "artifact_version": 2,
                "document_id_kind": document_ids["id_kind"],
                "documents": len(ranking),
                "ranking_density": "all canonical labels",
                "score_source": "sigmoid classifier probability at the label's taxonomy depth (HBGL Eq. 10)",
                "thresholds": args.ranking_thresholds,
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            score_capture.close()
            logger.info("Wrote ranking artifact for %d documents to %s", len(ranking), ranking_path)

        import pickle
        from eval import evaluate
        with open(args.add_vocab_file, 'rb') as f:
            label_map = pickle.load(f)

        def token_to_id(token):
            token = token.lower()
            try:
                token = int(token.replace('[a_', '').replace(']', ''))
                token = 0 if token >= len(label_map) else token
                return token
            except:
                return 0

        if args.model_type == 'roberta':
            def roberta_token_to_id(token):
                token = token.replace("<s>", '').replace('[A_', ' ').replace(']', ' ').split(' ')
                token = [int(i) for i in token if i != '']
                return token
            predict_labels = [roberta_token_to_id(i) for i in output_lines]
        else:
            predict_labels = [i.replace("\n", '').split(' ') for i in output_lines]
            predict_labels = [list(set([token_to_id(j) for j  in i])) for i in predict_labels]
        with open(args.input_file) as f:
            gd_labels = [json.loads(i)['tgt'] for i in f]
            gd_labels = [[token_to_id(j) for j  in i.split(' ')] for i in gd_labels]

        id2label = {token_to_id(label_map[k]): k for k in label_map}
        out = evaluate(predict_labels, gd_labels, id2label, as_sample=True)
        del out['full']
        print(out)
        return out

    if not found_checkpoint_flag:
        logger.info("Not found the model checkpoint file!")


if __name__ == "__main__":
    print(main())

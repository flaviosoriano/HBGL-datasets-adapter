#!/usr/bin/env python3
"""Evaluate an HGCLR or HBGL document-to-label ranking on a canonical fold."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from ranking_eval import build_test_relevance, evaluate_rankings, label_counts_from_relevance


class _RankingUnpickler(pickle.Unpickler):
    """Accept only plain built-in pickle values for external ranking artifacts."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(
            "ranking artifacts must contain only plain built-in values; blocked {}.{}".format(module, name)
        )


def _load_ranking_artifact(path):
    try:
        with Path(path).open("rb") as handle:
            ranking = _RankingUnpickler(handle).load()
    except (OSError, EOFError, pickle.UnpicklingError) as error:
        raise RuntimeError("could not safely load ranking artifact") from error
    if not isinstance(ranking, dict):
        raise ValueError("ranking artifact must be a dictionary")
    return ranking


def _labels_from_prepared_row(row):
    labels = set()
    for token in row["tgt"].split():
        if not token.startswith("[A_") or not token.endswith("]"):
            raise ValueError("prepared test target has a non-canonical label token: {!r}".format(token))
        labels.add(int(token[3:-1]))
    if not labels:
        raise ValueError("prepared test row has no labels")
    return labels


def _load_prepared_qrels(prepared_dir):
    prepared = Path(prepared_dir)
    try:
        document_ids = json.loads((prepared / "test_document_ids.json").read_text(encoding="utf-8"))
        corpus_statistics = json.loads((prepared / "corpus_label_counts.json").read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in (prepared / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    except (OSError, ValueError) as error:
        raise RuntimeError("could not load prepared ranking metadata") from error
    if document_ids.get("id_kind") not in {"idx", "text_idx"}:
        raise ValueError("prepared test document IDs have an invalid id_kind")
    if not isinstance(document_ids.get("ids"), list) or len(document_ids["ids"]) != len(rows):
        raise ValueError("prepared test document IDs do not align with test JSONL")
    label_counts_raw = corpus_statistics.get("label_counts")
    total_documents = corpus_statistics.get("documents")
    if not isinstance(label_counts_raw, dict) or not isinstance(total_documents, int) or total_documents <= 1:
        raise ValueError("prepared corpus label statistics are invalid")
    label_counts = {int(label_id): int(count) for label_id, count in label_counts_raw.items()}
    if any(count <= 0 for count in label_counts.values()):
        raise ValueError("prepared corpus label statistics contain a non-positive count")
    relevance = {}
    for document_id, row in zip(document_ids["ids"], rows):
        key = "text_{}".format(document_id)
        if key in relevance:
            raise ValueError("duplicate external document ID in prepared test split: {}".format(key))
        relevance[key] = _labels_from_prepared_row(row)
    return relevance, label_counts, total_documents


def _load_legacy_qrels(dataset_dir, dataset_name, fold):
    """Fallback for scoring existing HGCLR artifacts without prepared metadata."""
    root = Path(dataset_dir)
    fold_dir = root / "fold_{}".format(fold)
    try:
        with (root / "samples.pkl").open("rb") as handle:
            samples = pickle.load(handle)
        with (fold_dir / "test.pkl").open("rb") as handle:
            test_indices = pickle.load(handle)
        with (root / "relevance_map.pkl").open("rb") as handle:
            all_relevance_raw = pickle.load(handle)
    except (OSError, pickle.UnpicklingError) as error:
        raise RuntimeError("could not load canonical data") from error
    all_relevance = {int(document_id): set(label_ids) for document_id, label_ids in all_relevance_raw.items()}
    test_relevance = build_test_relevance(samples, test_indices, all_relevance, dataset_name)
    label_counts = label_counts_from_relevance(
        {"text_{}".format(key): labels for key, labels in all_relevance.items()}
    )
    return test_relevance, label_counts, len(all_relevance)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking-file", required=True, help="HGCLR/HBGL pickle ranking artifact")
    parser.add_argument("--dataset-dir", required=True, help="Canonical dataset root")
    parser.add_argument("--dataset-name", required=True, choices=("WOS-150-H2", "RCV1-103-H3"))
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--prepared-dir", default=None,
                        help="Prepared fold directory; avoids loading the full RCV1 corpus at evaluation time.")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--cutoffs", nargs="+", type=int, default=(1, 5, 10))
    parser.add_argument("--propensity-a", type=float, default=0.55)
    parser.add_argument("--propensity-b", type=float, default=1.5)
    args = parser.parse_args(argv)

    if any(cutoff <= 0 for cutoff in args.cutoffs):
        parser.error("--cutoffs values must be positive")
    rankings = _load_ranking_artifact(args.ranking_file)

    if args.prepared_dir:
        test_relevance, label_counts, total_documents = _load_prepared_qrels(args.prepared_dir)
        relevance_protocol = "prepared canonical test JSONL and corpus_label_counts.json"
    else:
        test_relevance, label_counts, total_documents = _load_legacy_qrels(
            args.dataset_dir, args.dataset_name, args.fold
        )
        relevance_protocol = "canonical samples.pkl and relevance_map.pkl"

    expected_documents = set(test_relevance)
    actual_documents = set(rankings)
    missing = sorted(expected_documents - actual_documents)
    unexpected = sorted(actual_documents - expected_documents)
    if missing or unexpected:
        raise ValueError(
            "ranking/test-fold document coverage mismatch: missing={}, unexpected={}".format(
                missing[:5], unexpected[:5]
            )
        )

    metrics = evaluate_rankings(
        rankings,
        test_relevance,
        label_counts=label_counts,
        total_documents=total_documents,
        cutoffs=args.cutoffs,
        propensity_a=args.propensity_a,
        propensity_b=args.propensity_b,
    )
    payload = {
        "artifact_version": 1,
        "dataset": args.dataset_name,
        "fold": args.fold,
        "documents": len(test_relevance),
        "cutoffs": args.cutoffs,
        "relevance_protocol": relevance_protocol,
        "propensity": {"a": args.propensity_a, "b": args.propensity_b, "corpus_documents": total_documents},
        "metrics": metrics,
    }
    destination = Path(args.output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()

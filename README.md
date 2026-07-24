# Exploiting Global and Local Hierarchies for Hierarchical Text Classifications


## Preprocess

We follow the  repositories  of [contrastive-htc](https://github.com/wzh9969/contrastive-htc) and [HDLTex](https://github.com/kk7nc/HDLTex) to get the preprocessed datasets in json format file {'token': List[str], 'label': List[str]}.

Please download the origin datasets and pre-process them using the code in the corresponding folder:

+ [WoS](https://github.com/kk7nc/HDLTex) : `cd data/WebOfScience/ & python preprocess_wos.py`
+ [NYT](https://catalog.ldc.upenn.edu/LDC2008T19): `cd data/nyt/ &  python preprocess_nyt.py`
+ [RCV1-V2](https://github.com/ductri/reuters_loader): `cd data/rcv1/ & python preprocess_rcv1.py . & python data_rcv1.py`

## Train & Evaluation

``` shell
bash run_rcv1.sh

bash run_wos.sh

bash run_nyt.sh
```


Our Code is based on [s2s-ft](https://github.com/microsoft/unilm/tree/master/s2s-ft)

## Canonical datasets and cross-validation folds

This checkout can train directly from the canonical datasets in the sibling
`../datasets/` directory, without downloading or regenerating the legacy HBGL
source files. The source PKLs are read-only; HBGL materializes compatible JSONL
files per dataset and fold under `resource/prepared-datasets/`.

Validate all source folds before training:

```shell
python3 dataset_adapter.py validate --dataset-dir ../datasets/WOS-150-H2 --dataset-name WOS-150-H2
python3 dataset_adapter.py validate --dataset-dir ../datasets/RCV1-103-H3 --dataset-name RCV1-103-H3
```

Prepare one fold explicitly (normally `run.py` does this automatically):

```shell
python3 dataset_adapter.py prepare \
  --dataset-dir ../datasets/RCV1-103-H3 \
  --dataset-name RCV1-103-H3 \
  --fold 0 \
  --prepared-data-dir resource/prepared-datasets
```

Train a single isolated fold:

```shell
bash run_fold.sh WOS-150-H2 0 wos-fold-0
bash run_fold.sh RCV1-103-H3 0 rcv1-fold-0
```

The wrappers `run_wos.sh FOLD [RUN_NAME]` and `run_rcv1.sh FOLD [RUN_NAME]`
call the same runner. Both wrappers default to `PER_GPU_TRAIN_BATCH_SIZE=32`,
validated on the RTX A6000; override it explicitly when reproducing a different
batch configuration. Set `HBGL_WANDB=1` to enable Weights & Biases logging.
Use a distinct run name for every fold: outputs, prepared artifacts, and
feature caches are deliberately fold-scoped.

### Dataset contract

The adapter expects `samples.pkl` plus `fold_N/{train,val,test}.pkl` in each
source directory. Fold IDs refer to positional `idx` in `samples.pkl`, **not**
`text_idx` (the latter is an external evaluation identifier).

WOS labels are ordered paths and are used to reconstruct its hierarchy. RCV1
labels can be siblings in arbitrary document order, so its hierarchy is rebuilt
from `data/rcv1/rcv1.taxonomy` and the versioned topic-code map instead. The
prepared `label_map.pkl` uses stable source label IDs (`[A_0]`, `[A_1]`, ...),
which makes checkpoints reproducible across folds.

Prepared artifacts include a manifest. If the source data changes, regenerate
with `--force-prepare`; the adapter will not silently reuse incompatible cache.

### HBGL ranking report with the HGCLR metric protocol

The canonical adapter writes `<split>_document_ids.json` beside each JSONL. It
preserves positional `idx` for WOS and the external `text_idx` for RCV1; it
never uses `text_idx` to index `samples.pkl`.

When `EXPORT_RANKINGS=1` (the `run_fold.sh` default), HBGL writes a **dense
HBGL-only** ranking report for each best checkpoint. It does not modify
`s2s_ft/modeling_decoding.py`: the test runner attaches a temporary forward
hook to the existing `cls` classifier. For hierarchy level `h`, it takes the
classifier vector selected by HBGL's own `hier_labels[h]` mask and records
`sigmoid(logit)` for every label at that label's taxonomy depth. This follows
the score in Eq. 10 of the HBGL paper; it never max-pools logits across decode
steps and never includes the separately appended EOS candidate.

The resulting artifact is:

```python
{"text_<external-document-id>": {"label_<source-label-id>": probability}}
```

It contains every canonical label, so class-filtered tail/head reporting has a
complete candidate set. Its companion metadata and the HBGL metrics JSON are
written under:

```text
models/<RUN_NAME>/rankings/best_micro.rnk
models/<RUN_NAME>/rankings/best_micro.rnk.metrics.json
models/<RUN_NAME>/rankings/best_macro.rnk
models/<RUN_NAME>/rankings/best_macro.rnk.metrics.json
```

`evaluate_hbgl_ranking.py` is intentionally exclusive to HBGL artifacts. It
reproduces HGCLR's head/tail filtering, full-corpus inverse-propensity weights
(`A=0.55`, `B=1.5`), rounding, and metric names: `precision@K`, `ndcg@K`,
`psprecision@K`, `psnDCG@K`, `Mac-F1@K`, and `Mic-F1@K`, with `K=1,5,10` by
default. The original HBGL paper used a different reported metric set; this is
a separate requested ranking report using HGCLR's calculation protocol.

```shell
python3 evaluate_hbgl_ranking.py \
  --ranking-file models/<RUN_NAME>/rankings/best_micro.rnk \
  --dataset-dir ../datasets/RCV1-103-H3 \
  --dataset-name RCV1-103-H3 \
  --fold 0 \
  --output-file models/<RUN_NAME>/rankings/best_micro.rnk.metrics.json \
  --thresholds 1 5 10
```

The ranking route requires greedy hierarchical soft-label decoding
(`--soft_label --soft_label_hier_real`) and one GPU. It validates that the
prepared taxonomy and HBGL's live hierarchy masks select exactly the same
labels at each level, and that ranking coverage equals the canonical test fold.
Set `EXPORT_RANKINGS=0` only when this report is intentionally not needed.

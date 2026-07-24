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
call the same runner. `run_rcv1.sh` defaults to `PER_GPU_TRAIN_BATCH_SIZE=32`,
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

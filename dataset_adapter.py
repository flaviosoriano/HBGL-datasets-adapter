"""Adapt canonical PKL dataset folds to the JSONL contract consumed by HBGL.

The source datasets are read-only. Prepared files are scoped by dataset and
fold, so feature caches and targets cannot be accidentally reused across folds.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import pickle
import shutil
import tempfile
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ARTIFACT_VERSION = 4
REQUIRED_SPLITS = ("train", "val", "test")
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RCV1_TAXONOMY = REPO_ROOT / "data" / "rcv1" / "rcv1.taxonomy"
DEFAULT_RCV1_TOPIC_CODES = REPO_ROOT / "data" / "rcv1" / "rcv1_topic_codes.json"


class DatasetValidationError(ValueError):
    """Raised when a canonical dataset does not satisfy HBGL adapter invariants."""


@dataclass(frozen=True)
class PreparedFold:
    path: Path
    reused: bool


def normalize_label(value: Any) -> str:
    if not isinstance(value, str):
        raise DatasetValidationError(f"label must be a string, received {type(value).__name__}")
    normalized = " ".join(value.split())
    if not normalized:
        raise DatasetValidationError("label cannot be empty")
    return normalized


def load_samples(dataset_dir: Path) -> list[dict[str, Any]]:
    path = dataset_dir / "samples.pkl"
    if not path.is_file():
        raise DatasetValidationError(f"missing samples file: {path}")
    with path.open("rb") as handle:
        samples = pickle.load(handle)
    if not isinstance(samples, list):
        raise DatasetValidationError(f"{path} must contain a list, got {type(samples).__name__}")
    return samples


def load_fold_ids(dataset_dir: Path, fold: int) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for split in REQUIRED_SPLITS:
        path = dataset_dir / f"fold_{fold}" / f"{split}.pkl"
        if not path.is_file():
            raise DatasetValidationError(f"missing fold {fold} {split} split: {path}")
        with path.open("rb") as handle:
            values = pickle.load(handle)
        if not isinstance(values, (list, tuple)):
            raise DatasetValidationError(
                f"fold {fold} {split} split must be a list or tuple, got {type(values).__name__}"
            )
        result[split] = list(values)
    return result


def build_source_id_to_label(samples: Iterable[dict[str, Any]]) -> dict[int, str]:
    id_to_label: dict[int, str] = {}
    for row, sample in enumerate(samples):
        try:
            ids = sample["labels_ids"]
            labels = sample["labels"]
        except (KeyError, TypeError) as error:
            raise DatasetValidationError(f"sample row {row} lacks labels_ids or labels") from error
        if not isinstance(ids, list) or not isinstance(labels, list) or len(ids) != len(labels):
            raise DatasetValidationError(f"sample row {row} has mismatched labels_ids/labels")
        for label_id, label in zip(ids, labels):
            if not isinstance(label_id, int):
                raise DatasetValidationError(f"sample row {row} label ID {label_id!r} is not an integer")
            canonical_label = normalize_label(label)
            existing = id_to_label.setdefault(label_id, canonical_label)
            if existing != canonical_label:
                raise DatasetValidationError(
                    f"label ID {label_id} maps to both {existing!r} and {canonical_label!r}"
                )
    return dict(sorted(id_to_label.items()))


def validate_dataset(
    samples: list[dict[str, Any]],
    split_ids: dict[str, list[int]],
    dataset_name: str,
    fold: int,
) -> None:
    if not samples:
        raise DatasetValidationError(f"{dataset_name} fold {fold} has no samples")
    if set(split_ids) != set(REQUIRED_SPLITS):
        raise DatasetValidationError(
            f"{dataset_name} fold {fold} requires splits {REQUIRED_SPLITS}, got {sorted(split_ids)}"
        )

    for row, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise DatasetValidationError(f"{dataset_name} sample row {row} is not a mapping")
        required = {"idx", "text", "labels_ids", "labels"}
        missing = required - set(sample)
        if missing:
            raise DatasetValidationError(f"{dataset_name} sample row {row} lacks {sorted(missing)}")
        if sample["idx"] != row:
            raise DatasetValidationError(
                f"{dataset_name} sample row {row} declares idx={sample['idx']!r}; samples must be positional"
            )
        if not isinstance(sample["text"], str):
            raise DatasetValidationError(f"{dataset_name} sample row {row} text is not a string")

    id_to_label = build_source_id_to_label(samples)
    if not id_to_label:
        raise DatasetValidationError(f"{dataset_name} has no labels")

    sets: dict[str, set[int]] = {}
    valid_indices = set(range(len(samples)))
    for split, indices in split_ids.items():
        if any(not isinstance(index, int) for index in indices):
            raise DatasetValidationError(f"{dataset_name} fold {fold} {split} contains a non-integer index")
        index_set = set(indices)
        if len(index_set) != len(indices):
            raise DatasetValidationError(f"{dataset_name} fold {fold} {split} contains duplicate indices")
        unknown = index_set - valid_indices
        if unknown:
            raise DatasetValidationError(
                f"{dataset_name} fold {fold} {split} contains invalid sample indices: {sorted(unknown)[:5]}"
            )
        sets[split] = index_set

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if overlap := sets[left] & sets[right]:
            raise DatasetValidationError(
                f"{dataset_name} fold {fold} splits overlap: {left}/{right}, examples={sorted(overlap)[:5]}"
            )
    covered = set().union(*sets.values())
    if covered != valid_indices:
        missing = valid_indices - covered
        extra = covered - valid_indices
        raise DatasetValidationError(
            f"{dataset_name} fold {fold} splits do not cover samples; "
            f"missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}"
        )


def _add_edge(taxonomy: dict[str, list[str]], parent: str, child: str) -> None:
    children = taxonomy.setdefault(parent, [])
    if child not in children:
        children.append(child)


def _validate_tree(taxonomy: dict[str, list[str]], labels: Iterable[str]) -> None:
    labels = set(labels)
    if "Root" not in taxonomy:
        raise DatasetValidationError("taxonomy does not define Root")
    attached: set[str] = set()
    for parent, children in taxonomy.items():
        for child in children:
            if child == "Root":
                raise DatasetValidationError("taxonomy cannot attach Root as a child")
            attached.add(child)
    if attached != labels:
        raise DatasetValidationError(
            "taxonomy does not attach every observed label; "
            f"missing={sorted(labels - attached)[:5]}"
        )

    visited: set[str] = set()
    active: set[str] = set()

    def visit(node: str) -> None:
        if node in active:
            raise DatasetValidationError(f"taxonomy contains a cycle at {node!r}")
        if node in visited:
            return
        active.add(node)
        for child in taxonomy.get(node, []):
            visit(child)
        active.remove(node)
        visited.add(node)

    visit("Root")
    if labels - visited:
        raise DatasetValidationError(f"taxonomy has disconnected labels: {sorted(labels - visited)[:5]}")


def compute_depths(taxonomy: dict[str, list[str]]) -> dict[str, int]:
    """Return the shortest root distance for every label in a hierarchy DAG."""
    depths: dict[str, int] = {}
    queue = deque([("Root", -1)])
    while queue:
        node, depth = queue.popleft()
        for child in taxonomy.get(node, []):
            candidate = depth + 1
            previous = depths.get(child)
            if previous is None or candidate < previous:
                depths[child] = candidate
                queue.append((child, candidate))
    return depths


def build_wos_taxonomy(samples: list[dict[str, Any]]) -> tuple[dict[str, list[str]], dict[int, str]]:
    id_to_label = build_source_id_to_label(samples)
    taxonomy: dict[str, list[str]] = {"Root": []}
    for row, sample in enumerate(samples):
        labels = [normalize_label(label) for label in sample["labels"]]
        if not labels:
            raise DatasetValidationError(f"WOS sample row {row} has no labels")
        _add_edge(taxonomy, "Root", labels[0])
        for parent, child in zip(labels, labels[1:]):
            _add_edge(taxonomy, parent, child)
    _validate_tree(taxonomy, id_to_label.values())
    return taxonomy, id_to_label


def load_code_taxonomy(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        raise DatasetValidationError(f"missing RCV1 taxonomy: {path}")
    taxonomy: dict[str, list[str]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = [part.strip() for part in line.split("\t") if part.strip()]
        if not parts:
            continue
        if len(parts) < 2:
            raise DatasetValidationError(f"{path}:{line_number} has no children")
        parent, *children = parts
        if parent in taxonomy:
            raise DatasetValidationError(f"duplicate RCV1 taxonomy parent {parent!r}")
        taxonomy[parent] = children
    if "Root" not in taxonomy:
        raise DatasetValidationError(f"{path} does not define Root")
    return taxonomy


def build_rcv1_taxonomy(
    samples: list[dict[str, Any]],
    code_taxonomy: dict[str, list[str]],
    code_to_label: dict[str, str],
) -> tuple[dict[str, list[str]], dict[int, str], list[dict[str, str]]]:
    id_to_label = build_source_id_to_label(samples)
    labels = set(id_to_label.values())
    label_to_code: dict[str, str] = {}
    for code, label in code_to_label.items():
        canonical_label = normalize_label(label)
        if canonical_label in label_to_code:
            raise DatasetValidationError(f"two RCV1 codes resolve to label {canonical_label!r}")
        label_to_code[canonical_label] = code
    missing_codes = sorted(labels - set(label_to_code))
    if missing_codes:
        raise DatasetValidationError(f"RCV1 labels missing topic codes: {missing_codes[:5]}")

    code_to_label_present = {
        label_to_code[label]: label for label in labels
    }
    known_codes = set(code_taxonomy)
    known_codes.update(child for children in code_taxonomy.values() for child in children)
    absent = sorted(set(code_to_label_present) - known_codes)
    if absent:
        raise DatasetValidationError(f"RCV1 labels absent from official taxonomy: {absent[:5]}")

    taxonomy: dict[str, list[str]] = {"Root": []}
    fallbacks: list[dict[str, str]] = []
    for parent_code, child_codes in code_taxonomy.items():
        if parent_code == "Root":
            destination = "Root"
        elif parent_code in code_to_label_present:
            destination = code_to_label_present[parent_code]
        else:
            destination = "Root"
            if any(child in code_to_label_present for child in child_codes):
                fallbacks.append({"missing_parent_code": parent_code, "attached_to": "Root"})
        for child_code in child_codes:
            child = code_to_label_present.get(child_code)
            if child is not None:
                _add_edge(taxonomy, destination, child)

    _validate_tree(taxonomy, labels)
    return taxonomy, id_to_label, fallbacks


def _source_manifest(dataset_dir: Path, dataset_name: str, fold: int) -> dict[str, Any]:
    files = [dataset_dir / "samples.pkl"] + [
        dataset_dir / f"fold_{fold}" / f"{split}.pkl" for split in REQUIRED_SPLITS
    ]
    metadata = []
    for path in files:
        if not path.is_file():
            raise DatasetValidationError(f"missing source artifact: {path}")
        stat = path.stat()
        metadata.append({"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "dataset_name": dataset_name,
        "dataset_dir": str(dataset_dir.resolve()),
        "fold": fold,
        "artifacts": metadata,
        "fingerprint": hashlib.sha256(encoded).hexdigest(),
    }


def _load_existing_manifest(path: Path) -> dict[str, Any] | None:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DatasetValidationError(f"prepared manifest is not valid JSON: {manifest_path}") from error


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def _write_taxonomy(path: Path, taxonomy: dict[str, list[str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for parent, children in taxonomy.items():
            if children:
                handle.write("\t".join([parent, *children]))
                handle.write("\n")


def sanitize_text(value: str) -> str:
    """Turn surrogate-escaped source bytes into valid Unicode for JSON/tokenizers."""
    return "".join(
        chr(ord(character) - 0xDC00)
        if 0xDC80 <= ord(character) <= 0xDCFF
        else character
        for character in value
    )


def build_split_document_ids(
    samples: list[dict[str, Any]], indices: Iterable[int], dataset_name: str
) -> dict[str, Any]:
    """Return evaluation IDs aligned to a prepared split without reindexing samples.

    Canonical folds always contain positional ``idx`` values. RCV1 ranking
    metadata is keyed by the external ``text_idx`` instead, so this mapping is
    deliberately a sidecar rather than an indexing mechanism.
    """
    if dataset_name == "RCV1-103-H3":
        id_kind = "text_idx"
    elif dataset_name == "WOS-150-H2":
        id_kind = "idx"
    else:
        raise DatasetValidationError(f"unsupported dataset name: {dataset_name!r}")

    document_ids: list[int] = []
    for index in indices:
        sample = samples[index]
        try:
            document_id = sample[id_kind]
        except KeyError as error:
            raise DatasetValidationError(
                f"{dataset_name} sample idx={index} lacks required evaluation ID {id_kind!r}"
            ) from error
        if not isinstance(document_id, int):
            raise DatasetValidationError(
                f"{dataset_name} sample idx={index} has non-integer {id_kind}={document_id!r}"
            )
        document_ids.append(document_id)
    return {"id_kind": id_kind, "ids": document_ids}


def build_corpus_label_counts(samples: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Count source label IDs across the immutable canonical corpus."""
    counts: Counter[int] = Counter()
    for sample in samples:
        labels = sample.get("labels_ids")
        if not isinstance(labels, list) or not all(isinstance(label_id, int) for label_id in labels):
            raise DatasetValidationError("sample has invalid labels_ids while building corpus counts")
        counts.update(labels)
    return {str(label_id): count for label_id, count in sorted(counts.items())}


def _sample_labels(sample: dict[str, Any], id_to_label: dict[int, str]) -> list[tuple[int, str]]:
    pairs: list[tuple[int, str]] = []
    seen: set[int] = set()
    for label_id, label in zip(sample["labels_ids"], sample["labels"]):
        canonical_label = normalize_label(label)
        if id_to_label.get(label_id) != canonical_label:
            raise DatasetValidationError(f"sample {sample['idx']} has inconsistent label mapping for {label_id}")
        if label_id not in seen:
            seen.add(label_id)
            pairs.append((label_id, canonical_label))
    return pairs


def _make_rows(
    samples: list[dict[str, Any]],
    indices: Iterable[int],
    split: str,
    id_to_label: dict[int, str],
    label_map: dict[str, str],
    depths: dict[str, int],
) -> Iterable[dict[str, Any]]:
    max_depth = max(depths.values())
    for index in indices:
        sample = samples[index]
        labels = _sample_labels(sample, id_to_label)
        labels.sort(key=lambda pair: (depths[pair[1]], pair[0]))
        tokens = [label_map[label] for _, label in labels]
        # HBGL's tokenizer receives ``src`` directly and requires text, not a
        # pre-tokenized JSON list.  Preserve whitespace normalization here.
        row: dict[str, Any] = {"src": " ".join(sanitize_text(sample["text"]).split())}
        if split == "train":
            target = [[] for _ in range(max_depth + 1)]
            for (_, label), token in zip(labels, tokens):
                target[depths[label]].append(token)
            row["tgt"] = target
        else:
            row["tgt"] = " ".join(tokens)
        yield row


def prepare_fold(
    dataset_dir: Path,
    dataset_name: str,
    fold: int,
    prepared_data_dir: Path,
    *,
    force: bool = False,
    rcv1_taxonomy_path: Path = DEFAULT_RCV1_TAXONOMY,
    rcv1_topic_codes_path: Path = DEFAULT_RCV1_TOPIC_CODES,
) -> PreparedFold:
    dataset_dir = Path(dataset_dir)
    prepared_data_dir = Path(prepared_data_dir)
    source = _source_manifest(dataset_dir, dataset_name, fold)
    target = prepared_data_dir / dataset_name / f"fold_{fold}"
    existing = _load_existing_manifest(target) if target.exists() else None
    if (
        existing
        and not force
        and existing.get("artifact_version") == ARTIFACT_VERSION
        and existing.get("source") == source
    ):
        return PreparedFold(target, reused=True)
    if target.exists() and not force:
        raise DatasetValidationError(
            f"prepared fold exists with a different manifest: {target}; pass force=True to replace it"
        )

    samples = load_samples(dataset_dir)
    split_ids = load_fold_ids(dataset_dir, fold)
    validate_dataset(samples, split_ids, dataset_name, fold)
    if dataset_name == "WOS-150-H2":
        taxonomy, id_to_label = build_wos_taxonomy(samples)
        fallbacks: list[dict[str, str]] = []
        taxonomy_source = "ordered sample label paths"
    elif dataset_name == "RCV1-103-H3":
        code_taxonomy = load_code_taxonomy(Path(rcv1_taxonomy_path))
        try:
            code_to_label = json.loads(Path(rcv1_topic_codes_path).read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise DatasetValidationError(f"missing RCV1 topic-code map: {rcv1_topic_codes_path}") from error
        taxonomy, id_to_label, fallbacks = build_rcv1_taxonomy(samples, code_taxonomy, code_to_label)
        taxonomy_source = str(Path(rcv1_taxonomy_path).resolve())
    else:
        raise DatasetValidationError(f"unsupported dataset name: {dataset_name!r}")

    depths = compute_depths(taxonomy)
    label_map = {label: f"[A_{label_id}]" for label_id, label in id_to_label.items()}
    target_parent = target.parent
    target_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target_parent))
    try:
        for split in REQUIRED_SPLITS:
            _write_jsonl(
                temporary / f"{split}.jsonl",
                _make_rows(samples, split_ids[split], split, id_to_label, label_map, depths),
            )
            (temporary / f"{split}_document_ids.json").write_text(
                json.dumps(build_split_document_ids(samples, split_ids[split], dataset_name), sort_keys=True) + "\n",
                encoding="utf-8",
            )
        (temporary / "corpus_label_counts.json").write_text(
            json.dumps({"documents": len(samples), "label_counts": build_corpus_label_counts(samples)}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with (temporary / "label_map.pkl").open("wb") as handle:
            pickle.dump(label_map, handle, protocol=4)
        _write_taxonomy(temporary / "label_taxonomy.tsv", taxonomy)
        manifest = {
            "artifact_version": ARTIFACT_VERSION,
            "source": source,
            "counts": {split: len(split_ids[split]) for split in REQUIRED_SPLITS},
            "labels": len(label_map),
            "max_depth": max(depths.values()),
            "taxonomy_source": taxonomy_source,
            "taxonomy_fallbacks": fallbacks,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if target.exists():
            backup = target.with_name(f".{target.name}.backup-{os.getpid()}")
            target.replace(backup)
            try:
                temporary.replace(target)
            except Exception:
                backup.replace(target)
                raise
            shutil.rmtree(backup)
        else:
            temporary.replace(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return PreparedFold(target, reused=False)


def _available_folds(dataset_dir: Path) -> list[int]:
    folds = []
    for path in dataset_dir.glob("fold_*"):
        if path.is_dir():
            try:
                folds.append(int(path.name[len("fold_"):]))
            except ValueError:
                continue
    return sorted(folds)


def validate_command(dataset_dir: Path, dataset_name: str) -> None:
    samples = load_samples(dataset_dir)
    id_to_label = build_source_id_to_label(samples)
    folds = _available_folds(dataset_dir)
    if not folds:
        raise DatasetValidationError(f"no fold_* directories in {dataset_dir}")
    for fold in folds:
        validate_dataset(samples, load_fold_ids(dataset_dir, fold), dataset_name, fold)
        print(f"{dataset_name} fold {fold}: valid")
    print(f"{dataset_name}: samples={len(samples)}, labels={len(id_to_label)}, folds={folds}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "prepare"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--dataset-dir", type=Path, required=True)
        subparser.add_argument("--dataset-name", choices=("WOS-150-H2", "RCV1-103-H3"), required=True)
        if command == "prepare":
            subparser.add_argument("--fold", type=int, required=True)
            subparser.add_argument("--prepared-data-dir", type=Path, default=Path("resource/prepared-datasets"))
            subparser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "validate":
        validate_command(args.dataset_dir, args.dataset_name)
    else:
        prepared = prepare_fold(
            args.dataset_dir,
            args.dataset_name,
            args.fold,
            args.prepared_data_dir,
            force=args.force,
        )
        state = "reused" if prepared.reused else "prepared"
        print(f"{state}: {prepared.path}")


if __name__ == "__main__":
    main()

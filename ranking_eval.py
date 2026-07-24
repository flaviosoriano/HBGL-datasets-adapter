"""Portable document-to-label ranking metrics for HGCLR/HBGL comparison.

The artifact schema intentionally matches HGCLR's pickle rankings:
``{"text_<external-id>": {"label_<source-id>": score}}``.
Metrics are calculated from the same document-label relevance relation for both
models, rather than from model-specific thresholded predictions.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, Mapping, Sequence, Set


def _label_id(label_key: str) -> int:
    prefix, separator, value = label_key.rpartition("_")
    if prefix != "label" or not separator:
        raise ValueError("invalid ranking label key: {!r}".format(label_key))
    try:
        return int(value)
    except ValueError as error:
        raise ValueError("invalid ranking label key: {!r}".format(label_key)) from error


def aggregate_label_scores(label_ids: Sequence[int], decode_steps: Sequence[Sequence[float]]) -> Dict[str, float]:
    """Collapse per-step decoder logits to one dense score per source label.

    A label can be considered at several generated positions. Taking its maximum
    raw logit retains the strongest model evidence while preserving all labels
    needed for top-k retrieval metrics.
    """
    if not label_ids:
        raise ValueError("label_ids cannot be empty")
    if len(set(label_ids)) != len(label_ids):
        raise ValueError("label_ids must be unique")
    if not decode_steps:
        raise ValueError("decode_steps cannot be empty")
    best = [float("-inf")] * len(label_ids)
    for step in decode_steps:
        if len(step) != len(label_ids):
            raise ValueError("decoder score width does not match label_ids")
        for position, score in enumerate(step):
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError("decoder score must be finite")
            best[position] = max(best[position], float(score))
    return {"label_{}".format(label_id): best[position] for position, label_id in enumerate(label_ids)}


def ordered_label_ids(label_scores: Mapping[str, float]) -> list[int]:
    """Return labels in descending score order with deterministic ID tie-breaking."""
    parsed = []
    for label_key, score in label_scores.items():
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("ranking score for {!r} must be finite".format(label_key))
        parsed.append((_label_id(label_key), float(score)))
    return [label_id for label_id, _ in sorted(parsed, key=lambda pair: (-pair[1], pair[0]))]


def inverse_propensity_weights(
    label_counts: Mapping[int, int], total_documents: int, a: float = 0.55, b: float = 1.5
) -> Dict[int, float]:
    """Compute the standard inverse-propensity weights used in XMLC metrics."""
    if total_documents <= 1:
        raise ValueError("total_documents must be greater than one")
    if a <= 0 or b < 0:
        raise ValueError("propensity parameters require a > 0 and b >= 0")
    constant = (math.log(total_documents) - 1.0) * math.pow(b + 1.0, a)
    weights = {}
    for label_id, count in label_counts.items():
        if not isinstance(count, int) or count <= 0:
            raise ValueError("label {!r} has invalid positive count {!r}".format(label_id, count))
        weights[label_id] = 1.0 + constant * math.pow(count + b, -a)
    return weights


def _discount(rank: int) -> float:
    return 1.0 / math.log2(rank + 1.0)


def _top_k(values: Sequence[int], cutoff: int) -> list[int]:
    return list(values[:cutoff])


def build_test_relevance(
    samples: Sequence[Mapping[str, int]],
    test_indices: Iterable[int],
    relevance_map: Mapping[int, Iterable[int]],
    dataset_name: str,
) -> Dict[str, Set[int]]:
    """Build HGCLR-format test qrels using the canonical identity for each dataset."""
    use_text_idx = dataset_name == "RCV1-103-H3"
    result = {}
    for positional_idx in test_indices:
        if not isinstance(positional_idx, int) or positional_idx < 0 or positional_idx >= len(samples):
            raise ValueError("invalid positional test index: {!r}".format(positional_idx))
        sample = samples[positional_idx]
        document_id = sample.get("text_idx") if use_text_idx else sample.get("idx")
        if not isinstance(document_id, int):
            field = "text_idx" if use_text_idx else "idx"
            raise ValueError("sample {} has invalid {}".format(positional_idx, field))
        key = "text_{}".format(document_id)
        if key in result:
            raise ValueError("duplicate external document ID in test split: {}".format(key))
        if document_id not in relevance_map:
            raise ValueError("test document {} absent from canonical relevance map".format(key))
        labels = set(relevance_map[document_id])
        if not labels:
            raise ValueError("test document {} has no relevance labels".format(key))
        result[key] = labels
    return result


def evaluate_rankings(
    rankings: Mapping[str, Mapping[str, float]],
    relevance: Mapping[str, Set[int]],
    *,
    label_counts: Mapping[int, int],
    total_documents: int,
    cutoffs: Iterable[int] = (1, 5, 10),
    propensity_a: float = 0.55,
    propensity_b: float = 1.5,
) -> Dict[str, float]:
    """Evaluate rankings with common P/nDCG and propensity-scored counterparts.

    Cutoffs are not silently clipped: a ranking shorter than ``k`` contributes
    zero-relevance placeholders after its final scored label, as in a top-k
    retrieval result. Every predicted document must have canonical relevance.
    """
    cutoffs = tuple(cutoffs)
    if not rankings:
        raise ValueError("rankings cannot be empty")
    if not cutoffs or any(not isinstance(cutoff, int) or cutoff <= 0 for cutoff in cutoffs):
        raise ValueError("cutoffs must be positive integers")

    missing_relevance = sorted(set(rankings) - set(relevance))
    if missing_relevance:
        raise ValueError("missing relevance for ranking documents: {}".format(missing_relevance[:5]))

    weights = inverse_propensity_weights(
        label_counts, total_documents, a=propensity_a, b=propensity_b
    )
    aggregates = {"precision": {}, "ndcg": {}, "psprecision": {}, "psnDCG": {}}

    for cutoff in cutoffs:
        precision_values = []
        ndcg_values = []
        psprecision_values = []
        psndcg_values = []
        for document_id, label_scores in rankings.items():
            gold = set(relevance[document_id])
            if not gold:
                raise ValueError("document {!r} has no relevance labels".format(document_id))
            unknown = gold - set(weights)
            if unknown:
                raise ValueError("document {!r} uses labels absent from label_counts: {}".format(document_id, sorted(unknown)[:5]))
            ranked = _top_k(ordered_label_ids(label_scores), cutoff)
            hits = [label_id in gold for label_id in ranked]
            precision_values.append(float(sum(hits)) / cutoff)

            dcg = sum((1.0 if hit else 0.0) * _discount(rank) for rank, hit in enumerate(hits, start=1))
            ideal_count = min(len(gold), cutoff)
            idcg = sum(_discount(rank) for rank in range(1, ideal_count + 1))
            ndcg_values.append(dcg / idcg)

            weighted_hits = [weights[label_id] if label_id in gold else 0.0 for label_id in ranked]
            ps_precision_denominator = sum(sorted((weights[label_id] for label_id in gold), reverse=True)[:cutoff])
            psprecision_values.append(sum(weighted_hits) / ps_precision_denominator)
            ps_dcg = sum(weight * _discount(rank) for rank, weight in enumerate(weighted_hits, start=1))
            ps_idcg = sum(
                weight * _discount(rank)
                for rank, weight in enumerate(
                    sorted((weights[label_id] for label_id in gold), reverse=True)[:cutoff], start=1
                )
            )
            psndcg_values.append(ps_dcg / ps_idcg)

        aggregates["precision"][cutoff] = sum(precision_values) / len(precision_values)
        aggregates["ndcg"][cutoff] = sum(ndcg_values) / len(ndcg_values)
        aggregates["psprecision"][cutoff] = sum(psprecision_values) / len(psprecision_values)
        aggregates["psnDCG"][cutoff] = sum(psndcg_values) / len(psndcg_values)

    result = {}
    for metric_name, values in aggregates.items():
        for cutoff, value in values.items():
            result["{}@{}".format(metric_name, cutoff)] = value
    return result


def label_counts_from_relevance(relevance: Mapping[str, Set[int]]) -> Dict[int, int]:
    """Count positive labels over the canonical corpus or selected reference set."""
    counts = Counter()
    for labels in relevance.values():
        counts.update(labels)
    return dict(counts)

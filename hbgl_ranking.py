"""HBGL-only ranking scores and the HGCLR evaluation protocol.

The score assigned to a label is the sigmoid probability produced at that
label's own hierarchy level (HBGL paper, Eq. 10).  This module deliberately
does not evaluate HGCLR artifacts; it evaluates HBGL rankings using HGCLR's
published project protocol and output names.
"""

from __future__ import annotations

import math
from collections import deque


def _label_id(label_key):
    prefix, separator, value = label_key.rpartition("_")
    if prefix != "label" or not separator:
        raise ValueError("invalid label key: {!r}".format(label_key))
    try:
        return int(value)
    except ValueError as error:
        raise ValueError("invalid label key: {!r}".format(label_key)) from error


def label_ids_by_depth_from_taxonomy(label_map, taxonomy_text):
    """Map canonical source label IDs to the decoder position for their depth.

    The prepared taxonomy uses the same Root=-1 convention as
    ``dataset_adapter.compute_depths``: direct children of Root are decoded at
    position zero.
    """
    taxonomy = {}
    for line_number, line in enumerate(taxonomy_text.splitlines(), 1):
        parts = [part.strip() for part in line.split("\t") if part.strip()]
        if not parts:
            continue
        if len(parts) < 2:
            raise ValueError("taxonomy line {} has no children".format(line_number))
        parent, children = parts[0], parts[1:]
        if parent in taxonomy:
            raise ValueError("taxonomy has duplicate parent {!r}".format(parent))
        taxonomy[parent] = children
    if "Root" not in taxonomy:
        raise ValueError("taxonomy does not define Root")

    depths = {}
    queue = deque([("Root", -1)])
    while queue:
        parent, depth = queue.popleft()
        for child in taxonomy.get(parent, ()):
            candidate = depth + 1
            prior = depths.get(child)
            if prior is None or candidate < prior:
                depths[child] = candidate
                queue.append((child, candidate))

    by_depth = {}
    for label_name, token in label_map.items():
        if label_name not in depths:
            raise ValueError("label {!r} is absent from taxonomy".format(label_name))
        if not isinstance(token, str) or not token.startswith("[A_") or not token.endswith("]"):
            raise ValueError("label map token is not canonical: {!r}".format(token))
        label_id = int(token[3:-1])
        by_depth.setdefault(depths[label_name], []).append(label_id)
    if not by_depth:
        raise ValueError("label map cannot be empty")
    maximum_depth = max(by_depth)
    levels = []
    for depth in range(maximum_depth + 1):
        if depth not in by_depth:
            raise ValueError("taxonomy has no exported labels at depth {}".format(depth))
        levels.append(sorted(by_depth[depth]))
    return levels


def build_document_label_scores(level_label_ids, level_probabilities):
    """Map one Eq.-10 probability vector per level to HGCLR ranking keys."""
    if len(level_label_ids) != len(level_probabilities):
        raise ValueError("score levels do not match hierarchy levels")
    scores = {}
    for label_ids, probabilities in zip(level_label_ids, level_probabilities):
        if len(label_ids) != len(probabilities):
            raise ValueError("level probability width does not match its labels")
        for label_id, probability in zip(label_ids, probabilities):
            if not isinstance(label_id, int):
                raise ValueError("source label IDs must be integers")
            if not isinstance(probability, (int, float)) or not math.isfinite(probability):
                raise ValueError("label probabilities must be finite")
            key = "label_{}".format(label_id)
            if key in scores:
                raise ValueError("a source label belongs to multiple hierarchy levels: {}".format(key))
            scores[key] = float(probability)
    if not scores:
        raise ValueError("cannot build a ranking without labels")
    return scores


def inverse_propensity_weights(label_counts, corpus_documents, a=0.55, b=1.5):
    """Use the exact A/B propensity equation used by HGCLR's EvalHelper."""
    if corpus_documents <= 1:
        raise ValueError("corpus_documents must be greater than one")
    constant = (math.log(corpus_documents) - 1.0) * math.pow(b + 1.0, a)
    result = {}
    for label_id, count in label_counts.items():
        if not isinstance(label_id, int) or not isinstance(count, int) or count <= 0:
            raise ValueError("label_counts must map integer labels to positive counts")
        result[label_id] = 1.0 + constant * math.pow(count + b, -a)
    return result


def _ordered_labels(label_scores):
    parsed = []
    for key, score in label_scores.items():
        label_id = _label_id(key)
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("ranking scores must be finite")
        parsed.append((label_id, float(score)))
    return [label_id for label_id, _ in sorted(parsed, key=lambda pair: (-pair[1], pair[0]))]


def _discount(rank):
    return 1.0 / math.log2(rank + 1.0)


def _mean(values):
    if not values:
        raise ValueError("HGCLR class filter produced no documents")
    return sum(values) / float(len(values))


def _round_percent(value):
    return round(100.0 * value, 1)


def _as_label_ids(relevance):
    return [_label_id(label_key) for label_key in relevance]


def _precision_and_ndcg(class_ranking, relevance, thresholds):
    result = {}
    for cutoff in thresholds:
        precisions = []
        ndcgs = []
        for document_id, label_scores in class_ranking.items():
            gold = set(_as_label_ids(relevance[document_id]))
            ranked = _ordered_labels(label_scores)[:cutoff]
            hits = [label_id in gold for label_id in ranked]
            precisions.append(float(sum(hits)) / cutoff)
            dcg = sum(_discount(rank) for rank, hit in enumerate(hits, 1) if hit)
            ideal = sum(_discount(rank) for rank in range(1, min(len(gold), cutoff) + 1))
            ndcgs.append(dcg / ideal)
        result["precision@{}".format(cutoff)] = _round_percent(_mean(precisions))
        result["ndcg@{}".format(cutoff)] = _round_percent(_mean(ndcgs))
    return result


def _ps_metrics(class_ranking, relevance, weights, thresholds):
    """Port EvalHelper.psprecision/psndcg with its aggregate-before-ratio semantics."""
    result = {}
    max_k = max(thresholds)
    predicted_prefix_precision = [[] for _ in range(max_k)]
    ideal_prefix_precision = [[] for _ in range(max_k)]
    predicted_ndcg = [[] for _ in range(max_k)]
    ideal_ndcg = [[] for _ in range(max_k)]

    for document_id, label_scores in class_ranking.items():
        gold = _as_label_ids(relevance[document_id])
        if not gold:
            raise ValueError("document {} has no relevance labels".format(document_id))
        if any(label_id not in weights for label_id in gold):
            raise ValueError("relevance contains a label without a corpus count")
        ranked = _ordered_labels(label_scores)[:max_k]
        predicted_weights = [weights[label_id] if label_id in gold else 0.0 for label_id in ranked]
        ideal_weights = sorted((weights[label_id] for label_id in gold), reverse=True)[:max_k]
        uniform_ideal = sum(_discount(rank) for rank in range(1, len(gold) + 1))
        running_predicted_weight = 0.0
        running_ideal_weight = 0.0
        running_predicted_dcg = 0.0
        running_ideal_dcg = 0.0
        for index in range(max_k):
            rank = index + 1
            predicted_weight = predicted_weights[index] if index < len(predicted_weights) else 0.0
            ideal_weight = ideal_weights[index] if index < len(ideal_weights) else 0.0
            running_predicted_weight += predicted_weight
            running_ideal_weight += ideal_weight
            predicted_prefix_precision[index].append(running_predicted_weight / rank)
            ideal_prefix_precision[index].append(running_ideal_weight / rank)
            running_predicted_dcg += predicted_weight * _discount(rank)
            running_ideal_dcg += ideal_weight * _discount(rank)
            normalizer = min(uniform_ideal, sum(_discount(position) for position in range(1, rank + 1)))
            predicted_ndcg[index].append(running_predicted_dcg / normalizer)
            ideal_ndcg[index].append(running_ideal_dcg / normalizer)

    for cutoff in thresholds:
        index = cutoff - 1
        result["psprecision@{}".format(cutoff)] = _round_percent(
            _mean(predicted_prefix_precision[index]) / _mean(ideal_prefix_precision[index])
        )
        result["psnDCG@{}".format(cutoff)] = _round_percent(
            _mean(predicted_ndcg[index]) / _mean(ideal_ndcg[index])
        )
    return result


def _f1(true_labels, predicted_labels):
    classes = sorted(set(true_labels) | set(predicted_labels))
    macro = []
    true_positive = false_positive = false_negative = 0
    for label in classes:
        tp = sum(true == label and predicted == label for true, predicted in zip(true_labels, predicted_labels))
        fp = sum(true != label and predicted == label for true, predicted in zip(true_labels, predicted_labels))
        fn = sum(true == label and predicted != label for true, predicted in zip(true_labels, predicted_labels))
        macro.append(0.0 if (2 * tp + fp + fn) == 0 else (2.0 * tp) / (2 * tp + fp + fn))
        true_positive += tp
        false_positive += fp
        false_negative += fn
    micro_denominator = 2 * true_positive + false_positive + false_negative
    micro = 0.0 if micro_denominator == 0 else (2.0 * true_positive) / micro_denominator
    return _mean(macro), micro


def _classification_metrics(class_ranking, relevance, thresholds):
    result = {}
    for cutoff in thresholds:
        true_labels = []
        predicted_labels = []
        for document_id, label_scores in class_ranking.items():
            top_labels = _ordered_labels(label_scores)[:cutoff]
            if not top_labels:
                raise ValueError("class ranking has no candidate labels for {}".format(document_id))
            relevant_labels = _as_label_ids(relevance[document_id])[:1]
            if not relevant_labels:
                raise ValueError("document {} has no relevance labels".format(document_id))
            target = relevant_labels[0]
            true_labels.append(target)
            predicted_labels.append(target if target in top_labels else top_labels[0])
        macro, micro = _f1(true_labels, predicted_labels)
        # EvalHelper rounds F1 to two decimals before converting to a percent.
        result["Mac-F1@{}".format(cutoff)] = round(100.0 * round(macro, 2), 1)
        result["Mic-F1@{}".format(cutoff)] = round(100.0 * round(micro, 2), 1)
    return result


def _class_ranking(ranking, label_classes, text_classes, label_class):
    result = {}
    for document_key, scores in ranking.items():
        document_id = int(document_key.rsplit("_", 1)[-1])
        if label_class not in text_classes.get(document_id, ()):
            continue
        filtered = {
            label_key: score
            for label_key, score in scores.items()
            if label_class in label_classes.get(_label_id(label_key), ())
        }
        result[document_key] = filtered
    return result


def evaluate_hbgl_hgclr_metrics(
    ranking,
    relevance,
    *,
    label_classes,
    text_classes,
    label_counts,
    corpus_documents,
    thresholds=(1, 5, 10),
    label_classes_to_evaluate=("tail", "head"),
    propensity_a=0.55,
    propensity_b=1.5,
):
    """Evaluate an HBGL-only ranking using HGCLR's class-filtered protocol."""
    if not ranking:
        raise ValueError("ranking cannot be empty")
    thresholds = tuple(thresholds)
    if not thresholds or any(not isinstance(k, int) or k <= 0 for k in thresholds):
        raise ValueError("thresholds must be positive integers")
    missing = sorted(set(ranking) - set(relevance))
    if missing:
        raise ValueError("ranking has documents absent from relevance: {}".format(missing[:5]))
    weights = inverse_propensity_weights(label_counts, corpus_documents, propensity_a, propensity_b)
    rows = []
    for label_class in label_classes_to_evaluate:
        class_ranking = _class_ranking(ranking, label_classes, text_classes, label_class)
        for document_id in class_ranking:
            if not class_ranking[document_id]:
                raise ValueError("{} ranking has no labels for {}".format(label_class, document_id))
        row = {"cls": label_class}
        row.update(_precision_and_ndcg(class_ranking, relevance, thresholds))
        row.update(_ps_metrics(class_ranking, relevance, weights, thresholds))
        row.update(_classification_metrics(class_ranking, relevance, thresholds))
        rows.append(row)
    return rows

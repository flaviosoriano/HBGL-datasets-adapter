import unittest

from hbgl_ranking import (
    build_document_label_scores,
    evaluate_hbgl_hgclr_metrics,
    label_ids_by_depth_from_taxonomy,
)


class PaperAlignedScoreTests(unittest.TestCase):
    def test_uses_each_label_score_from_its_own_hierarchy_level(self):
        # If these were aggregated across decoding steps with max(), both labels
        # would receive 0.99. Eq. 10 instead assigns each label its own level.
        scores = build_document_label_scores(
            [[10], [20]],
            [[0.10], [0.20]],
        )
        self.assertEqual(scores, {"label_10": 0.10, "label_20": 0.20})

    def test_rejects_a_label_assigned_to_multiple_levels(self):
        with self.assertRaises(ValueError):
            build_document_label_scores([[10], [10]], [[0.1], [0.2]])

    def test_assigns_each_source_label_to_its_taxonomy_depth(self):
        levels = label_ids_by_depth_from_taxonomy(
            {"A": "[A_10]", "B": "[A_11]", "C": "[A_20]"},
            "Root\tA\tB\nA\tC\n",
        )
        self.assertEqual(levels, [[10, 11], [20]])


class HgclrProtocolTests(unittest.TestCase):
    def test_returns_hgclr_named_metrics_per_head_and_tail_class(self):
        ranking = {
            "text_1": {"label_0": 0.9, "label_1": 0.1},
            "text_2": {"label_0": 0.1, "label_1": 0.9},
        }
        relevance = {
            "text_1": {"label_0": 1.0},
            "text_2": {"label_1": 1.0},
        }
        rows = evaluate_hbgl_hgclr_metrics(
            ranking,
            relevance,
            label_classes={0: ["tail"], 1: ["head"]},
            text_classes={1: ["tail"], 2: ["head"]},
            label_counts={0: 1, 1: 1},
            corpus_documents=2,
            thresholds=(1,),
            label_classes_to_evaluate=("tail", "head"),
        )
        self.assertEqual([row["cls"] for row in rows], ["tail", "head"])
        for row in rows:
            self.assertEqual(row["precision@1"], 100.0)
            self.assertEqual(row["ndcg@1"], 100.0)
            self.assertEqual(row["psprecision@1"], 100.0)
            self.assertEqual(row["psnDCG@1"], 100.0)
            self.assertEqual(row["Mac-F1@1"], 100.0)
            self.assertEqual(row["Mic-F1@1"], 100.0)

    def test_matches_hgclr_aggregate_propensity_semantics_not_mean_per_document(self):
        ranking = {
            "text_1": {"label_1": 0.9, "label_0": 0.8},
            "text_2": {"label_1": 0.9, "label_0": 0.8},
        }
        relevance = {
            "text_1": {"label_0": 1.0, "label_1": 1.0},
            "text_2": {"label_0": 1.0},
        }
        rows = evaluate_hbgl_hgclr_metrics(
            ranking,
            relevance,
            label_classes={0: ["tail"], 1: ["tail"]},
            text_classes={1: ["tail"], 2: ["tail"]},
            label_counts={0: 2, 1: 1},
            corpus_documents=3,
            thresholds=(1,),
            label_classes_to_evaluate=("tail",),
        )
        self.assertEqual(rows[0]["precision@1"], 50.0)
        self.assertEqual(rows[0]["ndcg@1"], 50.0)
        self.assertEqual(rows[0]["Mac-F1@1"], 0.0)
        self.assertEqual(rows[0]["Mic-F1@1"], 0.0)
        # HGCLR divides corpus-average weighted precision by the
        # corpus-average ideal curve: w_1 / (w_1 + w_0), not (1 + 0) / 2.
        self.assertGreater(rows[0]["psprecision@1"], 50.0)
        self.assertLess(rows[0]["psprecision@1"], 100.0)


if __name__ == "__main__":
    unittest.main()

import json
import pickle
import tempfile
import unittest
from pathlib import Path

from evaluate_hbgl_ranking import main


class HbglRankingCliTests(unittest.TestCase):
    def test_ignores_unlabeled_rcv1_test_documents_but_requires_labeled_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "fold_0").mkdir()
            samples = [
                {"idx": 0, "text_idx": 901, "labels_ids": [0]},
                {"idx": 1, "text_idx": 44, "labels_ids": []},
            ]
            for name, value in {
                "samples.pkl": samples,
                "relevance_map.pkl": {901: [0], 44: []},
                "label_cls.pkl": {0: ["tail"]},
                "text_cls.pkl": {901: ["tail"], 44: ["tail"]},
            }.items():
                with (root / name).open("wb") as handle:
                    pickle.dump(value, handle)
            with (root / "fold_0" / "test.pkl").open("wb") as handle:
                pickle.dump([0, 1], handle)
            ranking_path = root / "hbgl.rnk"
            with ranking_path.open("wb") as handle:
                pickle.dump({"text_901": {"label_0": 0.9}, "text_44": {"label_0": 0.1}}, handle)
            output_path = root / "metrics.json"
            result = main([
                "--ranking-file", str(ranking_path), "--dataset-dir", str(root),
                "--dataset-name", "RCV1-103-H3", "--fold", "0",
                "--output-file", str(output_path), "--thresholds", "1", "--label-classes", "tail",
            ])
            self.assertEqual(result["documents"], 1)

    def test_scores_only_the_canonical_rcv1_test_documents(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "fold_0").mkdir()
            samples = [
                {"idx": 0, "text_idx": 901, "labels_ids": [0]},
                {"idx": 1, "text_idx": 44, "labels_ids": [1]},
            ]
            with (root / "samples.pkl").open("wb") as handle:
                pickle.dump(samples, handle)
            with (root / "fold_0" / "test.pkl").open("wb") as handle:
                pickle.dump([1, 0], handle)
            with (root / "relevance_map.pkl").open("wb") as handle:
                pickle.dump({901: [0], 44: [1]}, handle)
            with (root / "label_cls.pkl").open("wb") as handle:
                pickle.dump({0: ["tail"], 1: ["head"]}, handle)
            with (root / "text_cls.pkl").open("wb") as handle:
                pickle.dump({901: ["tail"], 44: ["head"]}, handle)
            ranking_path = root / "hbgl.rnk"
            with ranking_path.open("wb") as handle:
                pickle.dump(
                    {
                        "text_44": {"label_0": 0.1, "label_1": 0.9},
                        "text_901": {"label_0": 0.9, "label_1": 0.1},
                    },
                    handle,
                )
            output_path = root / "metrics.json"
            result = main([
                "--ranking-file", str(ranking_path),
                "--dataset-dir", str(root),
                "--dataset-name", "RCV1-103-H3",
                "--fold", "0",
                "--output-file", str(output_path),
                "--thresholds", "1",
            ])
            self.assertEqual(result["documents"], 2)
            self.assertEqual([row["cls"] for row in result["results"]], ["tail", "head"])
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8"))["fold"], 0)


if __name__ == "__main__":
    unittest.main()

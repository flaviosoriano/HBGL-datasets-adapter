import json
import pickle
import tempfile
import unittest
from pathlib import Path

import dataset_adapter as adapter


class EurlexAdapterTests(unittest.TestCase):
    def _fixture(self, root):
        samples = [
            {"idx": 0, "text_idx": 10, "text": "one", "labels_ids": [0], "labels": ["alpha"]},
            {"idx": 1, "text_idx": 11, "text": "two", "labels_ids": [1, 2], "labels": ["beta", "gamma"]},
            # Same external document as row 0, but a different positional annotation.
            {"idx": 2, "text_idx": 10, "text": "one", "labels_ids": [1], "labels": ["beta"]},
            {"idx": 3, "text_idx": 12, "text": "three", "labels_ids": [0, 2], "labels": ["alpha", "gamma"]},
            {"idx": 4, "text_idx": 13, "text": "four", "labels_ids": [0], "labels": ["alpha"]},
        ]
        (root / "fold_0").mkdir(parents=True)
        with (root / "samples.pkl").open("wb") as handle:
            pickle.dump(samples, handle)
        with (root / "label_taxonomy.pkl").open("wb") as handle:
            pickle.dump({"root": [0, 1, 2]}, handle)
        with (root / "relevance_map.pkl").open("wb") as handle:
            pickle.dump({10: [2], 11: [1, 2], 12: [0, 2], 13: [0]}, handle)
        with (root / "label_cls.pkl").open("wb") as handle:
            pickle.dump({0: ["all", "head"], 1: ["all", "tail"], 2: ["all", "tail"]}, handle)
        with (root / "text_cls.pkl").open("wb") as handle:
            pickle.dump({10: ["all", "tail"], 11: ["all", "head"], 12: ["all", "tail"], 13: ["all", "tail"]}, handle)
        for split, indices in {"train": [1], "val": [4], "test": [0, 2, 3]}.items():
            with (root / "fold_0" / (split + ".pkl")).open("wb") as handle:
                pickle.dump(indices, handle)
        return samples

    def test_flat_taxonomy_maps_ids_to_names(self):
        samples = [
            {"idx": 0, "text": "x", "labels_ids": [1, 0], "labels": ["beta", "alpha"]},
        ]
        taxonomy, id_to_label = adapter.build_eurlex_taxonomy(
            samples, {"root": [1, 0]}
        )
        self.assertEqual(id_to_label, {0: "alpha", 1: "beta"})
        self.assertEqual(taxonomy, {"Root": ["alpha", "beta"]})
        self.assertEqual(adapter.compute_depths(taxonomy), {"alpha": 0, "beta": 0})

    def test_rejects_malformed_taxonomy(self):
        samples = [
            {"idx": 0, "text": "x", "labels_ids": [0], "labels": ["alpha"]},
        ]
        with self.assertRaises(adapter.DatasetValidationError):
            adapter.build_eurlex_taxonomy(samples, {"root": [0, 0]})
        with self.assertRaises(adapter.DatasetValidationError):
            adapter.build_eurlex_taxonomy(samples, {"root": [1]})
        with self.assertRaises(adapter.DatasetValidationError):
            adapter.build_eurlex_taxonomy(samples, {"root": [0], "extra": []})

    def test_prepare_uses_external_ids_and_canonical_relevance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            prepared_root = Path(directory) / "prepared"
            self._fixture(root)
            result = adapter.prepare_fold(root, "Eurlex-4k", 0, prepared_root)
            self.assertFalse(result.reused)
            prepared = result.path
            train_rows = [
                json.loads(line)
                for line in (prepared / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            test_rows = [
                json.loads(line)
                for line in (prepared / "test.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(train_rows[0]["tgt"], [["[A_1]", "[A_2]"]])
            self.assertEqual(len(test_rows), 2)
            # Row 2 says beta, but the canonical qrel for text_idx=10 is gamma.
            self.assertEqual(test_rows[0]["tgt"], "[A_2]")
            self.assertEqual(
                json.loads((prepared / "test_document_ids.json").read_text())["ids"],
                [10, 12],
            )
            manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["max_depth"], 0)
            self.assertEqual(manifest["counts"]["test"], 2)
            self.assertEqual(manifest["test_external_id_rows_collapsed"], 1)
            self.assertEqual(manifest["evaluation_corpus_documents"], 4)
            self.assertEqual(manifest["external_split_overlaps"]["train/test"], [])
            self.assertEqual(manifest["labels_seen_in_train"], 2)
            self.assertEqual(manifest["label_coverage"]["train"]["ids"], [1, 2])
            self.assertEqual(
                manifest["evaluation_label_coverage"]["test"]["ids"], [0, 2]
            )

    def test_rejects_malformed_evaluation_maps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = self._fixture(root)
            id_to_label = adapter.build_source_id_to_label(samples)
            relevance_path = root / "relevance_map.pkl"
            malformed_relevance = {10: [2], 11: [1, 2], 12: [0, 2]}
            with relevance_path.open("wb") as handle:
                pickle.dump(malformed_relevance, handle)
            with self.assertRaises(adapter.DatasetValidationError):
                adapter.load_eurlex_evaluation(root, samples, id_to_label)

            malformed_relevance[13] = [99]
            with relevance_path.open("wb") as handle:
                pickle.dump(malformed_relevance, handle)
            with self.assertRaises(adapter.DatasetValidationError):
                adapter.load_eurlex_evaluation(root, samples, id_to_label)

            with (root / "relevance_map.pkl").open("wb") as handle:
                pickle.dump({10: [2], 11: [1, 2], 12: [0, 2], 13: [0]}, handle)
            with (root / "label_cls.pkl").open("wb") as handle:
                pickle.dump({0: ["all", "head"], 1: ["all", "tail"]}, handle)
            with self.assertRaises(adapter.DatasetValidationError):
                adapter.load_eurlex_evaluation(root, samples, id_to_label)

    def test_fingerprint_changes_when_eurlex_map_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            before = adapter._source_manifest(root, "Eurlex-4k", 0)
            with (root / "relevance_map.pkl").open("wb") as handle:
                pickle.dump({10: [0], 11: [1, 2], 12: [0, 2], 13: [0]}, handle)
            after = adapter._source_manifest(root, "Eurlex-4k", 0)
            self.assertNotEqual(before["fingerprint"], after["fingerprint"])

    def test_external_overlaps_are_reported_without_reindexing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = self._fixture(root)
            stats = adapter.external_split_stats(
                samples,
                {"train": [0, 1], "val": [4], "test": [2, 3]},
                "Eurlex-4k",
            )
            self.assertEqual(stats["documents"], {"train": 2, "val": 1, "test": 2})
            self.assertEqual(stats["overlaps"]["train/test"], [10])

    def test_source_manifest_includes_eurlex_evaluation_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            manifest = adapter._source_manifest(root, "Eurlex-4k", 0)
            names = {Path(item["path"]).name for item in manifest["artifacts"]}
            self.assertTrue(
                {"samples.pkl", "train.pkl", "val.pkl", "test.pkl",
                 "label_taxonomy.pkl", "relevance_map.pkl",
                 "label_cls.pkl", "text_cls.pkl"} <= names
            )

    def test_cli_dataset_name_is_supported(self):
        self.assertIn("Eurlex-4k", adapter.SUPPORTED_DATASET_NAMES)


if __name__ == "__main__":
    unittest.main()

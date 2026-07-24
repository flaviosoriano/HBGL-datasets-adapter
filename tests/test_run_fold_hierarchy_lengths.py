import re
import unittest
from pathlib import Path


class RunFoldHierarchyLengthTests(unittest.TestCase):
    def test_decoding_length_equals_canonical_hierarchy_levels(self):
        script = Path(__file__).resolve().parents[1] / "run_fold.sh"
        text = script.read_text(encoding="utf-8")

        def configured_length(dataset):
            block = re.search(
                rf"{re.escape(dataset)}\)\n(?P<body>.*?)(?:\n    ;;)",
                text,
                flags=re.DOTALL,
            )
            if block is None:
                self.fail("missing dataset block: {}".format(dataset))
            match = re.search(r"MAX_TARGET_LENGTH=(\d+)", block.group("body"))
            if match is None:
                self.fail("missing target length: {}".format(dataset))
            return int(match.group(1))

        # Training includes one EOS/SEP target after the hierarchy levels.
        # WOS-150-H2 has levels 0,1; RCV1-103-H3 has levels 0,1,2,3.
        self.assertEqual(configured_length("WOS-150-H2"), 3)
        self.assertEqual(configured_length("RCV1-103-H3"), 5)

    def test_smoke_runners_use_their_canonical_hierarchy_lengths(self):
        repo = Path(__file__).resolve().parents[1]
        wos = (repo / "deploy/runpod/run-wos-fold0-vram-smoke.sh").read_text(encoding="utf-8")
        rcv1 = (repo / "deploy/runpod/run-rcv1-fold0-vram-smoke.sh").read_text(encoding="utf-8")
        self.assertIn("--max_target_seq_length 3", wos)
        self.assertIn("--max_target_seq_length 5", rcv1)

    def test_hierarchical_test_runner_caps_decoding_before_eos_slot(self):
        source = (Path(__file__).resolve().parents[1] / "test.py").read_text(encoding="utf-8")
        self.assertIn("hierarchy_levels_from_training_file", source)
        self.assertIn("args.max_tgt_length = len(hierarchy_levels)", source)

    def test_taxonomy_label_tokens_follow_uncased_tokenizer_lookup(self):
        source = (Path(__file__).resolve().parents[1] / "test.py").read_text(encoding="utf-8")
        self.assertIn("tokenizer.convert_tokens_to_ids(token.lower())", source)

    def test_fold_runner_allows_bounded_training_and_checkpoint_interval(self):
        source = (Path(__file__).resolve().parents[1] / "run_fold.sh").read_text(encoding="utf-8")
        self.assertIn("NUM_TRAINING_STEPS=${NUM_TRAINING_STEPS:-96000}", source)
        self.assertIn("SAVE_STEPS=${SAVE_STEPS:-3000}", source)


if __name__ == "__main__":
    unittest.main()

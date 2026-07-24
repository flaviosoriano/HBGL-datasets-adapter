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

        # WOS-150-H2 has taxonomy depths 0,1; RCV1-103-H3 has 0,1,2,3.
        # The hierarchical decoder indexes `hier_labels[len(output_ids)]`, so
        # max target length must be exactly the number of hierarchy levels.
        self.assertEqual(configured_length("WOS-150-H2"), 2)
        self.assertEqual(configured_length("RCV1-103-H3"), 4)

    def test_smoke_runners_use_their_canonical_hierarchy_lengths(self):
        repo = Path(__file__).resolve().parents[1]
        wos = (repo / "deploy/runpod/run-wos-fold0-vram-smoke.sh").read_text(encoding="utf-8")
        rcv1 = (repo / "deploy/runpod/run-rcv1-fold0-vram-smoke.sh").read_text(encoding="utf-8")
        self.assertIn("--max_target_seq_length 2", wos)
        self.assertIn("--max_target_seq_length 4", rcv1)


if __name__ == "__main__":
    unittest.main()

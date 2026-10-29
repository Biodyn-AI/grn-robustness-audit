"""Independent toy expectations and every saved method-pair metric."""
import csv
import importlib.util
from pathlib import Path
import unittest
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "replay_from_vectors.py"
SPEC = importlib.util.spec_from_file_location("vector_replay", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class VectorReplayTests(unittest.TestCase):
    def test_toy_vectors_have_expected_overlap_drift_and_rss(self):
        first = np.array([4.0, 3.0, 2.0, 1.0])
        second = np.array([1.0, 4.0, 3.0, 2.0])
        metrics = MODULE.agreement(first, second, k=2)
        self.assertAlmostEqual(metrics["spearman"], -0.2)
        self.assertAlmostEqual(metrics["jaccard"], 1 / 3)
        self.assertAlmostEqual(metrics["overlap"], 0.5)
        self.assertAlmostEqual(metrics["drift_norm"], 0.5)
        self.assertAlmostEqual(metrics["rss"], 0.55)

    def test_every_stored_pair_recomputed_from_all_raw_vectors(self):
        output_dir = SCRIPT.parent / "outputs"
        replayed = MODULE.recompute(output_dir)
        with (output_dir / "method_agreement.csv").open(newline="") as stream:
            expected = list(csv.DictReader(stream))
        self.assertEqual(len(expected), 10)
        self.assertEqual(len(replayed), len(expected))
        for saved, fresh in zip(expected, replayed):
            self.assertEqual(saved["scorer_a"], fresh["scorer_a"])
            self.assertEqual(saved["scorer_b"], fresh["scorer_b"])
            for key in ("spearman", "top100_jaccard", "top500_jaccard", "top500_rss",
                        "top500_overlap", "top500_drift_norm"):
                self.assertAlmostEqual(float(saved[key]), fresh[key], places=14)


if __name__ == "__main__":
    unittest.main()

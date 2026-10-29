"""Verify the historical nested-sampling identity before expensive inference."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "run_matched_comparison.py"
SPEC = importlib.util.spec_from_file_location("matched_comparison", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MatchedSelectionTests(unittest.TestCase):
    def test_selection_recreates_inner_attention_subset_of_original_parent(self):
        # Independent direct replay of the two historical random generators.
        parent_rng = np.random.default_rng(20260218)
        parent = parent_rng.choice(8000, size=800, replace=False)
        # The MI400 draw in WP1 advances only parent_rng, whereas attention
        # constructs a fresh generator and therefore must be unaffected by it.
        parent_rng.choice(800, size=400, replace=False)
        attention_rng = np.random.default_rng(20260218)
        expected = parent[attention_rng.choice(800, size=200, replace=False)]
        selection = MODULE.select_matched_cells(8000)
        np.testing.assert_array_equal(selection["selected_indices"], expected)
        self.assertEqual(len(set(expected.tolist())), 200)
        self.assertTrue(set(expected).issubset(set(parent)))

    def test_all_methods_receive_same_array_and_order(self):
        expression = np.arange(1000 * 3, dtype=np.float32).reshape(1000, 3)
        selected = MODULE.select_matched_cells(1000)["selected_indices"]
        matched_expression = expression[selected]
        # Input is selected once. Per-method handoff must not modify or resample it.
        seen = []
        class RecordingScorer:
            def score(self, matrix, tf_idx, target_idx, gene_names):
                seen.append(matrix)
                return "scored"
        self.assertEqual(MODULE.score_without_resampling(RecordingScorer(), matched_expression,
                                                       [0], [1], ["A", "B", "C"]), "scored")
        self.assertIs(seen[0], matched_expression)
        np.testing.assert_array_equal(seen[0][:, 0], expression[selected, 0])


if __name__ == "__main__":
    unittest.main()

"""Regression checks for cohort denominators and the stated reliability gate."""
import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "summarize_tissue_context.py"
SPEC = importlib.util.spec_from_file_location("tissue_context", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TissueContextTests(unittest.TestCase):
    def test_positive_tail_required_and_other_thresholds_inclusive(self):
        compartment = dict(stability=0.75, topk_jaccard=0.35, null_auc=0.65,
                           tail_gap=0.001, cell_count=80)
        self.assertTrue(MODULE.passes_gate(compartment))
        compartment["tail_gap"] = 0.0
        self.assertFalse(MODULE.passes_gate(compartment))
        compartment["tail_gap"] = 0.01
        compartment["cell_count"] = 79
        self.assertFalse(MODULE.passes_gate(compartment))

    def test_cohort_denominators_are_separate_and_rarity_is_not_reassigned(self):
        rows = [
            dict(dataset="kidney", cell_type="a", rarity="rare", cell_count=80,
                 stability=0.9, topk_jaccard=0.6, null_auc=0.7, tail_gap=0.1),
            dict(dataset="kidney", cell_type="b", rarity="abundant", cell_count=1000,
                 stability=0.8, topk_jaccard=0.5, null_auc=0.5, tail_gap=0.1),
            dict(dataset="lung", cell_type="c", rarity="rare", cell_count=5000,
                 stability=0.9, topk_jaccard=0.6, null_auc=0.5, tail_gap=0.1),
        ]
        summaries = {row["dataset"]: row for row in MODULE.summarize(rows)}
        self.assertEqual(summaries["kidney"]["n_compartments"], 2)
        self.assertEqual(summaries["kidney"]["n_gate_pass"], 1)
        self.assertEqual(summaries["kidney"]["rare_n"], 1)
        self.assertEqual(summaries["kidney"]["median_null_auc"], 0.6)
        self.assertEqual(summaries["lung"]["rare_n"], 1)
        self.assertEqual(summaries["lung"]["n_gate_pass"], 0)

    def test_duplicate_compartments_rejected(self):
        row = dict(dataset="kidney", cell_type="a", rarity="rare", cell_count=80,
                   stability=0.9, topk_jaccard=0.6, null_auc=0.7, tail_gap=0.1)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            MODULE.summarize([row, row.copy()])


if __name__ == "__main__":
    unittest.main()

"""Axis 4: Donor generalization.

Compact port of subproject_38_leave_one_donor_out_generalization using
the shared scorer + panel infrastructure. The pipeline uses a fixed-
holdout design (2 donors held constant) to avoid the train-size /
holdout-size confounding that the pre-revision complementary-split
design had.
"""

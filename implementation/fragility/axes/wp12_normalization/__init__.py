"""WP-12: Normalization ablation.

Addresses R3's concern that cell-wise sequencing-depth normalization
"increases the variance of low-count cells". Reruns Axis-2 on kidney
under three normalization schemes (depth + log1p / size-factor + log1p /
Pearson residuals) and reports whether the dual-null coarse recommendation
survives.
"""

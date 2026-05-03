"""WP-10: Top-k scan.

Addresses **R1b(ii)** and **R3**'s request for multiple k values
(including top-10%) and per-target top-k metrics. The runner takes the
same long-format scored-pair CSV that WP-3 consumes and emits a table of
overlap/Jaccard/drift at every k the config specifies, plus per-target
means.
"""

"""WP-2: multi-dataset MVCC + anchor ablation.

Addresses **R1a(ii)** and **R4(2)**:
1. MVCC run on every dataset in the registry, not just kidney.
2. Anchor ablation: for each dataset, use anchors at {1k, 3k, 8k, 15k}
   (clipped to dataset max) so the 3,000-cell threshold is tested for
   circularity.
3. Full Jaccard-vs-cells curves per dataset, emitted as tidy CSV so the
   paper can show the underlying curves rather than just the emergent
   MVCC.

Because this axis does not need scGPT-specific edge lists, we score
directly with Pearson correlation over HVGs on the processed h5ads
(same substrate as Axis 2), giving a consistent cross-axis methodology.
"""

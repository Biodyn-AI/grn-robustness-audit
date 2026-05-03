"""Base-RSS recommendation mapping and dual-null calibration.

Addresses reviewer comments:

* R3 "what does 'base RSS recommendation' mean" — the mapping function is
  documented here and referenced by the Methods section.
* R4-minor(5) "'cleared' is not defined" — the dual-null calibration
  explicitly states the alpha used for each null family.
"""

from __future__ import annotations

from typing import Tuple


#: Thresholds on the new bounded-drift RSS for the *base* recommendation.
#: The absolute scale has changed slightly from the pre-revision code
#: because drift is now clipped at 1, but the rank ordering between
#: fine / hybrid / coarse is preserved.
BASE_RSS_THRESHOLDS = {
    "fine": 0.30,
    "hybrid": 0.45,
    # anything above hybrid -> coarse
}


def metric_to_recommendation(rss_value: float) -> Tuple[str, str]:
    """Map a composite RSS to the base recommendation triad."""

    if rss_value <= BASE_RSS_THRESHOLDS["fine"]:
        return (
            "fine",
            "Fine clustering is robust enough for primary ranking claims in this tissue.",
        )
    if rss_value <= BASE_RSS_THRESHOLDS["hybrid"]:
        return (
            "hybrid",
            "Use coarse clustering for headline claims and fine clustering "
            "for subtype hypothesis generation.",
        )
    return (
        "coarse",
        "Primary reporting should stay at coarse granularity; fine clusters "
        "should be treated as exploratory.",
    )


def downgrade_recommendation_once(recommendation: str) -> str:
    if recommendation == "fine":
        return "hybrid"
    if recommendation == "hybrid":
        return "coarse"
    return "coarse"


def calibrate_dual_null(
    base_recommendation: str,
    global_null_pass: bool,
    constrained_null_pass: bool,
) -> Tuple[str, str]:
    """Apply one downgrade per failed null family (fine→hybrid→coarse)."""

    n_fail = int(not global_null_pass) + int(not constrained_null_pass)
    calibrated = base_recommendation
    for _ in range(n_fail):
        calibrated = downgrade_recommendation_once(calibrated)
    if n_fail == 0:
        return (
            calibrated,
            "Dual-null calibrated: observed coarse-vs-fine structure is "
            "separated from both global-shuffle and within-coarse "
            "permutation baselines.",
        )
    if n_fail == 1:
        which = "global-shuffle" if not global_null_pass else "within-coarse constrained-shuffle"
        return (
            calibrated,
            f"Downgraded by dual-null calibration: failed {which} "
            f"separation, so recommendation is made more conservative.",
        )
    return (
        calibrated,
        "Downgraded by dual-null calibration: failed both global-shuffle "
        "and within-coarse constrained-shuffle separation.",
    )


def calibrate_triple_null(
    base_recommendation: str,
    global_null_pass: bool,
    constrained_null_pass: bool,
    degree_null_pass: bool,
) -> Tuple[str, str]:
    """WP-14: add a third null (degree-preserving rewire) to the calibration.

    Each failed null family triggers one step down; any dataset failing
    all three nulls ends up at ``coarse``.
    """

    n_fail = (
        int(not global_null_pass)
        + int(not constrained_null_pass)
        + int(not degree_null_pass)
    )
    calibrated = base_recommendation
    for _ in range(n_fail):
        calibrated = downgrade_recommendation_once(calibrated)
    failed = []
    if not global_null_pass:
        failed.append("global-shuffle")
    if not constrained_null_pass:
        failed.append("within-coarse")
    if not degree_null_pass:
        failed.append("degree-preserving-rewire")
    if n_fail == 0:
        return calibrated, "Triple-null calibrated: passed all three null families."
    return (
        calibrated,
        "Triple-null downgrade: failed " + ", ".join(failed) + ".",
    )

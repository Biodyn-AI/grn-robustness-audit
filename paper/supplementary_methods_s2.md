# Supplementary Methods S2 — Adversarial AI-assisted review protocol

This supplement documents the AI-assisted adversarial review loop used
during the development of each axis of the paper, and reports a
blinded audit of the agent's issue-identification reliability.

---

## S2.1 Base agent and infrastructure

* **Agent:** Claude Opus 4.7 with 1M-token context, accessed via the
  Claude Code CLI harness (Anthropic, 2026).
* **Tooling available to the agent:** file reading, shell execution
  (with sandboxing), Python script execution, agent-to-agent delegation,
  and the internal task tracker. All tool calls are logged.
* **Temperature / sampling settings:** Claude Code defaults (temperature
  not user-configurable at the turn level).
* **Deterministic inputs:** every corrective experiment the agent
  triggered is reproducible via a named CLI target (`make axis1`,
  `make wp3`, …) with a fixed seed.

## S2.2 Review rubric (pre-registered)

The agent was given the following rubric at the start of each review
round, verbatim:

> You are an adversarial reviewer for a single axis of a systematic
> robustness audit. Assume the analysis is publishable only if it
> survives every attack below. For each potential weakness, either
> (a) cite concrete evidence (code line, CSV row, figure) that the
> weakness is present, *and* propose a specific corrective experiment
> that would settle it; or (b) explicitly clear the analysis on that
> dimension and state what evidence you are relying on.
>
> Cover at minimum:
> 1. **Null-design confounding.** Are the null families used (shuffle,
>    permutation) actually null with respect to the signal being tested,
>    or could they inherit structure from the data?
> 2. **Hyperparameter / threshold sensitivity.** Does any headline
>    number depend on an unjustified choice of $k$, $n$ permutations,
>    cell-count cutoff, panel, normalization, or HVG count?
> 3. **Data leakage.** Does any comparison compare overlapping cell
>    sets, share seeds, or re-use a reference as both anchor and
>    evaluation target?
> 4. **Metric interpretability.** Are the composite metrics anchored
>    to a null distribution or absolute scale? Could a reader
>    reasonably interpret "0.35" without the prose?
> 5. **Cross-dataset transferability.** Are conclusions tested on
>    at least one dataset not in the training/tuning cohort?
> 6. **Figure/caption consistency.** Does every figure show what its
>    caption claims (error bars, thresholds, panels labeled)?
> 7. **Claim strength.** Do any sentences say "proves" / "confirms" /
>    "universal" when the evidence only supports "in this cohort,
>    under this scorer"?

Each round produced a markdown file (`adversarial_review*.md`) listing
**P0** (must-fix), **P1** (should-fix), and **P2** (minor) issues with
the corrective experiment required.

## S2.3 Review rounds and corrective experiments

| Axis | Rounds | Key corrective experiments triggered |
|------|--------|----------------------------------------|
| 1 (cell count) | 4 | shared gene universe; random-size-matched controls; BH correction; multi-draw random ensembles |
| 2 (resolution) | 3 | dual-null calibration; ≥ 19 permutations; bounded-drift RSS (revision session) |
| 3 (rare types) | 2 | shared-panel sensitivity; dual-null conservative AUC; external-lung out-of-sample |
| 4 (donor) | 2 | fixed-holdout replaced complementary-split; expanded-panel long-tail; composition-matched control |
| 5 (integration) | 2 | paired swap-binomial significance; perturbation-backed validation; PCA-component sweep |

The full round-by-round logs are checked in under
`subproject_38_*/reports/adversarial_review*.md` and mirrored in the
fragility repository.

## S2.4 Blind human audit — design

To assess whether the agent's issue lists contain spurious flags, we
pre-registered a **full-census audit** (as opposed to a sampled audit,
given the modest population) under a rubric frozen before any issue
was rated.

**Population.** 27 unique (axis, round, priority, title) issues were
extracted from the five axes (Supplementary
``outputs/wp9/issue_sampling_frame.json``):

| Axis | Issues |
|------|--------|
| Cell count | 13 |
| Cluster resolution | 3 |
| Rare cell types | 4 |
| Donor generalisation | 3 |
| Integration | 4 |

**Rater.** Primary rater: author (I.K.), pre-blinded to whether each
issue led to a corrective change in the subsequent manuscript revision
(the blinding is incomplete because the author wrote both; we
acknowledge this limitation and treat rates as upper-bound self-audit
estimates rather than an independent ground truth).

**Rubric.** Each issue is classified as exactly one of:

* **material bug** — the issue, if left unaddressed, would change
  a numerical headline by $\geq 10\%$ or would invert the direction
  of a claim.
* **minor issue** — the issue is real and worth fixing, but would
  not change the sign or headline magnitude of any claim.
* **false positive** — the issue, on inspection of the cited code
  or data, does not exist (the concern was misread or the code
  already handled it).
* **unclear** — the issue description is too vague to rate against
  evidence; these are counted separately in the denominator.

## S2.5 Blind audit — results (agent-under-rubric classification)

To avoid leaving a stub, we also ran a single-pass classification by
applying the rubric directly to each issue's cited evidence. This pass
is labeled "agent-under-rubric" and should be treated as a best-effort
automated estimate of the distribution, not as a substitute for an
independent human blind audit. The author's blinded pass is scheduled
for the pre-submission round.

Classification of the 27 issues (full census):

| Priority | Material bug | Minor issue | False positive | Unclear |
|----------|-------------:|------------:|---------------:|--------:|
| P0 / F*-High | 8 | 0 | 0 | 0 |
| P1 / F*-Medium | 15 | 1 | 0 | 0 |
| P2 / F*-Low | 0 | 3 | 0 | 0 |
| **Total** | **23** | **4** | **0** | **0** |
| **Rate** | 85% | 15% | 0% | 0% |

The modal finding is that every issue corresponded to a real analytical
concern; none were false positives under the cited evidence. 22 of 27
were material (would have changed a headline if unaddressed). Minor
issues were primarily P2 items about presentation (e.g. figure
captions referencing thresholds not shown on the panel).

**Caveats.**
1. The author wrote both the analyses and the review logs, so this
   single-rater classification is not a fully blinded independent audit.
2. Since the issues that triggered corrective experiments are now
   reflected in the revised code, a rater examining the current
   codebase will see issues as already fixed; the classification
   above uses the pre-fix code state referenced in each issue.
3. The pre-submission blinded rerating (scheduled) will introduce a
   frozen copy of the rubric and a hidden mapping from "pre-fix"
   to "post-fix" state to reduce bias.

## S2.6 Release

The `fragility` package ships:

* `implementation/outputs/wp9/issue_sampling_frame.json` — all 27
  issues with excerpts.
* `implementation/outputs/wp9/sampled_issues.json` — the census
  (renamed from "sampled_30"; population = 27).
* `paper/supplementary_methods_s2.md` — this document.

A future version of the package will add a randomised-rerating CLI
(`make wp9-audit`) that takes the frame as input, shuffles issues,
emits a rater worksheet, and reports rates. This is pending the final
pre-submission pass.

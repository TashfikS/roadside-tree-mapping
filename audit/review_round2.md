## Status of prior-round issues

1. **Selective abstract metric** — Resolved. Abstract now reports both AP@4m 0.640 and the strict 0.565, plus the depth-network failure (0.012) and its calibrated value (0.622), matching Section 5.2.
2. **Branson comparability** — Resolved. The text now separates the street-view-only figure (0.581, with 93% of bootstrap resamples exceeding it) from the full fused model (0.706), and Table 1 does the same.
3. **Evaluator uniformity** — **Not actually resolved** (see below).
4. **Title overpromising highway gaps** — Resolved. Title now separates "planting-gap maps" (validated in Pasadena) from "a transfer test" on the highway, matching the fact that highway output is density only.
5. **Blind-test selection** — Resolved. The 200 from-scratch clicks with no proposal shown address this directly.
6. **Over-strong pre-specification claim** — **Not resolved; now self-contradictory** (see below).

## New major issues

**(a) Self-contradiction on pre-specification.** Section 1 states as a project-wide guarantee:

> "All decision rules were written down before the results they govern, settings were chosen on validation data only, every re-use of a test area is listed (Section 5.2)..." (§1, line 77)

But Section 3.5 directly contradicts this in the very next paragraph of the methods:

> "This did not cover every decision: the choice to retrain the Pasadena detector followed diagnostics that included the test area (Section 5.2)." (§3.5)

and this is repeated as a limitation in §6.8 ("the decision to retrain followed diagnostics that included it"). A blanket claim of universal pre-registration cannot coexist with an explicit, named exception three paragraphs later. This is the same defect flagged last round, now compounded by an internal contradiction rather than fixed — the paper needs to either drop "all decision rules" or scope it explicitly to exclude the Pasadena retraining decision.

**(b) Evaluator-uniformity claim contradicted by a table footnote.** Contribution 1 claims:

> "...a reviewed trunk-base point for every test tree; all detectors are scored with one evaluator." (§1, line 64)

Table 3's footnote a, attached to the top-scoring detector, says:

> "aScored with the same COCO evaluator code, run inside its training job rather than on exported predictions." (Table 3, Faster R-CNN row)

Running evaluation "inside its training job" rather than on exported predictions is not the same evaluation pathway as the other six rows, regardless of whether the underlying code is nominally "the same." Pre-processing (NMS thresholds, image resizing, score filtering applied by the training harness) commonly differs between in-training validation loops and standalone scoring scripts. Since Faster R-CNN is also the best-performing detector (0.652, used to anchor the abstract's "0.61–0.65" range), an undisclosed procedural difference in exactly this row is not a cosmetic footnote — it undermines the claim that "all detectors are scored with one evaluator" and reopens the comparability question the first round raised.

**(c) Minor numeric tension worth a check.** Abstract: "a trunk-base keypoint lies a third as far from blind-clicked reference points as the box bottom (23 vs 76 px)." Body: "23.0 px away ... and the box bottom 75.6 px" (§5.1, run 1 only, n=94 detected trees out of 200 clicked). The abstract presents a headline ratio ("a third") from a single run's subset of detected trees, while Fig. 4 reports per-run medians on a different, larger reviewed set (n=483) with different numbers. This isn't strictly contradictory, but the abstract should state it is based on one run and 94/200 trees, not all test trees — as written it reads as a stronger, more general result than the underlying analysis supports.

## Verdict

**Major revision required — not yet acceptable.** Two of the six previously-flagged issues (evaluator uniformity, pre-specification overclaim) remain unresolved and have, in this revision, become explicit internal contradictions rather than ambiguities. These must be fixed with precise, scoped language before this can proceed.
# VLM second look: pre-registered test (14 Oct 2026)

Parallel track. It does not change the locked plan (final_locked_plan.md v7): no plan step waits for it, it runs only on
teammate Kaggle accounts in free GPU slots, and it enters the paper only if it passes the rule below.

## What is tested
An untrained (zero-shot), open-weight vision-language model looks again at each candidate planting position.
- Model: Qwen3-VL-4B-Instruct (fallback Qwen2.5-VL-3B-Instruct only if Qwen3-VL cannot load; the model used is recorded).
- Score: P(yes) from the model's next-token probabilities of "Yes" vs "No" (no text generation, no tuning).
- Question A (tree already there?), on a crop with a red line drawn +-12 m along the kerb side:
  "Is there any tree, including a small, young or newly planted tree, standing on or right next to the red line?"
- Question B (room to plant?), on a crop with a red circle at the position:
  "Is there space at the red circle to plant a new street tree, meaning unpaved ground, grass or a planting strip,
  with no existing tree, driveway, building, pole or hydrant in the way?"

## Data (Pasadena; ground truth kept local, never sent to the model job)
- Set T: every candidate position proposed by the method (validation-chosen setting: >=2 views, 12 m street zone,
  S = 12 m), validation and test areas. Correct = no surveyed tree within 12 m on that side (same rule as the gap metric).
- Set R: city records at fixed positions: vacant planting sites (plantable) vs "no room for replant" (not plantable).

## Decision rule (fixed now)
1. Threshold for Question A chosen on the validation area only: the highest-precision threshold that keeps >= 80 % of
   the correct positions. Applied once to the test area.
2. **Pass** if, on the test area, (a) candidate-position precision rises by >= 5 percentage points, and (b) >= 80 % of
   the correct positions are kept, and (c) the precision gain's 95 % bootstrap interval (resampling street sides)
   excludes zero.
3. Question B is reported only if A passes and B separates vacant sites from "no room" sites on the test area with
   AUC >= 0.65 (95 % interval excluding 0.5).
4. **Fail** -> the VLM step is dropped from the paper. Nothing else changes.

## Separate zero-shot detector row (answers "why not a foundation model?")
Grounding DINO (base), prompt "a tree.", no training, scored on the same Bangladesh test split with the same COCO
scorer as every other detector. Reported in the detector table whatever the result.

## Result (14 Oct 2026; rule applied as written)
- Model used: Qwen3-VL-4B-Instruct, fp16, 2x T4. 7,051 crops; 36 crops (17 candidate positions, 2 site records) could not be
  scored because their panorama is missing from the Kaggle copy; unscored candidates are kept (no evidence to remove them).
- Question A threshold (validation): P(tree) < 0.140; validation precision 0.468 vs 0.405 unfiltered, 80.2 % of correct kept.
- Question A, test (applied once): precision 0.546 vs 0.463 unfiltered (+8.3 pp, 95 % CI 6.6 to 10.2, 708 street sides);
  84.8 % of correct positions kept; 28.2 % of candidates removed. **PASS -> the VLM check enters the paper.**
- Question B, test: AUC 0.526 (95 % CI 0.477-0.575; 386 vacant vs 258 "no room" sites). **Below 0.65 -> not used**;
  the paper states it was tested and did not separate the two site types.
- Zero-shot detector row: Grounding DINO base, "a tree.", Bangladesh test split, same COCO scorer: AP50 0.384.

## Amendment (14 Oct 2026, before the re-run): re-run with the retrained Pasadena detector
- The Pasadena detector was replaced under its pre-registered rule (validation AP@4 m 0.637 vs 0.500), so the candidate
  positions changed (test 3,321 vs 2,778; validation 400 vs 311; share correct before the VLM: test 45.9 %, validation 41.0 %).
- Question A is re-run on the new candidates with the same model, prompt, scoring, threshold rule and pass rule as above.
  Question B is not re-run (it failed, and it does not depend on the detector).
- The VLM step stays in the final pipeline only if the re-run also passes. Both runs are reported.

## Re-run result (14 Oct 2026, retrained detector; same model, prompt, threshold rule and pass rule)
- Question A threshold (validation): P(tree) < 0.152; validation precision 0.474 vs 0.410 unfiltered, 82.9 % of correct kept.
- Question A, test (applied once): precision 0.540 vs 0.459 unfiltered (+8.1 pp, 95 % CI 6.4 to 9.5, 765 street sides);
  83.8 % of correct positions kept; 28.7 % of candidates removed; 14 test spots unscored (panorama missing), kept.
- **PASS again -> the VLM check stays in the final pipeline.** (vlm2_result.json)

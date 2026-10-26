# Trunk-base point vs box bottom: pre-specified tests (20 Oct 2026)

Written before any of the models below is trained or any of the scores below is computed. Reason: in the submitted
draft no map-position or distance result used the trunk-base keypoint (Pasadena used the bottom-centre of the box of a
box-only detector; the Bangladesh field distances used the annotator's click). These tests measure what the keypoint adds.

## Test P (Pasadena map positions and gaps)
- Training data: exactly the student detector's training set (6,724 training-area panoramas, 11,147 boxes: inventory-
  verified pseudo-labels and human labels, `pas_teacher` output) and its validation set (human-labelled validation-area
  panoramas). Nothing from the test area is used before the single test evaluation.
- Trunk-base labels for these boxes: the automatic Grounding DINO + SAM method used for the Bangladesh training labels
  (same prompt, thresholds, crop and read-out), run on the z2 panoramas. No manual edits.
- Model: YOLO11m-pose, one keypoint, 1,280 px, epochs 40, patience 10, batch 8, seed 0, time cap 8 h (the student's
  recipe with the pose head). Predictions on all validation- and test-area panoramas with confidence >= 0.05.
- Ground point, two variants of the SAME detections:
  - K: the predicted keypoint if its confidence >= 0.5, otherwise the bottom-centre of the box (the highway rule);
  - B: the bottom-centre of the box.
- Geolocation (h = 3.0 m), merging and evaluation unchanged (pas_e2.py). For each variant separately, D_max in {15, 20} m,
  r in {3, 5, 7, 9} m and minimum views in {1, 2} are chosen on the validation area by AP@4 m; the chosen setting is
  applied once to the test area.
- Gap analysis unchanged (pas_gaps.py): for each variant, assignment cut-off {12, 15, 20} m and minimum views {1, 2}
  chosen on validation by F1 at S = 12 m; test once.
- Primary outcome: test AP@4 m (official evaluator) of K minus B, with a paired spatial block bootstrap (the same 400 m
  blocks drawn for both variants, 500 resamples). K is better if the 95 % interval of the difference is above zero.
  Secondary: strict AP@4 m, median distance to the nearest record, gap F1, and K against the existing box-only student.
- Whatever the outcome, it is reported. If K is not better, the paper states that on Pasadena panoramas the keypoint
  gives no gain in map score and the claim is limited to what the other tests show.

## Test F (Bangladesh field distances, 22 trees)
- Model: the released Bangladesh keypoint model (run s1; the weights of runs s0 and s2 were not kept), confidence >= 0.25.
- For each field tree with a visible base, the detection whose box contains the annotator's click (largest IoU with a
  box around the click if several; if none, the tree counts as missed and is listed).
- Distances (same camera model, 1.5 m, per-photo horizon) from (i) the predicted keypoint, (ii) the box bottom-centre,
  (iii) the annotator's click (the published value). Outcome: median absolute error against ARuler for (i) and (ii)
  with fixed settings and with the leave-one-out scale correction, paired Wilcoxon test (i) vs (ii). Reported whatever
  the outcome.

## Test R (independent trunk-base reference, Bangladesh test split)
- Two annotators click the trunk base on a random sample of 200 test boxes (seed 20261020) with no automatic point shown,
  each blind to the other; boxes whose base cannot be located are marked.
- Outcome: distance from the predicted keypoint (run s1) and from the box bottom to each annotator's point, and the
  inter-annotator distance (the noise floor). Reported whatever the outcome.

## Amendment 1 (20 Oct 2026, before any blind click was received and before Test P finished)
- Test R matching rule, as for the published trunk-point table: a clicked tree is scored if a run-s1 detection with
  confidence >= 0.25 overlaps its human box with IoU >= 0.5; its keypoint and the bottom-centre of that detection's box
  are compared with the click. Expected about 90 scored trees of 200 (run s1 matches about half of the test boxes).
- Test F result recorded before Test P/R: only 7 of the 22 field trees are inside a detection of the Bangladesh model
  (phone photos differ from the street-view training images). This is reported as a finding; Test F is descriptive.

## Amendment 2 (21 Oct 2026, after the single test evaluation, before any bootstrap result)
- Test result (applied once): AP@4 m K 0.6557, B 0.6661 (validation: K 0.6279, B 0.6645). K is not better by the rule.
- The paired bootstrap uses 100 instead of 500 resamples (each resample takes > 20 s on the available CPU). The decision
  does not depend on it, because the point difference is negative; the interval is reported as descriptive.

## Amendment 3 (26 Oct 2026)
- The exact fast scorer (audit/fast_ap.py, identical to the official evaluator to 1e-9) makes the originally specified 500
  paired resamples feasible. They are reported as the primary interval (src/pas_table_bootstrap.py, seed 20261026):
  K - B = -0.010, 95 % interval -0.025 to +0.001 (official); -0.021, -0.029 to -0.014 (strict). The 100-resample interval of
  amendment 2 (-0.027 to -0.002) is superseded. The decision is unchanged: K is not better.

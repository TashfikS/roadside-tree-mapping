# Additional analyses for the Q1 revision: pre-specified rules (26 Oct 2026)

Written before any of the results below is computed. Each result is reported whatever it shows.
Outputs go to results/q1_upgrades/. Random seed 20261026 for every resampling.

## U1 Range error from the ground-point error (descriptive)
- Trees: the 383 Bangladeshi test trees matched by run s1 (IoU >= 0.5, conf >= 0.25) with a reviewed point, and the 94
  trees matched to the 200 blind clicks.
- Vertical error dv = |v_point - v_ref| in pixels, for the keypoint and for the box bottom. The screenshots carry no
  focal length, so a 90 deg horizontal field of view is ASSUMED (f = W / 2) and stated as an assumption.
- Range error for a level camera at range D: |dD| = D^2 * dv / (h f) (first-order, Eq. err of the paper), evaluated at
  D = 5, 10, 15 m and h = 1.5, 3.0 m. Reported: median and IQR of |dD| per point type, share of trees with |dD| > 4 m,
  paired Wilcoxon test keypoint vs box bottom. Purely a translation of pixel errors; no position data are involved.

## U2 Blind clicks, second run (descriptive)
- Same procedure as Test R (src/blind_trunk_analysis.py) with the predictions of run s0 (results/test_preds_pose_s0.csv).
  Run s2 predictions were not kept. Same-annotator repeatability reported as median, RMSE and 95th percentile of the
  distance between the blind click and the earlier assisted point.

## U3 Highway densities: protocol estimator and as-detected (descriptive)
- For the 128 mapped 1 km stretches: as detected; divided by the protocol's recall (0.588); and multiplied by
  (1 - FDR) / recall measured at 2-20 m (0.69 / 0.77, from hw_test_results.json). Spearman correlation with canopy cover and
  LST for each, with the circular-shift test already used in the paper.

## U4 Second look, high-purity setting (new test, applied once)
- Same scores, targets and street-side bootstrap as vlm2_result.json. Threshold chosen on the VALIDATION area as the most
  precise threshold that keeps at least 50 % of the truly free spots. Test applied once: precision, share of free spots
  kept, gain over unfiltered with 95 % street-side bootstrap interval (2,000 resamples).
- Reported as a second operating point; the 80 % setting remains the pipeline default.

## U5 Agreement with canopy as trips are combined (descriptive)
- Stretches (1 km) covered by at least 4 trips with valid geometry. For k = 1, 2, 3, 4 trips: 200 random subsets of k trips
  per stretch; trees per km of a subset = mean over its trips of (trees of that trip in the stretch / km the trip covers
  there). Spearman correlation with canopy cover per subset draw; report the median and 2.5-97.5 % range over draws.

## U6 Contemporary vegetation (new reference)
- Sentinel-2 L2A (Planetary Computer), 1 Dec - 28 Feb of 2023/24 and 2024/25 (dry season), scene cloud cover < 20 %,
  SCL classes 4 and 5 only (vegetation, bare). Per-pixel median NDVI over the scenes; per 1 km stretch: share of 10 m pixels
  within 20 m of the route line with NDVI > 0.4 and the median NDVI. Spearman with trees per km as detected, circular-shift
  test (same as paper). Also with the 2015-2020 canopy map on the same stretches.
- If Planetary Computer cannot be reached, U6 is not reported.

## U7 Heat, tree density and distance to the cities (descriptive)
- d_city = distance along the route to the nearer end (Dhaka km 0 or Chattogram end). Partial Spearman correlation of
  LST with trees per km and with canopy, controlling for d_city (rank residuals), with the circular-shift test.
- Priority score per stretch: z(LST) - z(log(1 + trees per km)), with median/MAD standardisation. Stability: 1,000
  bootstrap resamples of the trips covering each stretch; for each of the top-15 stretches, the share of resamples in which
  it stays in the top 15. Overlap of the top 10 with the ten hottest stretches. Presented as a ranking tool, not a cause.

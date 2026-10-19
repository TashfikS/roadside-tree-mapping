# E2b: Mapillary position check on Pasadena (rules fixed 14 Oct 2026, before any result)

**Question.** Are tree positions computed from Mapillary photos accurate enough for the highway application?
E2 tested positions from Google panoramas and E3 tested distances from phone photos; the highway map uses Mapillary photos.

## Data
- Mapillary images captured 2015–2018 (closest to the 2016 inventory) inside the Pasadena validation and test areas:
  12,588 perspective images in 86 sequences (val 1,443; test 11,145). Selection uses only image metadata, never the inventory.
- Thinned to one image per >= 8 m along each sequence (same rule everywhere); 1024-px thumbnails.

## Pipeline (no training on Pasadena or Mapillary)
- Detector: the Bangladesh YOLO11m-pose run with the highest validation trunk-point score (pose mAP50-95):
  run 1 (0.844; run 0 0.749, run 2 0.838). Confidence >= 0.25.
- Position: the ray through the detected trunk point (pixel -> Mapillary intrinsics with k1, k2 undistortion -> camera
  orientation -> world) meets flat ground at the camera height below the Mapillary camera position (computed_geometry).
- Capture type from the sequence's median speed: >= 3 m/s vehicle, otherwise hand-held (1.5 m, fixed: only 476 images).
- Repeat sightings merged with the same greedy merge as E2.

## Settings chosen on the validation area only, then applied once to the test area
- Stage 1, geometry (criterion: median distance from single-image positions with range <= 20 m to the nearest surveyed
  tree, among those within 10 m):
  orientation {Mapillary SfM rotation (heading + pitch + roll), level camera (heading only)} x vehicle camera height
  {1.3, 1.6, 2.0, 2.4} m.
- Stage 2, merging (criterion: AP at 4 m, official evaluator; ties -> lower median): range cut {15, 20} m x merge radius
  {3, 5, 7} m x minimum views {1, 2}.

## Test-area metrics (reported whatever they are)
- Zone = within 20 m of a used image position. Ground truth = official test inventory inside the zone.
- AP at 4 m and maximum recall (official evaluator); median and P90 distance of detections matched within 10 m (the E2 gate
  metric); the same metrics for the Google-panorama pipeline (E2, deduplicated) inside the same zone.
- Chance level: the chosen setting with each detection's bearing replaced by a random bearing (same range, seed 0).
- Sensitivity: vehicle camera height +-0.3 m around the chosen value.

## Amendment (14 Oct 2026, before any validation or test score)
- Orientation is fixed to the Mapillary SfM rotation; the level-camera variant is reported as a comparison only.
  Reason: on 9 sample photos the SfM horizon matches the road's vanishing point, while the GoPro HERO5 sequences (4,696 of
  12,588 images, all in the test area) are mounted pitched up by about 21 deg, which a level camera cannot represent.
  The validation area has no GoPro images, so a validation-chosen orientation could not detect this.
- Disclosure: a 32-image geometry smoke test (mixed validation/test images) was run while debugging; no setting was
  chosen from it.

## Amendment 2 (14 Oct 2026, after the validation results, before any test result)
- Correction to amendment 1: the validation area does contain GoPro images (306 of 703). The original validation rule
  also selects the SfM rotation (stage 1: 4.79 m vs 5.82 m for the best level-camera setting), so amendment 1 does not
  change the chosen setting. Validation choices: SfM rotation, vehicle camera height 1.6 m, range cut 20 m, merge 5 m,
  minimum views 1 (mly_val_stage1.csv, mly_val_stage2.csv).
- Validation showed that the gate metric (median distance to the nearest surveyed tree among detections within 10 m) is
  close to its chance level for Mapillary positions (4.78 m vs 5.33 m with random bearings), because street trees are
  dense. The gate for individual positions therefore also requires: test AP at 4 m >= half the Google-panorama AP at 4 m
  in the same zone. If not met, the highway application reports trees per km and empty stretches per 50 m.
- Added test metric for that fallback: trees per 100 m x 100 m grid cell (cells with >= 3 used image positions),
  Mapillary vs surveyed inventory and Google-panorama pipeline vs inventory: Pearson r and median absolute difference.

## Decision (same gate as E2)
- Test median <= 5 m -> the highway application maps individual candidate planting positions from Mapillary.
- Test median > 5 m -> the highway application ranks 50 m road stretches instead. Same paper either way.

## Result (14 Oct 2026, test area, applied once)
| Positions (test zone: 2,655 surveyed trees, 5,994 Mapillary photos) | AP @ 4 m | Max recall | Median (matched <= 10 m) | P90 |
|---|---|---|---|---|
| Mapillary, chosen setting | 0.021 | 0.133 | 5.24 m | 8.53 m |
| Mapillary, level camera | 0.017 | 0.147 | 6.02 m | 8.98 m |
| Mapillary, camera height 1.3 / 1.9 m | 0.022 / 0.018 | 0.141 / 0.115 | 5.28 / 5.35 m | 8.69 / 8.69 m |
| Chance (random bearing) | 0.008 | 0.081 | 6.12 m | 9.14 m |
| Google panoramas (E2), same zone | 0.464 | 0.742 | 2.31 m | 6.71 m |
- Trees per 100 m cell (300 cells): Mapillary r 0.34, finds 47 % of the surveyed count; Google r 0.60, 100 %.
- **Gate: FAIL** (median > 5 m and AP far below half of Google's) -> no individual planting positions from Mapillary.
- Diagnosis (not used for any decision): surveyed trees projected into the photos land at the kerb where the trees
  stand (mly_projection_check.jpg), so camera poses look roughly right; but the detector boxes only 39 % of surveyed trees
  visible at 4-15 m (GoPro 46 %, phones 20 %, Garmin 11 %), and 58 % of photos have no detection at all.
  Bottleneck: detector transfer to these photos (US trees, other cameras), not the Mapillary camera positions.

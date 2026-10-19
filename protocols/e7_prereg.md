# E7: comparison with a single-image metric depth model (14 Oct 2026, before any depth result)

Question a reviewer will ask: why not take each tree's distance from a monocular depth network instead of the
flat-ground geometry with a known camera height? Reported whatever the result; no pass mark.

## Model
Depth Anything V2, metric, outdoor, Large (fallback Base, then Small, only if Large fails to load; the model used is
recorded). Zero-shot: no tuning, no scale correction from our data except where stated (leave-one-out, Bangladesh).

## A. Pasadena (same detections, same bearings; only the range differs)
- Detections: the retrained detector's raw detections on the validation and test panoramas (61,092 boxes).
- Perspective views rendered from each 1664 x 832 panorama: 90 x 90 deg, 518 x 518 px, pitch 0, centres every 45 deg;
  each box is read in the view whose centre is nearest to the box centre in azimuth.
- Range = horizontal distance from the depth map: Z * sqrt(1 + ((x - cx) / f)^2) at the read-out pixel(s).
  Two read-outs: (point) median of the 5 x 5 pixels at the box bottom-centre; (region) median over the central 30 % of the
  box width and the bottom 15 % of its height.
- Position = camera + range along the same bearing as the ground-plane method. Boxes whose bottom is above the horizon
  (no ground-plane solution) are kept for the depth method.
- The depth method gets the same validation sweep as ours (read-out, range cut 15/20 m, merge 3/5/7/9 m, views >= 1/2,
  chosen by AP@4 m on the validation area), then the test area once. Compared with the ground-plane result already
  measured on test (AP@4 m 0.641, median 2.16 m). Also reported: median absolute range difference to the ground-plane
  range on the same boxes.

## B. Bangladesh phone photos (the 22 ARuler trees)
- z-depth at the clicked trunk base (median of 5 x 5 pixels) -> horizontal distance using the photo's focal length
  (EXIF) and the horizon-based camera pitch already used for our distances.
- Compared with ARuler exactly as ours: median absolute error, share within 2 m, uncalibrated and with the same
  leave-one-out single scale factor. Ours: 2.34 m uncalibrated, 1.42 m leave-one-out (n = 22).

## Result B (14 Oct 2026): Bangladesh, 22 ARuler trees
Model loaded: Depth-Anything-V2-Metric-Outdoor-Large-hf (all 6 jobs). Depth read at the clicked trunk base; converted to
horizontal distance with the photo's focal length and the same horizon-based pitch as ours (sign verified: the same
conversion applied to the ground-plane point reproduces our distances to within 0.02 m).
| Method | Median abs. error | Within 2 m | Median signed |
|---|---|---|---|
| Ours, fixed settings (h 1.5 m) | 2.34 m | 41 % | -2.24 m |
| Ours, leave-one-out scale | **1.42 m** | **68 %** | -0.02 m |
| Depth Anything V2 L, as is | 4.15 m | 18 % | +4.15 m (over by 32 %) |
| Depth Anything V2 L, leave-one-out scale | 2.29 m | 45 % | -0.10 m |
Paired comparison of absolute errors (leave-one-out): ours smaller on 15 of 22, Wilcoxon p = 0.021; as is p = 0.054.
Rank correlation with ARuler: ours 0.905, depth model 0.768. (e7_bd_results.csv, e7_bd_summary.csv)

## Result A (14 Oct 2026): Pasadena, test area once (same 61,092 retrained-detector boxes and bearings)
| Range from | Setting (chosen on validation) | AP@4 m | Max recall | Median error (matched <= 10 m) | Within 2 m (of matched <= 4 m) |
|---|---|---|---|---|---|
| Ground plane, camera height 3.0 m (ours) | cut 15 m, merge 5 m, views >= 1 | **0.641** | 0.851 | **2.16 m** | 61.9 % |
| Depth Anything V2 L, as is (primary) | point read-out, cut 15 m, merge 3 m, views >= 1 | 0.012 | 0.106 | 5.99 m | 27.7 % |
| Depth Anything V2 L x 0.44 (secondary, added after the primary result; k chosen on validation) | as ours | 0.622 | 0.848 | 2.41 m | 57.9 % |
- Depth ranges are 2.25 x the ground-plane ranges (median over 45,569 boxes), and the ratio is constant from 4 to 40 m
  (2.16–2.29 per distance bin): a scale error, as expected when the camera's field of view differs from the training data.
- With a scale factor fitted to surveyed trees the depth model comes close but stays below the ground-plane method, which
  needs no surveyed trees (its scale comes from the known camera height).

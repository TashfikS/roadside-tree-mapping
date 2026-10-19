# Highway transfer study: pre-registration (14 Oct 2026, before any labels or results)

Question: does the detector find roadside trees in Mapillary photos of the Dhaka–Chattogram highway, and does
training with labelled highway photos raise that to a level where a planting-gap map is justified?

## Data
- Route: Dhaka–Chattogram driving route (OSRM), 239.5 km. Mapillary photos captured from 2022 on, within 25 m of
  the route: 33,069 photos in 119 sequences.
- Photo-quality filter (fixed before any detection on these photos): a sequence is usable if fewer than half of
  4 evenly spaced 256-px thumbnails are blank (share of pixels > 235 above 60 %, or grey-level SD < 12). Each selected
  photo is checked the same way (also rejected if more than 70 % of pixels are < 25). 360° panoramas are excluded.
  Usable coverage: 171.1 km of the 222.5 km photographed since 2022; 86 of 119 sequences.
- The quick look of 14 Oct (120 photos at 15 km intervals, model boxes viewed on 12 of them) found 58 % blank photos
  (87 % of one phone model) and 0.31 boxes per usable photo. Those 120 photos are excluded from every set, and no test
  photo lies within 200 m of them.

## Split by road position (no photo-level randomisation)
- 10 km blocks along the route. Odd blocks are test. Even blocks are the labelling pool: blocks 6 and 16
  (km 60–70 and 160–170) are validation, the other even blocks are training. Photos within 250 m of a block edge are dropped.
- One usable photo per road position, positions at least 244 m apart (training/validation) and 428 m apart (test),
  chosen at random among photos within ±40 m. Random subsample to 240 training, 60 validation, 150 test photos
  (seeds 20261014 and 7). Lists: hw_photos.csv.

## Labels (by the authors, two packages)
- Training + validation (300 photos, hw_boxes): a box around every roadside tree that can be made out as a separate
  tree, crown to trunk base, following the conventions of the Kaggle dataset. Editable suggestions are shown: Grounding
  DINO base, prompt "a tree.", score >= 0.30, plus the current detector at confidence >= 0.10. The share of final
  boxes that are unchanged suggestions is reported. Photos may be marked "no roadside trees" or "unusable".
- Test (150 photos, hw_test): a point where each such tree's trunk meets the ground; hidden-but-guessable bases are
  flagged. **No model output is shown.** Unusable photos are excluded and their count reported.
- Trunk points for the training boxes are generated automatically with the same method as the Kaggle training labels.

## Training
- YOLO11m-pose with the main model's recipe; training data = Kaggle training split + highway training photos;
  validation = Kaggle validation split + highway validation photos.
- Two configurations, one run each: (A) as is; (B) plus windscreen-style augmentation on training images only
  (glare spots, rain drops, motion blur, low contrast, a dark band for dashboard or sun visor). The configuration with
  the higher box AP50 on the **highway validation photos** is kept, then trained with 2 more seeds (3 runs).
- The Kaggle test split is re-scored with the new model and reported, whichever way it goes.

## Test measure (applied once, after the choice above)
- A clicked tree is found if the click lies inside a predicted box with confidence >= 0.25, the box enlarged by 3 % of
  the image width and height on each side.
- Decision metric: share of clicked trees found, among trees whose ground range from the camera is <= 20 m
  (the range used by the map; computed from the click with the same camera geometry as the map: Mapillary camera model,
  structure-from-motion orientation, camera height 1.6 m vehicle / 1.5 m hand-held). Mean of the 3 runs.
- Also reported: all ranges; visible vs hidden bases; old model (3 runs) vs new model (3 runs), mean ± SD across runs,
  95 % CI by photo-level bootstrap; boxes not containing any click per photo.
- Decision (fixed in plan v8, unchanged): >= 70 % -> map of empty 50 m stretches with P(plantable) and heat ranking;
  40–70 % -> trees per km (corrected for the measured miss rate) with heat ranking, no gap map;
  < 40 % -> no highway map; the transfer limit is reported as a finding.

## Repeat-trip reliability (after the model is fixed)
- Where two or more trips cover the same 100 m stretch, trees per 100 m are computed from each trip separately:
  reported as correlation and median absolute difference between trips. No threshold; descriptive.

## Release
- The 450 labelled highway photos and their labels: CC BY-SA 4.0, as on Mapillary, with each photographer credited
  (hw_attribution.csv).

## Amendment 1 (14 Oct 2026, after the first test pass, before any model is run on the test photos)
- First pass received: 150 photos, 83 with trunk points (356 points; 164 visible, 192 hidden but guessed), 20 with no
  roadside trees, 47 unusable (31 %; ZTE 7160N 31 of 66).
- **Second pass, once, by the author, on all 150 photos:** first-pass marks shown and editable, no model output, plus
  condition tags per photo (blurred or rain; crowded or merged trees; blocked by vehicles or objects). The second pass is
  the answer key and is frozen when received. The share of photos and marks changed between passes is reported.
- No model, including the current one, is run on the test photos until the answer key is frozen. After freezing, labels
  are not changed for any reason, including the test result. (A re-check only when results are low could only move
  labels toward the model, so it is not allowed.)
- Secondary analysis: share of trees found by condition tag (untagged vs each tag).
- **Usability check for the map (new component):** because 31 % of quality-filtered photos are unusable to a person,
  a small image classifier (usable / unusable) is trained on the 300 labelling photos' marks, its threshold chosen on the
  60 validation photos, and tested once on the 150 test photos' second-pass marks. The map uses only photos it accepts;
  50 m stretches with no accepted photo are reported as "no data", never as gaps.

## Amendment 2 (14 Oct 2026): answer key frozen, labelling convention for occluded trunks
- The author chose not to do the second pass. The first pass is the final answer key, frozen now, before any model was
  run on the test photos: hw_test_answers.csv, SHA-256 52967953cd0996a02338849814f78934cefe8c07eeb85c68f2f4b8c2893adda1.
- Condition tags are therefore not available. Replacement secondary analyses, fixed now: share of trees found by photo
  sharpness (variance of the Laplacian on the 1024-px grey photo, tertiles over the 83 photos with marks) and by number of
  marked trees per photo (1–2, 3–5, 6 or more), plus visible vs guessed trunk bases (164 vs 192).
- Box convention for the 300 labelling photos (the author's question, answered before any training): one box per tree that
  can be made out separately, including trees whose trunk is hidden by a vehicle or object; such boxes extend down to the
  estimated trunk base. One box for a group only when the trees cannot be told apart. Crowns that may not stand beside this
  road are skipped.
- Training trunk points stay automatic (same method as the Kaggle training labels); where no trunk is found in the lower
  part of a box, the point is marked not visible, so that box trains tree finding only.

## Amendment 3 (14 Oct 2026, before any highway training, full-highway detection or canopy data)
Three additions that need no new labels (author's decision: do all three).

### A. Self-training on unlabelled highway photos (third candidate)
- Highway detector candidates, one run each, compared by box AP50 on the 60 highway validation photos:
  (A) as is; (B) windscreen-style augmentation; (C) the better of A and B plus self-training.
- Self-training: the better of A/B (seed 0) detects trees on usable highway photos from 2022 on in the **training
  blocks only** (even blocks except 6 and 16, outside the 250 m edge buffers; never test or validation blocks).
  Detections with confidence >= t become pseudo-labels, where t is the lowest threshold giving box precision >= 0.80
  (IoU 0.5) on the highway validation photos. At most 3,000 pseudo-labelled photos, at least 30 m apart within a trip.
  Trunk points as for the human boxes. C is trained from the same starting weights on Kaggle training + highway
  training (human) + pseudo-labelled photos.
- C replaces the better of A/B only if its validation AP50 is higher by at least 0.02. The winner is trained with
  2 more seeds and scored once on the test photos.

### B. Combining trips for the highway map
- All usable highway photos (blank filter, then the usability check) are processed; trees are located per photo as in
  Pasadena (range <= 20 m) and merged within each trip (5 m radius, the setting chosen on Pasadena validation).
- Across trips, per 50 m stretch and side: "with trees" if any covering trip has a tree there; trees per km = median over
  covering trips. Stretches with no usable photo = "no data".
- Reported: share of stretches empty using one trip (the trip with most usable photos) vs all trips combined.
- Secondary use of the test clicks: a clicked tree (<= 20 m) is "found by any trip" if a merged tree from any trip lies
  within 8 m of its located position; chance level from the same rule with the clicked positions moved 50 m along the
  road. **The 70 % / 40 % decision stays on the single-photo measure.**

### C. Satellite canopy comparison (descriptive)
- Data: Meta/WRI 1 m canopy height maps (CC BY 4.0), newest public version that is accessible; observation dates reported.
- Unit: 100 m stretches; corridor 0–20 m from the route line, both sides combined (the route line follows one
  carriageway of a divided highway). Canopy cover = share of pixels >= 3 m tall.
- Compared once, after the map is final: Spearman correlation between our trees per 100 m (trips combined) and canopy
  cover; share of our empty stretches with canopy cover < 10 % vs the same share for stretches with trees. No pass mark.

## Amendment 4 (14 Oct 2026, before any model is run on the highway test photos): camera orientation
- Finding (camera metadata and the frozen clicks only, no model output): Mapillary's computed orientations for these
  sequences are not usable for ground-plane distances. 45 % of the 18,695 usable photos carry an exactly level default
  orientation (|pitch| < 0.01 deg); the rest scatter widely (SD 2–5 deg per camera model, some beyond 25 deg); the
  Samsung SM-S9080 / SM-G996U sequences (1,656 photos) are stored upside down relative to the served photos. With
  these orientations 118 of the 356 test clicks have no ground intersection, and the road's vanishing point lies at
  0.64 of the image height for vivo V2022 photos (median, 29 test photos) instead of the assumed 0.50, which would
  roughly halve distances.
- **Decision metric (replaces the 20 m filter):** share of ALL clicked trees found (visible and guessed trunk bases;
  356 trees in 83 photos). The labeller already targeted nearby trees. Breakdowns as fixed in amendment 2.
- **Map geometry (fixed now, before any detection on highway photos):** level camera, heading = computed compass
  angle, roll 0 (upside-down sequences rotated 180 deg), pitch per sequence from the horizon: median vanishing-point
  row over up to 30 evenly spaced photos of the sequence, using only estimates inside 0.30–0.90 of the image height;
  a sequence needs >= 8 such estimates with an interquartile range <= 0.08 of the image height, otherwise it is
  treated as "no data" for positions. Camera height as before (1.6 m vehicle, 1.5 m hand-held).
- Sanity check reported (descriptive): lateral offset of clicked test trees from the camera under this geometry.

## Amendment 5 (14 Oct 2026, before the re-check answers that label the training photos): usability check details
- Labels: the 300 labelling photos; unusable = re-marked X in the re-check (hw_recheck.csv, part 1); all others usable.
  Test labels: the frozen first-pass statuses of the 150 test photos (unusable = X; 47 photos).
- Features: 8 image statistics on the 1024-px photo (grey-level mean and SD, share of pixels > 235 and < 25, variance of
  the Laplacian, edge density, mean saturation, dark-band share in the bottom 20 %) plus a 1,280-d ImageNet-pretrained
  embedding (YOLO11s-cls, global-average-pooled penultimate layer), standardised.
- Model: L2 logistic regression; C from {0.01, 0.1, 1, 10} by 5-fold cross-validated AUC on the 240 training photos.
- Threshold: Youden's J on the 60 validation photos; if they contain fewer than 5 unusable photos, Youden's J on the
  cross-validated training predictions instead.
- Test once: AUC, share of usable and of unusable photos classified correctly. The map uses photos the check accepts.

## Amendment 6 (14 Oct 2026, before any temperature data): heat ranking
- Data: Landsat 8 and 9 Collection 2 Level-2 surface temperature (band ST_B10, kelvin = DN x 0.00341802 + 149.0),
  served free by Microsoft Planetary Computer (collection landsat-c2-l2).
- Season and years: hot pre-monsoon season, 1 March to 31 May, 2022–2025 (the period of the Mapillary photos).
  Scenes with cloud cover < 40 %; pixels flagged cloud, dilated cloud, cirrus or cloud shadow in QA_PIXEL are removed.
- Composite: per-pixel median of all clear observations, in degrees C; pixels with fewer than 3 clear observations are
  left empty.
- Unit: 1 km route stretches; value = median of composite pixels within 50 m of the route line.
- Use: every map stretch (gap or trees-per-km, whichever the transfer rule allows) gets the value of its 1 km stretch and
  is ranked from hottest to coolest. Descriptive only: Spearman correlation between stretch temperature and our trees per
  km, and between temperature and satellite canopy cover. No threshold.
- Correction to amendment 5 (14 Oct 2026, before any usability label exists): the YOLO11s-cls pooled embedding returned
  by the library has 512 dimensions, not 1,280 as written. Same model and layer; nothing else changes.
  Features for the 450 labelled photos: usab_features.py -> usab_features_450.parquet.

## Usability check: result (14 Oct 2026; amendment 5 applied as written, test once)
- Labels: 300 labelling photos (unusable: 71 of 240 training, 16 of 60 validation, all from the re-check); test: the
  frozen first-pass statuses (47 of 150 unusable).
- C = 0.01 (cross-validated AUC 0.884 on training); threshold P(usable) >= 0.553 (Youden's J on validation; validation
  AUC 0.619, 95.5 % of usable kept, 31.2 % of unusable rejected).
- **Test: AUC 0.775; 92.2 % of usable photos accepted, 44.7 % of unusable photos rejected; 80.7 % of photos accepted.**
  (usability_result.json, usability_model.pkl). The check removes about half of the unusable photos; the rest are
  handled by combining trips (amendment 3B) and are a stated limitation.

## Amendment 7 (14 Oct 2026, before any self-training or full-highway detection): clarifications
- "Usable" for self-training and for the map = passes the blank filter (amendment of 14 Oct) AND accepted by the
  usability check above. 360-degree panoramas excluded.
- Self-training candidates: usable photos from 2022 on, <= 25 m from the route, in training blocks (even blocks except
  6 and 16), > 250 m from block edges, not among the 450 study photos or the 120 quick-look photos. Spacing: within each
  trip, in capture order, keep a photo only if it is >= 30 m from the last kept one.
- A pseudo-labelled photo has >= 1 detection with confidence >= t; candidates without one are not used. If more than
  3,000 qualify, a random 3,000 are used (seed 20261015). Trunk points for pseudo-boxes: the same automatic method as
  for the human boxes, run inside the training job of candidate C.
- The map uses the final detector's seed-0 run. Photos are processed in chunks of <= 800 on Kaggle (job size limit);
  this changes nothing in the method.

## Amendment 8 (14 Oct 2026, before any full-highway detection): map construction details
- Photos: usable (amendment 7) photos of trips with a valid horizon (amendment 4); others are "no data".
- Detections: confidence >= 0.25 (as in the test measure). Point located: the detector's trunk keypoint when its
  confidence >= 0.5, otherwise the box bottom-centre.
- Geometry (amendment 4): level camera with the trip's horizon pitch, heading = computed compass angle, Mapillary camera
  model (perspective or fisheye), ground plane at 1.6 m (vehicle trips, median speed >= 3 m/s) or 1.5 m below the camera.
  Kept: horizontal range 2–20 m.
- Merge within each trip: the Pasadena-validated rule (greedy, score-ordered, 5 m radius).
- Road reference: the OSRM route line; side = sign of the cross product with the travel direction; trees > 25 m from the
  line dropped. 50 m pieces per side.
- Coverage: a trip covers a piece if it has a usable photo whose camera projects onto that piece. Pieces covered by no
  trip are "no data".
- Combining trips (amendment 3B): a piece-side has trees if any covering trip has a merged tree in it; trees per km per
  1 km stretch = sum over its pieces of the median count across covering trips, divided by the covered length.
- Outputs follow the transfer decision: >= 70 % -> empty pieces (no tree in any covering trip) with P(plantable)
  from Monte Carlo along-road position errors (Pasadena E2 error distribution) and heat ranking; 40–70 % -> trees per
  km divided by the measured recall (mean of the 3 runs), with heat ranking; < 40 % -> no map.
- E8 repeat-trip reliability: 100 m pieces covered by >= 2 trips; trees per 100 m per trip; Spearman correlation and
  median absolute difference between trip pairs.

## Result: A vs B (15 Oct 2026, highway validation photos only)
- A (as is) box AP50 0.561; **B (windscreen-style augmentation) 0.594** -> B is the better of A/B and the base of C.
- Bangladesh Google test split (native scorer, for information): A 0.671, B 0.654.

## Self-training: pseudo-labels (15 Oct 2026, rule applied as written)
- B (seed 0) on the 2,753 candidates (Kaggle CPU, 4 batches): detections on 2,162 photos.
- Threshold t = 0.329 (lowest confidence with box precision >= 0.80 on the 44 highway validation photos: 0.803;
  recall there 0.329). Pseudo-labelled: 1,123 photos, 1,668 boxes (below the 3,000 cap). hw_pseudo.json.
- C = B recipe + these photos (trunk points by the same automatic method inside the job), seed 0: rt-hwc-s0.
- Run before the C decision, with no effect on it: B seeds 1 and 2 (needed if C does not win), and map detections
  with B seed 0 (replaced by C seed 0 if C wins).

## Diagnostics on the provisional map (B seed 0; 15 Oct 2026; not decisions, repeated on the final model)
- Map: 11,075 photos, 40 trips, 126.3 km covered; 2,536 merged trees; empty 50 m piece-sides 75.0 % (one trip) vs 70.2 %
  (trips combined); median 12 trees per km (trips combined).
- E8 (pre-registered, 100 m): Spearman 0.11 over 1,131 trip pairs (345 pieces); median |difference| 1 tree per 100 m.
- Secondary, added after seeing E8: 1 km stretches (trips covering >= 0.5 km): Spearman 0.17 (111 pairs, 35 stretches);
  same-direction pairs 0.01 (59), opposite-direction pairs 0.31 (52). Opposite carriageways are therefore not the cause;
  per-trip counts are small (median 4 trees per km per trip), so counting noise dominates single-trip counts.

## Result: C vs B and final model (15 Oct 2026, highway validation photos only)
- C (B + self-training, 1,123 pseudo-labelled photos) box AP50 0.556 vs B 0.594 -> C does not replace B (needed >= 0.614).
- Final highway model: **B**, seeds 0/1/2 (highway validation box AP50 0.594 / 0.626 / 0.563).
- Next and only once: the three B runs and the three original main-model runs (YOLO11m-pose, no highway photos) on the
  150 test photos; hw_score_test.py.

## Result: highway transfer test (15 Oct 2026, scored once; hw_test_results.json, hw_test_results_runs.csv)
- Share of the 356 clicked trees (83 photos) inside a confident box: original main model 37.9 % (runs 50.3 / 26.7 / 36.8;
  95 % CI 29.8-46.7); highway model B **58.8 %** (61.0 / 57.6 / 57.9; 95 % CI 50.8-66.5).
- **Decision (pre-registered): 40-70 % -> trees per km with heat ranking; no individual planting gaps on the highway.**
  Trees per km are divided by the measured share found (0.588).

## Result: highway outputs (15 Oct 2026; map final = B seed 0; hw_outputs.json)
- Final map: 11,075 photos, 40 trips, 126.3 km covered; 2,536 merged trees; trees per km divided by 0.588.
- Heat ranking (amendment 6; 128 stretches with >= 0.5 km covered; hw_heat_ranking.csv): Spearman temperature vs our
  trees per km -0.21 (p 0.017); temperature vs satellite canopy -0.30 (p 0.0005). Hottest 10 stretches: median 2.9
  trees per km (corrected); coolest 10: 38.3.
- Satellite canopy (amendment 3C, compared once): 100 m stretches (n 1,281): Spearman our trees vs canopy cover 0.35
  (p 1e-38); 36.3 % of stretches have no tree in any trip; of these 59.8 % have canopy < 10 %, vs 37.6 % of stretches
  with trees (median canopy 5.4 % vs 17.2 %). At 1 km (secondary): Spearman 0.48 (p 8e-9).
- Route distances aligned through the shared polyline vertices (map 239.76 km local projection, UTM 239.12 km,
  geodesic 239.47 km).

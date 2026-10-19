# Data made for this study

All files are released under CC BY 4.0. Images are not included: the Bangladesh street-view images are in the
[Kaggle data set](https://www.kaggle.com/datasets/tashfiksalehin/roadside-trees-of-bangladesh-street-view), and the
Mapillary photos can be downloaded by id through the Mapillary API (CC BY-SA 4.0, contributor names in `creator`).

| File | Content | Paper |
|---|---|---|
| `bangladesh_split.csv` | route-grouped split of the 1,765 images: `folder`, `id`, image size, number of boxes, road-stretch `group`, `split` (train/val/test) | §3.1 |
| `trunk_pose_labels.json` | YOLO pose labels per image (`class xc yc w h ku kv visibility`, normalised): boxes plus trunk-base point | §4.1 |
| `bangladesh_test_trunk_review.csv` | the 789 test boxes with reviewed trunk-base points: human point (`hu`,`hv`), automatic proposal (`au`,`av`), box bottom (`bu`,`bv`), visibility (2 visible, 1 hidden but inferable, 0 skipped), `accepted` (proposal kept unchanged), proposal method; pixel coordinates | §3.1, §5.1 |
| `field_trees_bangladesh.csv` | 26 field trees: photo, ARuler distance, image size, clicked trunk base (pixels), visibility | §3.3, §5.3 |
| `field_gps_vs_aruler.csv` | field sites with phone GPS pins of camera and tree, ARuler distance and EXIF data | §5.3 |
| `pasadena_human_labels.json` | human tree boxes on Pasadena panoramas (training and validation areas) used by the teacher detector | §4.1 |
| `highway_photos.csv` | the 450 sampled highway photos: Mapillary `id`, `split`, camera make/model, capture time, position, km along the route, `creator` | §3.4 |
| `highway_train_labels.json` | labels of the 213 usable training/validation highway photos: `lines` in YOLO pose format (box + trunk-base point); `mask` lists boxes around groups of trees that could not be separated, which are blanked out during training (normalised xyxy) | §3.4, §5.5 |
| `highway_test_trunk_points.csv` | clicked trunk bases in the 150 test photos (`u`,`v` normalised; `visible`; photo `status`) | §3.4, §5.5 |
| `highway_test_status_pass1.csv` | first-pass status of every test photo (`done`, `none`, `unusable`), the frozen labels of the usability test | §3.4 |
| `highway_usability_labels.csv` | usable/unusable labels of the 300 training and validation photos | §3.4 |

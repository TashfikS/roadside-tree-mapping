# Results

Tables and summaries produced by the scripts in `src/` and the Kaggle jobs; the figure scripts and the paper's
numbers are read from these files.

| File | Content |
|---|---|
| `audit/audit_values.json` | every number quoted in the paper with its source file (collected for the internal audit) |
| `detector_unified_summary.csv`, `detector_unified_coco_scores.csv` | Table 3: all detectors scored with one COCO evaluator on the Bangladesh test split |
| `trunk_point_main_model_3runs.csv` | trunk-base point error (pixels) of the main model and of the box bottom, per run |
| `split_stats.json` | near-duplicate search and split sizes |
| `pas_test_gt.csv` | Pasadena test-area reference records used by the evaluator |
| `pas_e2_test_trees_dedup.csv`, `pas_student_test_trees.csv` | mapped trees on the Pasadena test area (first and retrained detector) |
| `pas_e2_dedup_comparison.csv`, `pas_student_decision.json` | merge settings chosen on validation and their test scores |
| `pas_e7_result.json`, `pas_e7_scaled_result.json`, `pas_e7_test_trees.csv` | monocular depth comparison on Pasadena |
| `mly_test_results.csv`, `mly_test_results.json` | Mapillary photos of Pasadena |
| `field_distances.csv`, `e3_summary_v2.csv`, `e7_bd_results.csv`, `e7_bd_summary.csv` | Bangladesh field distances: ours and the depth model |
| `pas_student_gap_validation_test.csv`, `pas_student_gap_city_sites_test.csv` | planting gaps against the inventory and the city's site records |
| `pas_density_validation.csv`, `pas_density_validation_student.csv` | trees per km and empty share per street side |
| `pas_detect_by_dbh.csv` | share of inventory trees found, by trunk diameter |
| `vlm_result.json`, `vlm2_result.json` | vision-language second look (first and retrained detector) |
| `usability_result.json` | highway photo usability check |
| `hw_val_box_ap50.json`, `detector_hw_effect.csv`, `hw_pseudo_info.json` | highway detector configurations on highway validation photos |
| `hw_test_results.json`, `hw_test_results_runs.csv`, `hw_test_preds_hwb_s0.csv` | highway transfer test (356 marked trees) |
| `hw_km.csv`, `hw_summary.json`, `hw_outputs.json`, `hw_heat_ranking.csv` | highway tree density per km and heat ranking |
| `chm_corridor_100m.csv`, `lst_corridor_1km.csv`, `lst_canopy_1km.csv` | satellite canopy cover and surface temperature along the route |
| `test_preds_pose_s0.csv` | Predictions of the main keypoint model (run s0) on the Bangladesh test split: image, normalised box, score, trunk-base point; used by `figures/make_fig_examples.py` |

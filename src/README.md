# Source code

Run from a scratch folder (`work/`) with `PYTHONPATH` pointing here; intermediate files are written to the current
folder. Paths to external data come from the environment variables listed in the main README.

**Geometry and shared modules**
- `treegeo.py`: ground-plane range and bearing for perspective photos and equirectangular panoramas; map coordinates.
- `e3_horizon.py`: horizon row from the dominant vanishing point of oblique line segments.
- `hw_geo.py`, `mly_geo.py`: Mapillary camera models (OpenSfM conventions) for the highway and Pasadena photos.
- `pas_gaps.py`: street pieces from the panorama network, side assignment, empty kerb length for a spacing `S`.
- `vlm_render.py`: crop rendering shared by the local check and the Kaggle job.
- `test_treegeo.py`: round-trip test of the geometry against the benchmark's own projection code.

**Pasadena (positions, depth, gaps)**
- `pas_e2.py <weights> <prefix>`: detection, geolocation, merging, evaluation with the benchmark evaluator.
- `pas_compare_student.py`: pre-specified comparison of the first and retrained detectors on validation.
- `pas_e7.py`, `pas_e7_scaled.py`: Depth Anything V2 in place of the flat-ground geometry.
- `pas_student_gaps.py`, `pas_density.py`, `pas_density_student.py`: planting gaps and per-side densities.
- `mly_setup.py`, `mly_detect.py`, `mly_test.py`: Mapillary photos of the Pasadena areas.
- `vlm_targets.py`, `vlm2_targets.py`, `vlm_check.py`: candidate spots and city sites for the vision–language test.

**Bangladesh field photos**
- `field_distances.py`: distance from one phone photo against ARuler (reproduces `results/field_distances.csv`).
- `make_review.py`: offline review page used to check trunk-base points and to click field trees.

**Dhaka–Chattogram highway**
- `n1_cov.py`: route (OSRM) and Mapillary coverage along it (needs `MAPILLARY`).
- `hw_download_all.py`: downloads the highway photos.
- `make_hw_packages.py`, `make_hw_labels.py`: labelling packages and training labels.
- `make_hw_train.py`, `make_hw_train_c.py`, `hw_selftrain_prep.py`, `make_hw_chunks.py`: Kaggle jobs for training and inference.
- `hw_test_predict.py`, `hw_score_test.py`: the transfer test on the 150 test photos.
- `usab_features.py`, `hw_usability.py`: photo usability check.
- `hw_horizon.py`, `hw_map.py`, `hw_outputs.py`: per-trip horizon, tree density per km, heat ranking and canopy comparison.
- `chm_corridor.py`, `lst_corridor.py`: satellite canopy cover and Landsat surface temperature along the route.

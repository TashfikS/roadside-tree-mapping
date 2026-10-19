# Model weights

Attached to the GitHub release `v1.0`. Download into `weights/`.

| File | Model | Trained on | Used in | SHA-256 |
|---|---|---|---|---|
| `bangladesh_trunkpose_yolo11m_s1.pt` | YOLO11m-pose, 1024 px | Bangladesh benchmark, training split (seed 1) | Table 3 (one of the three seeds); Mapillary photos of Pasadena (§5.2) | `b5e9de6bc952a401df8269726718cbc6b40ab536a67423da3ccd1fa69aa766fb` |
| `pasadena_retrained_student.pt` | YOLO11m, 1280 px | Pasadena inventory-verified pseudo-labels, training areas | retrained Pasadena detector (AP@4 m 0.64), planting gaps | `570bb3301579b4b1265d6923389344d5bc9823d9ff8b3cd2eea4ae19ec8edfb2` |
| `pasadena_first_yolo11s.pt` | YOLO11s, 832 px | Pasadena human-labelled panoramas, training areas | first Pasadena detector (AP@4 m 0.49) | `35d0f370ec8be05fc9ee25b11ba14f9b217462a6459d1b613b0446007328e1e2` |
| `highway_trunkpose_hwb_s0.pt` | YOLO11m-pose, 1024 px | Bangladesh benchmark + 169 highway photos (configuration B, seed 0) | highway map and test | `5dbc5a6971452500e02464c6dad05c40b9e16414fc82a6a76f67fae511bf2d4c` |
| `highway_trunkpose_hwb_s1.pt` | YOLO11m-pose, 1024 px | as above, seed 1 | highway test | `25fa45f96f13eaa87978d6bb91ba5da951fcb089230254c123f328dacb2767a9` |
| `highway_trunkpose_hwb_s2.pt` | YOLO11m-pose, 1024 px | as above, seed 2 | highway test | `8e518e56b526e4cfd7832b4b9e2248f0fa89c57a08460b903614448500ff016b` |

Check a download with `sha256sum weights/*.pt`.

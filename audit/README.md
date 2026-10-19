# Audit of the earlier result (90.7 % AP50)

The earlier version of this work (SSRN 4500701) reported AP50 = 90.7 % for YOLOv4. These scripts reproduce that
number with the original weights and show where it came from.

| File | What it does |
|---|---|
| `match_aug.py` | matches every flipped or blurred copy in the earlier data set to its original scene, and lists which validation images show a scene that is also in the training list |
| `infer_old.py` | runs the original YOLOv4 weights (OpenCV DNN, 832 × 832, score ≥ 0.005, NMS 0.45) on an image list |
| `cfg_thresh0005.cfg` | the original darknet configuration with the detection threshold used for scoring |
| `score_old.py` | darknet-equivalent all-point AP at IoU 0.5 and 0.75 |
| `fast_ap.py`, `audit_e2.py` | Pasadena AP@4 m re-implemented and checked against the benchmark's evaluator; spatial block bootstrap (400 m blocks) |
| `audit_hw.py` | independent re-scoring of the highway test |

Result (`../results/audit/audit_values.json`): the original weights give 91.3 % on a random sample of 700 images of the earlier validation list (91.8 %
of the list's images show a scene that is also in training), and 53.1 % on the earlier held-out test set.

# Kaggle GPU jobs

Kaggle scripts cannot read this repository, so `build_jobs.py` writes self-contained scripts with the split
(`data/bangladesh_split.csv`) and labels (`data/trunk_pose_labels.json`, `data/pasadena_human_labels.json`) embedded.
The generated `pose_s1`, `yolo11m_s1`, `yolo11m_s2`, `rtdetr_l_s0` and `frcnn` scripts are byte-identical to the scripts
that produced the paper's results.

```bash
python kaggle_jobs/build_jobs.py --user <kaggle-username> --out work/kjobs
kaggle kernels push -p work/kjobs/pose_s0          # needs the Kaggle CLI and an API token
kaggle kernels output <kaggle-username>/rt-pose-s0 -p work/kout/pose_s0
```

Attach the Bangladesh data set (`tashfiksalehin/roadside-trees-of-bangladesh-street-view`) to every Bangladesh job,
and a Kaggle copy of the Pasadena Urban Trees benchmark named `pasadena-urban-trees` (folder with `streetview/`,
`panos.json`, `rois.json`, `detection_datasets.json`) to the Pasadena jobs. Outputs are expected in `work/kout/<job>/`
by the scripts in `src/`.

| Job | Template | Settings |
|---|---|---|
| `split` | `split_by_route.py` | road-stretch groups from consecutive image numbers; perceptual-hash and thumbnail-correlation near-duplicate search |
| `trunk_labels` | `trunk_labels.py` | Grounding DINO base (prompt "a tree trunk.", threshold 0.2) inside the lower box, SAM ViT-B mask, mean column of the lowest 5 % of the mask (≥ 3 px) |
| `pose_s0`–`pose_s2` | `trunk_pose.py` | YOLO11m-pose, 1024 px, 120 epochs, early stopping 15, batch 8, horizontal flip 0.5, seeds 0–2 |
| `posel_s0`–`posel_s2` | `trunk_pose.py` | YOLO11l-pose, 1280 px, otherwise as above |
| `yolo11m_s1`, `yolo11m_s2` | `box_detector.py` | YOLO11m, 1024 px, 120 epochs, early stopping 30 |
| `yolo832_s0`–`yolo832_s2` | `box_detector.py` | YOLO11m, 832 px, early stopping 15 |
| `rtdetr_l_s0`–`rtdetr_l_s2` | `box_detector.py` | RT-DETR-L, 832 px, 80 epochs, AdamW, lr 1e-4, early stopping 15, batch 4 |
| `frcnn` | `faster_rcnn.py` | Faster R-CNN ResNet-50 FPN v2 (COCO weights), 1024 px, 30 epochs, SGD with one-cycle schedule, best validation epoch, seeds 0–2 |
| `gdino_zs` | `grounding_dino_zeroshot.py` | Grounding DINO base, prompt "a tree.", no training |
| `pas_teacher` | `pasadena_teacher.py` | teacher YOLO11m on human labels; a detection becomes a label if it lies within 4 m of an inventory tree and within 20 m of the camera (score ≥ 0.25), or has score ≥ 0.60; a panorama is used only if every inventory tree within 15 m is matched |
| `pas_student` | `pasadena_student.py` | YOLO11m retrained on the verified pseudo-labels (attach the `pas_teacher` output as a kernel source) |
| `e7_pas_a`, `e7_pas_b` | `depth_pasadena.py` | Depth Anything V2 metric outdoor Large on 90° perspective views (518 px) of each panorama; point and region read-out per box |
| `e7_bd_0`–`e7_bd_5` | `depth_bangladesh.py` | the same model at the clicked trunk bases of the 22 field photos (CPU) |
| `hw_suggest` | `highway_suggestions.py` | Grounding DINO box suggestions shown (editable) in the highway labelling package |
| `vlm_gap` | `vlm_second_look.py` | Qwen3-VL-4B-Instruct; crops built by `src/vlm2_targets.py` (`work/vlm2_crops.b64`) and rendered by `src/vlm_render.py` |

The highway models are built by `src/make_hw_train.py` (configurations A and B) and `src/make_hw_train_c.py`
(self-training, configuration C) from the generated `pose_s1` job; the highway photos are fetched inside the job
from their Mapillary thumbnail links; `trunk_labels_highway.py` adds automatic trunk-base points to the highway boxes.

The first trunk-base run (`pose_s0`) used early stopping 30; later runs used 15 so that each run fits one Kaggle
session. The YOLOv4 row of Table 3 uses the original darknet configuration (`audit/cfg_thresh0005.cfg`).

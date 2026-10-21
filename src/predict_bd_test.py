"""Predictions of a Bangladesh keypoint model on the test split (Test R). Output: results/test_preds_pose_s1.csv"""
import os, sys
import numpy as np, pandas as pd
from PIL import Image
from ultralytics import YOLO
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BD = os.environ.get('BD_DATASET_DIR', os.path.join(ROOT, 'data', 'external', 'roadside-trees-of-bangladesh-street-view'))
m = YOLO(os.environ.get('BD_POSE_WEIGHTS', os.path.join(ROOT, 'weights', 'bangladesh_trunkpose_yolo11m_s1.pt')))
S = pd.read_csv(os.path.join(ROOT, 'data', 'bangladesh_split.csv')); S = S[S.split == 'test']
rows = []
for r in S.itertuples():
    sub = 'Test' if r.folder.startswith('BD_Test') else 'Train'
    im = Image.open(f'{BD}/{sub}/{r.folder}/{r.id}.jpg').convert('RGB'); W, H = im.size
    res = m.predict(np.asarray(im)[:, :, ::-1].copy(), imgsz=1024, conf=0.001, verbose=False, device='cpu')[0]
    for (a, b, c, d), s, k in zip(res.boxes.xyxy.numpy(), res.boxes.conf.numpy(), res.keypoints.data.numpy()[:, 0, :]):
        rows.append(dict(image=f'{r.folder}_{r.id}.jpg', x0=a / W, y0=b / H, x1=c / W, y1=d / H, score=s, ku=k[0] / W, kv=k[1] / H, kconf=k[2]))
pd.DataFrame(rows).to_csv(os.path.join(ROOT, 'results', 'test_preds_pose_s1.csv'), index=False); print(len(S), len(rows))

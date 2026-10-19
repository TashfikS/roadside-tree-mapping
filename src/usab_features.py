"""Usability-check features (hw_prereg.md, amendment 5) for photos in a folder: 8 image statistics + YOLO11s-cls embedding."""
import os, sys, glob, numpy as np, pandas as pd, cv2
os.environ.setdefault('KMP_AFFINITY', 'disabled')
from ultralytics import YOLO
paths = sorted(glob.glob(sys.argv[1] + '/*.jpg')); out = sys.argv[2]
def stats(p):
    im = cv2.imread(p); g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32); hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h = g.shape[0]; bot = g[int(0.8 * h):]
    return [g.mean(), g.std(), (g > 235).mean(), (g < 25).mean(), cv2.Laplacian(g, cv2.CV_32F).var(),
            (cv2.Canny(g.astype(np.uint8), 100, 200) > 0).mean(), hsv[..., 1].mean(), (bot < 25).mean()]
S = np.array([stats(p) for p in paths], np.float32)
m = YOLO('yolo11s-cls.pt'); E = []
for i in range(0, len(paths), 32):
    E += [e.cpu().numpy().ravel() for e in m.embed(paths[i:i + 32], imgsz=224, verbose=False)]
E = np.stack(E)
cols = ['grey_mean', 'grey_sd', 'share_gt235', 'share_lt25', 'lap_var', 'edge_density', 'sat_mean', 'bottom_dark']
df = pd.concat([pd.DataFrame({'id': [os.path.basename(p)[:-4] for p in paths]}), pd.DataFrame(S, columns=cols),
                pd.DataFrame(E, columns=[f'e{j}' for j in range(E.shape[1])])], axis=1)
df.to_parquet(out); print(len(df), 'photos', E.shape[1], 'embedding dims')

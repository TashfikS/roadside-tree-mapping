"""Distance from one phone photo to a roadside tree, compared with the ARuler field measurement (paper Section 5.3).

For each field photo: focal length in pixels from the EXIF 35-mm-equivalent focal length, camera pitch from the horizon
(vanishing point of oblique line segments, e3_horizon.py), camera height 1.5 m, and the annotated trunk-base point.
Input : data/field_trees_bangladesh.csv and the field photos (FIELD_PHOTO_DIR; part of the Kaggle data set).
Output: results/field_distances.csv
"""
import os, sys, math
import numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import treegeo as tg
import e3_horizon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHOTOS = os.environ.get('FIELD_PHOTO_DIR', os.path.join(ROOT, 'data', 'external', 'field_photos'))
SENSOR_W, SENSOR_H, FULLFRAME_DIAG_MM = 4608, 3456, 43.267   # full 4:3 sensor; 16:9 and 1:1 modes are centre crops
H_CAM = 1.5

T = pd.read_csv(os.path.join(ROOT, 'data', 'field_trees_bangladesh.csv'))
rows, horizon = [], {}
for r in T.itertuples():
    p = os.path.join(PHOTOS, r.photo)
    f35 = float(Image.open(p).getexif().get_ifd(0x8769).get(0xA405) or 0)
    f_px = f35 / FULLFRAME_DIAG_MM * math.hypot(SENSOR_W, SENSOR_H)
    if p not in horizon: horizon[p] = e3_horizon.horizon_row(p)
    h = horizon[p]; cx, cy = r.width / 2, r.height / 2
    pitch = math.degrees(math.atan((h['vp_v'] - cy) / f_px)) if h else 0.0
    d = np.nan
    if r.visibility > 0:
        res = tg.locate_tree(r.trunk_u, r.trunk_v, dict(lat=0, lon=0, compass=0, f_px=f_px, cx=cx, cy=cy, h_cam=H_CAM, pitch=pitch))
        d = res['range_m'] if res else np.nan
    rows.append(dict(key=r.key, photo=r.photo, aruler_m=r.aruler_m, visible=r.visibility > 0, f_px=round(f_px, 1), pitch_deg=round(pitch, 2), distance_m=d))
D = pd.DataFrame(rows); D['error_m'] = D.distance_m - D.aruler_m
os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True); D.to_csv(os.path.join(ROOT, 'results', 'field_distances.csv'), index=False)
v = D[D.visible]
print(f"{len(v)} trees with a visible trunk base: median |error| {v.error_m.abs().median():.2f} m, median ratio {np.median(v.distance_m / v.aruler_m):.2f}")

"""Test F (protocols/trunk_ablation_prereg.md): field distances from the predicted trunk-base keypoint and from the box
bottom-centre of the same detection, against ARuler. Output: results/field_keypoint_ablation.csv"""
import os, sys, math
import numpy as np, pandas as pd
from PIL import Image
from scipy.stats import wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
import treegeo as tg, e3_horizon
from ultralytics import YOLO
PHOTOS = os.environ.get('FIELD_PHOTO_DIR', os.path.join(ROOT, 'data', 'external', 'field_photos'))
WEIGHTS = os.environ.get('BD_POSE_WEIGHTS', os.path.join(ROOT, 'weights', 'bangladesh_trunkpose_yolo11m_s1.pt'))
SENSOR_W, SENSOR_H, FF, H_CAM = 4608, 3456, 43.267, 1.5
T = pd.read_csv(os.path.join(ROOT, 'data', 'field_trees_bangladesh.csv')); T = T[T.visibility > 0].reset_index(drop=True)
m = YOLO(WEIGHTS); cache, rows = {}, []
for r in T.itertuples():
    p = os.path.join(PHOTOS, r.photo)
    if p not in cache:
        im = Image.open(p); assert im.size == (r.width, r.height), (p, im.size)
        arr = np.asarray(im.convert('RGB'))[:, :, ::-1].copy()
        res = m.predict(arr, imgsz=1024, conf=0.25, verbose=False, device='cpu')[0]
        ex = im.getexif().get_ifd(0x8769); f_px = float(ex.get(0xA405) or 0) / FF * math.hypot(SENSOR_W, SENSOR_H)
        h = e3_horizon.horizon_row(p); pitch = math.degrees(math.atan((h['vp_v'] - r.height / 2) / f_px)) if h else 0.0
        cache[p] = (res.boxes.xyxy.numpy(), res.keypoints.data.numpy()[:, 0, :], f_px, pitch)
    B, K, f_px, pitch = cache[p]
    inside = np.where((B[:, 0] <= r.trunk_u) & (r.trunk_u <= B[:, 2]) & (B[:, 1] <= r.trunk_v) & (r.trunk_v <= B[:, 3] + 0.03 * r.height))[0]
    cam = dict(lat=0, lon=0, compass=0, f_px=f_px, cx=r.width / 2, cy=r.height / 2, h_cam=H_CAM, pitch=pitch)
    dist = lambda u, v: (tg.locate_tree(u, v, cam) or {}).get('range_m', np.nan)
    row = dict(key=r.key, aruler_m=r.aruler_m, click_m=dist(r.trunk_u, r.trunk_v), n_boxes=len(B), found=len(inside) > 0)
    if len(inside):
        # several boxes contain the click: take the one whose bottom-centre is closest to the click
        j = inside[np.argmin(np.hypot((B[inside, 0] + B[inside, 2]) / 2 - r.trunk_u, B[inside, 3] - r.trunk_v))]
        ku, kv, kc = K[j]
        bu, bv = (B[j, 0] + B[j, 2]) / 2, B[j, 3]
        gu, gv = (ku, kv) if kc >= 0.5 else (bu, bv)
        row.update(kconf=kc, keypoint_m=dist(gu, gv), boxbottom_m=dist(bu, bv),
                   kp_px_from_click=math.hypot(gu - r.trunk_u, gv - r.trunk_v), bb_px_from_click=math.hypot(bu - r.trunk_u, bv - r.trunk_v))
    rows.append(row)
D = pd.DataFrame(rows)
def loo(d, a):
    out = np.full(len(d), np.nan); ok = ~np.isnan(d)
    for i in np.where(ok)[0]:
        o = ok.copy(); o[i] = False; out[i] = d[i] * np.median(a[o] / d[o])
    return out
for c in ('click_m', 'keypoint_m', 'boxbottom_m'):
    if c in D: D[c.replace('_m', '_loo_m')] = loo(D[c].values.astype(float), D.aruler_m.values)
D.to_csv(os.path.join(ROOT, 'results', 'field_keypoint_ablation.csv'), index=False)
f = D[D.found]
print(f'trees {len(D)}, found by a box {len(f)}')
for c in ('click', 'keypoint', 'boxbottom'):
    for s in ('_m', '_loo_m'):
        e = (f[c + s] - f.aruler_m).abs(); print(f'{c+s:16s} median |err| {e.median():.2f} m  ratio {np.median(f[c+"_m"] / f.aruler_m):.2f}')
for s in ('_m', '_loo_m'):
    a, b = (f['keypoint' + s] - f.aruler_m).abs(), (f['boxbottom' + s] - f.aruler_m).abs()
    print(s, 'keypoint vs box bottom: Wilcoxon p =', round(wilcoxon(a, b).pvalue, 4), ' keypoint better in', int((a < b).sum()), 'of', len(a))
print('px from click: keypoint median', round(f.kp_px_from_click.median(), 1), ' box bottom', round(f.bb_px_from_click.median(), 1))

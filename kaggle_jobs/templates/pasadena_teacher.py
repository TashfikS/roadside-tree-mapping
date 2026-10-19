"""Pasadena stage A: teacher detector on human labels -> inventory-verified pseudo-labels for the TRAIN region.
Rules fixed before running (no test-region data is touched):
  - teacher: YOLO11m, imgsz 1280, trained on human-labelled panoramas in the train ROIs, validated on the val ROI.
  - a teacher detection becomes a label if (score >= 0.25 AND geolocated within 4 m of an inventory TRAIN tree
    AND range <= 20 m) OR score >= 0.60 (likely real tree missing from the public inventory).
  - a panorama is used only if every inventory train tree within 15 m of the camera is matched by a kept detection
    (otherwise unlabeled trees would be taught as background).
  - human labels replace pseudo-labels on human-labelled panoramas.
"""
import os, glob, json, math, subprocess, time, shutil
subprocess.run("pip install -q ultralytics", shell=True)
import numpy as np
from ultralytics import YOLO

ROOT = glob.glob('/kaggle/input/**/pasadena-urban-trees', recursive=True)[0]
SV = ROOT + '/streetview'
panos = json.load(open(ROOT + '/panos.json')); rois = json.load(open(ROOT + '/rois.json'))
dd = json.load(open(ROOT + '/detection_datasets.json'))
HUMAN = json.loads(r'''__HUMAN__''')
R_EARTH = 6371008.8
W, H, HCAM = 1664, 832, 3.0


def destination(lat, lon, bearing_deg, dist_m):
    b = math.radians(bearing_deg); dn, de = dist_m * math.cos(b), dist_m * math.sin(b)
    return lat + math.degrees(dn / R_EARTH), lon + math.degrees(de / (R_EARTH * math.cos(math.radians(lat))))


def pano_pixel_to_geo(x, y, lat0, lon0, yaw_deg):
    look = x * 2 * math.pi / W; tilt = (H / 2 - y) * math.pi / H
    if tilt >= -1e-2: return None
    z = -HCAM / math.tan(tilt); az = look - math.pi + math.radians(yaw_deg)
    la, lo = destination(lat0, lon0, math.degrees(az), z)
    return la, lo, z


def tvt(la, lo):
    for r in rois:
        a, b, c, d = r['roi']
        if min(a, c) <= la <= max(a, c) and min(b, d) <= lo <= max(b, d): return r['train_val_test']


lat0 = 34.15
def enu(la, lo):
    return np.stack([np.radians(np.asarray(lo) + 118.13) * R_EARTH * math.cos(math.radians(lat0)), np.radians(np.asarray(la) - lat0) * R_EARTH], -1)


P = {}
for k, v in panos.items():
    f = f'{SV}/{k}_z2.jpg'
    if os.path.exists(f):
        la, lo = float(v['Location']['original_lat']), float(v['Location']['original_lng'])
        P[k] = dict(f=f, lat=la, lon=lo, yaw=float(v['Projection']['pano_yaw_deg']), tvt=tvt(la, lo))
print('panos', len(P), {t: sum(p['tvt'] == t for p in P.values()) for t in (0, 1, 2)}, flush=True)

# ---------- 1. teacher on human labels (train ROI -> train, val ROI -> val) ----------
D = '/kaggle/temp/human'
for s in ('train', 'val'):
    os.makedirs(f'{D}/images/{s}', exist_ok=True); os.makedirs(f'{D}/labels/{s}', exist_ok=True)
for pid, h in HUMAN.items():
    s = {0: 'train', 1: 'val'}.get(h['tvt'])
    if s is None or pid not in P: continue
    os.symlink(P[pid]['f'], f'{D}/images/{s}/{pid}.jpg'); open(f'{D}/labels/{s}/{pid}.txt', 'w').write('\n'.join(h['labels']) + '\n')
open(f'{D}/data.yaml', 'w').write(f'path: {D}\ntrain: images/train\nval: images/val\nnames:\n  0: tree\n')
t0 = time.time()
YOLO('yolo11m.pt').train(data=f'{D}/data.yaml', imgsz=1280, epochs=80, patience=20, batch=8, seed=0, deterministic=True, fliplr=0.5,
                         device=0, workers=4, project='/kaggle/working/runs', name='teacher', exist_ok=True, plots=False)
teacher = YOLO('/kaggle/working/runs/teacher/weights/best.pt')
shutil.copy('/kaggle/working/runs/teacher/weights/best.pt', '/kaggle/working/teacher_best.pt')
print('teacher minutes', (time.time() - t0) / 60, flush=True)

# ---------- 2. teacher on all train-ROI panoramas ----------
from scipy.spatial import cKDTree
train_ids = [k for k, p in P.items() if p['tvt'] == 0]
inv = dd['train']
kd_inv = cKDTree(enu([t['lat'] for t in inv], [t['lng'] for t in inv]))
labels, stats = {}, dict(n_panos=len(train_ids), used=0, skipped_incomplete=0, kept_matched=0, kept_highconf=0)
for i in range(0, len(train_ids), 32):
    ids = train_ids[i:i + 32]
    res = teacher.predict([P[k]['f'] for k in ids], imgsz=1280, conf=0.25, iou=0.5, device=0, verbose=False)
    for k, r in zip(ids, res):
        p = P[k]; cam = enu([p['lat']], [p['lon']])[0]
        near = kd_inv.query_ball_point(cam, 15.0)
        matched = set(); keep = []; nm = nh = 0
        sx, sy = W / r.orig_shape[1], H / r.orig_shape[0]
        for (x0, y0, x1, y1), s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            g = pano_pixel_to_geo((x0 + x1) / 2 * sx, y1 * sy, p['lat'], p['lon'], p['yaw'])
            ok = False
            if g is not None and g[2] <= 20:
                d, j = kd_inv.query(enu([g[0]], [g[1]])[0])
                if d <= 4.0: ok = True; matched.add(int(j))
            if ok or s >= 0.60:
                keep.append((x0 * sx, y0 * sy, x1 * sx, y1 * sy)); nm += ok; nh += (not ok)
        if not set(near) <= matched:
            stats['skipped_incomplete'] += 1; continue
        labels[k] = ['0 %.6f %.6f %.6f %.6f' % (((a + c) / 2) / W, ((b + d) / 2) / H, (c - a) / W, (d - b) / H) for a, b, c, d in keep]
        stats['used'] += 1; stats['kept_matched'] += nm; stats['kept_highconf'] += nh
    if i % 1600 == 0: print('pseudo', i, len(train_ids), stats, flush=True)
for pid, h in HUMAN.items():
    if h['tvt'] == 0: labels[pid] = h['labels']
stats['n_images'] = len(labels); stats['n_boxes'] = sum(len(v) for v in labels.values())
json.dump(labels, open('/kaggle/working/pseudo_labels.json', 'w'))
json.dump(stats, open('/kaggle/working/pseudo_stats.json', 'w'), indent=1); print(json.dumps(stats), flush=True)

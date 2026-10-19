"""E2: detector -> ground-plane geolocation -> multi-view merge -> evaluation on Pasadena.

Usage: python pas_e2.py WEIGHTS OUT_PREFIX
- Runs the detector on every panorama in the val ROI (ROI13) and the test ROIs (ROI14-19).
- Geolocates the bottom-centre of every box (h_cam = 3.0 m, dataset convention).
- Merges detections of the same tree across panoramas (greedy, 3 m, score-ordered; position = score-weighted mean).
- Chooses the max-range cut-off on VAL only (from 15/20/25/30/inf m, by AP@4 m), then applies it once to TEST.
- Reports: AP@4 m with the dataset's own evaluator (Branson et al. 2018 protocol), and
  the gate metric: median error of merged trees matched within 10 m.
"""
import os, sys, json, math, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import treegeo as tg
P = os.environ.get('PASADENA_DIR', 'data/external/PasadenaUrbanTrees_Detection')
sys.path.insert(0, P + "/PythonAPI")
from parse_dataset import ParseDataset
from eval import EvaluatePasadenaUrbanTrees

WEIGHTS, OUT = sys.argv[1], sys.argv[2]
D = P + "/PasadenaUrbanTrees"
ds = ParseDataset(D)
if not hasattr(ds, 'detection_datasets'):
    ds.parse_dataset()
panos, rois = ds.panos, ds.rois
W, H, HCAM = 1664, 832, 3.0


def roi_of(la, lo):
    for r in rois:
        a, b, c, d = r['roi']
        if min(a, c) <= la <= max(a, c) and min(b, d) <= lo <= max(b, d):
            return r['train_val_test']
    return None


avail = {os.path.basename(f).split('_z2')[0]: f for f in glob.glob(D + '/streetview/*_z2.jpg')}
pz = pd.DataFrame([dict(pano=k, lat=float(v['Location']['original_lat']), lon=float(v['Location']['original_lng']),
                        yaw=float(v['Projection']['pano_yaw_deg'])) for k, v in panos.items() if k in avail])
pz['tvt'] = [roi_of(a, b) for a, b in zip(pz.lat, pz.lon)]
pz = pz[pz.tvt.isin([1, 2])].reset_index(drop=True)

# ---------- 1. detection (cached) ----------
raw_path = OUT + '_raw_dets.parquet'
if not os.path.exists(raw_path):
    import torch
    torch.set_num_threads(4)
    from ultralytics import YOLO
    model = YOLO(WEIGHTS)
    rows = []
    files = [avail[p] for p in pz.pano]
    for i in range(0, len(files), 16):
        res = model.predict(files[i:i + 16], imgsz=832, conf=0.05, iou=0.5, device='cpu', verbose=False)
        for j, r in enumerate(res):
            b = r.boxes
            for (x0, y0, x1, y1), s in zip(b.xyxy.numpy(), b.conf.numpy()):
                rows.append(dict(pano=pz.pano[i + j], x0=x0 * W / r.orig_shape[1], y0=y0 * H / r.orig_shape[0],
                                 x1=x1 * W / r.orig_shape[1], y1=y1 * H / r.orig_shape[0], score=float(s)))
        if i % 800 == 0:
            print('detected', i, len(files), flush=True)
    pd.DataFrame(rows).to_parquet(raw_path)
det = pd.read_parquet(raw_path).merge(pz, on='pano')

# ---------- 2. geolocation ----------
g = [tg.pano_pixel_to_geo((a + b) / 2, c, W, H, la, lo, yw, h_cam=HCAM)
     for a, b, c, la, lo, yw in zip(det.x0, det.x1, det.y1, det.lat, det.lon, det.yaw)]
det['ok'] = [x is not None for x in g]
det = det[det.ok].copy()
gg = [x for x in g if x is not None]
det['tlat'] = [x['lat'] for x in gg]; det['tlon'] = [x['lon'] for x in gg]; det['range_m'] = [x['range_m'] for x in gg]

lat0 = 34.15; R = 6371008.8
def enu(la, lo):
    return np.stack([np.radians(np.asarray(lo) + 118.13) * R * math.cos(math.radians(lat0)), np.radians(np.asarray(la) - lat0) * R], -1)

# ---------- 3. multi-view merge ----------
def merge(d, radius=3.0):
    """Greedy score-ordered merge: each detection joins the nearest existing centre within `radius`
    (centre = running score-weighted mean), else starts a new tree. Grid hashing makes it O(n)."""
    d = d.sort_values('score', ascending=False)
    X = enu(d.tlat.values, d.tlon.values); S = d.score.values
    cen, wsum, best, nv = [], [], [], []
    grid = {}
    cell = radius
    def key(x): return (int(math.floor(x[0] / cell)), int(math.floor(x[1] / cell)))
    for x, s in zip(X, S):
        gx, gy = key(x); bk, bd = -1, radius + 1e-9
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for k in grid.get((gx + dx, gy + dy), ()):
                    dd = math.hypot(cen[k][0] - x[0], cen[k][1] - x[1])
                    if dd < bd: bk, bd = k, dd
        if bk >= 0 and bd <= radius:
            old = key(cen[bk]); cen[bk] = (cen[bk] * wsum[bk] + x * s) / (wsum[bk] + s); wsum[bk] += s; nv[bk] += 1
            new_ = key(cen[bk])
            if new_ != old:
                grid[old].remove(bk); grid.setdefault(new_, []).append(bk)
            continue
        cen.append(np.array(x, dtype=float)); wsum.append(s); best.append(s); nv.append(1)
        grid.setdefault((gx, gy), []).append(len(cen) - 1)
    C = np.array(cen)
    lon = C[:, 0] / (R * math.cos(math.radians(lat0))) * 180 / math.pi - 118.13
    lat = C[:, 1] / R * 180 / math.pi + lat0
    return pd.DataFrame(dict(lat=lat, lng=lon, score=best, n_views=nv))


def evaluate(trees, split):
    ev = EvaluatePasadenaUrbanTrees(ds, 4.0, dataset_id=split)
    dets = trees[['lat', 'lng', 'score']].to_dict('records')
    ap, prec, rec, matches = ev.evaluate_detection_average_precision(dets)
    gt = pd.DataFrame(ds.detection_datasets[split])
    from scipy.spatial import cKDTree
    kd = cKDTree(enu(gt.lat.values, gt.lng.values))
    dist, _ = kd.query(enu(trees.lat.values, trees.lng.values))
    m10 = dist[dist <= 10]
    return dict(AP_4m=round(float(ap), 4), max_recall=round(float(max(rec) if rec else 0), 4), n_trees=len(trees),
                gate_median_matched10=round(float(np.median(m10)), 2) if len(m10) else None,
                p90_matched10=round(float(np.quantile(m10, .9)), 2) if len(m10) else None,
                frac_within2_of_matched4=round(float((dist[dist <= 4] <= 2).mean()), 3) if (dist <= 4).any() else None)

# ---------- 4. choose range cut-off on VAL, apply once to TEST ----------
out = {}
val = {}
for cut in [15, 20, 25, 30, 1e9]:
    t = merge(det[(det.tvt == 1) & (det.range_m <= cut)])
    val[str(cut)] = evaluate(t, 'val')
best_cut = max(val, key=lambda k: val[k]['AP_4m'])
out['val_sweep'] = val; out['chosen_cut_m'] = best_cut
test_trees = merge(det[(det.tvt == 2) & (det.range_m <= float(best_cut))])
out['test'] = evaluate(test_trees, 'test')
out['test_no_cut'] = evaluate(merge(det[det.tvt == 2]), 'test')
test_trees.to_csv(OUT + '_test_trees.csv', index=False)
json.dump(out, open(OUT + '_results.json', 'w'), indent=1)
print(json.dumps(out, indent=1))

"""E7 Pasadena (rules: e7_prereg.md, part A). Same retrained-detector boxes and bearings as the ground-plane method; only
the range comes from Depth Anything V2 (metric, outdoor). Validation sweep chooses read-out, range cut, merge radius and
minimum views by AP@4 m; the chosen setting is applied once to the test area. Run in a fresh process (needs the drive)."""
import os, sys, json, math, glob
import numpy as np, pandas as pd
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pas_e2.py')).read(); head = src[:src.index("# ---------- 4.")]
sys.argv = ['pas_e2.py', 'kout/pas_student/student_best.pt', 'pas_student']
g = dict(__file__=os.path.abspath('pas_e2.py'), __name__='pas_e7_head'); exec(head, g)
merge, evaluate, pz, tg = g['merge'], g['evaluate'], g['pz'], g['tg']
W, H, HCAM = 1664, 832, 3.0

raw = pd.read_parquet('pas_student_raw_dets.parquet').reset_index(drop=True); raw['idx'] = np.arange(len(raw))
dep = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('kout/e7_pas/e7_pas_depth_*.csv'))], ignore_index=True)
assert dep.idx.is_unique and dep.idx.max() < len(raw), 'depth rows do not map onto the raw detections'
d = raw.merge(dep, on='idx', how='inner').merge(pz, on='pano', how='inner')
u = (d.x0.values + d.x1.values) / 2
d['az'] = np.degrees(u * 2 * math.pi / W - math.pi + np.radians(d.yaw.values)) % 360
tilt = (H / 2 - d.y1.values) * math.pi / H
d['r_ground'] = np.where(tilt < -1e-2, -HCAM / np.tan(np.minimum(tilt, -1e-2)), np.nan)
info = dict(n_raw=len(raw), n_depth=len(dep), n_joined=len(d), depth_point_nan=int(d.r_point.isna().sum()), depth_region_nan=int(d.r_region.isna().sum()))

def place(df, col):
    x = df[df[col].notna() & (df[col] > 0)].copy()
    ll = [tg.destination(a, b, c, r) for a, b, c, r in zip(x.lat, x.lon, x.az, x[col])]
    x['tlat'] = [p[0] for p in ll]; x['tlon'] = [p[1] for p in ll]; x['range_m'] = x[col]
    return x

rows = []
for col in ('r_point', 'r_region'):
    X = place(d, col)
    for cut in (15, 20):
        for rad in (3, 5, 7, 9):
            t = merge(X[(X.tvt == 1) & (X.range_m <= cut)], radius=rad)
            for mv in (1, 2):
                rows.append(dict(readout=col, cut=cut, radius=rad, min_views=mv, **evaluate(t[t.n_views >= mv], 'val')))
V = pd.DataFrame(rows); V.to_csv('pas_e7_val_sweep.csv', index=False)
b = V.loc[V.AP_4m.idxmax()]
X = place(d, b.readout); t = merge(X[(X.tvt == 2) & (X.range_m <= b.cut)], radius=int(b.radius)); t = t[t.n_views >= int(b.min_views)]
test = evaluate(t, 'test'); t.to_csv('pas_e7_test_trees.csv', index=False)
both = d[d.r_ground.notna() & d[b.readout].notna() & (d.r_ground <= 20)]
rd = (both[b.readout] - both.r_ground)
out = dict(info=info, val_best=b.to_dict(), test_depth=test,
           test_ground_plane_reference=json.load(open('pas_student_decision.json'))['test_student'],
           range_vs_ground_plane=dict(n=len(both), median_abs_diff_m=round(float(rd.abs().median()), 2), median_signed_m=round(float(rd.median()), 2),
                                      median_ratio=round(float((both[b.readout] / both.r_ground).median()), 3)))
json.dump(out, open('pas_e7_result.json', 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))

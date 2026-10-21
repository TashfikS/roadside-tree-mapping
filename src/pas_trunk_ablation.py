"""Test P (protocols/trunk_ablation_prereg.md): trunk-base keypoint (K) vs box bottom-centre (B) of the SAME detections
of the Pasadena keypoint model. For each variant: D_max / merge radius / minimum views chosen on validation by AP@4 m,
test applied once; paired 400 m spatial block bootstrap of the test AP@4 m difference (official and strict rule).

Usage: PASADENA_DIR=... python src/pas_trunk_ablation.py work/kout/pas_trunk_pose/pas_pose_raw_dets.csv [n_boot]
Outputs: results/pas_trunk_ablation.json, results/pas_trunk_ablation_val_sweep.csv, results/pas_trunk_{K,B}_test_trees.csv
"""
import os, sys, json, math, ast
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RAW = sys.argv[1]; NBOOT = int(sys.argv[2]) if len(sys.argv) > 2 else 500
KCONF = 0.5
raw = pd.read_csv(RAW)
variants = {}
for name in ('K', 'B'):
    d = raw.copy(); w = d.x1 - d.x0
    if name == 'K':
        use = d.kconf >= KCONF
        gu = np.where(use, d.ku, (d.x0 + d.x1) / 2); gv = np.where(use, d.kv, d.y1)
    else:
        gu = (d.x0 + d.x1) / 2; gv = d.y1
    d['x0'], d['x1'], d['y1'] = gu - w / 2, gu + w / 2, gv          # pas_e2 geolocates ((x0+x1)/2, y1)
    p = os.path.join(ROOT, 'work', f'pas_trunk_{name}_raw_dets.parquet'); os.makedirs(os.path.dirname(p), exist_ok=True)
    d[['pano', 'x0', 'y0', 'x1', 'y1', 'score']].to_parquet(p); variants[name] = p
print('detections', len(raw), 'keypoint used (conf >= 0.5):', round(float((raw.kconf >= KCONF).mean()), 3), flush=True)

src = open(os.path.join(HERE, 'pas_e2.py')).read(); head = src[:src.index("# ---------- 4.")]
G, rows = {}, []
for name, p in variants.items():
    out_prefix = p[:-len('_raw_dets.parquet')]
    sys.argv = ['pas_e2.py', 'unused.pt', out_prefix]
    g = dict(__file__=os.path.join(HERE, 'pas_e2.py'), __name__='pas_e2_head'); exec(head, g); G[name] = g
    det, merge, evaluate = g['det'], g['merge'], g['evaluate']
    for cut in (15, 20):
        for rad in (3, 5, 7, 9):
            t = merge(det[(det.tvt == 1) & (det.range_m <= cut)], radius=rad)
            for mv in (1, 2):
                rows.append(dict(variant=name, cut=cut, radius=rad, min_views=mv, **evaluate(t[t.n_views >= mv], 'val')))
    print(name, 'validation sweep done', flush=True)
V = pd.DataFrame(rows); V.to_csv(os.path.join(ROOT, 'results', 'pas_trunk_ablation_val_sweep.csv'), index=False)
best = V.loc[V.groupby('variant').AP_4m.idxmax()].set_index('variant')
out = dict(keypoint_conf_threshold=KCONF, share_keypoint_used=float((raw.kconf >= KCONF).mean()), val_best=best.reset_index().to_dict('records'), test={})
TT = {}
for name in ('K', 'B'):
    b = best.loc[name]; g = G[name]
    t = g['merge'](g['det'][(g['det'].tvt == 2) & (g['det'].range_m <= b.cut)], radius=int(b.radius)); t = t[t.n_views >= int(b.min_views)]
    out['test'][name] = g['evaluate'](t, 'test'); TT[name] = t
    t.to_csv(os.path.join(ROOT, 'results', f'pas_trunk_{name}_test_trees.csv'), index=False)
    print('TEST (once)', name, out['test'][name], flush=True)

# ---- strict rule, nearest-record error and paired block bootstrap (same procedure as audit/audit_e2.py)
g = G['K']; ds = g['ds']; EvaluatePasadenaUrbanTrees = g['EvaluatePasadenaUrbanTrees']
GT0 = list(ds.detection_datasets['test'])
NT = {'UNDER PROCESS FOR PLANT', 'NO ROOM FOR REPLANT', 'PLANTING SITE-SMALL', 'PLANTING SITE-MEDIUM', 'PLANTING SITE-LARGE', 'STUMP', 'NO TREE AT THIS SITE', 'INSUFFICIENT PLANTING SITE'}
def cn(t):
    e = t['extended'] if isinstance(t['extended'], dict) else ast.literal_eval(t['extended']); return (e.get('Common_Name') or '').upper().strip()
def ap(dets, gt, thr=4.0):
    ds.detection_datasets['audit'] = gt
    a, p, r, m = EvaluatePasadenaUrbanTrees(ds, thr, 'audit').evaluate_detection_average_precision(dets)
    return float(a)
R = 6371008.8; lat0 = 34.15
def en(la, lo): return np.c_[np.radians(np.asarray(lo, float) + 118.13) * R * math.cos(math.radians(lat0)), np.radians(np.asarray(la, float) - lat0) * R]
strict = lambda G_: [dict(t, diff=False) for t in G_ if cn(t) not in NT]
from scipy.spatial import cKDTree
gxy = en([t['lat'] for t in GT0], [t['lng'] for t in GT0]); istree = np.array([cn(t) not in NT for t in GT0]); kd = cKDTree(gxy[istree])
DETS = {k: v[['lat', 'lng', 'score']].to_dict('records') for k, v in TT.items()}
for k, v in TT.items():
    dd, _ = kd.query(en(v.lat, v.lng))
    out['test'][k].update(AP4_strict=round(ap(DETS[k], strict(GT0)), 4), median_nearest_real_tree_within10=round(float(np.median(dd[dd <= 10])), 3),
                          share_within2_of_within4=round(float((dd[dd <= 4] <= 2).mean()), 3))
B_ = 400.0; gb = [tuple(x) for x in np.floor(gxy / B_).astype(int)]
pb = {k: [tuple(x) for x in np.floor(en(v.lat, v.lng) / B_).astype(int)] for k, v in TT.items()}
blocks = sorted(set(gb) | set(pb['K']) | set(pb['B'])); gidx, pidx = {}, {'K': {}, 'B': {}}
for i, b in enumerate(gb): gidx.setdefault(b, []).append(i)
for k in pb:
    for i, b in enumerate(pb[k]): pidx[k].setdefault(b, []).append(i)
step_lat = max(t['lat'] for t in GT0) - min(t['lat'] for t in GT0) + 0.02; step_lng = max(t['lng'] for t in GT0) - min(t['lng'] for t in GT0) + 0.02; NCOL = 12
rng = np.random.default_rng(20261020); diff = {'official': [], 'strict': []}
for rep in range(NBOOT):
    pick = rng.choice(len(blocks), len(blocks), replace=True); Gs, Ds = [], {'K': [], 'B': []}
    for k, bi in enumerate(pick):
        b = blocks[bi]; dla, dlo = (k // NCOL) * step_lat, (k % NCOL) * step_lng
        Gs += [dict(GT0[i], lat=GT0[i]['lat'] + dla, lng=GT0[i]['lng'] + dlo) for i in gidx.get(b, [])]
        for v in ('K', 'B'):
            Ds[v] += [dict(lat=DETS[v][i]['lat'] + dla, lng=DETS[v][i]['lng'] + dlo, score=DETS[v][i]['score']) for i in pidx[v].get(b, [])]
    diff['official'].append(ap(Ds['K'], Gs) - ap(Ds['B'], Gs)); sg = strict(Gs); diff['strict'].append(ap(Ds['K'], sg) - ap(Ds['B'], sg))
    if rep % 10 == 9: print('boot', rep + 1, flush=True)
out['paired_bootstrap_K_minus_B'] = {k: dict(point=round(out['test']['K']['AP_4m' if k == 'official' else 'AP4_strict'] - out['test']['B']['AP_4m' if k == 'official' else 'AP4_strict'], 4),
                                             ci95=[round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)],
                                             share_positive=round(float(np.mean(np.array(v) > 0)), 3), n_rep=len(v)) for k, v in diff.items()}
out['paired_bootstrap_K_minus_B']['n_blocks'] = len(blocks)
out['K_better_by_rule'] = bool(out['paired_bootstrap_K_minus_B']['official']['ci95'][0] > 0)
json.dump(out, open(os.path.join(ROOT, 'results', 'pas_trunk_ablation.json'), 'w'), indent=1, default=float)
print(json.dumps({k: out[k] for k in ('test', 'paired_bootstrap_K_minus_B', 'K_better_by_rule')}, indent=1, default=float))

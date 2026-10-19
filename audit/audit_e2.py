"""Audit of E2 (Pasadena positions, test area): official evaluator vs strict rule (all real trees count, no 'difficult'
exemption, non-tree records removed), error distribution, and a 400 m spatial block bootstrap of AP@4 m.
Output: audit/audit_e2.json"""
import os
import sys, json, ast, math, time, numpy as np, pandas as pd
from scipy.spatial import cKDTree
PA = os.environ.get('PASADENA_DIR', 'data/external/PasadenaUrbanTrees_Detection')
sys.path.insert(0, PA + "/PythonAPI")
from parse_dataset import ParseDataset; from eval import EvaluatePasadenaUrbanTrees
ds = ParseDataset(PA + "/PasadenaUrbanTrees")
if not hasattr(ds, 'detection_datasets'): ds.parse_dataset()
GT0 = list(ds.detection_datasets['test'])
NT = {'UNDER PROCESS FOR PLANT', 'NO ROOM FOR REPLANT', 'PLANTING SITE-SMALL', 'PLANTING SITE-MEDIUM', 'PLANTING SITE-LARGE', 'STUMP', 'NO TREE AT THIS SITE', 'INSUFFICIENT PLANTING SITE'}
def cn(t):
    e = t['extended'] if isinstance(t['extended'], dict) else ast.literal_eval(t['extended']); return (e.get('Common_Name') or '').upper().strip()
ISTREE = np.array([cn(t) not in NT for t in GT0]); DIFF = np.array([bool(t['diff']) for t in GT0])
def pairs(m):
    out = []
    for x in m:
        try: i, j = x
        except Exception: continue
        if isinstance(i, (int, np.integer)) and isinstance(j, (int, np.integer)) and i >= 0 and j >= 0: out.append((int(i), int(j)))
    return out
def ap(dets, gt, thr=4.0):
    ds.detection_datasets['audit'] = gt
    a, p, r, m = EvaluatePasadenaUrbanTrees(ds, thr, 'audit').evaluate_detection_average_precision(dets)
    return float(a), (float(max(r)) if len(r) else 0.0), m
R = 6371008.8; lat0 = 34.15
def en(la, lo): return np.c_[np.radians(np.asarray(lo, float) + 118.13) * R * math.cos(math.radians(lat0)), np.radians(np.asarray(la, float) - lat0) * R]
gxy = en([t['lat'] for t in GT0], [t['lng'] for t in GT0])
out = dict(test_records=len(GT0), real_trees=int(ISTREE.sum()), non_tree_records=int((~ISTREE).sum()), difficult_records=int(DIFF.sum()),
           difficult_real_trees=int((DIFF & ISTREE).sum()), non_difficult_trees=int((~DIFF & ISTREE).sum()), non_tree_flagged_difficult=int((DIFF & ~ISTREE).sum()))
print(out, flush=True)
strict_gt = [dict(t, diff=False) for t, k in zip(GT0, ISTREE) if k]
for name, f in (('retrained (final)', 'pas_student_test_trees.csv'), ('first detector, validation-chosen merge', 'pas_e2_test_trees_dedup.csv')):
    t0 = time.time(); P = pd.read_csv(f); dets = P[['lat', 'lng', 'score']].to_dict('records')
    a_off, r_off, m_off = ap(dets, GT0); a_st, r_st, m_st = ap(dets, strict_gt)
    pxy = en(P.lat, P.lng); d, _ = cKDTree(gxy[ISTREE]).query(pxy)
    print('raw match examples', m_off[:3], flush=True)
    m_off, m_st = pairs(m_off), pairs(m_st)
    mo = np.array([np.hypot(*(pxy[i] - gxy[j])) for i, j in m_off]) if len(m_off) else np.array([])
    print('official pairs', len(m_off), 'max matched distance', round(float(mo.max()), 3) if len(mo) else None, flush=True)
    sx = en([t['lat'] for t in strict_gt], [t['lng'] for t in strict_gt])
    ms = np.array([np.hypot(*(pxy[i] - sx[j])) for i, j in m_st]) if len(m_st) else np.array([])
    print('strict pairs', len(m_st), 'max matched distance', round(float(ms.max()), 3) if len(ms) else None, flush=True)
    out[name] = dict(n_mapped=len(P), AP4_official=round(a_off, 4), max_recall_official=round(r_off, 4),
                     AP4_strict_all_real_trees=round(a_st, 4), max_recall_strict=round(r_st, 4),
                     nearest_real_tree_within_m={str(k): round(float((d <= k).mean()), 3) for k in (1, 2, 4, 10)},
                     median_nearest_among_within10=round(float(np.median(d[d <= 10])), 3), p90_nearest_among_within10=round(float(np.percentile(d[d <= 10], 90)), 3),
                     one_to_one_matches_official=int(len(mo)), median_error_matched_official=round(float(np.median(mo)), 3) if len(mo) else None,
                     one_to_one_matches_strict=int(len(ms)), median_error_matched_strict=round(float(np.median(ms)), 3) if len(ms) else None,
                     seconds=round(time.time() - t0, 1))
    print(name, json.dumps(out[name]), flush=True)
# spatial block bootstrap (400 m blocks), final map; sampled copies are placed on a 2-D grid far apart so they never interact
P = pd.read_csv('pas_student_test_trees.csv'); dets = P[['lat', 'lng', 'score']].to_dict('records'); pxy = en(P.lat, P.lng)
B = 400.0; gb = [tuple(x) for x in np.floor(gxy / B).astype(int)]; pb = [tuple(x) for x in np.floor(pxy / B).astype(int)]
blocks = sorted(set(gb) | set(pb)); gidx, pidx = {}, {}
for i, b in enumerate(gb): gidx.setdefault(b, []).append(i)
for i, b in enumerate(pb): pidx.setdefault(b, []).append(i)
ext_lat = max(t['lat'] for t in GT0) - min(t['lat'] for t in GT0); ext_lng = max(t['lng'] for t in GT0) - min(t['lng'] for t in GT0)
step_lat, step_lng = ext_lat + 0.02, ext_lng + 0.02; NCOL = 12
rng = np.random.default_rng(20261019); res = {'official': [], 'strict': []}; NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 100
for rep in range(NREP):
    pick = rng.choice(len(blocks), len(blocks), replace=True); G, D = [], []
    for k, bi in enumerate(pick):
        b = blocks[bi]; dla, dlo = (k // NCOL) * step_lat, (k % NCOL) * step_lng
        G += [dict(GT0[i], lat=GT0[i]['lat'] + dla, lng=GT0[i]['lng'] + dlo) for i in gidx.get(b, [])]
        D += [dict(lat=dets[i]['lat'] + dla, lng=dets[i]['lng'] + dlo, score=dets[i]['score']) for i in pidx.get(b, [])]
    res['official'].append(ap(D, G)[0]); res['strict'].append(ap(D, [dict(t, diff=False) for t in G if cn(t) not in NT])[0])
    if rep % 20 == 19: print('boot', rep + 1, flush=True)
out['bootstrap_400m_blocks'] = {k: dict(ci95=[round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)], sd=round(float(np.std(v)), 4), n_rep=len(v))
                                for k, v in res.items()}
out['bootstrap_400m_blocks']['n_blocks'] = len(blocks)
print(json.dumps(out['bootstrap_400m_blocks'])); json.dump(out, open('audit/audit_e2.json', 'w'), indent=1)

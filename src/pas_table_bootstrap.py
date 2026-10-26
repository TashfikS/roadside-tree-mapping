"""Pasadena comparison table (q1_upgrades_prereg.md, section on the ablation table): test AP@4 m (official and strict) of
every Pasadena map with a 400 m spatial block bootstrap that uses the SAME block draws for all maps (paired), with the
exact fast scorer (audit/fast_ap.py, checked against the official evaluator first). Output: results/q1_upgrades/pas_table.json"""
import os, sys, json, ast, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, os.path.join(ROOT, 'audit'))
src = open(os.path.join(ROOT, 'audit', 'fast_ap.py')).read(); exec(src[:src.index("if __name__ == '__main__':")])
PA = os.environ['PASADENA_DIR']; sys.path.insert(0, PA + '/PythonAPI')
from parse_dataset import ParseDataset; from eval import EvaluatePasadenaUrbanTrees
ds = ParseDataset(PA + '/PasadenaUrbanTrees')
if not hasattr(ds, 'detection_datasets'): ds.parse_dataset()
GT0 = list(ds.detection_datasets['test'])
NT = {'UNDER PROCESS FOR PLANT', 'NO ROOM FOR REPLANT', 'PLANTING SITE-SMALL', 'PLANTING SITE-MEDIUM', 'PLANTING SITE-LARGE', 'STUMP', 'NO TREE AT THIS SITE', 'INSUFFICIENT PLANTING SITE'}
cn = lambda t: ((t['extended'] if isinstance(t['extended'], dict) else ast.literal_eval(t['extended'])).get('Common_Name') or '').upper().strip()
glat = np.array([t['lat'] for t in GT0]); glng = np.array([t['lng'] for t in GT0]); gdiff = np.array([bool(t['diff']) for t in GT0]); ist = np.array([cn(t) not in NT for t in GT0])
MAPS = {'first_detector': 'pas_e2_test_trees_dedup.csv', 'retrained_box': 'pas_student_test_trees.csv', 'pose_box_bottom_B': 'pas_trunk_B_test_trees.csv',
        'pose_keypoint_K': 'pas_trunk_K_test_trees.csv', 'depth_as_is': 'pas_e7_test_trees.csv'}
M = {k: pd.read_csv(os.path.join(ROOT, 'results', f)) for k, f in MAPS.items() if os.path.exists(os.path.join(ROOT, 'results', f))}
out = {'point': {}}
for k, P in M.items():
    a, r = fast_ap(P.lat, P.lng, P.score, glat, glng, gdiff); s, rs = fast_ap(P.lat, P.lng, P.score, glat[ist], glng[ist], np.zeros(ist.sum(), bool))
    full = EvaluatePasadenaUrbanTrees(ds, 4.0, 'test').evaluate_detection_average_precision(P[['lat', 'lng', 'score']].to_dict('records'))[0]
    assert abs(full - a) < 1e-9, (k, full, a)
    out['point'][k] = dict(AP4=round(a, 4), recall=round(r, 4), AP4_strict=round(s, 4), recall_strict=round(rs, 4), n_trees=len(P)); print(k, out['point'][k], flush=True)
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 500; B = 400.0; gxy = enu(glat, glng); gb = [tuple(x) for x in np.floor(gxy / B).astype(int)]
pb = {k: [tuple(x) for x in np.floor(enu(P.lat, P.lng) / B).astype(int)] for k, P in M.items()}
blocks = sorted(set(gb).union(*[set(v) for v in pb.values()])); gi = {}
for i, b in enumerate(gb): gi.setdefault(b, []).append(i)
pi = {k: {} for k in M}
for k in M:
    for i, b in enumerate(pb[k]): pi[k].setdefault(b, []).append(i)
rng = np.random.default_rng(20261026); V = {k: {'official': [], 'strict': []} for k in M}
for rep in range(NB):
    pick = rng.choice(len(blocks), len(blocks), replace=True)
    gl, gg, gd, gt = [], [], [], []; dl = {k: [] for k in M}; dg = {k: [] for k in M}; dsc = {k: [] for k in M}
    for j, bi in enumerate(pick):
        b = blocks[bi]; ola, olo = (j // 20) * 0.1, (j % 20) * 0.1; ii = gi.get(b, [])
        gl += list(glat[ii] + ola); gg += list(glng[ii] + olo); gd += list(gdiff[ii]); gt += list(ist[ii])
        for k, P in M.items():
            q = pi[k].get(b, [])
            dl[k] += list(P.lat.values[q] + ola); dg[k] += list(P.lng.values[q] + olo); dsc[k] += list(P.score.values[q])
    gl, gg, gd, gt = map(np.array, (gl, gg, gd, gt))
    for k in M:
        V[k]['official'].append(fast_ap(np.array(dl[k]), np.array(dg[k]), np.array(dsc[k]), gl, gg, gd)[0])
        V[k]['strict'].append(fast_ap(np.array(dl[k]), np.array(dg[k]), np.array(dsc[k]), gl[gt], gg[gt], np.zeros(gt.sum(), bool))[0])
    if rep % 50 == 49: print('boot', rep + 1, flush=True)
ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
out['ci95'] = {k: {r: ci(V[k][r]) for r in ('official', 'strict')} for k in M}
d = lambda a, b, r: np.array(V[a][r]) - np.array(V[b][r])
out['paired_differences'] = {f'{a}_minus_{b}': {r: dict(ci95=ci(d(a, b, r)), share_positive=round(float((d(a, b, r) > 0).mean()), 3)) for r in ('official', 'strict')}
                             for a, b in (('pose_keypoint_K', 'pose_box_bottom_B'), ('retrained_box', 'first_detector'), ('pose_box_bottom_B', 'retrained_box')) if a in M and b in M}
out['share_above_branson_0581'] = {k: round(float((np.array(V[k]['official']) > 0.581).mean()), 3) for k in M}
out['sd'] = {k: {r: round(float(np.std(V[k][r])), 4) for r in ('official', 'strict')} for k in M}
out['n_rep'] = NB; out['n_blocks'] = len(blocks)
json.dump(out, open(os.path.join(ROOT, 'results', 'q1_upgrades', 'pas_table.json'), 'w'), indent=1); print(json.dumps({k: out[k] for k in ('ci95', 'paired_differences')}, indent=0))

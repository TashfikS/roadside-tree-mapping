"""Exact fast re-implementation of EvaluatePasadenaUrbanTrees.evaluate_detection_average_precision (PythonAPI/eval.py):
detections ranked by score (stable); each takes the nearest REMAINING non-difficult tree with haversine distance < thr
(R = 6371000 m, same formula); otherwise a remaining difficult tree within thr is consumed without counting; otherwise FP.
AP = sum of precision at each true positive / number of non-difficult trees.
Candidates come from a KD-tree in a local tangent plane with a 0.1 m margin, then exact haversine (identical results).
Self-check: must reproduce the official evaluator on the real data before the bootstrap is run.
Usage: python fast_ap.py [n_boot]   -> audit/audit_e2_bootstrap.json"""
import os
import sys, json, ast, math, time
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
PA = os.environ.get('PASADENA_DIR', 'data/external/PasadenaUrbanTrees_Detection')
ER = 6371000.0


def hav(lat, lng, A):
    aa = (np.square(np.sin((A[:, 0] - lat) * math.pi / 360.0)) +
          (math.cos(math.radians(lat)) * np.cos(A[:, 0] * math.pi / 180.0)) * np.square(np.sin((A[:, 1] - lng) * math.pi / 360.0)))
    return 2 * ER * np.arctan2(np.sqrt(aa), np.sqrt(1 - aa))


def enu(la, lo, lat0=34.15, lon0=-118.13):
    return np.c_[np.radians(np.asarray(lo, float) - lon0) * ER * math.cos(math.radians(lat0)), np.radians(np.asarray(la, float) - lat0) * ER]


def fast_ap(det_lat, det_lng, det_score, gt_lat, gt_lng, gt_diff, thr=4.0):
    det_lat, det_lng, det_score = map(np.asarray, (det_lat, det_lng, det_score))
    G = np.c_[gt_lat, gt_lng].astype(float); diff = np.asarray(gt_diff, bool)
    num_gt = int((~diff).sum())
    order = np.argsort(-det_score, kind='stable')
    lat0 = float(np.mean(G[:, 0])) if len(G) else 0.0; lon0 = float(np.mean(G[:, 1])) if len(G) else 0.0
    kd = cKDTree(enu(G[:, 0], G[:, 1], lat0, lon0))
    cand = kd.query_ball_point(enu(det_lat[order], det_lng[order], lat0, lon0), r=thr + 0.1)
    alive = np.ones(len(G), bool); tp = 0; num = 0; psum = 0.0
    for k, i in enumerate(order):
        c = np.asarray(cand[k], int)
        if len(c):
            c = c[alive[c]]
        hit = None
        if len(c):
            d = hav(det_lat[i], det_lng[i], G[c])
            nd = ~diff[c]
            if nd.any():
                dn = d[nd]; j = int(np.argmin(dn))
                if dn[j] < thr: hit = ('tp', c[nd][j])
            if hit is None and (~nd).any():
                dd = d[~nd]; j = int(np.argmin(dd))
                if dd[j] < thr: hit = ('diff', c[~nd][j])
        if hit is None:
            num += 1
        elif hit[0] == 'tp':
            tp += 1; num += 1; psum += tp / num; alive[hit[1]] = False
        else:
            alive[hit[1]] = False
    return psum / max(num_gt, 1), tp / max(num_gt, 1)


if __name__ == '__main__':
    sys.path.insert(0, PA + "/PythonAPI")
    from parse_dataset import ParseDataset
    ds = ParseDataset(PA + "/PasadenaUrbanTrees")
    if not hasattr(ds, 'detection_datasets'): ds.parse_dataset()
    GT0 = list(ds.detection_datasets['test'])
    NT = {'UNDER PROCESS FOR PLANT', 'NO ROOM FOR REPLANT', 'PLANTING SITE-SMALL', 'PLANTING SITE-MEDIUM', 'PLANTING SITE-LARGE', 'STUMP', 'NO TREE AT THIS SITE', 'INSUFFICIENT PLANTING SITE'}
    def cn(t):
        e = t['extended'] if isinstance(t['extended'], dict) else ast.literal_eval(t['extended']); return (e.get('Common_Name') or '').upper().strip()
    glat = np.array([t['lat'] for t in GT0]); glng = np.array([t['lng'] for t in GT0]); gdiff = np.array([bool(t['diff']) for t in GT0])
    istree = np.array([cn(t) not in NT for t in GT0])
    ref = {'pas_student_test_trees.csv': (0.6405, 0.8513, 0.5653, 0.6969), 'pas_e2_test_trees_dedup.csv': (0.4906, 0.7945, 0.4493, 0.6331)}
    out = {'self_check': {}}
    for f, (a_o, r_o, a_s, r_s) in ref.items():
        P = pd.read_csv(f)
        t0 = time.time(); ao, ro = fast_ap(P.lat, P.lng, P.score, glat, glng, gdiff)
        as_, rs_ = fast_ap(P.lat, P.lng, P.score, glat[istree], glng[istree], np.zeros(istree.sum(), bool))
        ok = abs(ao - a_o) < 6e-5 and abs(ro - r_o) < 6e-5 and abs(as_ - a_s) < 6e-5 and abs(rs_ - r_s) < 6e-5
        out['self_check'][f] = dict(fast_official=round(ao, 6), fast_official_recall=round(ro, 6), fast_strict=round(as_, 6), fast_strict_recall=round(rs_, 6),
                                    evaluator=[a_o, r_o, a_s, r_s], identical_to_4dp=bool(ok), seconds=round(time.time() - t0, 2))
        print(f, out['self_check'][f], flush=True)
        # exact check against the official evaluator itself (full precision) for the official rule
        from eval import EvaluatePasadenaUrbanTrees
        a_full = EvaluatePasadenaUrbanTrees(ds, 4.0, 'test').evaluate_detection_average_precision(P[['lat', 'lng', 'score']].to_dict('records'))[0]
        out['self_check'][f]['evaluator_full_precision'] = float(a_full); out['self_check'][f]['abs_diff'] = float(abs(a_full - ao))
        print('  official evaluator full precision', a_full, '| fast', ao, '| diff', abs(a_full - ao), flush=True)
        assert abs(a_full - ao) < 1e-9, 'fast re-implementation differs from the official evaluator'
    NB = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    assert np.ptp(glat) < 0.09 and np.ptp(glng) < 0.09, (np.ptp(glat), np.ptp(glng))   # copies 0.1 deg apart cannot interact
    B = 400.0; gxy = enu(glat, glng)
    res = {}
    for f in ref:
        P = pd.read_csv(f); pxy = enu(P.lat, P.lng)
        gb = [tuple(x) for x in np.floor(gxy / B).astype(int)]; pb = [tuple(x) for x in np.floor(pxy / B).astype(int)]
        blocks = sorted(set(gb) | set(pb)); gi, pi = {}, {}
        for i, b in enumerate(gb): gi.setdefault(b, []).append(i)
        for i, b in enumerate(pb): pi.setdefault(b, []).append(i)
        rng = np.random.default_rng(20261019); vals = {'official': [], 'strict': []}
        for rep in range(NB):
            pick = rng.choice(len(blocks), len(blocks), replace=True)
            Gl, Gg, Gd, Gt, Dl, Dg, Ds = [], [], [], [], [], [], []
            for k, bi in enumerate(pick):
                b = blocks[bi]; off_la = (k // 20) * 0.1; off_lo = (k % 20) * 0.1      # copies ~11 km apart: never interact
                g = gi.get(b, []); p = pi.get(b, [])
                Gl.append(glat[g] + off_la); Gg.append(glng[g] + off_lo); Gd.append(gdiff[g]); Gt.append(istree[g])
                Dl.append(P.lat.values[p] + off_la); Dg.append(P.lng.values[p] + off_lo); Ds.append(P.score.values[p])
            Gl, Gg, Gd, Gt, Dl, Dg, Ds = map(np.concatenate, (Gl, Gg, Gd, Gt, Dl, Dg, Ds))
            vals['official'].append(fast_ap(Dl, Dg, Ds, Gl, Gg, Gd)[0])
            vals['strict'].append(fast_ap(Dl, Dg, Ds, Gl[Gt], Gg[Gt], np.zeros(Gt.sum(), bool))[0])
            if rep % 200 == 199: print(f, 'boot', rep + 1, flush=True)
        res[f] = {k: dict(ci95=[round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)], sd=round(float(np.std(v)), 4)) for k, v in vals.items()}
        res[f]['n_blocks'] = len(blocks); res[f]['n_rep'] = NB
        if f == 'pas_student_test_trees.csv': VALS_FINAL = vals
        else: VALS_FIRST = vals
        print(f, json.dumps(res[f]), flush=True)
    out['bootstrap_400m_blocks'] = res
    out['share_of_bootstrap_reps_final_official_above_branson_0.581'] = round(float(np.mean(np.array(VALS_FINAL['official']) > 0.581)), 4)
    out['note'] = 'Blocks resampled independently for each map (not paired).'
    json.dump(out, open('audit/audit_e2_bootstrap.json', 'w'), indent=1); print(json.dumps({k: v for k, v in out.items() if k != 'self_check'}, indent=1))

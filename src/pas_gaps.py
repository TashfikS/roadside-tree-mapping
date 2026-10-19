"""Planting-gap validation on the Pasadena TEST area: gaps from detected trees vs. gaps from the surveyed inventory.

Rules fixed before looking at results:
  streets   : Google panorama link graph (test-ROI panoramas), split into chains at intersections; unnamed links and
              freeways (I-*, *Fwy, *Freeway) excluded.
  assignment: each tree -> nearest chain segment within 20 m; side = sign of the cross product with travel direction.
  plantable : a position s on a chain side is plantable at spacing S if no tree on that side lies within S of it and s is
              >= 10 m from either chain end (intersection margin). Evaluated on a 1-m grid, only on chain sides that hold
              >= 1 inventory tree (i.e. sides with evidence of a planting strip).
  counts    : a plantable run of length l admits floor(l/S)+1 trees (= floor(g/S)-1 for an interior gap g).
"""
import os
import json, math, collections, sys
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

D = os.path.join(os.environ.get('PASADENA_DIR', 'data/external/PasadenaUrbanTrees_Detection'), 'PasadenaUrbanTrees')
panos = json.load(open(D + '/panos.json')); rois = json.load(open(D + '/rois.json')); dd = json.load(open(D + '/detection_datasets.json'))
R, LAT0 = 6371008.8, 34.15
import os
ASSIGN_MAX, END_MARGIN, GRID = float(os.environ.get('GAP_ASSIGN_MAX', 20)), 10.0, 1.0
AREA = int(os.environ.get('GAP_AREA', 2))  # 1 = validation ROI, 2 = test ROI


def enu(la, lo):
    return np.stack([np.radians(np.asarray(lo, float) + 118.13) * R * math.cos(math.radians(LAT0)), np.radians(np.asarray(la, float) - LAT0) * R], -1)


def tvt(la, lo):
    for r in rois:
        a, b, c, d = r['roi']
        if min(a, c) <= la <= max(a, c) and min(b, d) <= lo <= max(b, d): return r['train_val_test']


def is_fwy(n): return n == '' or n.startswith(('I-', 'SR-', 'CA-', 'US-')) or 'Fwy' in n or 'Freeway' in n


# ---------- street chains ----------
xy = {k: enu(float(v['Location']['lat']), float(v['Location']['lng'])) for k, v in panos.items()}
name = {k: v['Location'].get('description', '') for k, v in panos.items()}
test = {k for k, v in panos.items() if tvt(float(v['Location']['lat']), float(v['Location']['lng'])) == AREA and not is_fwy(name[k])}
nbr = collections.defaultdict(set); inter = set()
for k in test:
    for l in panos[k].get('Links', []):
        b = l['panoId']
        if b in test and name[b] == name[k]: nbr[k].add(b); nbr[b].add(k)
        elif l.get('description', '') not in ('', name[k]): inter.add(k)
brk = {k for k in test if len(nbr[k]) != 2 or k in inter}
chains, seen = [], set()
def walk(a, b):
    path = [a, b]
    while path[-1] not in brk:
        nx = [n for n in nbr[path[-1]] if n != path[-2]]
        if not nx or nx[0] in path: break
        path.append(nx[0])
    return path
for a in brk:
    for b in nbr[a]:
        e = tuple(sorted((a, b)))
        if e in seen: continue
        p = walk(a, b)
        for u, v in zip(p, p[1:]): seen.add(tuple(sorted((u, v))))
        chains.append(p)
seg = []
for ci, p in enumerate(chains):
    P = np.array([xy[k] for k in p]); s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    for j in range(len(p) - 1): seg.append((ci, P[j], P[j + 1], s[j]))
chain_len = np.array([sum(np.linalg.norm(np.diff(np.array([xy[k] for k in p]), axis=0), axis=1)) for p in chains])
mid = np.array([(a + b) / 2 for _, a, b, _ in seg]); kd = cKDTree(mid)


def assign(T):
    """T: (N,2) ENU -> DataFrame chain, side (+1 left / -1 right), s, dist."""
    out = []
    _, cand = kd.query(T, k=min(12, len(seg)))
    for t, cs in zip(T, cand):
        best = (1e9, None)
        for c in cs:
            ci, a, b, s0 = seg[c]; ab = b - a; L2 = ab @ ab
            u = np.clip(((t - a) @ ab) / L2, 0, 1) if L2 > 0 else 0.0; q = a + u * ab; d = np.linalg.norm(t - q)
            if d < best[0]: best = (d, (ci, 1 if (ab[0] * (t - a)[1] - ab[1] * (t - a)[0]) > 0 else -1, s0 + u * math.sqrt(L2)))
        out.append((*best[1], best[0]) if best[0] <= ASSIGN_MAX else (-1, 0, np.nan, best[0]))
    return pd.DataFrame(out, columns=['chain', 'side', 's', 'dist'])


def plantable_mask(svals, L, S):
    g = np.arange(END_MARGIN, L - END_MARGIN + 1e-9, GRID)
    if len(g) == 0: return g, np.zeros(0, bool)
    if len(svals) == 0: return g, np.ones(len(g), bool)
    sv = np.sort(svals); i = np.searchsorted(sv, g)
    dl = np.where(i > 0, g - sv[np.maximum(i - 1, 0)], np.inf); dr = np.where(i < len(sv), sv[np.minimum(i, len(sv) - 1)] - g, np.inf)
    return g, np.minimum(dl, dr) >= S


def runs_count(mask, S):
    n, run = 0, 0
    for m in np.r_[mask, False]:
        if m: run += 1
        elif run: n += int(((run - 1) * GRID) // S) + 1; run = 0
    return n


def evaluate(inv_a, pred_a, S):
    tp = fp = fn = 0; rows = []
    keys = set(map(tuple, inv_a.loc[inv_a.chain >= 0, ['chain', 'side']].values))
    for ci, sd in keys:
        L = chain_len[ci]
        si = inv_a.s[(inv_a.chain == ci) & (inv_a.side == sd)].values; sp = pred_a.s[(pred_a.chain == ci) & (pred_a.side == sd)].values
        _, mi = plantable_mask(si, L, S); _, mp = plantable_mask(sp, L, S)
        tp += (mi & mp).sum(); fp += (~mi & mp).sum(); fn += (mi & ~mp).sum()
        rows.append(dict(chain=ci, side=sd, L=L, n_inv_trees=len(si), n_pred_trees=len(sp), add_inv=runs_count(mi, S), add_pred=runs_count(mp, S)))
    r = pd.DataFrame(rows); pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return dict(S=S, chain_sides=len(r), km_evaluated=round(float(r.L.sum()) / 1000, 1),
                plantable_km_inv=round(float(tp + fn) * GRID / 1000, 2), plantable_km_pred=round(float(tp + fp) * GRID / 1000, 2),
                precision=round(pr, 3), recall=round(rc, 3), F1=round(2 * pr * rc / max(pr + rc, 1e-9), 3),
                addable_inv=int(r.add_inv.sum()), addable_pred=int(r.add_pred.sum()),
                pearson_r_per_side=round(float(np.corrcoef(r.add_inv, r.add_pred)[0, 1]), 3),
                MAE_per_side=round(float((r.add_inv - r.add_pred).abs().mean()), 2)), r


NONTREE = ['UNDER PROCESS FOR PLANT', 'NO ROOM FOR REPLANT', 'PLANTING SITE-SMALL', 'PLANTING SITE-MEDIUM', 'PLANTING SITE-LARGE',
           'STUMP', 'NO TREE AT THIS SITE', 'INSUFFICIENT PLANTING SITE']
VACANT = ['UNDER PROCESS FOR PLANT', 'PLANTING SITE-SMALL', 'PLANTING SITE-MEDIUM', 'PLANTING SITE-LARGE']


def load_inventory():
    import ast
    inv = pd.DataFrame(dd['test' if AREA == 2 else 'val'])
    ext = inv.extended.apply(lambda e: e if isinstance(e, dict) else ast.literal_eval(e))
    inv['cn'] = ext.apply(lambda e: (e.get('Common_Name') or '').upper().strip())
    return inv


def site_hits(sites_a, tree_a, pred_a, S):
    """Share of city sites (on evaluated sides) inside the predicted / inventory-derived plantable mask, and chance level."""
    keys = set(map(tuple, tree_a.loc[tree_a.chain >= 0, ['chain', 'side']].values))
    hp = hi = n = 0; cover = tot = 0
    for ci, sd in keys:
        L = chain_len[ci]
        g, mi = plantable_mask(tree_a.s[(tree_a.chain == ci) & (tree_a.side == sd)].values, L, S)
        _, mp = plantable_mask(pred_a.s[(pred_a.chain == ci) & (pred_a.side == sd)].values, L, S)
        cover += mp.sum(); tot += len(mp)
        ss = sites_a.s[(sites_a.chain == ci) & (sites_a.side == sd)].values
        ss = ss[(ss >= END_MARGIN) & (ss <= L - END_MARGIN)]
        if len(g) == 0 or len(ss) == 0: continue
        k = np.clip(np.round((ss - END_MARGIN) / GRID).astype(int), 0, len(g) - 1)
        hp += mp[k].sum(); hi += mi[k].sum(); n += len(ss)
    return dict(n_sites=int(n), in_pred_gap=round(hp / max(n, 1), 3), in_inventory_gap=round(hi / max(n, 1), 3), chance=round(cover / max(tot, 1), 3))


if __name__ == '__main__':
    pred_csv = sys.argv[1] if len(sys.argv) > 1 else 'pas_e2_test_trees_dedup.csv'
    inv = load_inventory(); pred = pd.read_csv(pred_csv)
    trees = inv[~inv.cn.isin(NONTREE)].reset_index(drop=True)
    IA = assign(enu(trees.lat, trees.lng)); PA = assign(enu(pred.lat, pred.lng))
    print(f'chains {len(chains)}  km {chain_len.sum()/1000:.1f}  trees {len(trees)} (non-tree records removed: {len(inv)-len(trees)})  '
          f'inv assigned {(IA.chain>=0).mean():.3f}  pred assigned {(PA.chain>=0).mean():.3f}', flush=True)
    kdi = cKDTree(enu(trees.lat, trees.lng)); d, j = kdi.query(enu(pred.lat, pred.lng))
    m = (d <= 4) & (PA.chain.values >= 0) & (IA.chain.values[j] >= 0)
    side_ok = float(((PA.chain.values == IA.chain.values[j]) & (PA.side.values == IA.side.values[j]))[m].mean())
    res = []
    for S in (10, 12, 15):
        r, per = evaluate(IA, PA, S); r['side_agreement_matched'] = round(side_ok, 3); res.append(r)
        if S == 12: per.to_csv(pred_csv.replace('.csv', '_gaps_per_side.csv'), index=False)
    empty = PA.copy(); empty['chain'] = -1
    for S in (10, 12, 15):
        b, _ = evaluate(IA, empty, S); b['S'] = f'{S} (baseline: no trees detected)'; res.append(b)
    out = pd.DataFrame(res); print(out.to_string(index=False))
    out.to_csv(pred_csv.replace('.csv', f'_gap_validation_cut{int(ASSIGN_MAX)}.csv'), index=False)
    sites = []
    for nm, sel in (('city vacant planting sites', inv.cn.isin(VACANT)), ('city "no room for replant" sites', inv.cn == 'NO ROOM FOR REPLANT'), ('stumps', inv.cn == 'STUMP')):
        SA = assign(enu(inv.lat[sel], inv.lng[sel]))
        for S in (10, 12, 15): sites.append(dict(site_type=nm, S=S, **site_hits(SA, IA, PA, S)))
    st = pd.DataFrame(sites); print(st.to_string(index=False))
    st.to_csv(pred_csv.replace('.csv', f'_city_sites_cut{int(ASSIGN_MAX)}.csv'), index=False)

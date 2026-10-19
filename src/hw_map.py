import os
"""Highway tree map (hw_prereg.md, amendments 3B, 4, 7, 8). Usage: python hw_map.py OUT_PREFIX DETS_GLOB [RECALL]
DETS: id,x0,y0,x1,y1,score,ku,kv,kconf,img_w,img_h (normalised). RECALL: measured share of trees found (test), used only
for the corrected trees per km. Writes OUT_PREFIX_trees.csv, _pieces.csv, _km.csv, _e8.json."""
import glob, json, math, os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import hw_geo as hg

OUT, DG = sys.argv[1], sys.argv[2]; RECALL = float(sys.argv[3]) if len(sys.argv) > 3 else None
CONF, KCONF, RMIN, RMAX, MERGE_R, LAT_MAX, PIECE = 0.25, 0.5, 2.0, 20.0, 5.0, 25.0, 50.0

meta = pd.read_parquet('hw_all_meta.parquet'); cams = hg.prepare(meta)
U = pd.read_parquet('hw_all_usability.parquet').set_index('id')
hz = pd.read_csv('hw_seq_horizon.csv').set_index('sequence')
cams = cams.join(U[['accepted']], how='inner')
cams['h_frac'] = cams.sequence.map(hz.horizon); cams['seq_valid'] = cams.sequence.map(hz.seq_valid).fillna(False).astype(bool)
mapc = cams[cams.accepted & cams.seq_valid]

# route polyline (ENU), along-route distance and side
line = np.array(json.load(open('n1_coverage.json'))['line']); L = hg.enu(line[:, 1], line[:, 0])
seg = np.diff(L, axis=0); sl = np.hypot(seg[:, 0], seg[:, 1]); s0 = np.r_[0, np.cumsum(sl)]
dd = np.arange(0, s0[-1], 2.0); Dp = np.c_[np.interp(dd, s0, L[:, 0]), np.interp(dd, s0, L[:, 1])]; kd = cKDTree(Dp)
tang = np.gradient(Dp, axis=0); tang /= np.linalg.norm(tang, axis=1, keepdims=True)
def project(E, N):
    d, j = kd.query(np.c_[E, N]); v = np.c_[E, N] - Dp[j]
    side = np.sign(tang[j, 0] * v[:, 1] - tang[j, 1] * v[:, 0])           # +1 left of the Dhaka->Chattogram direction
    return dd[j], d * side

# detections -> ground points
D = pd.concat([pd.read_csv(f, dtype={'id': str}) for f in sorted(glob.glob(DG))], ignore_index=True)
D = D[(D.score >= CONF) & D.id.isin(mapc.index)].copy()
kp = D.kconf >= KCONF
D['pu'] = np.where(kp, D.ku, (D.x0 + D.x1) / 2); D['pv'] = np.where(kp, D.kv, D.y1); D['point'] = np.where(kp, 'keypoint', 'box bottom')
c = mapc.loc[D.id]
pitch = np.array([hg.pitch_from_horizon(h, H, W, f, k1, k2, fe) for h, H, W, f, k1, k2, fe in
                  zip(c.h_frac, D.img_h, D.img_w, c.f, c.k1, c.k2, c.fisheye)])
R = np.stack([hg.level_rotation(hd, p) for hd, p in zip(c.computed_compass_angle.values, pitch)])
E, N, rng = hg.locate_level(c.camE.values, c.camN.values, R, c.f.values, c.k1.values, c.k2.values, c.fisheye.values,
                            D.pu.values, D.pv.values, D.img_w.values, D.img_h.values, np.where(c.vehicle.values, 1.6, 1.5))
D['E'], D['N'], D['range_m'], D['sequence'] = E, N, rng, c.sequence.values
D = D[(D.range_m >= RMIN) & (D.range_m <= RMAX)]

def merge(d, radius=MERGE_R):
    """Pasadena-validated rule: greedy score-ordered, each detection joins the nearest centre within radius."""
    d = d.sort_values('score', ascending=False); cen, ws, best, nv = [], [], [], []
    for x, y, s in zip(d.E.values, d.N.values, d.score.values):
        if cen:
            C = np.array(cen); dist = np.hypot(C[:, 0] - x, C[:, 1] - y); k = int(dist.argmin())
            if dist[k] <= radius:
                cen[k] = (C[k] * ws[k] + np.array([x, y]) * s) / (ws[k] + s); ws[k] += s; nv[k] += 1; continue
        cen.append(np.array([x, y], float)); ws.append(s); best.append(s); nv.append(1)
    C = np.array(cen) if cen else np.zeros((0, 2))
    return pd.DataFrame(dict(E=C[:, 0], N=C[:, 1], score=best, n_views=nv))

T = []
for sq, g in D.groupby('sequence'):
    m = merge(g); m['sequence'] = sq; T.append(m)
T = pd.concat(T, ignore_index=True) if T else pd.DataFrame(columns=['E', 'N', 'score', 'n_views', 'sequence'])
T['s'], T['lat_off'] = project(T.E.values, T.N.values); T = T[T.lat_off.abs() <= LAT_MAX].copy()
T['side'] = np.where(T.lat_off > 0, 'left', 'right'); T['piece'] = (T.s // PIECE).astype(int)
T['lat'], T['lon'] = hg.to_ll(T.E.values, T.N.values); T.to_csv(f'{OUT}_trees.csv', index=False)

# coverage: a trip covers a piece if one of its map photos projects onto it
cs, _ = project(mapc.camE.values, mapc.camN.values)
cov = pd.DataFrame(dict(sequence=mapc.sequence.values, piece=(cs // PIECE).astype(int))).drop_duplicates()
cnt = T.groupby(['piece', 'side', 'sequence']).size().to_dict()
n_photos = mapc.sequence.value_counts().to_dict(); trips_of = cov.groupby('piece').sequence.apply(list).to_dict()
rows = []
for pc in sorted(trips_of):
    trips = trips_of[pc]; main = max(trips, key=lambda t: n_photos.get(t, 0))       # covering trip with most map photos
    for sd in ('left', 'right'):
        ns = [int(cnt.get((pc, sd, t), 0)) for t in trips]
        rows.append(dict(piece=pc, side=sd, s_start=pc * PIECE, n_trips=len(trips), has_trees=bool(max(ns) > 0),
                         median_count=float(np.median(ns)), single_trip_count=int(cnt.get((pc, sd, main), 0))))
P = pd.DataFrame(rows); P.to_csv(f'{OUT}_pieces.csv', index=False)
P['km'] = (P.s_start // 1000).astype(int)
K = P.groupby('km').agg(pieces_covered=('piece', lambda x: x.nunique()), trees_median_sum=('median_count', 'sum'),
                        empty_share_combined=('has_trees', lambda x: 1 - x.mean()), empty_share_single=('single_trip_count', lambda x: (x == 0).mean())).reset_index()
K['covered_km'] = K.pieces_covered * PIECE / 1000; K['trees_per_km'] = K.trees_median_sum / K.covered_km
if RECALL: K['trees_per_km_corrected'] = K.trees_per_km / RECALL
K.to_csv(f'{OUT}_km.csv', index=False)

# E8: 100 m pieces covered by >= 2 trips; trees per 100 m per trip; all trip pairs
T['p100'] = (T.s // 100).astype(int); cov['p100'] = cov.piece // 2
c100 = T.groupby(['p100', 'sequence']).size()
pairs = []
for p, g in cov.groupby('p100'):
    tr = sorted(g.sequence.unique())
    for i in range(len(tr)):
        for j in range(i + 1, len(tr)): pairs.append((int(c100.get((p, tr[i]), 0)), int(c100.get((p, tr[j]), 0))))
pr = np.array(pairs) if pairs else np.zeros((0, 2))
e8 = dict(n_pairs=len(pairs), n_pieces=int((cov.groupby('p100').sequence.nunique() >= 2).sum()),
          spearman=float(spearmanr(pr[:, 0], pr[:, 1]).correlation) if len(pr) > 2 else None,
          median_abs_diff=float(np.median(np.abs(pr[:, 0] - pr[:, 1]))) if len(pr) else None,
          mean_trees_per_100m=float(pr.mean()) if len(pr) else None)
# E8 secondary (added after the 100 m result was seen on the provisional map): 1 km stretches, trees per covered km per
# trip, trips covering >= 10 of the 20 pieces of a stretch; all trip pairs per stretch
T['km'] = (T.s // 1000).astype(int); cov['km'] = cov.piece // 20
ck = cov.groupby(['km', 'sequence']).piece.nunique(); tk = T.groupby(['km', 'sequence']).size()
pairs1 = []
for k_, g_ in ck[ck >= 10].groupby(level=0):
    tr = [s_ for _, s_ in g_.index]; dens = [tk.get((k_, s_), 0) / (g_.loc[(k_, s_)] * PIECE / 1000) for s_ in tr]
    for i in range(len(tr)):
        for jj in range(i + 1, len(tr)): pairs1.append((dens[i], dens[jj]))
p1 = np.array(pairs1) if pairs1 else np.zeros((0, 2))
e8['km_secondary'] = dict(n_pairs=len(pairs1), n_stretches=int((ck[ck >= 10].groupby(level=0).size() >= 2).sum()),
                          spearman=float(spearmanr(p1[:, 0], p1[:, 1]).correlation) if len(p1) > 2 else None,
                          median_abs_diff_per_km=float(np.median(np.abs(p1[:, 0] - p1[:, 1]))) if len(p1) else None,
                          median_trees_per_km=float(np.median(p1)) if len(p1) else None)
# diagnostic: travel direction per trip (camera heading vs route direction); E8 at 1 km for same-direction pairs only
_, _j = kd.query(np.c_[mapc.camE.values, mapc.camN.values]); rb = np.degrees(np.arctan2(tang[_j, 0], tang[_j, 1]))
mapc = mapc.assign(dirn=np.cos(np.radians(mapc.computed_compass_angle.values - rb)))
tdir = mapc.groupby('sequence').dirn.median(); trip_dir = np.sign(tdir).to_dict()
pairs_s, pairs_o = [], []
for k_, g_ in ck[ck >= 10].groupby(level=0):
    tr = [s_ for _, s_ in g_.index]; dens = [tk.get((k_, s_), 0) / (g_.loc[(k_, s_)] * PIECE / 1000) for s_ in tr]
    for i in range(len(tr)):
        for jj in range(i + 1, len(tr)): (pairs_s if trip_dir[tr[i]] == trip_dir[tr[jj]] else pairs_o).append((dens[i], dens[jj]))
for nm, pp in (('same_direction', pairs_s), ('opposite_direction', pairs_o)):
    pp = np.array(pp) if pp else np.zeros((0, 2))
    e8[f'km_{nm}'] = dict(n_pairs=len(pp), spearman=float(spearmanr(pp[:, 0], pp[:, 1]).correlation) if len(pp) > 2 else None,
                          median_abs_diff_per_km=float(np.median(np.abs(pp[:, 0] - pp[:, 1]))) if len(pp) else None)
e8['trips_by_direction'] = {str(int(k)): int(v) for k, v in pd.Series(trip_dir).value_counts().items()}
summary = dict(photos_in_map=int(len(mapc)), trips_in_map=int(mapc.sequence.nunique()), detections_used=int(len(D)),
               keypoint_share=float((D.point == 'keypoint').mean()) if len(D) else None, trees=int(len(T)),
               pieces_covered=int(P.piece.nunique()), covered_km=float(P.piece.nunique() * PIECE / 1000),
               empty_share_single_trip=float((P.single_trip_count == 0).mean()), empty_share_combined=float(1 - P.has_trees.mean()),
               trees_per_km_median=float(K.trees_per_km.median()) if len(K) else None, e8=e8)
json.dump(summary, open(f'{OUT}_summary.json', 'w'), indent=1); print(json.dumps(summary, indent=1))

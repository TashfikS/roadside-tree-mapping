import os
"""Highway outputs allowed by the transfer decision (trees per km, no gaps): heat ranking (amendment 6) and the satellite
canopy comparison (amendment 3C). Distances along the shared route polyline are converted between the map's local
projection, UTM 46N (temperature stretches) and geodesic length (canopy stretches) through the polyline vertices.
Writes hw_heat_ranking.csv, hw_canopy_100m.csv, hw_outputs.json."""
import json, math, os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from rasterio.warp import transform as wtransform
import hw_geo as hg

line = np.array(json.load(open('n1_coverage.json'))['line'])                       # (lon, lat)
idx = np.arange(len(line))
def cum(P): return np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
s_map = cum(hg.enu(line[:, 1], line[:, 0]))                                          # as in hw_map.py
X, Y = wtransform('EPSG:4326', 'EPSG:32646', line[:, 0].tolist(), line[:, 1].tolist()); s_utm = cum(np.c_[X, Y])   # as in lst_corridor.py
la1, la2, lo1, lo2 = map(np.radians, (line[:-1, 1], line[1:, 1], line[:-1, 0], line[1:, 0]))
s_hav = np.r_[0, np.cumsum(2 * 6371008.8 * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2)))]  # as in chm_corridor.py
def map_to(s, target): return np.interp(np.interp(s, s_map, idx), idx, target)
print('route km: map %.2f, UTM %.2f, geodesic %.2f' % (s_map[-1] / 1000, s_utm[-1] / 1000, s_hav[-1] / 1000))

P = pd.read_csv('map_final/hw_pieces.csv'); K = pd.read_csv('map_final/hw_km.csv')
LST = pd.read_csv('lst_corridor_1km.csv').set_index('stretch'); CH = pd.read_csv('chm_corridor_100m.csv')
RECALL = json.load(open('hw_test_results.json'))['new']['mean']

# ---- heat ranking: each 1 km map stretch gets the temperature of the UTM 1 km stretch containing its midpoint
K['s_mid'] = K.km * 1000 + 500; K['lst_stretch'] = (map_to(K.s_mid.values, s_utm) // 1000).astype(int)
K['lst_c'] = K.lst_stretch.map(LST.lst_median_c)
CH['s_mid'] = (CH.s_start_m + CH.s_end_m) / 2; inv_h = lambda s: np.interp(np.interp(s, s_hav, idx), idx, s_map)
CH['map_km'] = (inv_h(CH.s_mid.values) // 1000).astype(int)
can_km = CH.groupby('map_km').canopy_ge3.mean() * 100
K['canopy_pct'] = K.km.map(can_km)
H = K[K.covered_km >= 0.5].copy()                                                    # stretches with at least half a km covered
H['trees_per_km_corrected'] = H.trees_per_km / RECALL
H = H.sort_values('lst_c', ascending=False).reset_index(drop=True); H['heat_rank'] = np.arange(1, len(H) + 1)
H[['heat_rank', 'km', 'lst_c', 'trees_per_km', 'trees_per_km_corrected', 'canopy_pct', 'covered_km', 'empty_share_combined']].to_csv('hw_heat_ranking.csv', index=False)
r1 = spearmanr(H.lst_c, H.trees_per_km, nan_policy='omit'); r2 = spearmanr(H.lst_c, H.canopy_pct, nan_policy='omit')
r3 = spearmanr(H.trees_per_km, H.canopy_pct, nan_policy='omit')

# ---- canopy comparison at 100 m (geodesic bins of the canopy table): our trees per 100 m, trips combined
P['s_mid'] = P.s_start + 25; P['hbin'] = (map_to(P.s_mid.values, s_hav) // 100).astype(int)
G = P.groupby('hbin').agg(trees=('median_count', 'sum'), any_tree=('has_trees', 'max'), n_piece_sides=('piece', 'size'))
CH['hbin'] = (CH.s_start_m // 100).astype(int); M = G.join(CH.set_index('hbin')[['canopy_ge3']], how='inner')
M['canopy_pct'] = M.canopy_ge3 * 100; M = M[M.n_piece_sides >= 2]
r4 = spearmanr(M.trees, M.canopy_pct)
empty, withtree = M[~M.any_tree.astype(bool)], M[M.any_tree.astype(bool)]
M.reset_index().to_csv('hw_canopy_100m.csv', index=False)
out = dict(recall_used=RECALL, heat=dict(n_stretches=len(H), spearman_lst_trees=float(r1.correlation), p_lst_trees=float(r1.pvalue),
                                          spearman_lst_canopy=float(r2.correlation), p_lst_canopy=float(r2.pvalue),
                                          spearman_trees_canopy_1km=float(r3.correlation), p_trees_canopy_1km=float(r3.pvalue),
                                          hottest10_median_trees_per_km_corrected=float(H.head(10).trees_per_km_corrected.median()),
                                          coolest10_median_trees_per_km_corrected=float(H.tail(10).trees_per_km_corrected.median())),
           canopy_100m=dict(n_stretches=int(len(M)), spearman_trees_canopy=float(r4.correlation), p=float(r4.pvalue),
                            empty_share=float(len(empty) / len(M)), empty_with_canopy_lt10=float((empty.canopy_pct < 10).mean()),
                            withtrees_with_canopy_lt10=float((withtree.canopy_pct < 10).mean()),
                            canopy_median_empty=float(empty.canopy_pct.median()), canopy_median_withtrees=float(withtree.canopy_pct.median())))
json.dump(out, open('hw_outputs.json', 'w'), indent=1); print(json.dumps(out, indent=1))
print(H.head(8)[['heat_rank', 'km', 'lst_c', 'trees_per_km_corrected', 'canopy_pct']].round(1).to_string(index=False))

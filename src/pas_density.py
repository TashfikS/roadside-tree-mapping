"""Per street side (test area, validation-chosen setting): trees per km and empty share at S = 10/12/15 m, detected vs survey."""
import os, sys, numpy as np, pandas as pd
os.environ['GAP_AREA'] = '2'; os.environ['GAP_ASSIGN_MAX'] = '12'; sys.path.insert(0, '.')
import pas_gaps as pg
inv = pg.load_inventory(); trees = inv[~inv.cn.isin(pg.NONTREE)].reset_index(drop=True); pred = pd.read_csv('gaps_test_v2.csv')
IA = pg.assign(pg.enu(trees.lat, trees.lng)); PA = pg.assign(pg.enu(pred.lat, pred.lng))
rows = []
for ci, sd in sorted(set(map(tuple, IA.loc[IA.chain >= 0, ['chain', 'side']].values))):
    L = pg.chain_len[ci]
    if L < 100: continue
    si = IA.s[(IA.chain == ci) & (IA.side == sd)].values; sp = PA.s[(PA.chain == ci) & (PA.side == sd)].values
    r = dict(chain=ci, side=sd, km=L / 1000, inv_per_km=len(si) / L * 1000, det_per_km=len(sp) / L * 1000)
    for S in (10, 12, 15):
        _, mi = pg.plantable_mask(si, L, S); _, mp = pg.plantable_mask(sp, L, S)
        r[f'inv_empty_{S}'] = mi.mean() if len(mi) else np.nan; r[f'det_empty_{S}'] = mp.mean() if len(mp) else np.nan
    rows.append(r)
D = pd.DataFrame(rows); D.to_csv('pas_density_per_side.csv', index=False)
out = [dict(metric='trees per km', sides=len(D), km=round(D.km.sum(), 1), r=round(np.corrcoef(D.inv_per_km, D.det_per_km)[0, 1], 3),
            median_abs_diff=round(float(np.median(np.abs(D.det_per_km - D.inv_per_km))), 1), survey_median=round(D.inv_per_km.median(), 1), detected_median=round(D.det_per_km.median(), 1))]
for S in (10, 12, 15):
    a, b = D[f'inv_empty_{S}'] * 100, D[f'det_empty_{S}'] * 100; m = a.notna() & b.notna()
    out.append(dict(metric=f'empty share % (S={S} m)', sides=int(m.sum()), km=round(D.km[m].sum(), 1), r=round(np.corrcoef(a[m], b[m])[0, 1], 3),
                    median_abs_diff=round(float(np.median(np.abs(b[m] - a[m]))), 1), survey_median=round(a[m].median(), 1), detected_median=round(b[m].median(), 1)))
O = pd.DataFrame(out); O.to_csv('pas_density_validation.csv', index=False); print(O.to_string(index=False))

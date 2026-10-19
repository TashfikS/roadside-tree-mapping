import os
"""E2b test area, applied once with the validation-chosen setting (mly_prereg.md). Writes mly_test_results.csv/json."""
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mly_setup.py')).read())
D = load_dets(); Dt = D[D.id.map(cams.tvt) == 2]
assert Dt.id.nunique() == int((cams.tvt == 2).sum()), (Dt.id.nunique(), int((cams.tvt == 2).sum()))
CH = dict(orient='sfm', h=1.6, cut=20, radius=5, mv=1)
def run(orient=CH['orient'], h=CH['h'], rng=None):
    L = mg.locate(Dt, cams, orient=orient, h_vehicle=h, rng=rng); T = mg.merge(L[L.range_m <= CH['cut']], CH['radius'])
    return T[T.n_views >= CH['mv']]
res = {}
res['Mapillary (chosen setting)'] = ev(T0 := run(), 'test')
res['Mapillary, level camera (comparison)'] = ev(run(orient='level'), 'test')
res['Mapillary, camera height 1.3 m'] = ev(run(h=1.3), 'test')
res['Mapillary, camera height 1.9 m'] = ev(run(h=1.9), 'test')
res['Chance (random bearing)'] = ev(run(rng=np.random.default_rng(0)), 'test')
G2 = in_zone(pd.read_csv('pas_e2_test_trees_dedup.csv'), 'test'); res['Google panoramas (E2, same zone)'] = ev(G2, 'test')
R = pd.DataFrame(res).T
# fallback metric: trees per 100 m x 100 m cell with >= 3 used image positions (test area)
def cell(E, N): return np.floor(np.asarray(E) / 100).astype(int) * 100000 + np.floor(np.asarray(N) / 100).astype(int)
c2 = cams[cams.tvt == 2]; cc = pd.Series(cell(c2.camE, c2.camN)).value_counts(); cells = set(cc[cc >= 3].index)
gt = ZONE['test']; n_inv = pd.Series(cell(gt[:, 0], gt[:, 1])).value_counts()
def cnt(T):
    P = mg.enu(T.lat.values, T.lng.values); return pd.Series(cell(P[:, 0], P[:, 1])).value_counts()
cnts = pd.DataFrame({'inventory': n_inv, 'mapillary': cnt(T0), 'google': cnt(G2)}).fillna(0)
cnts = cnts[cnts.index.isin(cells)]
cmp = {k: dict(pearson_r=round(float(np.corrcoef(cnts.inventory, cnts[k])[0, 1]), 3), median_abs_diff=float(np.median(np.abs(cnts[k] - cnts.inventory))),
               ratio_total=round(float(cnts[k].sum() / cnts.inventory.sum()), 3)) for k in ('mapillary', 'google')}
gate_med = R.loc['Mapillary (chosen setting)', 'median_m10']; ap_m = R.loc['Mapillary (chosen setting)', 'AP_4m']; ap_g = R.loc['Google panoramas (E2, same zone)', 'AP_4m']
gate = bool(gate_med <= 5 and ap_m >= 0.5 * ap_g)
R.to_csv('mly_test_results.csv')
json.dump(dict(setting=CH, zone_gt=len(ZONE['test']), images=int((cams.tvt == 2).sum()), results=res, cells=len(cnts), counts=cmp,
               gate=dict(median_m10=gate_med, AP_mly=ap_m, AP_google=ap_g, individual_positions=gate)), open('mly_test_results.json', 'w'), indent=1, default=float)
print(R.to_string()); print('cells', len(cnts), cmp); print('GATE individual positions:', gate)

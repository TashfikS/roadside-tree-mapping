"""E7 Pasadena, SECONDARY analysis added after the primary result (e7_prereg.md): one global scale factor k for the depth
ranges, chosen on the VALIDATION area by AP@4 m (k = 0.36-0.60, step 0.02), other settings fixed at those chosen for the
ground-plane method (point read-out, range cut 15 m, merge 5 m, views >= 1); then the test area once."""
import os, sys, json, math, glob
import numpy as np, pandas as pd
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pas_e7.py')).read(); pre = src[:src.index("rows = []")]
g = dict(__file__=os.path.abspath('pas_e7.py'), __name__='e7s'); exec(pre, g)
d, place, merge, evaluate = g['d'], g['place'], g['merge'], g['evaluate']
dec = json.load(open('pas_student_decision.json'))['val_best']; sb = [r for r in dec if r['detector'] == 'student'][0]
cut, rad, mv = int(sb['cut']), int(sb['radius']), int(sb['min_views'])
rows = []
for k in np.round(np.arange(0.36, 0.601, 0.02), 2):
    d['r_s'] = d.r_point * k; X = place(d, 'r_s'); t = merge(X[(X.tvt == 1) & (X.range_m <= cut)], radius=rad)
    rows.append(dict(k=k, **evaluate(t[t.n_views >= mv], 'val'))); print(rows[-1], flush=True)
V = pd.DataFrame(rows); V.to_csv('pas_e7_scaled_val.csv', index=False); kb = float(V.loc[V.AP_4m.idxmax(), 'k'])
d['r_s'] = d.r_point * kb; X = place(d, 'r_s'); t = merge(X[(X.tvt == 2) & (X.range_m <= cut)], radius=rad); t = t[t.n_views >= mv]
res = dict(k=kb, setting=dict(cut=cut, radius=rad, min_views=mv), test=evaluate(t, 'test'))
json.dump(res, open('pas_e7_scaled_result.json', 'w'), indent=1, default=float); print(json.dumps(res, default=float))

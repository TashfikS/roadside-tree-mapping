"""Pre-registered rule: the retrained (student) Pasadena detector replaces the current one only if it scores higher on the
VALIDATION area. Both detectors get the same validation sweep (cut 15/20 m, merge 3/5/7/9 m, views >= 1/2); the best
validation AP@4 m of each is compared. Only if the student wins is its chosen setting applied once to the test area."""
import os, sys, json
import pandas as pd
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pas_e2.py')).read(); head = src[:src.index("# ---------- 4.")]
if not os.path.exists('pas_student_raw_dets.parquet'):
    pd.read_csv('kout/pas_student/student_raw_dets.csv').to_parquet('pas_student_raw_dets.parquet')
rows, G = [], {}
for name, out, w in (('current', 'pas_e2', 'pas_y11s_best.pt'), ('student', 'pas_student', 'kout/pas_student/student_best.pt')):
    sys.argv = ['pas_e2.py', w, out]; g = dict(__file__=os.path.abspath('pas_e2.py'), __name__='pas_e2_head'); exec(head, g); G[name] = g
    det, merge, evaluate = g['det'], g['merge'], g['evaluate']
    for cut in (15, 20):
        for rad in (3, 5, 7, 9):
            t = merge(det[(det.tvt == 1) & (det.range_m <= cut)], radius=rad)
            for mv in (1, 2):
                rows.append(dict(detector=name, cut=cut, radius=rad, min_views=mv, **evaluate(t[t.n_views >= mv], 'val')))
    print(name, 'done', flush=True)
V = pd.DataFrame(rows); V.to_csv('pas_student_val_sweep.csv', index=False)
best = V.loc[V.groupby('detector').AP_4m.idxmax()].set_index('detector')
cols = ['cut', 'radius', 'min_views', 'n_trees', 'AP_4m', 'max_recall', 'gate_median_matched10']
print(best[[c for c in cols if c in best.columns]].to_string())
out = dict(val_best=best.reset_index().to_dict('records'))
if best.loc['student', 'AP_4m'] > best.loc['current', 'AP_4m']:
    b = best.loc['student']; g = G['student']
    t = g['merge'](g['det'][(g['det'].tvt == 2) & (g['det'].range_m <= b.cut)], radius=int(b.radius)); t = t[t.n_views >= int(b.min_views)]
    out['decision'] = 'student replaces current (higher validation AP@4 m)'
    out['test_student'] = g['evaluate'](t, 'test'); t.to_csv('pas_student_test_trees.csv', index=False)
    print('TEST (once):', out['test_student'])
else:
    out['decision'] = 'current detector stays (student not higher on validation)'
print(out['decision'])
json.dump(out, open('pas_student_decision.json', 'w'), indent=1, default=float)

"""E5 with the retrained (student) detector, same procedure as before: tree sets from the E2 setting chosen on validation
(cut 15 m, merge 5 m), views >= 1 or 2; gap assignment cut-off 12/15/20 m; choice by F1 at S = 12 m on the VALIDATION
area; then the chosen setting is applied once to the test area."""
import os, sys, subprocess
import pandas as pd
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pas_e2.py')).read(); head = src[:src.index("# ---------- 4.")]
sys.argv = ['pas_e2.py', 'kout/pas_student/student_best.pt', 'pas_student']; g = dict(__file__=os.path.abspath('pas_e2.py'), __name__='x'); exec(head, g)
det, merge = g['det'], g['merge']
for area, code in (('val', 1), ('test', 2)):
    t = merge(det[(det.tvt == code) & (det.range_m <= 15)], radius=5)
    for v in (1, 2): t[t.n_views >= v][['lat', 'lng', 'score', 'n_views']].to_csv(f'gaps_{area}_student_v{v}.csv', index=False)
def run(area, cut, f):
    env = dict(os.environ, GAP_AREA=str(area), GAP_ASSIGN_MAX=str(cut))
    subprocess.run([sys.executable, 'pas_gaps.py', f], env=env, check=True, capture_output=True)
    return pd.read_csv(f.replace('.csv', f'_gap_validation_cut{cut}.csv'))
rows = []
for v in (1, 2):
    for c in (12, 15, 20):
        d = run(1, c, f'gaps_val_student_v{v}.csv'); r = d[d.S.astype(str) == '12'].iloc[0]
        rows.append(dict(views=v, cut=c, precision=r.precision, recall=r.recall, F1=r.F1))
VG = pd.DataFrame(rows); VG.to_csv('pas_student_gaps_val_sweep.csv', index=False); print(VG.to_string(index=False))
b = VG.loc[VG.F1.idxmax()]; v, c = int(b.views), int(b.cut); print('chosen on validation: views >=', v, 'cut', c)
T = run(2, c, f'gaps_test_student_v{v}.csv'); print(T.drop(columns=[x for x in ('km_evaluated',) if x in T]).to_string(index=False))
S = pd.read_csv(f'gaps_test_student_v{v}_city_sites_cut{c}.csv'); print(S.to_string(index=False))
T.to_csv('pas_student_gap_validation_test.csv', index=False); S.to_csv('pas_student_gap_city_sites_test.csv', index=False)

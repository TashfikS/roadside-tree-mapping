import os
"""Highway transfer test (hw_prereg.md; decision metric as amended in amendment 4). Written before any model was run on the
test photos. Refuses to run unless the answer key matches its recorded checksum and the final model choice is recorded.

found  = the click lies inside a predicted box with confidence >= 0.25, the box enlarged by 3 % of the image width and
         height on each side (coordinates are normalised, so 0.03 on each axis).
metric = share of ALL clicked trees found (356 trees in 83 photos), per run; decision on the mean of the 3 new runs:
         >= 70 % gap map; 40-70 % trees per km; < 40 % no highway map.
Usage: python hw_score_test.py   (reads hw_final_model.json and hw_test_preds_<run>.csv for every run listed there)
"""
import hashlib, json, os
import numpy as np, pandas as pd, cv2

KEY, SHA = 'hw_test_answers_pass1.csv', None
for line in open('hw_prereg.md'):
    if 'SHA-256' in line and 'hw_test_answers.csv' in line: SHA = line.strip().rstrip('.').split()[-1]
assert SHA and hashlib.sha256(open(KEY, 'rb').read()).hexdigest() == SHA, 'answer key differs from the frozen file'
assert os.path.exists('hw_final_model.json'), 'final model not chosen yet: the test is scored only after the choice'
F = json.load(open('hw_final_model.json'))                  # {"old": [runs], "new": [runs], "choice": "..."}

A = pd.read_csv(KEY, dtype={'id': str})
C = A[(A.status == 'done') & A.u.notna()].reset_index(drop=True)
sharp = pd.Series({i: cv2.Laplacian(cv2.imread(f'hw/img/{i}.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32), cv2.CV_32F).var()
                   for i in C.id.unique()})
C['sharp_tertile'] = C.id.map(pd.qcut(sharp, 3, labels=['blurriest third', 'middle third', 'sharpest third']))
npp = C.groupby('id').size()
C['trees_in_photo'] = C.id.map(pd.cut(npp, [0, 2, 5, np.inf], labels=['1-2', '3-5', '6 or more']))
C['base'] = C.visible.map({2: 'visible', 1: 'guessed'})


def found(run):
    """Per clicked tree: True if inside an enlarged confident box. Also: confident boxes containing no click, per photo."""
    P = pd.read_csv(f'hw_test_preds_{run}.csv', dtype={'id': str}); P = P[P.score >= 0.25]
    G = {k: g for k, g in P.groupby('id')}
    f = pd.Series(False, index=C.index); spare = {}
    for i, g in C.groupby('id'):
        b = G.get(i)
        if b is None: spare[i] = 0; continue
        inside = ((g.u.values[:, None] >= b.x0.values - 0.03) & (g.u.values[:, None] <= b.x1.values + 0.03) &
                  (g.v.values[:, None] >= b.y0.values - 0.03) & (g.v.values[:, None] <= b.y1.values + 0.03))
        f.loc[g.index] = inside.any(1); spare[i] = int((~inside.any(0)).sum())
    return f, spare


rows, FOUND = [], {}
for grp in ('old', 'new'):
    for run in F[grp]:
        f, spare = found(run); FOUND[run] = f.values
        r = dict(group=grp, run=run, found_all=round(float(f.mean()), 4),
                 boxes_without_click_per_photo=round(float(np.mean(list(spare.values()))), 2))
        for col in ('base', 'sharp_tertile', 'trees_in_photo'):
            for lev, ix in C.groupby(col, observed=True).groups.items():
                r[f'{col}={lev}'] = round(float(f.loc[ix].mean()), 4)
        rows.append(r)
R = pd.DataFrame(rows); R.to_csv('hw_test_results_runs.csv', index=False)
# 95 % interval by photo-level bootstrap of the mean over runs
rng = np.random.default_rng(20261015); up = C.id.unique(); by = {p: np.where(C.id.values == p)[0] for p in up}
boot = {g: [] for g in ('old', 'new')}
for _ in range(2000):
    s = np.concatenate([by[p] for p in rng.choice(up, len(up))])
    for g in boot: boot[g].append(np.mean([FOUND[r][s].mean() for r in F[g]]))
S = {}
for g in boot:
    v = R[R.group == g].found_all
    S[g] = dict(runs=list(F[g]), mean=round(float(v.mean()), 4), sd=round(float(v.std(ddof=1)), 4) if len(v) > 1 else None,
                ci95=[round(float(np.percentile(boot[g], 2.5)), 4), round(float(np.percentile(boot[g], 97.5)), 4)])
m = S['new']['mean']
S['decision'] = 'gap map (>= 70 %)' if m >= 0.70 else ('trees per km, no gap map (40-70 %)' if m >= 0.40 else 'no highway map (< 40 %)')
S.update(n_trees=int(len(C)), n_photos=int(len(up)), choice=F.get('choice'))
json.dump(S, open('hw_test_results.json', 'w'), indent=1)
print(R.to_string(index=False)); print(json.dumps(S, indent=1))

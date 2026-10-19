"""Self-training step 2 (hw_prereg.md, amendments 3A and 7). Usage: python hw_selftrain_prep.py RUN DETS_GLOB
RUN: the better of A/B (e.g. hwb_s0); kout/RUN/hwval_preds_RUN.csv gives the threshold t = lowest confidence with box
precision >= 0.80 at IoU 0.5 on the highway validation photos (score-ordered one-to-one matching, as in COCO).
DETS_GLOB: detections of RUN on the self-training candidates. Writes hw_pseudo.json {photo id: [[x0,y0,x1,y1], ...]} and
hw_pseudo_info.json."""
import glob, json, sys
import numpy as np, pandas as pd

RUN, DG = sys.argv[1], sys.argv[2]
L = json.load(open('hw_train_labels.json'))
GT = {f'HW_{k}': np.array([[float(x) for x in ln.split()[1:5]] for ln in v['lines']]).reshape(-1, 4)
      for k, v in L.items() if v['split'] == 'val'}
GT = {k: np.c_[g[:, 0] - g[:, 2] / 2, g[:, 1] - g[:, 3] / 2, g[:, 0] + g[:, 2] / 2, g[:, 1] + g[:, 3] / 2] for k, g in GT.items()}
P = pd.read_csv(f'kout/{RUN}/hwval_preds_{RUN}.csv'); P['stem'] = P.image.str[:-4]
assert set(P.stem) <= set(GT), 'prediction for a photo that is not a highway validation photo'

def iou(b, G):
    ix = np.clip(np.minimum(b[2], G[:, 2]) - np.maximum(b[0], G[:, 0]), 0, None); iy = np.clip(np.minimum(b[3], G[:, 3]) - np.maximum(b[1], G[:, 1]), 0, None)
    i = ix * iy; return i / ((b[2] - b[0]) * (b[3] - b[1]) + (G[:, 2] - G[:, 0]) * (G[:, 3] - G[:, 1]) - i + 1e-12)

# true/false positive flag for every prediction (matching done once, in descending score over all photos)
P = P.sort_values('score', ascending=False).reset_index(drop=True); used = {k: np.zeros(len(g), bool) for k, g in GT.items()}; tp = []
for r in P.itertuples():
    g = GT[r.stem]
    if len(g) == 0: tp.append(False); continue
    o = iou((r.x0, r.y0, r.x1, r.y1), g); o[used[r.stem]] = -1; j = int(o.argmax())
    if o[j] >= 0.5: used[r.stem][j] = True; tp.append(True)
    else: tp.append(False)
P['tp'] = tp
# precision of all predictions with score >= t, for every candidate t (cumulative because P is sorted by score)
prec = np.cumsum(P.tp.values) / np.arange(1, len(P) + 1)
sc = P.score.values; group_end = np.r_[sc[1:] != sc[:-1], True]       # evaluate only complete groups of tied scores
ok = np.where((prec >= 0.80) & group_end)[0]
assert len(ok), 'no threshold reaches precision 0.80'
t = float(P.score.values[ok.max()])                     # lowest score whose cumulative set still has precision >= 0.80
n_gt = sum(len(g) for g in GT.values()); rec_t = float(P.tp[P.score >= t].sum() / n_gt)
D = pd.concat([pd.read_csv(f, dtype={'id': str}) for f in sorted(glob.glob(DG))], ignore_index=True)
cand = pd.read_csv('hw_selftrain_candidates.csv', dtype={'id': str})
assert set(D.id) <= set(cand.id), 'detections on photos outside the candidate list'
K = D[D.score >= t]; ps = {i: g[['x0', 'y0', 'x1', 'y1']].round(5).values.tolist() for i, g in K.groupby('id')}
if len(ps) > 3000:
    keep = set(np.random.default_rng(20261015).choice(sorted(ps), 3000, replace=False)); ps = {k: v for k, v in ps.items() if k in keep}
info = dict(run=RUN, threshold=round(t, 4), val_precision_at_t=round(float(prec[ok.max()]), 4), val_recall_at_t=round(rec_t, 4),
            val_photos=len(GT), val_boxes=n_gt, candidates=int(len(cand)), candidates_detected=int(D.id.nunique()),
            pseudo_photos=len(ps), pseudo_boxes=int(sum(len(v) for v in ps.values())))
json.dump(ps, open('hw_pseudo.json', 'w')); json.dump(info, open('hw_pseudo_info.json', 'w'), indent=1); print(json.dumps(info, indent=1))

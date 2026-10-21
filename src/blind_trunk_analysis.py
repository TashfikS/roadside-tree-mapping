"""Test R (protocols/trunk_ablation_prereg.md): trunk-base accuracy against points clicked from scratch (no proposal shown).
Usage: python src/blind_trunk_analysis.py data/blind_trunk_<name1>.csv [data/blind_trunk_<name2>.csv]
Inputs: data/bangladesh_test_trunk_review.csv (boxes), results/test_preds_pose_s1.csv (keypoint model run s1)
Output: results/blind_trunk_analysis.json
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TR = pd.read_csv(os.path.join(ROOT, 'data', 'bangladesh_test_trunk_review.csv')).set_index('key')
PP = pd.read_csv(os.path.join(ROOT, 'results', 'test_preds_pose_s1.csv')); PP = PP[PP.score >= 0.25]
A = [pd.read_csv(f) for f in sys.argv[1:]]


def matched_prediction(key):
    r = TR.loc[key]; img = key.rsplit('/', 1)[0].replace('/', '_') + '.jpg'; q = PP[PP.image == img]
    if not len(q): return None
    W, H = r.W, r.H; b = np.c_[q.x0 * W, q.y0 * H, q.x1 * W, q.y1 * H]; g = np.array([r.x0b, r.y0b, r.x1b, r.y1b])
    ix = np.clip(np.minimum(b[:, 2], g[2]) - np.maximum(b[:, 0], g[0]), 0, None); iy = np.clip(np.minimum(b[:, 3], g[3]) - np.maximum(b[:, 1], g[1]), 0, None)
    I = ix * iy; iou = I / ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) + (g[2] - g[0]) * (g[3] - g[1]) - I); j = int(iou.argmax())
    if iou[j] < 0.5: return None
    k = q.iloc[j]; return dict(ku=k.ku * W, kv=k.kv * H, bu=(b[j, 0] + b[j, 2]) / 2, bv=b[j, 3], box_h=r.box_h,
                               au=r.au, av=r.av)   # au/av: the automatic proposal shown in the first review


out = dict(n_annotators=len(A))
for a in A:
    who = str(a.annotator.iloc[0]); rows = []
    for r in a[a.status != 'none'].itertuples():
        m = matched_prediction(r.key)
        if m is None: continue
        rows.append(dict(key=r.key, status=r.status, e_kp=np.hypot(m['ku'] - r.u, m['kv'] - r.v), e_bb=np.hypot(m['bu'] - r.u, m['bv'] - r.v),
                         e_auto=np.hypot(m['au'] - r.u, m['av'] - r.v), box_h=m['box_h']))
    E = pd.DataFrame(rows)
    res = dict(n_clicked=int((a.status != 'none').sum()), n_cannot_locate=int((a.status == 'none').sum()), n_matched=len(E))
    for nm, d in (('all', E), ('visible', E[E.status == 'visible']), ('hidden', E[E.status == 'hidden'])):
        if len(d) < 3: continue
        res[nm] = dict(n=len(d), keypoint_median_px=round(float(d.e_kp.median()), 1), boxbottom_median_px=round(float(d.e_bb.median()), 1),
                       auto_proposal_median_px=round(float(d.e_auto.median()), 1),
                       keypoint_rel_pct=round(float((100 * d.e_kp / d.box_h).median()), 1), boxbottom_rel_pct=round(float((100 * d.e_bb / d.box_h).median()), 1),
                       keypoint_closer_share=round(float((d.e_kp < d.e_bb).mean()), 3), wilcoxon_p=float(wilcoxon(d.e_kp, d.e_bb).pvalue))
    out[who] = res
if len(A) == 2:
    m = A[0].merge(A[1], on='key', suffixes=('_1', '_2')); both = m[(m.status_1 != 'none') & (m.status_2 != 'none')]
    d = np.hypot(both.u_1 - both.u_2, both.v_1 - both.v_2); bh = TR.loc[both.key, 'box_h'].values
    out['inter_annotator'] = dict(n=len(both), median_px=round(float(np.median(d)), 1), median_rel_pct=round(float(np.median(100 * d / bh)), 1),
                                  p90_px=round(float(np.percentile(d, 90)), 1),
                                  status_agreement=round(float((m.status_1 == m.status_2).mean()), 3))
json.dump(out, open(os.path.join(ROOT, 'results', 'blind_trunk_analysis.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))

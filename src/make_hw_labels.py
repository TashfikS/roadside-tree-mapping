import os
"""Assemble highway training labels (hw_prereg.md, amendments 2 and 3) -> hw_train_labels.json, hw_usability_labels.csv.

Inputs: hw_boxes_answers.csv (author's boxes), hw_recheck.csv (author's re-check: part 1 = 'no roadside trees' photos,
keep or unusable; part 2 = suspected group boxes, keep or group), kout/hw_trunk/hw_trunk_auto.csv (automatic trunk points,
same method as the Kaggle training labels), hw_photos.csv (split, thumbnail URL).
Rules:
  - photos re-marked unusable: excluded from detector training; used as 'unusable' for the usability check;
  - 'no roadside trees' photos kept: empty label files (negatives);
  - boxes re-marked as groups: removed from the labels and their area greyed out in training (other boxes restored);
  - trunk point visibility 2 if found by the detector + SAM method, 0 otherwise (box trains tree finding only), also 0 if
    the point falls outside the image.
"""
import json, sys
import numpy as np, pandas as pd

ANS, RECHECK = sys.argv[1], sys.argv[2]
P = pd.read_csv('hw_photos.csv', dtype={'id': str}).set_index('id')
B = pd.read_csv(ANS, dtype={'id': str})
R = pd.read_csv(RECHECK)
T = pd.read_csv('kout/hw_trunk/hw_trunk_auto.csv', dtype={'id': str}).set_index('bid')

done = B[B.status == 'done'].reset_index(drop=True)                       # row order = bid used by the trunk job
assert len(done) == len(T) and (done.id.values == T.id.values).all(), 'box order differs from the trunk job'
st = B.groupby('key').agg(id=('id', 'first'), status=('status', 'first'))
r1 = R[R.part == 1].set_index('key').answer
r2 = R[R.part == 2].copy(); r2['bid'] = r2.key.str.split('#box').str[1].astype(int); groups = set(r2.bid[r2.answer == 'group'])

out, usab = {}, []
for key, s in st.iterrows():
    pid = s.id; split = P.loc[pid, 'split']; url = P.loc[pid, 'url']
    if s.status == 'none' and r1.get(key) == 'unusable':
        usab.append(dict(id=pid, split=split, usable=0)); continue
    usab.append(dict(id=pid, split=split, usable=1))
    lines, mask = [], []
    for bid in done.index[done.key == key]:
        b = done.loc[bid]
        if bid in groups:
            mask.append([float(b.x0), float(b.y0), float(b.x1), float(b.y1)]); continue
        xc, yc, w, h = (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2, b.x1 - b.x0, b.y1 - b.y0
        t = T.loc[bid]; vis = 2 if str(t.method).startswith('gdino') else 0
        ku, kv = (float(t.trunk_u), float(t.trunk_v)) if vis else (0.0, 0.0)
        if vis and not (0 <= ku <= 1 and 0 <= kv <= 1): ku, kv, vis = 0.0, 0.0, 0
        lines.append(f'0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {ku:.6f} {kv:.6f} {vis}')
    out[pid] = dict(split=split, url=url, lines=lines, mask=mask)
json.dump(out, open('hw_train_labels.json', 'w'))
U = pd.DataFrame(usab); U.to_csv('hw_usability_labels.csv', index=False)
nb = sum(len(v['lines']) for v in out.values())
print('photos for training', len(out), {s: sum(v['split'] == s for v in out.values()) for s in ('train', 'val')},
      '| boxes', nb, '| group boxes masked', sum(len(v['mask']) for v in out.values()),
      '| negatives', sum(len(v['lines']) == 0 for v in out.values()), '| unusable', int((U.usable == 0).sum()),
      '| trunk visible', sum(l.endswith(' 2') for v in out.values() for l in v['lines']))

"""Original thesis dataset: map every flipped/blurred copy to its source image and measure leakage in the original
train/valid lists (Final Dataset & Essential Files/train.txt, valid.txt).
Matching: 64x32 grey thumbnails, correlation with each source image and with its mirror image (either orientation);
check: labels identical or mirrored (tolerance 0.01). Output: audit/old_valid_leakage.csv, audit/aug_map.json, audit/match_summary.json"""
import os, json, time, numpy as np, pandas as pd
from PIL import Image, ImageOps
U = os.path.join(os.environ.get('THESIS_DIR', 'data/external'), 'To Upload', 'New_Dataset'); F = os.path.join(os.environ.get('THESIS_DIR', 'data/external'), 'Final Dataset & Essential Files')
def thumb(p, flip=False):
    im = Image.open(p).convert('L').resize((64, 32), Image.BILINEAR)
    if flip: im = ImageOps.mirror(im)
    a = np.asarray(im, np.float32).ravel(); return (a - a.mean()) / (a.std() + 1e-6)
def labels(p):
    t = p[:-4] + '.txt'
    return sorted(tuple(round(float(x), 3) for x in l.split()[1:5]) for l in open(t) if l.strip()) if os.path.exists(t) else None
def same(P, R): return len(P) == len(R) and all(max(abs(a - b) for a, b in zip(u, v)) <= 0.01 for u, v in zip(P, R))
t0 = time.time(); MAP, Q, CONF = {}, {}, {}
for src, fl, bl in (('BD_740', 'Flip_BD_740', 'Blur_BD_740'), ('BD_755', 'Flip_BD_755', 'Blur_BD_755'),
                    ('PasadenaUrbanTrees', 'PasadenaUrbanTrees_Flip', 'PasadenaUrbanTrees_Blur'),
                    ('Test_Set/Test_Set_270', 'Test_Set/Flip_Test_Set_270', 'Test_Set/Blur_Test_Set_270')):
    so = sorted(f for f in os.listdir(f'{U}/{src}') if f.endswith('.jpg'))
    A = np.stack([thumb(f'{U}/{src}/{f}') for f in so]); Af = np.stack([thumb(f'{U}/{src}/{f}', True) for f in so])
    for f in so: MAP[f'{src}/{f}'] = f'{src}/{f}'
    for aug in (fl, bl):
        fs = sorted(f for f in os.listdir(f'{U}/{aug}') if f.endswith('.jpg'))
        B = np.stack([thumb(f'{U}/{aug}/{f}') for f in fs]); S = np.maximum(B @ A.T, B @ Af.T) / B.shape[1]
        o = np.argsort(-S, 1); r = np.arange(len(fs)); best = S[r, o[:, 0]]; sec = S[r, o[:, 1]]; lab_ok = []
        for f, j, b in zip(fs, o[:, 0], best):
            MAP[f'{aug}/{f}'] = f'{src}/{so[j]}'; CONF[f'{aug}/{f}'] = float(b)
            la, lo = labels(f'{U}/{aug}/{f}'), labels(f'{U}/{src}/{so[j]}')
            if la is not None and lo is not None:
                lab_ok.append(same(la, lo) or same(la, sorted((round(1 - x, 3), y, w, h) for x, y, w, h in lo)))
        Q[aug] = dict(n=len(fs), n_src=len(so), corr_median=round(float(np.median(best)), 3), corr_min=round(float(best.min()), 3),
                      share_corr_ge_0_9=round(float((best >= 0.9).mean()), 3), label_match=round(float(np.mean(lab_ok)), 3), n_label_checked=len(lab_ok))
def loc(l): return l.strip().replace('/Thesis/MyDrive/Thesis/New_Dataset/', '')
tr = [loc(l) for l in open(f'{F}/train.txt') if l.strip()]; va = [loc(l) for l in open(f'{F}/valid.txt') if l.strip()]
src_tr = {MAP[x] for x in tr if x in MAP}; src_tr_conf = {MAP[x] for x in tr if x in MAP and CONF.get(x, 1.0) >= 0.9}
V = pd.DataFrame(dict(img=va)); V['src'] = V.img.map(MAP); V['conf'] = V.img.map(lambda x: CONF.get(x, 1.0))
V['group'] = V.img.str.split('/').str[0].str.replace('Flip_|Blur_|_Flip|_Blur', '', regex=True)
V['twin_in_train'] = V.src.isin(src_tr); V['twin_in_train_conf'] = (V.conf >= 0.9) & V.src.isin(src_tr_conf)
tst = {MAP[k] for k in MAP if k.startswith('Test_Set/')}
out = dict(seconds=round(time.time() - t0), matching=Q, n_train=len(tr), n_valid=len(va), valid_mapped=int(V.src.notna().sum()),
           valid_twin_in_train=int(V.twin_in_train.sum()), valid_twin_in_train_pct=round(float(V.twin_in_train.mean() * 100), 1),
           valid_twin_conservative=int(V.twin_in_train_conf.sum()), valid_twin_conservative_pct=round(float(V.twin_in_train_conf.mean() * 100), 1),
           by_group=V.groupby('group').twin_in_train.mean().round(3).to_dict(),
           train_folders=pd.Series([x.split('/')[0] for x in tr]).value_counts().to_dict(),
           test_set_sources_in_train_or_valid=len(tst & (src_tr | set(V.src.dropna()))))
V.to_csv('audit/old_valid_leakage.csv', index=False); json.dump(MAP, open('audit/aug_map.json', 'w'))
json.dump(out, open('audit/match_summary.json', 'w'), indent=1); print(json.dumps(out, indent=1))

"""Per-sequence horizon for the highway map (hw_prereg.md, amendment 4): median vanishing-point row over up to 30 evenly
spaced photos per sequence; estimates outside 0.30-0.90 of the image height are discarded; a sequence is valid if it has
>= 8 estimates and IQR <= 0.08. Uses the same vanishing-point estimator as the Bangladesh distance test (e3_horizon.py).
Output: hw_seq_horizon.csv, hw_seq_horizon_photos.csv."""
import os, sys, time, urllib.request, concurrent.futures as cf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import e3_horizon
from PIL import Image

M = pd.read_parquet('hw_all_meta.parquet')
M = M[M.camera_type.isin(['perspective', 'fisheye'])].sort_values(['sequence', 'captured_at'])
pick = M.groupby('sequence', group_keys=False).apply(lambda d: d.iloc[np.unique(np.linspace(0, len(d) - 1, min(30, len(d))).astype(int))])
os.makedirs('hw_hz', exist_ok=True)
def get(r):
    p = f'hw_hz/{r.id}.jpg'
    if os.path.exists(p) and os.path.getsize(p) > 0: return p
    for k in range(3):
        try:
            with urllib.request.urlopen(r.url1024, timeout=60) as f: open(p, 'wb').write(f.read()); return p
        except Exception: time.sleep(3)
    return None
with cf.ThreadPoolExecutor(12) as ex: paths = list(ex.map(get, pick.itertuples()))
print('downloaded', sum(p is not None for p in paths), 'of', len(pick), flush=True)
rows = []
for (r, p) in zip(pick.itertuples(), paths):
    if p is None: continue
    try:
        h = e3_horizon.horizon_row(p); H = Image.open(p).size[1]
        rows.append(dict(id=r.id, sequence=r.sequence, make=r.make, model=r.model, vp_frac=(h['vp_v'] / H) if h else np.nan,
                         n_inl=h['n_inliers'] if h else 0))
    except Exception as e:
        rows.append(dict(id=r.id, sequence=r.sequence, make=r.make, model=r.model, vp_frac=np.nan, n_inl=0))
P = pd.DataFrame(rows); P.to_csv('hw_seq_horizon_photos.csv', index=False)
P['valid'] = P.vp_frac.between(0.30, 0.90)
S = P.groupby('sequence').agg(make=('make', 'first'), model=('model', 'first'), n_photos=('id', 'size'), n_valid=('valid', 'sum'),
                              horizon=('vp_frac', lambda x: float(np.median(x[(x >= 0.30) & (x <= 0.90)])) if ((x >= 0.30) & (x <= 0.90)).any() else np.nan),
                              iqr=('vp_frac', lambda x: float(np.subtract(*np.percentile(x[(x >= 0.30) & (x <= 0.90)], [75, 25]))) if ((x >= 0.30) & (x <= 0.90)).sum() >= 2 else np.nan))
S['seq_valid'] = (S.n_valid >= 8) & (S.iqr <= 0.08)
S.to_csv('hw_seq_horizon.csv')
n_img = M.groupby('sequence').size()
print('sequences', len(S), 'valid', int(S.seq_valid.sum()), '| photos in valid sequences', int(n_img[S.index[S.seq_valid]].sum()), 'of', len(M))
print(S.groupby(['make', 'model']).agg(seqs=('seq_valid', 'size'), valid=('seq_valid', 'sum'), horizon=('horizon', 'median'), iqr=('iqr', 'median')).round(3).to_string())

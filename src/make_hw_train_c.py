import os
"""Candidate C (hw_prereg.md, amendments 3A and 7): the better of A/B plus self-training. Same job as make_hw_train.py
(same data, recipe and outputs), plus pseudo-labelled highway training-block photos read from the detection-batch job
outputs; their trunk points come from the same automatic method as the human boxes (Grounding DINO + SAM, identical crop
and thresholds), run inside this job before training.
Usage: python make_hw_train_c.py BASE_CFG SEED ACCOUNT PSEUDO_JSON CHUNK_KERNELS
  BASE_CFG A or B; PSEUDO_JSON {photo id: [[x0,y0,x1,y1], ...]} (normalised); CHUNK_KERNELS comma-separated slugs on ACCOUNT."""
import re, sys, json, zlib, base64

BASE, SEED, ACCT, PJ = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
text = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'make_hw_train.py')).read()
tmpl = re.search(r"src = r'''(.*?)'''\n", text, re.S).group(1)
th = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kaggle_jobs', 'templates', 'trunk_labels_highway.py')).read(); TRUNK_HEAD = th[:th.index('import io, urllib.request, collections')]
assert '__' not in TRUNK_HEAD.replace('__init__', '')
BLOCK = r'''
# ---- candidate C: pseudo-labelled training-block photos (trunk points: same automatic method as the human boxes) ----
import collections
PSEUDO = json.loads(zlib.decompress(base64.b64decode("__PSEUDO__")))
__TRUNK_HEAD__
def trunk_point(img, x0, y0, x1, y1):
    W_, H_ = img.size; xc = (x0 + x1) / 2
    cx0 = max(0, x0 - 0.1 * (x1 - x0)); cx1 = min(W_, x1 + 0.1 * (x1 - x0))
    cy0 = max(0, y0 + 0.3 * (y1 - y0)); cy1 = min(H_, y1 + 0.15 * (y1 - y0))
    crop = img.crop((int(cx0), int(cy0), int(cx1), int(cy1)))
    try:
        tb, ts = gdino(crop)
        if len(ts):
            ok = ts >= 0.8 * ts.max(); k = np.argmax(np.where(ok, tb[:, 3], -1)); trunk_box = tb[k]
            mk = sam_mask(crop, trunk_box); ys, xs = np.nonzero(mk)
            if len(ys) > 30:
                ymax = ys.max(); band = ys >= ymax - max(3, 0.05 * (ys.max() - ys.min()))
                return cx0 + xs[band].mean(), cy0 + ymax, 'gdino_sam'
            return cx0 + (trunk_box[0] + trunk_box[2]) / 2, cy0 + trunk_box[3], 'gdino_box'
    except Exception as e:
        return xc, y1, 'error'
    return xc, y1, 'box_bottom'
import urllib.request, concurrent.futures as _cf
os.makedirs('/kaggle/temp/pseudo', exist_ok=True)
def _get(pid):
    p = f'/kaggle/temp/pseudo/{pid}.jpg'
    for k in range(4):
        try:
            with urllib.request.urlopen(PSEUDO[pid]['u'], timeout=60) as f: open(p, 'wb').write(f.read()); return pid, p
        except Exception: time.sleep(3)
    return pid, None
with _cf.ThreadPoolExecutor(12) as _ex: IMGS = {k: v for k, v in _ex.map(_get, list(PSEUDO)) if v}
nps, meth, t1 = 0, collections.Counter(), time.time()
for pid, rec in PSEUDO.items():
    boxes = rec['b']
    if pid not in IMGS: continue
    img = Image.open(IMGS[pid]).convert('RGB'); W_, H_ = img.size; lines = []
    for (a, b, c, d) in boxes:
        u, v, mth = trunk_point(img, a * W_, b * H_, c * W_, d * H_); meth[mth] += 1
        vis = 2 if mth.startswith('gdino') else 0
        ku, kv = (u / W_, v / H_) if vis else (0.0, 0.0)
        if vis and not (0 <= ku <= 1 and 0 <= kv <= 1): ku, kv, vis = 0.0, 0.0, 0
        lines.append(f'0 {(a + c) / 2:.6f} {(b + d) / 2:.6f} {c - a:.6f} {d - b:.6f} {ku:.6f} {kv:.6f} {vis}')
    os.symlink(IMGS[pid], f'{D}/images/train/PS_{pid}.jpg'); open(f'{D}/labels/train/PS_{pid}.txt', 'w').write("\n".join(lines) + "\n"); nps += 1
PSEUDO_INFO = dict(n_listed=len(PSEUDO), n_used=nps, n_images_found=len(IMGS), methods=dict(meth), minutes=(time.time() - t1) / 60)
print('pseudo-labelled photos', PSEUDO_INFO, flush=True)
del gm, sm; torch.cuda.empty_cache()
'''
new = tmpl.replace('yaml = "kpt_shape', BLOCK + 'yaml = "kpt_shape', 1)
assert new != tmpl
new = new.replace("res = dict(config=CFG, seed=SEED, failed_downloads=fail)", "res = dict(config='C (base ' + CFG + ')', seed=SEED, failed_downloads=fail, pseudo=PSEUDO_INFO)", 1)
assert 'pseudo=PSEUDO_INFO' in new
text2 = text.replace(tmpl, new, 1)
text2 = text2.replace("name = f'hw{CFG.lower()}_s{SEED}'", "name = f'hwc_s{SEED}'", 1)
text2 = text2.replace("('__HW__', HW)):", "('__HW__', HW), ('__PSEUDO__', PSEUDO_PACKED), ('__TRUNK_HEAD__', TRUNK_HEAD)):", 1)
text2 = text2.replace('"kernel_sources": []', '"kernel_sources": KSRC', 1)
assert all(s in text2 for s in ("hwc_s{SEED}", "PSEUDO_PACKED", '"kernel_sources": KSRC'))
import pandas as _pd
_url = _pd.read_csv('hw_selftrain_candidates.csv', dtype={'id': str}).set_index('id').url
PSEUDO_PACKED = base64.b64encode(zlib.compress(json.dumps({k: dict(u=_url[k], b=v) for k, v in json.load(open(PJ)).items()}).encode(), 9)).decode()
KSRC = []
sys.argv = ['make_hw_train.py', BASE, SEED, ACCT]
exec(compile(text2, 'make_hw_train_c', 'exec'), dict(__name__='__main__', PSEUDO_PACKED=PSEUDO_PACKED, TRUNK_HEAD=TRUNK_HEAD, KSRC=KSRC))

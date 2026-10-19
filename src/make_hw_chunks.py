"""Kaggle CPU jobs that run a trained highway detector on highway photos fetched from their public thumbnail links
(amendment 7: chunks of <= 800 photos because of the job size limit; nothing else differs from a single run).
Usage: python make_hw_chunks.py PHOTOS_CSV TAG ACCOUNT TRAIN_KERNEL WEIGHTS_FILE SAVE_IMAGES CHUNK
  PHOTOS_CSV: columns id,url. TRAIN_KERNEL: slug of the training kernel on the SAME account (its output holds WEIGHTS_FILE).
  SAVE_IMAGES: 1 keeps the downloaded photos in the job output (needed for self-training), 0 does not.
Each job writes dets_<TAG>_<k>.csv: id,x0,y0,x1,y1,score,ku,kv,kconf,img_w,img_h (normalised; conf >= 0.05, 1024 px)."""
import json, os, sys, zlib, base64
import pandas as pd

PH, TAG, ACCT, TK, WF, SAVE, CH = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]), int(sys.argv[7])
P = pd.read_csv(PH, dtype={'id': str})
SRC = r'''import os, io, json, glob, time, zlib, base64, subprocess, urllib.request, concurrent.futures as cf
subprocess.run("pip install -q ultralytics", shell=True)
from ultralytics import YOLO
ITEMS = json.loads(zlib.decompress(base64.b64decode("__ITEMS__")))
TAG, SAVE = "__TAG__", __SAVE__
W = glob.glob('/kaggle/input/**/__WF__', recursive=True)[0]
D = '/kaggle/working/img' if SAVE else '/kaggle/temp/img'; os.makedirs(D, exist_ok=True)
def get(it):
    p = f"{D}/{it['id']}.jpg"
    for k in range(4):
        try:
            with urllib.request.urlopen(it['url'], timeout=60) as f: open(p, 'wb').write(f.read()); return p
        except Exception: time.sleep(3)
    return None
t0 = time.time()
with cf.ThreadPoolExecutor(12) as ex: paths = [p for p in ex.map(get, ITEMS) if p]
print('downloaded', len(paths), 'of', len(ITEMS), round(time.time() - t0), 's', flush=True)
import gc, torch, psutil
torch.set_num_threads(4)
m = YOLO(W); n = 0
with open(f'/kaggle/working/dets_{TAG}.csv', 'w') as f:
    f.write('id,x0,y0,x1,y1,score,ku,kv,kconf,img_w,img_h\n')
    for b0 in range(0, len(paths), 16):
        for r in m.predict(paths[b0:b0 + 16], imgsz=1024, conf=0.05, device='cpu', verbose=False, batch=4):
            H, Wd = r.orig_shape; pid = os.path.basename(r.path)[:-4]; n += 1
            k = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None
            for j, ((a, b, c, d), s) in enumerate(zip(r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy())):
                ku, kv, kc = (k[j, 0, 0] / Wd, k[j, 0, 1] / H, k[j, 0, 2]) if k is not None else (0, 0, 0)
                f.write(f"{pid},{a:.5f},{b:.5f},{c:.5f},{d:.5f},{s:.4f},{ku:.5f},{kv:.5f},{kc:.3f},{Wd},{H}\n")
        f.flush(); gc.collect()
        if b0 % 160 == 0: print(b0, len(paths), round((time.time() - t0) / 60, 1), 'min', 'RAM %', psutil.virtual_memory().percent, flush=True)
json.dump(dict(tag=TAG, weights=W, n_items=len(ITEMS), n_downloaded=len(paths), n_predicted=n, minutes=(time.time() - t0) / 60),
          open(f'/kaggle/working/info_{TAG}.json', 'w'))
print('done', n, round((time.time() - t0) / 60, 1), 'min', flush=True)
'''
made = []
for k, s in enumerate(range(0, len(P), CH)):
    part = P.iloc[s:s + CH]; tag = f'{TAG}_{k}'
    blob = base64.b64encode(zlib.compress(json.dumps(part[['id', 'url']].to_dict('records')).encode(), 9)).decode()
    code = SRC.replace('__ITEMS__', blob).replace('__TAG__', tag).replace('__SAVE__', str(SAVE)).replace('__WF__', WF)
    d = f'kjobs/{tag}'; os.makedirs(d, exist_ok=True); open(f'{d}/train.py', 'w').write(code)
    json.dump({"id": f"{ACCT}/rt-{tag.replace('_', '-')}", "title": f"rt-{tag.replace('_', '-')}", "code_file": "train.py",
               "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_internet": True,
               "dataset_sources": [], "competition_sources": [], "kernel_sources": [f"{ACCT}/{TK}"]}, open(f'{d}/kernel-metadata.json', 'w'))
    made.append((tag, len(part), os.path.getsize(f'{d}/train.py') // 1024))
print(len(made), 'jobs; photos', len(P), '; largest job KB', max(m[2] for m in made)); print(made[:3])

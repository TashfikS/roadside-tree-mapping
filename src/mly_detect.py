"""Download Mapillary thumbnails listed in a URL CSV and run the Bangladesh YOLO11m-pose model (CPU).
usage: python mly_detect.py <urls.csv> <img_dir> <weights.pt> <out.csv>"""
import os, sys, csv, time, concurrent.futures as cf, urllib.request
urls_csv, img_dir, weights, out_csv = sys.argv[1:5]
os.makedirs(img_dir, exist_ok=True)
rows = list(csv.DictReader(open(urls_csv)))
def get(r):
    p = f"{img_dir}/{r['id']}.jpg"
    if os.path.exists(p) and os.path.getsize(p) > 1000: return p
    for k in range(3):
        try:
            with urllib.request.urlopen(r['url1024'], timeout=60) as f: b = f.read()
            open(p + '.tmp', 'wb').write(b); os.replace(p + '.tmp', p); return p
        except Exception: time.sleep(2 + 3 * k)
    return None
t0 = time.time()
with cf.ThreadPoolExecutor(16) as ex: paths = list(ex.map(get, rows))
ok = [(r['id'], p) for r, p in zip(rows, paths) if p]
print('downloaded', len(ok), 'of', len(rows), round((time.time() - t0) / 60, 1), 'min', flush=True)
import torch; torch.set_num_threads(4)
from ultralytics import YOLO
m = YOLO(weights)
done = set()
if os.path.exists(out_csv): done = {l.split(',')[0] for l in open(out_csv).read().splitlines()[1:]}
fo = open(out_csv, 'a')
if not done: fo.write('id,x0,y0,x1,y1,score,ku,kv,kconf,img_w,img_h\n')
todo = [(i, p) for i, p in ok if i not in done]
for b in range(0, len(todo), 16):
    chunk = todo[b:b + 16]
    for (i, p), r in zip(chunk, m.predict([p for _, p in chunk], imgsz=1024, conf=0.25, device='cpu', verbose=False)):
        h, w = r.orig_shape
        if r.boxes is None or len(r.boxes) == 0: fo.write(f'{i},,,,,,,,,{w},{h}\n'); continue
        kp = r.keypoints.xyn.cpu().numpy(); kc = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
        for j, ((a, bb, c, d), s) in enumerate(zip(r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy())):
            fo.write(f'{i},{a:.5f},{bb:.5f},{c:.5f},{d:.5f},{s:.4f},{kp[j,0,0]:.5f},{kp[j,0,1]:.5f},{(kc[j,0] if kc is not None else 1):.3f},{w},{h}\n')
    fo.flush()
    if b % 800 == 0: print('det', b, len(todo), round((time.time() - t0) / 60, 1), 'min', flush=True)
print('done', round((time.time() - t0) / 60, 1), 'min', flush=True)

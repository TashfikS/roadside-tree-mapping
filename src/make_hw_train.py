"""Build a Kaggle job for the highway detector (hw_prereg.md): main-model recipe (YOLO11m-pose, 1024 px, 120 epochs,
patience 15, batch 8) on Kaggle train + highway train; validation = Kaggle val + highway val.
Usage: python make_hw_train.py CONFIG SEED ACCOUNT        CONFIG in {A, B}  (B = windscreen-style augmentation)
Inputs: hw_train_labels.json (photo id -> {split, url, lines, mask}), kjobs/pose_s1/train.py (Kaggle split + labels).
Outputs of the job: best weights, results JSON (highway-val box AP50 = the selection metric), Kaggle test predictions,
highway-val predictions. Highway TEST photos are not touched by these jobs.
"""
import json, os, re, sys

CFG, SEED, ACCT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
name = f'hw{CFG.lower()}_s{SEED}'
base = open('kjobs/pose_s1/train.py').read()
import ast, zlib, base64
def _lit(pat):                                       # evaluate the literal exactly as Python did in the original job
    m = re.search(pat, base, re.S); pre, body = m.group(1), m.group(2)
    return ast.literal_eval(pre + '"""' + body + '"""')
def _pack(text): return base64.b64encode(zlib.compress(text.encode(), 9)).decode()
split_csv = _pack(_lit(r'SPLIT_CSV = (r?)"""(.*?)"""'))
labels = _pack(_lit(r'LABELS = json\.loads\((r?)"""(.*?)"""\)'))
HW = _pack(open('hw_train_labels.json').read())

src = r'''import os, glob, json, csv, io, time, subprocess, math, random, urllib.request
subprocess.run("pip install -q ultralytics", shell=True)
import numpy as np, cv2
from PIL import Image
from ultralytics import YOLO
CFG, SEED, NAME = "__CFG__", __SEED__, "__NAME__"
import zlib, base64
SPLIT_CSV = zlib.decompress(base64.b64decode("__SPLIT__")).decode()
LABELS = json.loads(zlib.decompress(base64.b64decode("__LABELS__")))
HW = json.loads(zlib.decompress(base64.b64decode("__HW__")))
R = glob.glob('/kaggle/input/**/roadside-trees-of-bangladesh-street-view', recursive=True)[0]
src = {'BD_740': f'{R}/Train/BD_740', 'BD_755': f'{R}/Train/BD_755', 'BD_Test_Set_270': f'{R}/Test/BD_Test_Set_270'}
D = '/kaggle/temp/pose'
for s in ['train', 'val', 'test', 'hwval']: os.makedirs(f'{D}/images/{s}', exist_ok=True); os.makedirs(f'{D}/labels/{s}', exist_ok=True)
for r in csv.DictReader(io.StringIO(SPLIT_CSV)):
    st = f"{r['folder']}_{r['id']}"; os.symlink(f"{src[r['folder']]}/{r['id']}.jpg", f"{D}/images/{r['split']}/{st}.jpg")
    open(f"{D}/labels/{r['split']}/{st}.txt", 'w').write("\n".join(LABELS.get(st, [])) + "\n")
# highway photos: download, grey out ignored group-box regions, write labels (box + trunk point)
fail = []
for pid, h in HW.items():
    im = None
    for k in range(4):
        try:
            with urllib.request.urlopen(h['url'], timeout=60) as f: im = np.array(Image.open(io.BytesIO(f.read())).convert('RGB')); break
        except Exception: time.sleep(3)
    if im is None: fail.append(pid); continue
    H_, W_ = im.shape[:2]
    for (a, b, c, d) in h.get('mask', []):
        keep = im.copy(); x0, y0, x1, y1 = int(a * W_), int(b * H_), int(c * W_), int(d * H_); im[y0:y1, x0:x1] = 114
        for ln in h['lines']:
            _, xc, yc, w, hh = map(float, ln.split()[:5]); p0, q0, p1, q1 = int((xc - w / 2) * W_), int((yc - hh / 2) * H_), int((xc + w / 2) * W_), int((yc + hh / 2) * H_)
            im[q0:q1, p0:p1] = keep[q0:q1, p0:p1]
    dirs = ['train'] if h['split'] == 'train' else ['val', 'hwval']
    for sd in dirs:
        Image.fromarray(im).save(f'{D}/images/{sd}/HW_{pid}.jpg', quality=95)
        open(f'{D}/labels/{sd}/HW_{pid}.txt', 'w').write("\n".join(h['lines']) + "\n")
print('highway photos', len(HW), 'failed downloads', len(fail), flush=True)
yaml = "kpt_shape: [1, 3]\nflip_idx: [0]\nnames:\n  0: tree\n"
open(f'{D}/data.yaml', 'w').write(f"path: {D}\ntrain: images/train\nval: images/val\ntest: images/test\n" + yaml)
open(f'{D}/hwval.yaml', 'w').write(f"path: {D}\ntrain: images/train\nval: images/hwval\n" + yaml)

# config B: windscreen-style augmentation on training samples only (applied before the colour jitter)
def windscreen(im, labels):
    h, w = im.shape[:2]; out = im.astype(np.float32)
    fx = [f for f in ('band', 'contrast', 'glare', 'rain', 'motion') if random.random() < 0.35] or [random.choice(['band', 'contrast', 'glare', 'rain', 'motion'])]
    ins = labels.get('instances'); kp = getattr(ins, 'keypoints', None); norm = bool(getattr(ins, 'normalized', False))
    if 'band' in fx:
        dark = random.uniform(5, 40)
        if random.random() < 0.7:
            y0 = int(h * (1 - random.uniform(0.05, 0.2))); out[y0:] = out[y0:] * 0.15 + dark * 0.85
            if kp is not None and len(kp): kp[..., 2][kp[..., 1] * (h if norm else 1) >= y0] = 0
        else:
            y1 = int(h * random.uniform(0.05, 0.15)); out[:y1] = out[:y1] * 0.15 + dark * 0.85
            if kp is not None and len(kp): kp[..., 2][kp[..., 1] * (h if norm else 1) <= y1] = 0
    if 'contrast' in fx:
        m = out.mean(); out = m + (out - m) * random.uniform(0.4, 0.8); hz = random.uniform(0, 0.3); out = out * (1 - hz) + hz * random.uniform(180, 230)
    if 'glare' in fx:
        yy, xx = np.ogrid[:h, :w]
        for _ in range(random.randint(1, 2)):
            cx, cy, r = random.uniform(0, w), random.uniform(0, 0.6 * h), random.uniform(0.08, 0.3) * max(h, w)
            g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (r / 2) ** 2))[..., None]; a = random.uniform(0.4, 0.9); out = out * (1 - a * g) + 255 * a * g
    if 'rain' in fx:
        lay = np.zeros((h, w), np.float32)
        for _ in range(random.randint(30, 200)): cv2.circle(lay, (random.randint(0, w - 1), random.randint(0, h - 1)), random.randint(2, max(3, int(0.012 * w))), 1.0, -1)
        for _ in range(random.randint(0, 80)):
            x, y = random.randint(0, w - 1), random.randint(0, h - 1); cv2.line(lay, (x, y), (x + random.randint(-6, 6), y + random.randint(10, 40)), 0.6, 1)
        lay = cv2.GaussianBlur(lay, (0, 0), random.uniform(1.0, 2.5))[..., None]; bl = cv2.GaussianBlur(out, (0, 0), 3.0)
        out = out * (1 - lay) + (0.6 * bl + 0.4 * 200) * lay
    if 'motion' in fx:
        k = random.choice([5, 7, 9, 11, 15]); K = np.zeros((k, k), np.float32); K[k // 2, :] = 1
        K = cv2.warpAffine(K, cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), random.uniform(0, 180), 1), (k, k)); K /= K.sum() + 1e-6
        out = cv2.filter2D(out, -1, K)
    return np.clip(out, 0, 255).astype(np.uint8)
if CFG == 'B':
    from ultralytics.data import augment as _AU
    _orig = _AU.RandomHSV.__call__
    def _hsv(self, labels):
        if random.random() < 0.5:
            try: labels['img'] = windscreen(labels['img'], labels)
            except Exception: pass
        return _orig(self, labels)
    _AU.RandomHSV.__call__ = _hsv

t0 = time.time(); m = YOLO('yolo11m-pose.pt')
m.train(data=f'{D}/data.yaml', imgsz=1024, epochs=120, patience=15, batch=8, seed=SEED, deterministic=True, fliplr=0.5, device=0, workers=4,
        project='/kaggle/working/runs', name=NAME, exist_ok=True, plots=True)
best = YOLO(f'/kaggle/working/runs/{NAME}/weights/best.pt'); res = dict(config=CFG, seed=SEED, failed_downloads=fail)
v = best.val(data=f'{D}/hwval.yaml', split='val', imgsz=1024, batch=8, device=0, project='/kaggle/working/runs', name='eval_hwval', exist_ok=True)
res['hwval'] = dict(box_AP50=float(v.box.map50), box_AP50_95=float(v.box.map), pose_AP50=float(v.pose.map50))
for sp in ['val', 'test']:
    v = best.val(data=f'{D}/data.yaml', split=sp, imgsz=1024, batch=8, device=0, project='/kaggle/working/runs', name=f'eval_{sp}', exist_ok=True)
    res[sp] = dict(box_mAP50=float(v.box.map50), box_mAP50_95=float(v.box.map), pose_mAP50=float(v.pose.map50), pose_mAP50_95=float(v.pose.map))
for sd, fn in (('test', f'test_preds_{NAME}.csv'), ('hwval', f'hwval_preds_{NAME}.csv')):
    with open(f'/kaggle/working/{fn}', 'w') as f:
        f.write('image,x0,y0,x1,y1,score,ku,kv,kconf,img_w,img_h\n')
        for r in best.predict(f'{D}/images/{sd}', imgsz=1024, conf=0.001, device=0, verbose=False, stream=True):
            H, W = r.orig_shape
            for (a, b, c, d), s, k in zip(r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.keypoints.data.cpu().numpy()):
                f.write(f"{os.path.basename(r.path)},{a:.5f},{b:.5f},{c:.5f},{d:.5f},{s:.4f},{k[0,0]/W:.5f},{k[0,1]/H:.5f},{k[0,2]:.3f},{W},{H}\n")
import shutil; shutil.copy(f'/kaggle/working/runs/{NAME}/weights/best.pt', f'/kaggle/working/{NAME}_best.pt')
res['train_minutes'] = (time.time() - t0) / 60; json.dump(res, open(f'/kaggle/working/results_{NAME}.json', 'w'), indent=1); print(json.dumps(res, indent=1))
'''
for a, b in (('__CFG__', CFG), ('__SEED__', str(SEED)), ('__NAME__', name), ('__SPLIT__', split_csv), ('__LABELS__', labels), ('__HW__', HW)):
    assert src.count(a) >= 1, a
    src = src.replace(a, b)
os.makedirs(f'kjobs/{name}', exist_ok=True)
open(f'kjobs/{name}/train.py', 'w').write(src)
json.dump({"id": f"{ACCT}/rt-{name.replace('_', '-')}", "title": f"rt-{name.replace('_', '-')}", "code_file": "train.py", "language": "python",
           "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
           "dataset_sources": ["tashfiksalehin/roadside-trees-of-bangladesh-street-view"], "competition_sources": [], "kernel_sources": []},
          open(f'kjobs/{name}/kernel-metadata.json', 'w'))
print(name, os.path.getsize(f'kjobs/{name}/train.py'))

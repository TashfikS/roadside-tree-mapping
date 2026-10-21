"""Test P (protocols/trunk_ablation_prereg.md): trunk-base keypoint detector for Pasadena panoramas.
Stage 1: automatic trunk-base labels (Grounding DINO + SAM, same method as the Bangladesh labels) for every box of the
student's training set (pas_teacher pseudo_labels.json) and of the human-labelled validation panoramas.
Stage 2: YOLO11m-pose with the student's recipe. Stage 3: raw detections with keypoints on all val + test panoramas.
Attach: dataset pasadena-urban-trees, kernel output pas-teacher. Outputs: pas_trunk_labels.json, pas_pose_raw_dets.csv,
pas_pose_best.pt, pas_pose_info.json.
"""
import os, glob, json, time, shutil, subprocess, collections
subprocess.run("pip install -q ultralytics", shell=True)
subprocess.run("pip install -q -U transformers accelerate", shell=True)
import numpy as np, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor

ROOT = glob.glob('/kaggle/input/**/pasadena-urban-trees', recursive=True)[0]
SV = ROOT + '/streetview'
panos = json.load(open(ROOT + '/panos.json')); rois = json.load(open(ROOT + '/rois.json'))
PL = json.load(open(glob.glob('/kaggle/input/**/pseudo_labels.json', recursive=True)[0]))
HUMAN = json.loads(r'''__HUMAN__''')
W_OUT = '/kaggle/working'


def tvt(la, lo):
    for r in rois:
        a, b, c, d = r['roi']
        if min(a, c) <= la <= max(a, c) and min(b, d) <= lo <= max(b, d): return r['train_val_test']


P = {}
for k, v in panos.items():
    f = f'{SV}/{k}_z2.jpg'
    if os.path.exists(f): P[k] = dict(f=f, tvt=tvt(float(v['Location']['original_lat']), float(v['Location']['original_lng'])))
SETS = {'train': {pid: L for pid, L in PL.items() if pid in P and P[pid]['tvt'] == 0},
        'val': {pid: h['labels'] for pid, h in HUMAN.items() if h['tvt'] == 1 and pid in P}}

# ---------------- stage 1: automatic trunk-base labels (identical read-out to kaggle_jobs/templates/trunk_labels.py)
dev = 'cuda'
gp = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
gm = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(dev).eval()
sp = SamProcessor.from_pretrained("facebook/sam-vit-base")
sm = SamModel.from_pretrained("facebook/sam-vit-base").to(dev).eval()


def gdino(img, text="a tree trunk."):
    inp = gp(images=img, text=text, return_tensors="pt").to(dev)
    with torch.no_grad(): out = gm(**inp)
    kw = dict(target_sizes=[img.size[::-1]], text_threshold=0.2)
    try: r = gp.post_process_grounded_object_detection(out, inp.input_ids, threshold=0.2, **kw)[0]
    except TypeError: r = gp.post_process_grounded_object_detection(out, inp.input_ids, box_threshold=0.2, **kw)[0]
    return r['boxes'].cpu().numpy(), r['scores'].cpu().numpy()


def sam_mask(img, box):
    inp = sp(img, input_boxes=[[list(map(float, box))]], return_tensors="pt").to(dev)
    with torch.no_grad(): o = sm(**inp, multimask_output=False)
    m = sp.image_processor.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0]
    return m[0, 0].numpy().astype(bool)


def trunk_point(img, xc, yc, w, h):
    W, H = img.size
    x0, x1 = (xc - w / 2) * W, (xc + w / 2) * W; y0, y1 = (yc - h / 2) * H, (yc + h / 2) * H
    cx0 = max(0, x0 - 0.1 * (x1 - x0)); cx1 = min(W, x1 + 0.1 * (x1 - x0))
    cy0 = max(0, y0 + 0.3 * (y1 - y0)); cy1 = min(H, y1 + 0.15 * (y1 - y0))
    if cx1 - cx0 < 4 or cy1 - cy0 < 4: return xc, y1 / H, 'box_bottom'
    crop = img.crop((int(cx0), int(cy0), int(cx1), int(cy1)))
    try:
        tb, ts = gdino(crop)
        if len(ts):
            ok = ts >= 0.8 * ts.max(); k = np.argmax(np.where(ok, tb[:, 3], -1)); trunk_box = tb[k]
            m = sam_mask(crop, trunk_box); ys, xs = np.nonzero(m)
            if len(ys) > 30:
                ymax = ys.max(); band = ys >= ymax - max(3, 0.05 * (ys.max() - ys.min()))
                return (cx0 + xs[band].mean()) / W, (cy0 + ymax) / H, 'gdino_sam'
            return (cx0 + (trunk_box[0] + trunk_box[2]) / 2) / W, (cy0 + trunk_box[3]) / H, 'gdino_box'
    except Exception:
        pass
    return xc, y1 / H, 'box_bottom'


t0 = time.time(); KL = {}; meth = collections.Counter(); n = 0
for s, D_ in SETS.items():
    for pid, L in D_.items():
        lines = []
        if L:
            img = Image.open(P[pid]['f']).convert('RGB')
            for l in L:
                _, xc, yc, w, h = map(float, l.split()[:5])
                u, v, mth = trunk_point(img, xc, yc, w, h); meth[mth] += 1; n += 1
                u = min(max(u, 0.0), 1.0); v = min(max(v, 0.0), 1.0)
                lines.append(f'0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {u:.6f} {v:.6f} 2')
                if n % 1000 == 0: print('labels', n, round((time.time() - t0) / 60, 1), dict(meth), flush=True)
        KL[pid] = dict(set=s, lines=lines)
json.dump(KL, open(f'{W_OUT}/pas_trunk_labels.json', 'w'))
label_minutes = (time.time() - t0) / 60
print('labels done', n, dict(meth), round(label_minutes, 1), flush=True)
del gm, sm; torch.cuda.empty_cache()

# ---------------- stage 2: pose detector, student recipe
from ultralytics import YOLO
D = '/kaggle/temp/pose'
for s in ('train', 'val'):
    os.makedirs(f'{D}/images/{s}', exist_ok=True); os.makedirs(f'{D}/labels/{s}', exist_ok=True)
for pid, r in KL.items():
    os.symlink(P[pid]['f'], f'{D}/images/{r["set"]}/{pid}.jpg')
    open(f'{D}/labels/{r["set"]}/{pid}.txt', 'w').write('\n'.join(r['lines']) + ('\n' if r['lines'] else ''))
open(f'{D}/data.yaml', 'w').write(f'path: {D}\ntrain: images/train\nval: images/val\nkpt_shape: [1, 3]\nflip_idx: [0]\nnames:\n  0: tree\n')
t1 = time.time()
YOLO('yolo11m-pose.pt').train(data=f'{D}/data.yaml', imgsz=1280, epochs=40, patience=10, batch=8, seed=0, deterministic=True, fliplr=0.5,
                              device=0, workers=4, time=8.0, project=f'{W_OUT}/runs', name='pas_pose', exist_ok=True, plots=False)
best = f'{W_OUT}/runs/pas_pose/weights/best.pt'; shutil.copy(best, f'{W_OUT}/pas_pose_best.pt')
train_minutes = (time.time() - t1) / 60

# ---------------- stage 3: raw detections with keypoints on val + test panoramas
m = YOLO(best)
ids = [k for k, p in P.items() if p['tvt'] in (1, 2)]
with open(f'{W_OUT}/pas_pose_raw_dets.csv', 'w') as fo:
    fo.write('pano,x0,y0,x1,y1,score,ku,kv,kconf\n')
    for i in range(0, len(ids), 32):
        res = m.predict([P[k]['f'] for k in ids[i:i + 32]], imgsz=1280, conf=0.05, iou=0.5, device=0, verbose=False)
        for k, r in zip(ids[i:i + 32], res):
            sx, sy = 1664 / r.orig_shape[1], 832 / r.orig_shape[0]
            kp = r.keypoints.data.cpu().numpy() if r.keypoints is not None else np.zeros((0, 1, 3))
            for (a, b, c, d), s, q in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), kp):
                fo.write(f'{k},{a*sx:.1f},{b*sy:.1f},{c*sx:.1f},{d*sy:.1f},{s:.4f},{q[0,0]*sx:.1f},{q[0,1]*sy:.1f},{q[0,2]:.4f}\n')
json.dump(dict(label_minutes=label_minutes, train_minutes=train_minutes, n_label_boxes=n, label_methods=dict(meth),
               n_train=sum(r['set'] == 'train' for r in KL.values()), n_val=sum(r['set'] == 'val' for r in KL.values()), n_pred_panos=len(ids)),
          open(f'{W_OUT}/pas_pose_info.json', 'w'), indent=1)
print('done', flush=True)

"""Pasadena stage B: student detector on inventory-verified pseudo-labels (train ROI) -> raw detections on val + test ROIs.
Validation for model selection = human-labelled panoramas in the val ROI. Test ROI images are used only for the final prediction.
"""
import os, glob, json, math, subprocess, time, shutil
subprocess.run("pip install -q ultralytics", shell=True)
from ultralytics import YOLO

ROOT = glob.glob('/kaggle/input/**/pasadena-urban-trees', recursive=True)[0]
SV = ROOT + '/streetview'
panos = json.load(open(ROOT + '/panos.json')); rois = json.load(open(ROOT + '/rois.json'))
PL = json.load(open(glob.glob('/kaggle/input/**/pseudo_labels.json', recursive=True)[0]))
HUMAN = json.loads(r'''__HUMAN__''')


def tvt(la, lo):
    for r in rois:
        a, b, c, d = r['roi']
        if min(a, c) <= la <= max(a, c) and min(b, d) <= lo <= max(b, d): return r['train_val_test']


P = {}
for k, v in panos.items():
    f = f'{SV}/{k}_z2.jpg'
    if os.path.exists(f): P[k] = dict(f=f, tvt=tvt(float(v['Location']['original_lat']), float(v['Location']['original_lng'])))
D = '/kaggle/temp/student'
for s in ('train', 'val'):
    os.makedirs(f'{D}/images/{s}', exist_ok=True); os.makedirs(f'{D}/labels/{s}', exist_ok=True)
for pid, L in PL.items():
    if pid in P and P[pid]['tvt'] == 0:
        os.symlink(P[pid]['f'], f'{D}/images/train/{pid}.jpg'); open(f'{D}/labels/train/{pid}.txt', 'w').write('\n'.join(L) + ('\n' if L else ''))
for pid, h in HUMAN.items():
    if h['tvt'] == 1 and pid in P:
        os.symlink(P[pid]['f'], f'{D}/images/val/{pid}.jpg'); open(f'{D}/labels/val/{pid}.txt', 'w').write('\n'.join(h['labels']) + '\n')
open(f'{D}/data.yaml', 'w').write(f'path: {D}\ntrain: images/train\nval: images/val\nnames:\n  0: tree\n')
print('train', len(os.listdir(f'{D}/images/train')), 'val', len(os.listdir(f'{D}/images/val')), flush=True)
t0 = time.time()
YOLO('yolo11m.pt').train(data=f'{D}/data.yaml', imgsz=1280, epochs=40, patience=10, batch=8, seed=0, deterministic=True, fliplr=0.5,
                         device=0, workers=4, time=8.0, project='/kaggle/working/runs', name='student', exist_ok=True, plots=False)
best = '/kaggle/working/runs/student/weights/best.pt'; shutil.copy(best, '/kaggle/working/student_best.pt')
print('train minutes', (time.time() - t0) / 60, flush=True)
m = YOLO(best)
ids = [k for k, p in P.items() if p['tvt'] in (1, 2)]
with open('/kaggle/working/student_raw_dets.csv', 'w') as fo:
    fo.write('pano,x0,y0,x1,y1,score\n')
    for i in range(0, len(ids), 32):
        res = m.predict([P[k]['f'] for k in ids[i:i + 32]], imgsz=1280, conf=0.05, iou=0.5, device=0, verbose=False)
        for k, r in zip(ids[i:i + 32], res):
            sx, sy = 1664 / r.orig_shape[1], 832 / r.orig_shape[0]
            for (a, b, c, d), s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                fo.write(f'{k},{a*sx:.1f},{b*sy:.1f},{c*sx:.1f},{d*sy:.1f},{s:.4f}\n')
json.dump(dict(train_minutes=(time.time() - t0) / 60, n_pred_panos=len(ids)), open('/kaggle/working/student_info.json', 'w'))
print('done', flush=True)

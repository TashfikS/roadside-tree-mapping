"""E7 Pasadena: Depth Anything V2 metric (outdoor) ranges for every retrained-detector box (rules: e7_prereg.md).
Output /kaggle/working/e7_pas_depth.csv: idx, view, px, py, z_point, r_point, r_region (metres). No ground truth here."""
import os, glob, json, zlib, base64, time, subprocess, math, collections
subprocess.run("pip install -q -U transformers accelerate", shell=True)
import numpy as np, torch, cv2
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
ROWS = json.loads(zlib.decompress(base64.b64decode("__ROWS__")))
ROOT = glob.glob('/kaggle/input/**/pasadena-urban-trees', recursive=True)[0]; SV = ROOT + '/streetview'
W, H, S = 1664, 832, 518; F = (S / 2) / math.tan(math.radians(45)); C = (S - 1) / 2; NV = 8
used = None
for mid in ('depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf', 'depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf',
            'depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf'):
    try:
        proc = AutoImageProcessor.from_pretrained(mid); model = AutoModelForDepthEstimation.from_pretrained(mid).to('cuda').half().eval(); used = mid; break
    except Exception as e: print('LOAD FAIL', mid, repr(e)[:200], flush=True)
assert used
mean = torch.tensor(proc.image_mean, device='cuda').view(1, 3, 1, 1).half(); std = torch.tensor(proc.image_std, device='cuda').view(1, 3, 1, 1).half()
# remap grid for a view centred on column 0 (pitch 0)
px, py = np.meshgrid(np.arange(S, dtype=np.float32), np.arange(S, dtype=np.float32))
dx, dy = (px - C) / F, (py - C) / F
az = np.arctan2(dx, 1.0); el = -np.arctan2(dy, np.hypot(dx, 1.0))
MX0 = (az / (2 * np.pi) * W).astype(np.float32); MY = (H / 2 - el / np.pi * H).astype(np.float32)
def view(img, j):
    mx = ((MX0 + j * W / NV) % W).astype(np.float32)
    return cv2.remap(img, mx, MY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
def to_view(u, v, j):
    daz = ((u - j * W / NV + W / 2) % W - W / 2) / W * 2 * np.pi; e = (H / 2 - v) / H * np.pi
    return C + F * np.tan(daz), C - F * np.tan(e) / np.cos(daz)
@torch.no_grad()
def depth(batch):
    x = torch.from_numpy(np.stack(batch)).to('cuda').permute(0, 3, 1, 2).half() / 255.0
    x = (x - mean) / std
    d = model(pixel_values=x).predicted_depth
    if d.shape[-2:] != (S, S): d = torch.nn.functional.interpolate(d.unsqueeze(1).float(), size=(S, S), mode='bilinear', align_corners=False)[:, 0]
    return d.float().cpu().numpy()
by = collections.defaultdict(list)
for i, (p, x0, y0, x1, y1) in enumerate(ROWS): by[p].append(i)
t0 = time.time(); out = open('/kaggle/working/e7_pas_depth.csv', 'w'); out.write('idx,view,px,py,z_point,r_point,r_region\n')
jobs = []
def flush():
    if not jobs: return
    D = depth([v for v, _ in jobs])
    for dm, (_, items) in zip(D, jobs):
        for i, j in items:
            p, x0, y0, x1, y1 = ROWS[i]; u = (x0 + x1) / 2
            qx, qy = to_view(u, y1, j); zp = rp = rr = float('nan')
            if 2 <= qx <= S - 3 and 2 <= qy <= S - 3:
                a, b = int(round(qx)), int(round(qy)); zp = float(np.median(dm[b - 2:b + 3, a - 2:a + 3])); rp = zp * math.sqrt(1 + ((qx - C) / F) ** 2)
            us = np.linspace(u - 0.15 * (x1 - x0), u + 0.15 * (x1 - x0), 7); vs = np.linspace(y1 - 0.15 * (y1 - y0), y1, 4)
            U, V = np.meshgrid(us, vs); QX, QY = to_view(U.ravel(), V.ravel(), j)
            ok = (QX >= 0) & (QX <= S - 1) & (QY >= 0) & (QY <= S - 1)
            if ok.sum() >= 4:
                z = dm[np.round(QY[ok]).astype(int), np.round(QX[ok]).astype(int)]; rr = float(np.median(z * np.sqrt(1 + ((QX[ok] - C) / F) ** 2)))
            out.write(f"{i},{j},{qx:.1f},{qy:.1f},{zp:.3f},{rp:.3f},{rr:.3f}\n")
    jobs.clear()
missing = 0
for n, (p, idx) in enumerate(by.items()):
    f = f'{SV}/{p}_z2.jpg'
    if not os.path.exists(f): missing += len(idx); continue
    img = np.asarray(Image.open(f).convert('RGB').resize((W, H)))
    need = collections.defaultdict(list)
    for i in idx:
        u = (ROWS[i][1] + ROWS[i][3]) / 2; need[int(round(u / (W / NV))) % NV].append(i)
    for j, items in need.items():
        jobs.append((view(img, j), [(i, j) for i in items]))
        if len(jobs) >= 16: flush()
    if n % 500 == 0: print(n, len(by), round((time.time() - t0) / 60, 1), 'min', flush=True); out.flush()
flush(); out.close()
json.dump(dict(model=used, n_boxes=len(ROWS), n_panos=len(by), missing_boxes=missing, minutes=(time.time() - t0) / 60), open('/kaggle/working/e7_info.json', 'w'), indent=1)
print('done', used, missing, round((time.time() - t0) / 60, 1), flush=True)

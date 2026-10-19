import os, glob, csv, json, time, random, subprocess
subprocess.run("pip install -q -U transformers accelerate", shell=True)
import numpy as np, torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor
dev = 'cuda'
R = glob.glob('/kaggle/input/**/roadside-trees-of-bangladesh-street-view', recursive=True)[0]
src = {'BD_740': f'{R}/Train/BD_740', 'BD_755': f'{R}/Train/BD_755', 'BD_Test_Set_270': f'{R}/Test/BD_Test_Set_270'}
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
rows = []; viz = []; t0 = time.time(); n = 0
os.makedirs('/kaggle/working/viz', exist_ok=True)
for folder, d in src.items():
    for jp in sorted(glob.glob(d + '/*.jpg'), key=lambda p: int(os.path.basename(p)[:-4])):
        iid = int(os.path.basename(jp)[:-4]); img = Image.open(jp).convert('RGB'); W, H = img.size
        boxes = [list(map(float, l.split()[1:5])) for l in open(jp[:-4] + '.txt') if l.strip()]
        for b, (xc, yc, w, h) in enumerate(boxes):
            n += 1
            x0, x1 = (xc - w / 2) * W, (xc + w / 2) * W; y0, y1 = (yc - h / 2) * H, (yc + h / 2) * H
            cx0 = max(0, x0 - 0.1 * (x1 - x0)); cx1 = min(W, x1 + 0.1 * (x1 - x0))
            cy0 = max(0, y0 + 0.3 * (y1 - y0)); cy1 = min(H, y1 + 0.15 * (y1 - y0))
            crop = img.crop((int(cx0), int(cy0), int(cx1), int(cy1)))
            u = v = None; method = 'box_bottom'; score = 0.0; trunk_box = None
            try:
                tb, ts = gdino(crop)
                if len(ts):
                    ok = ts >= 0.8 * ts.max()
                    k = np.argmax(np.where(ok, tb[:, 3], -1))
                    trunk_box = tb[k]; score = float(ts[k])
                    m = sam_mask(crop, trunk_box)
                    ys, xs = np.nonzero(m)
                    if len(ys) > 30:
                        ymax = ys.max(); band = ys >= ymax - max(3, 0.05 * (ys.max() - ys.min()))
                        u = cx0 + xs[band].mean(); v = cy0 + ymax; method = 'gdino_sam'
                    else:
                        u = cx0 + (trunk_box[0] + trunk_box[2]) / 2; v = cy0 + trunk_box[3]; method = 'gdino_box'
            except Exception as e:
                method = 'error:' + str(e)[:60]
            if u is None: u, v = xc * W, y1
            rows.append(dict(folder=folder, id=iid, box_index=b, xc=xc, yc=yc, w=w, h=h, trunk_u=round(u / W, 6), trunk_v=round(v / H, 6),
                             score=round(score, 3), method=method, width=W, height=H))
            if random.Random(n).random() < 0.012 and len(viz) < 60:
                vi = img.copy(); dr = ImageDraw.Draw(vi)
                dr.rectangle([x0, y0, x1, y1], outline=(255, 0, 255), width=4)
                if trunk_box is not None: dr.rectangle([cx0 + trunk_box[0], cy0 + trunk_box[1], cx0 + trunk_box[2], cy0 + trunk_box[3]], outline=(0, 200, 255), width=3)
                dr.ellipse([u - 12, v - 12, u + 12, v + 12], outline=(0, 255, 0), width=5)
                vi.crop((int(cx0), int(max(0, y0)), int(cx1), int(cy1))).save(f'/kaggle/working/viz/{folder}_{iid}_{b}_{method}.jpg', quality=80)
                viz.append(1)
        if n % 500 < len(boxes): print(n, round(time.time() - t0), flush=True)
with open('/kaggle/working/trunk_auto.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
import collections
print(json.dumps(dict(n=len(rows), methods=collections.Counter(r['method'].split(':')[0] for r in rows), minutes=(time.time() - t0) / 60)))

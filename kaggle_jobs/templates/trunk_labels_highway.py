import os, glob, csv, json, time, random, subprocess
subprocess.run("pip install -q -U transformers accelerate", shell=True)
import numpy as np, torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor
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
import io, urllib.request, collections
ITEMS = json.loads(r"__ITEMS__")
def fetch(u):
    for k in range(4):
        try:
            with urllib.request.urlopen(u, timeout=60) as f: return Image.open(io.BytesIO(f.read())).convert('RGB')
        except Exception: time.sleep(3)
rows = []; t0 = time.time(); cache = {}
os.makedirs('/kaggle/working/viz', exist_ok=True)
for n, it in enumerate(ITEMS):
    if it['id'] not in cache: cache = {it['id']: fetch(it['url'])}
    img = cache[it['id']]
    if img is None: rows.append(dict(bid=it['bid'], key=it['key'], id=it['id'], trunk_u='', trunk_v='', score=0, method='fetch_failed', width='', height='')); continue
    W, H = img.size
    # identical crop and thresholds to the Kaggle training labels (trunk_auto)
    x0, x1, y0, y1 = it['x0'] * W, it['x1'] * W, it['y0'] * H, it['y1'] * H; xc = (x0 + x1) / 2
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
    if u is None: u, v = xc, y1
    rows.append(dict(bid=it['bid'], key=it['key'], id=it['id'], trunk_u=round(u / W, 6), trunk_v=round(v / H, 6), score=round(score, 3), method=method, width=W, height=H))
    if n % 12 == 0 and n < 600:
        vi = img.copy(); dr = ImageDraw.Draw(vi); dr.rectangle([x0, y0, x1, y1], outline=(255, 0, 255), width=3)
        if trunk_box is not None: dr.rectangle([cx0 + trunk_box[0], cy0 + trunk_box[1], cx0 + trunk_box[2], cy0 + trunk_box[3]], outline=(0, 200, 255), width=2)
        dr.ellipse([u - 8, v - 8, u + 8, v + 8], outline=(0, 255, 0), width=4); vi.save(f"/kaggle/working/viz/{it['bid']}_{method}.jpg", quality=80)
with open('/kaggle/working/hw_trunk_auto.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(json.dumps(dict(n=len(rows), methods=collections.Counter(r['method'].split(':')[0] for r in rows), minutes=(time.time() - t0) / 60)))

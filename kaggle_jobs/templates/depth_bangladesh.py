"""E7 Bangladesh: Depth Anything V2 metric (outdoor) z-depth at the clicked trunk bases (rules: e7_prereg.md). CPU."""
import io, json, base64, subprocess, time
subprocess.run("pip install -q -U transformers", shell=True)
import numpy as np, torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
PH = json.loads(r"""__PH__"""); IT = json.loads(r"""__IT__""")
used = None
for mid in ('depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf', 'depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf',
            'depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf'):
    try: proc = AutoImageProcessor.from_pretrained(mid); model = AutoModelForDepthEstimation.from_pretrained(mid).eval(); used = mid; break
    except Exception as e: print('LOAD FAIL', mid, repr(e)[:200], flush=True)
assert used
torch.set_num_threads(4); D = {}
for name, p in PH.items():
    im = Image.open(io.BytesIO(base64.b64decode(p['b64']))).convert('RGB')
    with torch.no_grad():
        d = model(**proc(images=im, return_tensors='pt')).predicted_depth
    D[name] = torch.nn.functional.interpolate(d.unsqueeze(1).float(), size=(im.height, im.width), mode='bilinear', align_corners=False)[0, 0].numpy()
rows = []
for it in IT:
    dm = D[it['photo']]; a, b = int(round(it['u'])), int(round(it['v']))
    z = float(np.median(dm[max(0, b - 2):b + 3, max(0, a - 2):a + 3]))
    rows.append(dict(key=it['key'], z_m=round(z, 3), depth_shape=list(dm.shape)))
json.dump(dict(model=used, rows=rows), open('/kaggle/working/e7_bd_depth___PART__.json', 'w'), indent=1); print(json.dumps(rows))

"""Box suggestions for the highway labelling package: Grounding DINO (base), prompt 'a tree.', no training.
Images are fetched from their public Mapillary thumbnail links. Output: /kaggle/working/hw_suggest_gdino.csv (normalised xyxy + score)."""
import io, csv, json, time, subprocess, urllib.request
subprocess.run("pip install -q -U transformers", shell=True)
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
ITEMS = json.loads(r"""__ITEMS__""")
mid = 'IDEA-Research/grounding-dino-base'
proc = AutoProcessor.from_pretrained(mid); model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to('cuda').eval()
t0 = time.time(); ok = 0; fail = []
with open('/kaggle/working/hw_suggest_gdino.csv', 'w') as fo:
    fo.write('id,x0,y0,x1,y1,score\n')
    for it in ITEMS:
        img = None
        for k in range(3):
            try:
                with urllib.request.urlopen(it['url'], timeout=60) as f: img = Image.open(io.BytesIO(f.read())).convert('RGB'); break
            except Exception: time.sleep(3)
        if img is None: fail.append(it['id']); continue
        W, H = img.size
        inp = proc(images=img, text='a tree.', return_tensors='pt').to('cuda')
        with torch.no_grad(): out = model(**inp)
        try: res = proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=0.05, text_threshold=0.05, target_sizes=[(H, W)])[0]
        except TypeError: res = proc.post_process_grounded_object_detection(out, inp.input_ids, box_threshold=0.05, text_threshold=0.05, target_sizes=[(H, W)])[0]
        ok += 1
        for (a, b, c, d), s in zip(res['boxes'].tolist(), res['scores'].tolist()):
            fo.write(f"{it['id']},{a/W:.5f},{b/H:.5f},{c/W:.5f},{d/H:.5f},{s:.4f}\n")
json.dump(dict(model=mid, prompt='a tree.', n_ok=ok, failed=fail, minutes=(time.time() - t0) / 60), open('/kaggle/working/hw_suggest_info.json', 'w'))
print('done', ok, len(fail), flush=True)

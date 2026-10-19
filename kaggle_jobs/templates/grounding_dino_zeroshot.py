"""Zero-shot detector row: Grounding DINO (base), prompt 'a tree.', no training. Bangladesh test split only.
Output: /kaggle/working/test_preds_gdino.csv in the same format as every other detector (normalised xyxy + score)."""
import os, glob, csv, io, json, subprocess, time
subprocess.run("pip install -q -U transformers", shell=True)
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
SPLIT_CSV = """__SPLIT__"""
R = glob.glob('/kaggle/input/**/roadside-trees-of-bangladesh-street-view', recursive=True)[0]
src = {'BD_740': f'{R}/Train/BD_740', 'BD_755': f'{R}/Train/BD_755', 'BD_Test_Set_270': f'{R}/Test/BD_Test_Set_270'}
rows = [r for r in csv.DictReader(io.StringIO(SPLIT_CSV)) if r['split'] == 'test']
mid = 'IDEA-Research/grounding-dino-base'
proc = AutoProcessor.from_pretrained(mid); model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to('cuda').eval()
t0 = time.time()
with open('/kaggle/working/test_preds_gdino.csv', 'w') as fo:
    fo.write('image,x0,y0,x1,y1,score\n')
    for r in rows:
        img = Image.open(f"{src[r['folder']]}/{r['id']}.jpg").convert('RGB'); W, H = img.size
        inp = proc(images=img, text='a tree.', return_tensors='pt').to('cuda')
        with torch.no_grad(): out = model(**inp)
        try: res = proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=0.05, text_threshold=0.05, target_sizes=[(H, W)])[0]
        except TypeError: res = proc.post_process_grounded_object_detection(out, inp.input_ids, box_threshold=0.05, text_threshold=0.05, target_sizes=[(H, W)])[0]
        for (a, b, c, d), s in zip(res['boxes'].tolist(), res['scores'].tolist()):
            fo.write(f"{r['folder']}_{r['id']}.jpg,{a/W:.5f},{b/H:.5f},{c/W:.5f},{d/H:.5f},{s:.4f}\n")
json.dump(dict(model=mid, prompt='a tree.', n_images=len(rows), minutes=(time.time() - t0) / 60), open('/kaggle/working/gdino_info.json', 'w'))
print('done', len(rows), flush=True)

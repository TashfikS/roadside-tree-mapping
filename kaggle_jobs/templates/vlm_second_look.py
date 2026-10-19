"""Zero-shot VLM second look on Pasadena candidate planting positions and city planting-site records.
Model, prompts and scoring are fixed in vlm_prereg.md. No ground truth is present in this job.
Output: /kaggle/working/vlm_scores.csv  (key, question, p_yes) written incrementally.
"""
import os, glob, json, zlib, base64, time, subprocess, math, sys
subprocess.run("pip install -q -U transformers accelerate", shell=True)
import numpy as np, torch
from PIL import Image

# ---------------- renderer (identical to vlm_render.py) ----------------
__RENDER__
# -----------------------------------------------------------------------

CROPS = json.loads(zlib.decompress(base64.b64decode("__CROPS__")))
ROOT = glob.glob('/kaggle/input/**/pasadena-urban-trees', recursive=True)[0]; SV = ROOT + '/streetview'
Q = {'line': "This is part of a street-level panorama. A red line is drawn on the ground along one side of the street. "
             "Is there any tree, including a small, young or newly planted tree, standing on or right next to the red line? Answer Yes or No.",
     'circle': "This is part of a street-level panorama. A red circle marks a spot on the ground beside the street. "
               "Is there space at the red circle to plant a new street tree, meaning unpaved ground, grass or a planting strip, "
               "with no existing tree, driveway, building, pole or hydrant in the way? Answer Yes or No."}
MODELS = ['Qwen/Qwen3-VL-4B-Instruct', 'Qwen/Qwen2.5-VL-3B-Instruct']
from transformers import AutoProcessor, AutoModelForImageTextToText
print('GPUs', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)


def load(mid, dtype):
    kw = dict(device_map='auto')
    if dtype == torch.float32 and torch.cuda.device_count() == 1: kw['max_memory'] = {0: '13GiB', 'cpu': '28GiB'}
    try: m = AutoModelForImageTextToText.from_pretrained(mid, dtype=dtype, **kw)
    except TypeError: m = AutoModelForImageTextToText.from_pretrained(mid, torch_dtype=dtype, **kw)
    return m.eval(), AutoProcessor.from_pretrained(mid)


def token_ids(tok, words):
    out = set()
    for w in words:
        ids = tok.encode(w, add_special_tokens=False)
        if len(ids) == 1: out.add(ids[0])
    return sorted(out)


pano_cache = {}
def crop_img(c):
    if c['pano'] not in pano_cache:
        pano_cache.clear(); pano_cache[c['pano']] = Image.open(f"{SV}/{c['pano']}_z2.jpg").convert('RGB')
    return render(pano_cache[c['pano']], c['x_c'], c['hw'], c['y0'], c['y1'], c['ov'], c['kind'])


@torch.no_grad()
def p_yes(img, q):
    msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': img}, {'type': 'text', 'text': q}]}]
    inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to(model.device)
    lg = model(**inp).logits[0, -1].float()
    return float(torch.sigmoid(torch.logsumexp(lg[Y], 0) - torch.logsumexp(lg[N], 0)))


used = None
for mid in MODELS:
    for dt in (torch.float16, torch.float32):
        try:
            model, proc = load(mid, dt); tok = proc.tokenizer
            Y, N = token_ids(tok, ['Yes', 'yes', ' Yes', ' yes']), token_ids(tok, ['No', 'no', ' No', ' no'])
            Y, N = torch.tensor(Y, device=model.device), torch.tensor(N, device=model.device)
            test = [p_yes(crop_img(c), Q[c['kind']]) for c in CROPS[:6]]
            msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': crop_img(CROPS[0])}, {'type': 'text', 'text': 'Describe this image in one sentence.'}]}]
            inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to(model.device)
            cap = proc.batch_decode(model.generate(**inp, max_new_tokens=40)[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)[0]
            print('SELFTEST', mid, dt, [round(t, 3) for t in test], '|', cap, flush=True)
            if all(np.isfinite(test)) and len(cap.strip()) > 10 and len(set(np.round(test, 4))) > 1: used = (mid, str(dt)); break
        except Exception as e:
            print('LOAD/SELFTEST FAIL', mid, dt, repr(e)[:300], flush=True)
        model = None; torch.cuda.empty_cache()
    if used: break
assert used, 'no model passed the self-test'
json.dump(dict(model=used[0], dtype=used[1], n_crops=len(CROPS), prompts=Q), open('/kaggle/working/vlm_info.json', 'w'), indent=1)

t0 = time.time(); CROPS.sort(key=lambda c: c['pano'])
with open('/kaggle/working/vlm_scores.csv', 'w') as fo:
    fo.write('key,question,p_yes\n')
    for i, c in enumerate(CROPS):
        try: p = p_yes(crop_img(c), Q[c['kind']])
        except Exception as e: p = float('nan'); print('ERR', c['key'], repr(e)[:120], flush=True)
        fo.write(f"{c['key']},{c['kind']},{p:.5f}\n")
        if i % 200 == 0: fo.flush(); print(i, len(CROPS), round((time.time() - t0) / 60, 1), 'min', flush=True)
        if time.time() - t0 > 10.5 * 3600: print('TIME BUDGET STOP', i, flush=True); break
print('done', round((time.time() - t0) / 60, 1), 'min', flush=True)

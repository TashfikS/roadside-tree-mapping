"""Build an offline review package: review/review.html + review/img/*.jpg + items embedded as JSON.

Item types
- 'trunk' : test-split tree box with an automatic trunk-base proposal (accept with Enter, or click to correct)
- 'e3'    : field photo + ARuler screenshot; the user clicks the trunk base of the measured tree
Every item stores the crop geometry so clicks map back to ORIGINAL image pixels.
"""
import os, json, glob, re, datetime as dt
import pandas as pd
from PIL import Image

import sys
OUT = sys.argv[1] if len(sys.argv)>1 else 'review'
ONLY = sys.argv[2] if len(sys.argv)>2 else 'all'; os.makedirs(OUT + '/img', exist_ok=True)
items = []

def ts(n):
    for pat, fmt in [(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})', '%Y-%m-%d-%H-%M-%S'), (r'(\d{4}-\d{2}-\d{2}_\d{6})', '%Y-%m-%d_%H%M%S'),
                     (r'(\d{8}-\d{6})', '%Y%m%d-%H%M%S'), (r'(\d{8}_\d{6})', '%Y%m%d_%H%M%S')]:
        m = re.search(pat, n)
        if m:
            return dt.datetime.strptime(m.group(1), fmt)

# ---------------- E3 items ----------------
if ONLY in ('all','e3'):
    ex = pd.DataFrame(json.load(open('field_screens_extracted.json')))
    ex['name'] = ex.file.map(os.path.basename); ex['t'] = ex.name.map(ts)
    ar = ex[ex.type == 'aruler'].sort_values('t')
    photos = []
    for p in glob.glob('field/feb/T*/IMG_*.jpg') + glob.glob(os.path.join(os.environ.get('FIELD_PHOTO_DIR', 'data/external/field_photos'), '**', '*.jpg'), recursive=True):
        b = os.path.basename(p)
        if re.search(r'(IMG_\d{8}_\d{6}|^\d{8}_\d{6})', b):
            photos.append(dict(photo=p, t=ts(b)))
    ph = pd.DataFrame(photos).drop_duplicates('t').sort_values('t').reset_index(drop=True)
    for k, a in enumerate(ar.itertuples()):
        same_site = ph[(ph.t - a.t).abs() <= pd.Timedelta(minutes=6)]
        if '/feb/' in a.file:
            site = a.file.split('/')[2]
            same_site = ph[ph.photo.str.contains(f'/feb/{site}/')]
        before = same_site[same_site.t <= a.t]
        cand = before.iloc[-1] if len(before) else same_site.iloc[0]
        im = Image.open(cand.photo).convert('RGB'); W, H = im.size; s = 2304 / max(W, H)
        small = im.resize((round(W * s), round(H * s)), Image.LANCZOS)
        pf = f'img/e3_{k:02d}_photo.jpg'; small.save(f'{OUT}/{pf}', quality=88)
        sc = Image.open(a.file).convert('RGB'); sc.thumbnail((500, 900)); sf = f'img/e3_{k:02d}_aruler.jpg'; sc.save(f'{OUT}/{sf}', quality=80)
        items.append(dict(kind='e3', key=f'e3/{k:02d}', img=pf, aux=sf, x0=0, y0=0, scale=s, orig_w=W, orig_h=H,
                          photo=os.path.relpath(cand.photo), aruler_file=os.path.relpath(a.file), aruler_cm=float(a.aruler_cm),
                          prompt=f'Click the trunk base of the tree you measured as {a.aruler_cm/100:.2f} m (ARuler screenshot on the right). Press S if that tree is not in this photo.'))

# ---------------- trunk items (test split only) ----------------
if ONLY in ('all','trunk') and os.path.exists('trunk_auto.csv'):
    ND = os.environ.get('BD_DATASET_DIR', 'data/external/roadside-trees-of-bangladesh-street-view')
    src = {'BD_740': f'{ND}/Train/BD_740', 'BD_755': f'{ND}/Train/BD_755', 'BD_Test_Set_270': f'{ND}/Test/BD_Test_Set_270'}
    ta = pd.read_csv('trunk_auto.csv'); sp = pd.read_csv('split_route_proxy.csv')
    ta = ta.merge(sp[['folder', 'id', 'split']], on=['folder', 'id'])
    ta = ta[ta.split == 'test'].sort_values(['folder', 'id', 'box_index'])
    cache = {}
    for r in ta.itertuples():
        p = f"{src[r.folder]}/{r.id}.jpg"
        if p not in cache:
            cache = {p: Image.open(p).convert('RGB')}
        im = cache[p]; W, H = im.size
        x0b, x1b = (r.xc - r.w / 2) * W, (r.xc + r.w / 2) * W; y0b, y1b = (r.yc - r.h / 2) * H, (r.yc + r.h / 2) * H
        mx = 0.35 * (x1b - x0b) + 40
        cx0, cx1 = max(0, x0b - mx), min(W, x1b + mx); cy0, cy1 = max(0, y0b + 0.4 * (y1b - y0b) - 40), min(H, y1b + 0.25 * (y1b - y0b) + 60)
        crop = im.crop((int(cx0), int(cy0), int(cx1), int(cy1))); s = min(1.0, 1400 / max(crop.size))
        crop = crop.resize((max(1, round(crop.width * s)), max(1, round(crop.height * s))), Image.LANCZOS)
        f = f'img/t_{r.folder}_{r.id}_{r.box_index}.jpg'; crop.save(f'{OUT}/{f}', quality=88)
        items.append(dict(kind='trunk', key=f'{r.folder}/{r.id}/{r.box_index}', img=f, x0=int(cx0), y0=int(cy0), scale=s, orig_w=W, orig_h=H,
                          box=[(x0b - int(cx0)) * s, (y0b - int(cy0)) * s, (x1b - int(cx0)) * s, (y1b - int(cy0)) * s],
                          prop=[(r.trunk_u * W - int(cx0)) * s, (r.trunk_v * H - int(cy0)) * s], method=r.method,
                          prompt='Is the green cross on the point where the trunk meets the ground? Enter = yes. Otherwise click the correct point (Shift+click if hidden but guessable, S if impossible).'))

tpl = open('review_template.html').read()
open(f'{OUT}/review.html', 'w').write(tpl.replace('/*ITEMS*/[]', json.dumps(items)).replace("const STORE='review_v1'", f"const STORE='review_{OUT}'"))
print(len(items), sum(i['kind'] == 'e3' for i in items), sum(i['kind'] == 'trunk' for i in items))

import os
import sys, os, json, zlib, base64, numpy as np, pandas as pd
sys.path.insert(0, '.'); os.environ['GAP_AREA'] = '2'; os.environ['GAP_ASSIGN_MAX'] = '12'
import pas_gaps as pg, vlm_render as vr
from PIL import Image, ImageDraw
crops = {c['key']: c for c in json.loads(zlib.decompress(base64.b64decode(open('vlm_crops.b64').read())))}
T = pd.read_csv('vlm_targets_gt.csv'); inv = pg.load_inventory(); trees = inv[~inv.cn.isin(pg.NONTREE)]
TE = pg.enu(trees.lat, trees.lng)
rng = np.random.default_rng(1)
t = T[(T.area == 'test') & (T.set == 'T')]
pick = list(t[t.correct == False].sample(4, random_state=1).key) + list(t[t.correct == True].sample(4, random_state=2).key)
r = T[(T.area == 'test') & (T.set == 'R')]
pick_r = list(r[r.site_type == 'vacant'].sample(2, random_state=3).key) + list(r[r.site_type == 'no_room'].sample(2, random_state=4).key)
tiles = []
for key, kind in [(k, 'line') for k in pick] + [(k, 'circle') for k in pick_r]:
    c = crops[f'{key}/{kind}']; k = c['pano']; v = pg.panos[k]
    cam = pg.enu(float(v['Location']['original_lat']), float(v['Location']['original_lng'])); yaw = float(v['Projection']['pano_yaw_deg'])
    img = Image.open(f"{pg.D}/streetview/{k}_z2.jpg"); im = vr.render(img, c['x_c'], c['hw'], c['y0'], c['y1'], c['ov'], kind)
    near = TE[np.linalg.norm(TE - cam, axis=1) < 30]
    if len(near):
        px, z = vr.ground_to_pixel(cam, yaw, near); dx = ((px[:, 0] - c['x_c'] + vr.W / 2) % vr.W) - vr.W / 2
        d = ImageDraw.Draw(im)
        for u, y in zip(dx, px[:, 1]):
            X, Y = (u + c['hw']) * 2, (y - c['y0']) * 2
            if 0 <= X < im.width and 0 <= Y < im.height: d.ellipse([X - 6, Y - 6, X + 6, Y + 6], outline=(0, 255, 0), width=3)
    row = T.set_index('key').loc[key]
    lab = f"{key.split('/')[1]} {'correct' if row.get('correct') == True else ('WRONG d_inv=%s' % row.get('d_inv') if kind == 'line' else row.get('site_type'))}  cam {row.cam_dist} m"
    ImageDraw.Draw(im).rectangle([0, 0, 520, 26], fill=(0, 0, 0)); ImageDraw.Draw(im).text((6, 6), lab, fill=(255, 255, 0))
    im.thumbnail((700, 420)); tiles.append(im)
Wd = 700; S = Image.new('RGB', (2 * Wd + 10, 6 * 430), 'white')
for i, im in enumerate(tiles): S.paste(im, ((i % 2) * (Wd + 10), (i // 2) * 430))
S.save('vlm_check_sheet.jpg', quality=82); print(S.size, len(tiles))

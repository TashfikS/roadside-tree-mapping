"""Build the two highway packages from hw_photos.csv.

hw_boxes/ : 300 training + validation photos with editable box suggestions (Grounding DINO >= 0.30 and the current
            detector >= 0.10, merged), for box labelling.
hw_test/  : 150 test photos, no model output, for trunk-base clicks.
Suggestion rules are fixed in hw_prereg.md.
"""
import os, json, shutil, random
import numpy as np, pandas as pd
from PIL import Image

P = pd.read_csv('hw_photos.csv', dtype={'id': str})
G = pd.read_csv('kout/hw_suggest/hw_suggest_gdino.csv', dtype={'id': str})
D = pd.read_csv('hw_suggest_det.csv', dtype={'id': str})


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-12)


def nms(rows, thr):
    keep = []
    for r in sorted(rows, key=lambda r: -r[4]):
        if all(iou(r, k) < thr for k in keep): keep.append(r)
    return keep


def suggestions(i):
    det = [(r.x0, r.y0, r.x1, r.y1, r.score, 'det') for r in D[(D.id == i) & (D.score >= 0.10)].itertuples()]
    det = nms(det, 0.5)
    g = [(r.x0, r.y0, r.x1, r.y1, r.score, 'gdino') for r in G[(G.id == i) & (G.score >= 0.30)].itertuples()
         if (r.x1 - r.x0) * (r.y1 - r.y0) <= 0.40 and (r.y1 - r.y0) >= 0.04]
    g = [b for b in nms(g, 0.5) if all(iou(b, d) < 0.4 for d in det)]
    return sorted(det + g, key=lambda r: -r[4])[:12]


def build(name, rows, mode, title, help_html, filename, sugg):
    out = f'{name}/img'; shutil.rmtree(name, ignore_errors=True); os.makedirs(out)
    items = []
    for k, r in enumerate(rows.itertuples(), 1):
        src = f'hw/img/{r.id}.jpg'; shutil.copy(src, f'{out}/{r.id}.jpg'); w, h = Image.open(src).size
        it = dict(key=f'{name}/{k:03d}', id=r.id, img=f'img/{r.id}.jpg', w=w, h=h)
        if sugg:
            it['sugg'] = [[round(b[0] * w, 1), round(b[1] * h, 1), round(b[2] * w, 1), round(b[3] * h, 1), b[5]] for b in suggestions(r.id)]
        items.append(it)
    t = open('hw_tool_template.html').read()
    for a, b in (('/*TITLE*/', title), ('/*HELP*/', help_html), ('/*MODE*/', mode), ('/*STORE*/', f'{name}_v1'),
                 ('/*FILENAME*/', filename), ('/*ITEMS*/[]', json.dumps(items)), ('/*TAGS*/[]', '[]')):
        assert t.count(a) == 1, a
        t = t.replace(a, b)
    open(f'{name}/label.html', 'w').write(t)
    return items


HELP_BOXES = (
    "<b>Draw a box around every roadside tree you can make out as a separate tree</b>, on both sides of the road, from the "
    "top of the crown down to where the trunk meets the ground. Use the same rules as your Kaggle dataset. "
    "<b>Yellow dashed boxes are suggestions</b> from untrained detectors: keep good ones, delete wrong ones, fix loose ones, "
    "and add every tree they missed. Boxes you draw or fix turn green.<br>"
    "<b>Mouse:</b> drag = new box &middot; click a box = select it (click again to reach an overlapping one) &middot; drag a "
    "cyan corner or edge = resize. A magnifier follows the mouse.<br>"
    "<b>Keys:</b> <b>Delete</b> or <b>D</b> = delete the selected box &middot; <b>Z</b> = undo &middot; <b>Enter</b> = save and next "
    "&middot; <b>&larr;</b> = back &middot; <b>N</b> = no roadside trees &middot; <b>X</b> = unusable photo (too blurred, blank, "
    "or not this road) &middot; <b>H</b> = hide boxes for a moment.<br>"
    "Skip tree lines too far away to separate. Progress is saved in this browser automatically. "
    "<b>Teammates:</b> person 1 does photos 1&ndash;100, person 2 does 101&ndash;200, person 3 does 201&ndash;300 (use "
    "<i>Go to photo #</i>); each presses <b>Save CSV</b> and sends the file.")
HELP_POINTS = (
    "<b>Click the point where each roadside tree's trunk meets the ground</b>: every tree beside this road that you can make "
    "out as a separate tree, on both sides. <b>Shift+click</b> if the base is hidden but you can guess where it is "
    "(orange mark). Click an existing mark to remove it. A magnifier follows the mouse.<br>"
    "<b>Keys:</b> <b>Enter</b> = save and next &middot; <b>&larr;</b> = back &middot; <b>Z</b> = undo &middot; <b>N</b> = no "
    "roadside trees &middot; <b>X</b> = unusable photo.<br>"
    "This is the test, so no model output is shown. Progress is saved in this browser automatically; when all 150 are "
    "done, press <b>Save CSV</b> and send the file.")

if __name__ == '__main__':
    lab = P[P.split.isin(['train', 'val'])].sort_values('km')
    test = P[P.split == 'test'].sample(frac=1, random_state=11)
    ib = build('hw_boxes', lab, 'boxes', 'Highway photos: tree boxes (300)', HELP_BOXES, 'hw_boxes_answers.csv', True)
    it = build('hw_test', test, 'points', 'Highway photos: trunk bases (150)', HELP_POINTS, 'hw_test_answers.csv', False)
    ns = [len(i['sugg']) for i in ib]; src = pd.Series([b[4] for i in ib for b in i['sugg']]).value_counts().to_dict()
    att = P[['id', 'split', 'creator', 'captured_at', 'make', 'model', 'lat', 'lon', 'km']].copy()
    att['source_url'] = 'https://www.mapillary.com/app/?pKey=' + att.id; att['license'] = 'CC BY-SA 4.0'
    att.to_csv('hw_attribution.csv', index=False)
    print(len(ib), len(it), 'suggestions per photo: mean', round(np.mean(ns), 2), 'zero', sum(n == 0 for n in ns), 'by source', src)

"""Build self-contained Kaggle GPU scripts from the templates in kaggle_jobs/templates/.

Kaggle scripts cannot read this repository, so the split, labels and helper code are embedded into each generated
script. Every job is written to <out>/<name>/train.py together with a kernel-metadata.json.

Usage
-----
    python kaggle_jobs/build_jobs.py --user <kaggle-username> --out work/kjobs            # all paper jobs
    python kaggle_jobs/build_jobs.py --user <kaggle-username> --out work/kjobs --only pose_s0 pose_s1
    kaggle kernels push -p work/kjobs/pose_s0

Jobs (paper Table 3 and Sections 4.1, 4.4, 5.4)
    split               route-grouped split of the Bangladesh images (writes split_route_proxy.csv, split_stats.json)
    trunk_labels        automatic trunk-base points for every box (Grounding DINO + SAM)
    pose_s{0,1,2}       YOLO11m-pose, 1024 px (main model)          posel_s{0,1,2}  YOLO11l-pose, 1280 px
    yolo11m_s{1,2}      YOLO11m, 1024 px                             yolo832_s{0,1,2} YOLO11m, 832 px
    rtdetr_l_s{0,1,2}   RT-DETR-L, 832 px                            frcnn           Faster R-CNN, seeds 0-2
    gdino_zs            Grounding DINO, zero-shot
    pas_teacher         Pasadena teacher + inventory-verified pseudo-labels
    pas_student         Pasadena detector retrained on the pseudo-labels (attach the pas_teacher output)
    vlm_gap             vision-language second look (needs work/vlm2_crops.b64 from src/vlm2_targets.py)
    e7_pas_a, e7_pas_b  Depth Anything V2 on the Pasadena boxes (needs work/kout/pas_student/student_raw_dets.csv)
    e7_bd_{0..5}        Depth Anything V2 on the Bangladesh field photos (needs FIELD_PHOTO_DIR)
    hw_suggest          box suggestions for highway labelling (needs work/hw_photos_urls.csv with thumbnail links)
"""
import argparse, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
T = lambda n: open(os.path.join(HERE, 'templates', n)).read()
DATA = lambda n: os.path.join(ROOT, 'data', n)
BD = 'tashfiksalehin/roadside-trees-of-bangladesh-street-view'
PAS = 'pasadena-urban-trees'          # Kaggle copy of the Pasadena Urban Trees benchmark, uploaded by the user


def split_csv():
    return open(DATA('bangladesh_split.csv')).read()


def pose(name, seed, model='yolo11m-pose.pt', imgsz=1024, patience=15):
    s = T('trunk_pose.py').split('"""', 2)[2]          # drop the template docstring
    for k, v in (('__SPLIT_CSV__', split_csv()), ('__LABELS_JSON__', json.dumps(json.load(open(DATA('trunk_pose_labels.json'))))),
                 ('__PATIENCE__', str(patience)), ('__SEED__', str(seed)), ('__NAME__', name), ('__MODEL__', model), ('__IMGSZ__', str(imgsz))):
        s = s.replace(k, v)
    return s, [BD]


def box(name, seed, model, imgsz, patience=30):
    s = T('box_detector.py')
    for k, v in (('__SPLIT_CSV__', split_csv()), ('__RT_PATIENCE__', '15' if patience == 15 else '20'), ('__BATCH__', '4' if 'rtdetr' in model else '8'), ('__PATIENCE__', str(patience)), ('__SEED__', str(seed)), ('__NAME__', name),
                 ('__MODEL__', model), ('__IMGSZ__', str(imgsz))):
        s = s.replace(k, v)
    return s, [BD]


def human():
    return json.dumps(json.load(open(DATA('pasadena_human_labels.json'))))


JOBS = {
    'split': lambda: (T('split_by_route.py'), [BD]),
    'trunk_labels': lambda: (T('trunk_labels.py'), [BD]),
    'frcnn': lambda: (T('faster_rcnn.py').replace('__SPLIT_CSV__', split_csv()), [BD]),
    'gdino_zs': lambda: (T('grounding_dino_zeroshot.py').replace('__SPLIT__', split_csv()), [BD]),
    'pas_teacher': lambda: (T('pasadena_teacher.py').replace('__HUMAN__', human()), [PAS]),
    'pas_student': lambda: (T('pasadena_student.py').replace('__HUMAN__', human()), [PAS]),
    'vlm_gap': lambda: (T('vlm_second_look.py').replace('__RENDER__', open(os.path.join(ROOT, 'src', 'vlm_render.py')).read())
                        .replace('__CROPS__', open(os.path.join(ROOT, 'work', 'vlm2_crops.b64')).read().strip()), [PAS]),
}
for s in (0, 1, 2):
    JOBS[f'pose_s{s}'] = (lambda s=s: pose(f'pose_s{s}', s))
    JOBS[f'posel_s{s}'] = (lambda s=s: pose(f'posel_s{s}', s, 'yolo11l-pose.pt', 1280))
    JOBS[f'yolo832_s{s}'] = (lambda s=s: box(f'yolo832_s{s}', s, 'yolo11m.pt', 832, 15))
    JOBS[f'rtdetr_l_s{s}'] = (lambda s=s: box(f'rtdetr_l_s{s}', s, 'rtdetr-l.pt', 832, 15))
for s in (1, 2):
    JOBS[f'yolo11m_s{s}'] = (lambda s=s: box(f'yolo11m_s{s}', s, 'yolo11m.pt', 1024))


# ---------------------------------------------------------------- jobs that need outputs of earlier steps
def depth_pasadena(part):
    """Depth Anything V2 ranges for every retrained-detector box (paper Section 5.2); two halves to fit Kaggle's size limit."""
    import base64, zlib
    import numpy as np, pandas as pd
    sd = pd.read_csv(os.path.join(ROOT, 'work', 'kout', 'pas_student', 'student_raw_dets.csv'))
    pn = list(dict.fromkeys(sd.pano)); half = len(pn) // 2; pl = {'a': pn[:half], 'b': pn[half:]}[part]
    sel = sd.pano.isin(set(pl)).values; sub = sd[sel]; gidx = np.where(sel)[0]; pi_ = {p: i for i, p in enumerate(pl)}
    A_ = np.stack([sub.pano.map(pi_).values.astype(np.uint16)] +
                  [np.clip(np.round(sub[c].values), 0, 65535).astype(np.uint16) for c in ('x0', 'y0', 'x1', 'y1')], 0)
    enc = lambda b: base64.b64encode(zlib.compress(b, 9)).decode()
    s = T('depth_pasadena.py').replace('ROWS = json.loads(zlib.decompress(base64.b64decode("__ROWS__")))',
        '_A = np.frombuffer(zlib.decompress(base64.b64decode("__ARR__")), dtype=np.uint16).reshape(5, -1)\n'
        '_P = zlib.decompress(base64.b64decode("__PN__")).decode().split("\\n")\n'
        'GIDX = np.frombuffer(zlib.decompress(base64.b64decode("__GIDX__")), dtype=np.uint32)\n'
        'ROWS = [[_P[a], float(b), float(c), float(d), float(e)] for a, b, c, d, e in _A.T]')
    s = s.replace('out.write(f"{i},{j},', 'out.write(f"{GIDX[i]},{j},').replace('e7_pas_depth.csv', f'e7_pas_depth_{part}.csv') \
         .replace('e7_info.json', f'e7_info_{part}.json')
    s = s.replace('__ARR__', enc(A_.tobytes())).replace('__PN__', enc('\n'.join(pl).encode())).replace('__GIDX__', enc(gidx.astype(np.uint32).tobytes()))
    return s, [PAS]


def depth_bangladesh(part):
    """Depth Anything V2 at the clicked trunk bases of the field photos (CPU); photos resized to 518 px on the short side."""
    import base64, io
    import pandas as pd
    from PIL import Image
    photo_dir = os.environ.get('FIELD_PHOTO_DIR', os.path.join(ROOT, 'data', 'external', 'field_photos'))
    F = pd.read_csv(DATA('field_trees_bangladesh.csv')); F = F[F.visibility > 0]
    names = list(dict.fromkeys(F.photo)); mine = names[part::6]; photos, items = {}, []
    for r in F[F.photo.isin(mine)].itertuples():
        im = Image.open(os.path.join(photo_dir, r.photo)).convert('RGB'); sc = 518 / min(im.size)
        if r.photo not in photos:
            nw, nh = int(round(im.width * sc)), int(round(im.height * sc)); b = io.BytesIO()
            im.resize((nw, nh), Image.LANCZOS).save(b, 'JPEG', quality=82)
            photos[r.photo] = dict(b64=base64.b64encode(b.getvalue()).decode(), w=nw, h=nh)
        items.append(dict(key=r.key, photo=r.photo, u=round(r.trunk_u * sc, 2), v=round(r.trunk_v * sc, 2), k=sc))
    s = T('depth_bangladesh.py').replace('__PH__', json.dumps(photos)).replace('__IT__', json.dumps(items)).replace('__PART__', str(part))
    return s, []


def highway_suggestions():
    """Grounding DINO box suggestions for the highway labelling package (photos fetched from Mapillary thumbnail links)."""
    import pandas as pd
    P = pd.read_csv(os.path.join(ROOT, 'work', 'hw_photos_urls.csv'), dtype={'id': str})
    return T('highway_suggestions.py').replace('__ITEMS__', json.dumps(P[P.split.isin(['train', 'val'])][['id', 'url']].to_dict('records'))), []


JOBS.update({'e7_pas_a': lambda: depth_pasadena('a'), 'e7_pas_b': lambda: depth_pasadena('b'), 'hw_suggest': highway_suggestions})
for p in range(6):
    JOBS[f'e7_bd_{p}'] = (lambda p=p: depth_bangladesh(p))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--user', required=True, help='Kaggle username that will own the kernels')
    ap.add_argument('--out', default=os.path.join(ROOT, 'work', 'kjobs'))
    ap.add_argument('--only', nargs='*', help='job names (default: all)')
    a = ap.parse_args()
    for name in a.only or sorted(JOBS):
        try:
            src, data = JOBS[name]()
        except FileNotFoundError as e:
            print(f'{name:14s} skipped: input not built yet ({os.path.relpath(e.filename, ROOT)})'); continue
        d = os.path.join(a.out, name); os.makedirs(d, exist_ok=True)
        open(os.path.join(d, 'train.py'), 'w').write(src)
        slug = 'rt-' + name.replace('_', '-')
        json.dump(dict(id=f'{a.user}/{slug}', title=slug, code_file='train.py', language='python', kernel_type='script',
                       is_private=True, enable_gpu=not name.startswith('e7_bd'), enable_internet=True, dataset_sources=data, competition_sources=[],
                       kernel_sources=[]), open(os.path.join(d, 'kernel-metadata.json'), 'w'), indent=1)
        print(f'{name:14s} -> {d}  ({len(src) // 1024} KB)')

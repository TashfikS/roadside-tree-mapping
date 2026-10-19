import os
os.environ.setdefault('KMP_AFFINITY', 'disabled'); os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('MKL_NUM_THREADS', '2')
import sys, json, math, importlib, numpy as np, pandas as pd
sys.path.insert(0, '.'); import mly_geo as mg; importlib.reload(mg)
from scipy.spatial import cKDTree
PA = os.environ.get('PASADENA_DIR', 'data/external/PasadenaUrbanTrees_Detection')
sys.path.insert(0, PA + "/PythonAPI")
from parse_dataset import ParseDataset; from eval import EvaluatePasadenaUrbanTrees
Mk = pd.read_parquet('mly_pas_keep.parquet'); cams = mg.prepare(Mk)
inv = json.load(open('pas_inv_valtest.json')); gtE = {s: mg.enu([t['lat'] for t in inv[s]], [t['lng'] for t in inv[s]]) for s in ('val', 'test')}
ds = ParseDataset(PA + "/PasadenaUrbanTrees")
if not hasattr(ds, 'detection_datasets'): ds.parse_dataset()
camP = np.c_[cams.camE.values, cams.camN.values]; ZONE = {}
for s, k in (('val', 1), ('test', 2)):
    g = ds.detection_datasets[s]; G = mg.enu([t['lat'] for t in g], [t['lng'] for t in g])
    d, _ = cKDTree(camP[cams.tvt.values == k]).query(G); sel = [t for t, dd in zip(g, d) if dd <= 20]
    ds.detection_datasets[s + '_mly'] = sel; ZONE[s] = mg.enu([t['lat'] for t in sel], [t['lng'] for t in sel])


def ev(trees, split):
    E = EvaluatePasadenaUrbanTrees(ds, 4.0, dataset_id=split + '_mly')
    ap, prec, rec, _ = E.evaluate_detection_average_precision(trees[['lat', 'lng', 'score']].to_dict('records'))
    P = mg.enu(trees.lat.values, trees.lng.values); dist = cKDTree(ZONE[split]).query(P)[0] if len(P) else np.array([]); m10 = dist[dist <= 10]
    return dict(AP_4m=round(float(ap), 3), max_recall=round(float(max(rec)) if len(rec) else 0, 3), n_trees=len(trees),
                median_m10=round(float(np.median(m10)), 2) if len(m10) else None, p90_m10=round(float(np.quantile(m10, .9)), 2) if len(m10) else None)


def in_zone(trees, split, r=20):
    k = {'val': 1, 'test': 2}[split]; P = mg.enu(trees.lat.values, trees.lng.values)
    d = cKDTree(camP[cams.tvt.values == k]).query(P)[0]; return trees[d <= r]


def load_dets():
    D = pd.read_csv('mly_pas_dets.csv', dtype={'id': str}); return D[D.id.isin(cams.index)]

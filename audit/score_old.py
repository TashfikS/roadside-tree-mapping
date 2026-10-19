"""Darknet-equivalent AP (`detector map -points 0`) for detections from infer_old.py: all detections (score >= 0.005, NMS 0.45)
sorted by score over all processed images; a detection is TP if its best-IoU ground-truth box in the same image has
IoU > thr and is not yet taken; AP = area under the monotone precision envelope over all recall points.
Usage: python score_old.py preds.csv [subset_regex ...]   (images = those listed in preds.csv.done)"""
import sys, re, os, numpy as np, pandas as pd
def gt_of(p):
    t = p[:-4] + '.txt'
    rows = [l.split() for l in open(t) if l.strip()] if os.path.exists(t) else []
    return np.array([[float(r[1]) - float(r[3]) / 2, float(r[2]) - float(r[4]) / 2, float(r[1]) + float(r[3]) / 2, float(r[2]) + float(r[4]) / 2] for r in rows]).reshape(-1, 4)
def iou(b, G):
    ix = np.clip(np.minimum(b[2], G[:, 2]) - np.maximum(b[0], G[:, 0]), 0, None); iy = np.clip(np.minimum(b[3], G[:, 3]) - np.maximum(b[1], G[:, 1]), 0, None)
    i = ix * iy; return i / ((b[2] - b[0]) * (b[3] - b[1]) + (G[:, 2] - G[:, 0]) * (G[:, 3] - G[:, 1]) - i + 1e-12)
def ap_all_points(P, paths, thr=0.5):
    GT = {p: gt_of(p) for p in paths}; npos = sum(len(g) for g in GT.values())
    P = P[P.path.isin(set(paths))].sort_values('score', ascending=False, kind='mergesort')
    taken = {p: np.zeros(len(g), bool) for p, g in GT.items()}; tp = np.zeros(len(P), int)
    for k, r in enumerate(P.itertuples()):
        G = GT[r.path]
        if len(G) == 0: continue
        o = iou(np.array([r.x0, r.y0, r.x1, r.y1]), G); j = int(o.argmax())
        if o[j] > thr and not taken[r.path][j]: taken[r.path][j] = True; tp[k] = 1
    ctp = np.cumsum(tp); cfp = np.cumsum(1 - tp); rec = ctp / max(npos, 1); prec = ctp / np.maximum(ctp + cfp, 1e-12)
    mrec = np.r_[0, rec, 1]; mpre = np.r_[0, prec, 0]
    for i in range(len(mpre) - 2, -1, -1): mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])), int(npos), len(paths), float(rec[-1]) if len(rec) else 0.0
if __name__ == '__main__':
    P = pd.read_csv(sys.argv[1]); paths = open(sys.argv[1] + '.done').read().split('\n'); paths = [p for p in paths if p]
    for s in (sys.argv[2:] or ['.*']):
        ps = [p for p in paths if re.search(s, p)]
        if not ps: continue
        a50 = ap_all_points(P, ps, 0.5); a75 = ap_all_points(P, ps, 0.75)
        print(f'{s:28s} images {a50[2]:5d} boxes {a50[1]:5d}  AP50 {a50[0]*100:6.2f} %  AP75 {a75[0]*100:6.2f} %  max recall@0.5 {a50[3]:.3f}  min score {P[P.path.isin(set(ps))].score.min():.4f}')

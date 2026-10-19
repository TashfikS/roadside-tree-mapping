"""Highway test audit (post hoc and descriptive; the pre-set decision used found_all only, and nothing here changes it).
1. Reproduce found_all per run from re-run predictions (same weights, photos and script); must equal hw_test_results_runs.csv.
2. Precision proxy over ALL usable test photos (status 'done' or 'none'): share of confident boxes (score >= 0.25) that
   contain >= 1 clicked trunk base (box enlarged by 0.03 on each side, as in the decision metric). Boxes without a click
   are false positives OR trees that were not clicked, so the proxy is a lower bound on precision.
3. Range window of the map (2-20 m): clicks and boxes in photos of trips with a valid horizon are located with the map's
   own geometry (hw_geo level camera, trip horizon pitch, camera height 1.6 m vehicle / 1.5 m hand-held; trunk point =
   keypoint if kconf >= 0.5 else box bottom-centre). Recall and precision proxy inside the window.
Output: audit/audit_hw.json, audit/audit_hw_runs.csv"""
import os, sys, json
import numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0, os.getcwd())
import hw_geo as hg

CONF, KCONF, RMIN, RMAX, PAD = 0.25, 0.5, 2.0, 20.0, 0.03
A = pd.read_csv('hw_test_answers.csv', dtype={'id': str})
usable = A[A.status.isin(['done', 'none'])].id.unique()
C = A[(A.status == 'done') & A.u.notna()].reset_index(drop=True)
saved = pd.read_csv('audit/hw_test_results_runs_saved.csv').set_index('run')
size = {i: Image.open(f'hw/img/{i}.jpg').size for i in A.id.unique()}

meta = pd.read_parquet('hw_all_meta.parquet'); meta['id'] = meta.id.astype(str)
cams = hg.prepare(meta)                                   # all highway photos, exactly as hw_map.py (trip speed sets vehicle vs hand-held)
cams = cams[cams.index.isin(A.id.unique())]
hz = pd.read_csv('hw_seq_horizon.csv').set_index('sequence')
cams['h_frac'] = cams.sequence.map(hz.horizon); cams['seq_valid'] = cams.sequence.map(hz.seq_valid).fillna(False).astype(bool)


def ranges(ids, u, v):
    """Map geometry for normalised points (u, v) in photos `ids`; NaN where the trip has no valid horizon."""
    ids = np.asarray(ids); out = np.full(len(ids), np.nan)
    ok = np.array([i in cams.index and bool(cams.loc[i, 'seq_valid']) for i in ids])
    if not ok.any(): return out
    c = cams.loc[ids[ok]]; W = np.array([size[i][0] for i in ids[ok]], float); H = np.array([size[i][1] for i in ids[ok]], float)
    pitch = np.array([hg.pitch_from_horizon(h, hh, ww, f, k1, k2, fe) for h, hh, ww, f, k1, k2, fe in
                      zip(c.h_frac, H, W, c.f, c.k1, c.k2, c.fisheye)])
    R = np.stack([hg.level_rotation(hd, p) for hd, p in zip(c.computed_compass_angle.values, pitch)])
    _, _, rng = hg.locate_level(c.camE.values, c.camN.values, R, c.f.values, c.k1.values, c.k2.values, c.fisheye.values,
                                np.asarray(u)[ok], np.asarray(v)[ok], W, H, np.where(c.vehicle.values, 1.6, 1.5))
    out[ok] = rng; return out


C['range_m'] = ranges(C.id.values, C.u.values, C.v.values)
C['in_window'] = (C.range_m >= RMIN) & (C.range_m <= RMAX)
rows = []
for run in ('hwb_s0', 'hwb_s1', 'hwb_s2'):
    P = pd.read_csv(f'hwrerun/hw_test_preds_{run}.csv', dtype={'id': str}); P = P[(P.score >= CONF) & P.id.isin(usable)].reset_index(drop=True)
    kp = P.kconf >= KCONF; P['pu'] = np.where(kp, P.ku, (P.x0 + P.x1) / 2); P['pv'] = np.where(kp, P.kv, P.y1)
    P['range_m'] = ranges(P.id.values, P.pu.values, P.pv.values); P['in_window'] = (P.range_m >= RMIN) & (P.range_m <= RMAX)
    found = np.zeros(len(C), bool); has_click = np.zeros(len(P), bool)
    for i, g in C.groupby('id'):
        b = P[P.id == i]
        if not len(b): continue
        ins = ((g.u.values[:, None] >= b.x0.values - PAD) & (g.u.values[:, None] <= b.x1.values + PAD) &
               (g.v.values[:, None] >= b.y0.values - PAD) & (g.v.values[:, None] <= b.y1.values + PAD))
        found[g.index] = ins.any(1); has_click[b.index] = ins.any(0)
    w = C.in_window.values; rv = C.range_m.notna().values; pw = P.in_window.values
    rows.append(dict(run=run, found_all=round(float(found.mean()), 4), saved_found_all=float(saved.loc[run, 'found_all']),
                     n_clicks=int(len(C)), n_clicks_ranged=int(rv.sum()), n_clicks_in_window=int(w.sum()),
                     recall_in_window=round(float(found[w].mean()), 4), recall_ranged_outside_window=round(float(found[rv & ~w].mean()), 4) if (rv & ~w).any() else None,
                     n_boxes=int(len(P)), boxes_with_click_share=round(float(has_click.mean()), 4),
                     boxes_on_photos_without_trees=int(P.id.isin(A[A.status == 'none'].id).sum()),
                     n_boxes_in_window=int(pw.sum()), boxes_in_window_with_click_share=round(float(has_click[pw].mean()), 4) if pw.any() else None,
                     keypoint_share=round(float(kp.mean()), 3)))
    # 4. trunk point vs box bottom, in metres: for each clicked tree inside a confident box (highest-scoring box containing
    #    the click), range from the human click vs range from the predicted keypoint and from the box bottom-centre
    best = {}
    for i, g in C.groupby('id'):
        b = P[P.id == i]
        if not len(b): continue
        ins = ((g.u.values[:, None] >= b.x0.values - PAD) & (g.u.values[:, None] <= b.x1.values + PAD) &
               (g.v.values[:, None] >= b.y0.values - PAD) & (g.v.values[:, None] <= b.y1.values + PAD))
        for r_, ci in enumerate(g.index):
            if ins[r_].any(): best[ci] = b.index[ins[r_]][np.argmax(b.score.values[ins[r_]])]
    ci = np.array(list(best)); bi = np.array([best[k] for k in ci])
    if len(ci):
        ok = C.loc[ci, 'in_window'].values
        ci, bi = ci[ok], bi[ok]
        kp_ok = (P.loc[bi, 'kconf'].values >= KCONF)
        d_kp = ranges(C.loc[ci, 'id'].values, P.loc[bi, 'ku'].values, P.loc[bi, 'kv'].values)
        d_bb = ranges(C.loc[ci, 'id'].values, ((P.loc[bi, 'x0'] + P.loc[bi, 'x1']) / 2).values, P.loc[bi, 'y1'].values)
        d_ck = C.loc[ci, 'range_m'].values
        vis = (C.loc[ci, 'visible'].values == 2)
        e_kp, e_bb = np.abs(d_kp - d_ck), np.abs(d_bb - d_ck)
        rows[-1].update(n_trunk_pairs=int(len(ci)), keypoint_conf_share=round(float(kp_ok.mean()), 3),
                        kp_minus_click_median_abs_m=round(float(np.nanmedian(e_kp)), 2), boxbottom_minus_click_median_abs_m=round(float(np.nanmedian(e_bb)), 2),
                        kp_minus_click_median_abs_m_visible_base=round(float(np.nanmedian(e_kp[vis])), 2) if vis.any() else None,
                        boxbottom_minus_click_median_abs_m_visible_base=round(float(np.nanmedian(e_bb[vis])), 2) if vis.any() else None,
                        kp_better_share=round(float(np.nanmean(e_kp < e_bb)), 3),
                        boxbottom_signed_median_m=round(float(np.nanmedian(d_bb - d_ck)), 2), kp_signed_median_m=round(float(np.nanmedian(d_kp - d_ck)), 2),
                        kp_minus_click_rel_median_pct=round(float(np.nanmedian(e_kp / d_ck) * 100), 1), boxbottom_minus_click_rel_median_pct=round(float(np.nanmedian(e_bb / d_ck) * 100), 1))
R = pd.DataFrame(rows); R.to_csv('audit/audit_hw_runs.csv', index=False)
m = R.mean(numeric_only=True)
summ = dict(runs=rows, reproduced=bool((R.found_all - R.saved_found_all).abs().max() < 1e-4),
            mean_recall_all=round(float(m.found_all), 4), mean_recall_in_window=round(float(m.recall_in_window), 4),
            mean_precision_proxy_all=round(float(m.boxes_with_click_share), 4), mean_precision_proxy_in_window=round(float(m.boxes_in_window_with_click_share), 4),
            photos_usable=int(len(usable)), photos_with_trees=int(C.id.nunique()),
            photos_valid_trip=int(sum(bool(cams.loc[i, 'seq_valid']) for i in usable if i in cams.index)),
            click_range_quantiles_m={str(q): round(float(C.range_m.quantile(q)), 1) for q in (0.1, 0.25, 0.5, 0.75, 0.9)})
summ['implied_density_factor_vs_map'] = round(float((m.boxes_in_window_with_click_share / m.recall_in_window) / (1 / m.found_all)), 3)
json.dump(summ, open('audit/audit_hw.json', 'w'), indent=1); print(R.to_string(index=False)); print(json.dumps({k: v for k, v in summ.items() if k != 'runs'}, indent=1))

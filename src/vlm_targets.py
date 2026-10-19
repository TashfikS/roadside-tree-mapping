"""Targets for the VLM second-look test on Pasadena (validation + test areas).

Set T: every candidate planting position our method proposes (validation-chosen setting: >=2 views, 12 m street zone,
       spacing S = 12 m; positions at run start + k*S inside each predicted plantable run).
       Ground truth (kept local, never sent to the VLM job): correct = no inventory tree within S on that side.
Set R: city records at fixed positions: vacant planting sites (plantable), 'no room for replant' (not plantable), stumps.
Each target gets a 'line' crop (+-12 m along the kerb side, for the tree question) and/or a 'circle' crop (room question).
"""
import os, sys, json, math, zlib, base64
import numpy as np, pandas as pd
import vlm_render as vr

S, HALF = 12.0, 12.0
rows, crops = [], []
for AREA, an in ((1, 'val'), (2, 'test')):
    os.environ['GAP_AREA'] = str(AREA); os.environ['GAP_ASSIGN_MAX'] = '12'
    sys.modules.pop('pas_gaps', None)
    import pas_gaps as pg
    inv = pg.load_inventory()
    import ast
    ext = inv.extended.apply(lambda e: e if isinstance(e, dict) else ast.literal_eval(e))
    inv['dbh'] = pd.to_numeric(ext.apply(lambda e: e.get('Trunk_Diameter')), errors='coerce')
    trees = inv[~inv.cn.isin(pg.NONTREE)].reset_index(drop=True)
    pred = pd.read_csv(f'gaps_{an}_v2.csv')
    IA = pg.assign(pg.enu(trees.lat, trees.lng)); PA = pg.assign(pg.enu(pred.lat, pred.lng))
    CP = [np.array([pg.xy[k] for k in p]) for p in pg.chains]
    CS = [np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))] for P in CP]
    cam = {k: pg.enu(float(v['Location']['original_lat']), float(v['Location']['original_lng'])) for k, v in pg.panos.items() if k in pg.test}
    yaw = {k: float(pg.panos[k]['Projection']['pano_yaw_deg']) for k in cam}

    def at(ci, s):
        P, Sc = CP[ci], CS[ci]; s = np.clip(np.atleast_1d(s), 0, Sc[-1])
        j = np.clip(np.searchsorted(Sc, s, side='right') - 1, 0, len(Sc) - 2)
        seg = P[j + 1] - P[j]; L = np.maximum(np.linalg.norm(seg, axis=1), 1e-9)
        return P[j] + ((s - Sc[j]) / L)[:, None] * seg, seg / L[:, None]

    def best_pano(ci, p):
        ks = [k for k in pg.chains[ci] if k in cam]
        d = np.array([np.linalg.norm(cam[k] - p) for k in ks]); d[d < 3] = np.inf
        return ks[int(np.argmin(d))] if np.isfinite(d).any() else None

    def add_crop(key, k, pts, kind):
        px, z = vr.ground_to_pixel(cam[k], yaw[k], pts)
        x_c, hw, y0, y1, ov = vr.window(px, kind, x_c=px[len(px) // 2, 0])
        crops.append(dict(key=key, pano=k, kind=kind, x_c=round(x_c, 1), hw=round(hw, 1), y0=round(y0, 1), y1=round(y1, 1),
                          ov=np.round(ov, 1).tolist()))

    off_side = PA[PA.chain >= 0].groupby(['chain', 'side']).dist.median().to_dict()
    off_def = float(PA[PA.chain >= 0].dist.median())
    keys = sorted(set(map(tuple, IA.loc[IA.chain >= 0, ['chain', 'side']].values)))
    mp_store = {}
    for ci, sd in keys:
        L = pg.chain_len[ci]
        si = IA.s[(IA.chain == ci) & (IA.side == sd)].values; sp = PA.s[(PA.chain == ci) & (PA.side == sd)].values
        dbh_i = trees.dbh.values[np.where((IA.chain == ci) & (IA.side == sd))[0]]
        g, mi = pg.plantable_mask(si, L, S); _, mp = pg.plantable_mask(sp, L, S); mp_store[(ci, sd)] = (g, mp)
        off = off_side.get((ci, sd), off_def)
        i = 0
        while i < len(g):
            if not mp[i]: i += 1; continue
            j = i
            while j + 1 < len(g) and mp[j + 1]: j += 1
            for s_pos in np.arange(g[i], g[j] + 1e-9, S):
                gi = int(round((s_pos - g[0]) / pg.GRID))
                q, t = at(ci, s_pos); n = np.array([-t[0, 1], t[0, 0]]); P0 = q[0] + sd * off * n
                k = best_pano(ci, P0)
                if k is None: continue
                ss = np.arange(max(0, s_pos - HALF), min(L, s_pos + HALF) + 1e-9, 2.0)
                qq, tt = at(ci, ss); LP = qq + sd * off * np.stack([-tt[:, 1], tt[:, 0]], 1)
                key = f'{an}/T/{ci}/{sd}/{int(s_pos)}'
                dd = np.abs(si - s_pos) if len(si) else np.array([np.inf])
                rows.append(dict(key=key, area=an, set='T', chain=ci, side=sd, s=round(float(s_pos), 1), pano=k,
                                 correct=bool(mi[gi]), d_inv=round(float(dd.min()), 1),
                                 dbh_nearest=float(dbh_i[int(dd.argmin())]) if len(si) else np.nan,
                                 cam_dist=round(float(np.linalg.norm(cam[k] - P0)), 1), lat_off=round(float(off), 1)))
                add_crop(key + '/line', k, LP, 'line'); add_crop(key + '/circle', k, P0[None], 'circle')
            i = j + 1
    for typ, sel in (('vacant', inv.cn.isin(pg.VACANT)), ('no_room', inv.cn == 'NO ROOM FOR REPLANT'), ('stump', inv.cn == 'STUMP')):
        E = pg.enu(inv.lat[sel], inv.lng[sel]); SA = pg.assign(E)
        for e, (_, a) in zip(E, SA.iterrows()):
            if a.chain < 0: continue
            k = best_pano(int(a.chain), e)
            if k is None: continue
            g, mp = mp_store.get((int(a.chain), int(a.side)), (None, None))
            in_gap = None
            if g is not None and len(g) and g[0] <= a.s <= g[-1]: in_gap = bool(mp[int(round((a.s - g[0]) / pg.GRID))])
            key = f'{an}/R/{typ}/{len(rows)}'
            rows.append(dict(key=key, area=an, set='R', site_type=typ, chain=int(a.chain), side=int(a.side), s=round(float(a.s), 1), pano=k,
                             in_pred_gap=in_gap, cam_dist=round(float(np.linalg.norm(cam[k] - e)), 1)))
            add_crop(key + '/circle', k, e[None], 'circle')
T = pd.DataFrame(rows); T.to_csv('vlm_targets_gt.csv', index=False)
blob = base64.b64encode(zlib.compress(json.dumps(crops).encode(), 9)).decode()
open('vlm_crops.b64', 'w').write(blob)
print(T.groupby(['area', 'set']).size().to_dict(), 'crops', len(crops), 'blob KB', len(blob) // 1024)
print(T[T.set == 'T'].groupby('area').correct.astype(float).mean().round(3).to_dict(), T[T.set == 'R'].groupby(['area', 'site_type']).size().to_dict())
print('cam_dist median/p90', T.cam_dist.median(), T.cam_dist.quantile(.9), 'lateral offset median', T.lat_off.median())

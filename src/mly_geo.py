"""E2b: tree positions from Mapillary perspective photos (rules: mly_prereg.md).

Camera model (OpenSfM, as used by Mapillary): X_cam = R X_world + t, world = local ENU (x east, y north, z up); camera axes
x right, y down, z forward = rows of R. Normalised image coordinates use the long image side; principal point at the centre;
radial distortion x_d = x_u (1 + k1 r^2 + k2 r^4).
"""
import math, sys, os
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

R_E, LAT0, LON0 = 6371008.8, 34.15, -118.13


def enu(lat, lon):
    return np.c_[np.radians(np.asarray(lon, float) - LON0) * R_E * math.cos(math.radians(LAT0)),
                 np.radians(np.asarray(lat, float) - LAT0) * R_E]


def to_ll(E, N):
    return np.asarray(N) / R_E * 180 / math.pi + LAT0, np.asarray(E) / (R_E * math.cos(math.radians(LAT0))) * 180 / math.pi + LON0


def rot_level(compass_deg):
    th = math.radians(compass_deg)
    fwd = np.array([math.sin(th), math.cos(th), 0.0]); right = np.array([math.cos(th), -math.sin(th), 0.0])
    return np.stack([right, np.array([0.0, 0.0, -1.0]), fwd])


def undistort(xd, yd, k1, k2, it=25):
    x, y = xd.copy(), yd.copy()
    for _ in range(it):
        r2 = x * x + y * y; d = 1 + k1 * r2 + k2 * r2 * r2
        x, y = xd / d, yd / d
    return x, y


def prepare(meta):
    """Per-image camera: ENU position, rotations (SfM and level), intrinsics, capture type."""
    m = meta.copy()
    m['id'] = m['id'].astype(str)
    P = enu(m.lat, m.lon); m['camE'], m['camN'] = P[:, 0], P[:, 1]
    m['f'] = [p[0] for p in m.camera_parameters]; m['k1'] = [p[1] for p in m.camera_parameters]; m['k2'] = [p[2] for p in m.camera_parameters]
    m['R_sfm'] = list(Rot.from_rotvec(np.stack(m.computed_rotation.values)).as_matrix())
    m['R_level'] = [rot_level(c) for c in m.computed_compass_angle]
    m['vehicle'] = m.seq_speed >= 3.0
    return m.set_index('id')


def locate(dets, cams, orient='sfm', h_vehicle=2.0, h_hand=1.5, rng=None):
    """dets: id, ku, kv, img_w, img_h, score -> adds E, N, range_m (ground-plane intersection). rng: random-bearing baseline."""
    d = dets.dropna(subset=['ku']).copy(); d['id'] = d['id'].astype(str)
    c = cams.loc[d.id]
    W, H = d.img_w.values.astype(float), d.img_h.values.astype(float); S = np.maximum(W, H)
    xd = (d.ku.values * W - W / 2) / S / c.f.values; yd = (d.kv.values * H - H / 2) / S / c.f.values
    x, y = undistort(xd, yd, c.k1.values, c.k2.values)
    Rs = np.stack(c['R_sfm' if orient == 'sfm' else 'R_level'].values)              # (n,3,3) world->camera
    ray = np.einsum('nij,ni->nj', Rs, np.c_[x, y, np.ones_like(x)])                # R^T @ ray_cam
    h = np.where(c.vehicle.values, h_vehicle, h_hand)
    ok = ray[:, 2] < -1e-6
    t = np.where(ok, -h / np.where(ok, ray[:, 2], -1), np.nan)
    dE, dN = t * ray[:, 0], t * ray[:, 1]
    d['range_m'] = np.hypot(dE, dN)
    if rng is not None:
        b = rng.uniform(0, 2 * np.pi, len(d)); dE, dN = d.range_m.values * np.sin(b), d.range_m.values * np.cos(b)
    d['E'] = c.camE.values + dE; d['N'] = c.camN.values + dN
    d['tlat'], d['tlon'] = to_ll(d.E.values, d.N.values)
    return d[ok & np.isfinite(d.range_m.values)]


def nearest_m(E, N, gtE):
    kd = cKDTree(gtE); dist, _ = kd.query(np.c_[E, N]); return dist


def merge(d, radius):
    """Greedy score-ordered merge (identical rule to pas_e2.merge)."""
    d = d.sort_values('score', ascending=False)
    X = np.c_[d.E.values, d.N.values]; S = d.score.values
    cen, wsum, best, nv, grid = [], [], [], [], {}
    def key(x): return (int(math.floor(x[0] / radius)), int(math.floor(x[1] / radius)))
    for x, s in zip(X, S):
        gx, gy = key(x); bk, bd = -1, radius + 1e-9
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for k in grid.get((gx + dx, gy + dy), ()):
                    dd = math.hypot(cen[k][0] - x[0], cen[k][1] - x[1])
                    if dd < bd: bk, bd = k, dd
        if bk >= 0 and bd <= radius:
            old = key(cen[bk]); cen[bk] = (cen[bk] * wsum[bk] + x * s) / (wsum[bk] + s); wsum[bk] += s; nv[bk] += 1
            new_ = key(cen[bk])
            if new_ != old: grid[old].remove(bk); grid.setdefault(new_, []).append(bk)
            continue
        cen.append(np.array(x, float)); wsum.append(s); best.append(s); nv.append(1); grid.setdefault((gx, gy), []).append(len(cen) - 1)
    C = np.array(cen) if cen else np.zeros((0, 2))
    lat, lng = to_ll(C[:, 0], C[:, 1]) if len(C) else (np.array([]), np.array([]))
    return pd.DataFrame(dict(lat=lat, lng=lng, E=C[:, 0] if len(C) else [], N=C[:, 1] if len(C) else [], score=best, n_views=nv))

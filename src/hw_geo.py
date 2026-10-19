"""Highway (Dhaka–Chattogram) camera geometry for Mapillary photos: same OpenSfM conventions as mly_geo.py
(X_cam = R X_world + t; world = local ENU; camera axes x right, y down, z forward; normalised coordinates use the long
image side; principal point at the centre), plus the OpenSfM fisheye model.
  perspective: x_d = x_u (1 + k1 r^2 + k2 r^4),           r = |x_u|
  fisheye:     x_d = (theta / r) x_u (1 + k1 theta^2 + k2 theta^4),   theta = atan(r)
Ground-plane intersection with the camera height fixed by capture type (vehicle 1.6 m, hand-held 1.5 m; hw_prereg.md).
"""
import math
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot

R_E, LAT0, LON0 = 6371008.8, 23.0, 91.0


def enu(lat, lon):
    return np.c_[np.radians(np.asarray(lon, float) - LON0) * R_E * math.cos(math.radians(LAT0)),
                 np.radians(np.asarray(lat, float) - LAT0) * R_E]


def to_ll(E, N):
    return np.asarray(N) / R_E * 180 / math.pi + LAT0, np.asarray(E) / (R_E * math.cos(math.radians(LAT0))) * 180 / math.pi + LON0


def distort(x, y, k1, k2, fisheye):
    r = np.hypot(x, y)
    if fisheye:
        th = np.arctan(r); s = np.where(r > 1e-12, th / np.maximum(r, 1e-12), 1.0) * (1 + k1 * th ** 2 + k2 * th ** 4)
    else:
        s = 1 + k1 * r ** 2 + k2 * r ** 4
    return x * s, y * s


def undistort(xd, yd, k1, k2, fisheye, it=30):
    xd, yd = np.asarray(xd, float), np.asarray(yd, float); k1 = np.asarray(k1, float); k2 = np.asarray(k2, float)
    fe = np.broadcast_to(np.asarray(fisheye, bool), xd.shape)
    x, y = xd.copy(), yd.copy()
    # perspective: fixed-point iteration (as in mly_geo.py)
    for _ in range(it):
        r2 = x * x + y * y; d = 1 + k1 * r2 + k2 * r2 * r2
        x = np.where(fe, x, xd / d); y = np.where(fe, y, yd / d)
    # fisheye: solve theta (1 + k1 theta^2 + k2 theta^4) = r_d by Newton, then r = tan(theta)
    rd = np.hypot(xd, yd); th = rd.copy()
    for _ in range(it):
        f = th * (1 + k1 * th ** 2 + k2 * th ** 4) - rd; df = 1 + 3 * k1 * th ** 2 + 5 * k2 * th ** 4
        th = np.clip(th - f / np.where(np.abs(df) > 1e-9, df, 1e-9), 0, math.pi / 2 - 1e-3)
    scale = np.where(rd > 1e-12, np.tan(th) / np.maximum(rd, 1e-12), 1.0)
    x = np.where(fe, xd * scale, x); y = np.where(fe, yd * scale, y)
    return x, y


def prepare(meta, vehicle_speed=3.0):
    """meta: id, lat, lon, captured_at (ms), sequence, camera_type, camera_parameters, computed_rotation, width, height."""
    m = meta[meta.camera_type.isin(['perspective', 'fisheye']) & meta.camera_parameters.notna() & meta.computed_rotation.notna()].copy()
    m['id'] = m['id'].astype(str)
    P = enu(m.lat, m.lon); m['camE'], m['camN'] = P[:, 0], P[:, 1]
    cp = [list(p) + [0.0] * (3 - len(p)) for p in m.camera_parameters]
    m['f'] = [p[0] for p in cp]; m['k1'] = [p[1] for p in cp]; m['k2'] = [p[2] for p in cp]
    m['fisheye'] = m.camera_type.eq('fisheye')
    m['R_sfm'] = list(Rot.from_rotvec(np.stack([np.asarray(r, float) for r in m.computed_rotation])).as_matrix())
    m = m.sort_values(['sequence', 'captured_at'])
    st = np.hypot(m.camE.diff(), m.camN.diff()); dt = m.captured_at.diff() / 1000.0
    same = m.sequence.eq(m.sequence.shift())
    m['speed'] = np.where(same & (dt > 0) & (dt < 30), st / dt.where(dt > 0), np.nan)
    sp = m.groupby('sequence').speed.median(); m['seq_speed'] = m.sequence.map(sp)
    m['vehicle'] = m.seq_speed.fillna(0) >= vehicle_speed
    return m.set_index('id')


def locate_px(cams, ids, u, v, img_w, img_h, h_vehicle=1.6, h_hand=1.5):
    """Normalised pixel (u, v in 0..1 of an image of size img_w x img_h) -> ground point E, N and horizontal range."""
    c = cams.loc[list(map(str, ids))]
    W, H = np.asarray(img_w, float), np.asarray(img_h, float); S = np.maximum(W, H)
    xd = (np.asarray(u) * W - W / 2) / S / c.f.values; yd = (np.asarray(v) * H - H / 2) / S / c.f.values
    x, y = undistort(xd, yd, c.k1.values, c.k2.values, c.fisheye.values)
    Rs = np.stack(c.R_sfm.values)
    ray = np.einsum('nij,ni->nj', Rs, np.c_[x, y, np.ones_like(x)])          # R^T ray_cam (world direction)
    h = np.where(c.vehicle.values, h_vehicle, h_hand)
    ok = ray[:, 2] < -1e-6
    t = np.where(ok, -h / np.where(ok, ray[:, 2], -1), np.nan)
    dE, dN = t * ray[:, 0], t * ray[:, 1]
    return c.camE.values + dE, c.camN.values + dN, np.hypot(dE, dN)


def pitch_from_horizon(h_frac, H, W, f, k1, k2, fisheye):
    """Camera pitch (radians, positive = pitched up) from the horizon row (fraction of image height), roll 0."""
    S = max(W, H); yd = (h_frac * H - H / 2) / S / f
    _, yu = undistort(np.array([0.0]), np.array([yd]), np.array([k1]), np.array([k2]), np.array([fisheye]))
    return float(math.atan(yu[0]))


def level_rotation(heading_deg, pitch_rad):
    """World(ENU)->camera rotation for a camera with the given heading (clockwise from north) and pitch, roll 0."""
    th = math.radians(heading_deg); p = pitch_rad
    right = np.array([math.cos(th), -math.sin(th), 0.0]); f0 = np.array([math.sin(th), math.cos(th), 0.0]); up = np.array([0.0, 0.0, 1.0])
    z = math.cos(p) * f0 + math.sin(p) * up; y = math.sin(p) * f0 - math.cos(p) * up
    return np.stack([right, y, z])


def locate_level(camE, camN, R, f, k1, k2, fisheye, u, v, W, H, h):
    """Arrays per detection: camera position, rotation (n,3,3), intrinsics, normalised pixel, image size, camera height."""
    W, H = np.asarray(W, float), np.asarray(H, float); S = np.maximum(W, H)
    xd = (np.asarray(u) * W - W / 2) / S / f; yd = (np.asarray(v) * H - H / 2) / S / f
    x, y = undistort(xd, yd, k1, k2, fisheye)
    ray = np.einsum('nij,ni->nj', R, np.c_[x, y, np.ones_like(x)])
    ok = ray[:, 2] < -1e-6
    t = np.where(ok, -np.asarray(h, float) / np.where(ok, ray[:, 2], -1), np.nan)
    dE, dN = t * ray[:, 0], t * ray[:, 1]
    return camE + dE, camN + dN, np.hypot(dE, dN)


def _selftest_level():
    rng = np.random.default_rng(1)
    for fe in (False, True):
        k1, k2 = (-0.03, 0.004) if not fe else (0.02, -0.003)
        hd, p, h, f, W, H = 37.0, math.radians(4.0), 1.6, 0.55, 1024, 768
        R = level_rotation(hd, p)
        # horizon row from the rotation, then pitch back from it
        yh = math.tan(p); _, ydh = distort(np.array([0.0]), np.array([yh]), k1, k2, fe); hf = (ydh[0] * f * W + H / 2) / H
        assert abs(pitch_from_horizon(hf, H, W, f, k1, k2, fe) - p) < 1e-6
        P = np.c_[rng.uniform(-8, 8, 50), rng.uniform(3, 18, 50)]                 # ground points in a forward-right frame
        th = math.radians(hd); Ew = P[:, 1] * math.sin(th) + P[:, 0] * math.cos(th); Nw = P[:, 1] * math.cos(th) - P[:, 0] * math.sin(th)
        Xc = (R @ np.c_[Ew, Nw, np.full(50, -h)].T).T; xn, yn = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]
        xd, yd = distort(xn, yn, k1, k2, fe); u = (xd * f * W + W / 2) / W; v = (yd * f * W + H / 2) / H
        E, N, r = locate_level(0.0, 0.0, np.repeat(R[None], 50, 0), np.full(50, f), np.full(50, k1), np.full(50, k2), np.full(50, fe), u, v,
                               np.full(50, W), np.full(50, H), np.full(50, h))
        assert np.abs(E - Ew).max() < 1e-5 and np.abs(N - Nw).max() < 1e-5, (fe, np.abs(E - Ew).max())
    return 'ok'


def _selftest():
    rng = np.random.default_rng(0)
    for fe in (False, True):
        k1, k2 = (-0.05, 0.01) if not fe else (0.02, -0.004)
        x, y = rng.uniform(-0.8, 0.8, 500), rng.uniform(-0.6, 0.6, 500)
        xd, yd = distort(x, y, k1, k2, fe); xu, yu = undistort(xd, yd, np.full(500, k1), np.full(500, k2), np.full(500, fe))
        assert np.abs(xu - x).max() < 1e-6 and np.abs(yu - y).max() < 1e-6, (fe, np.abs(xu - x).max())
    # synthetic camera: level, facing east, height 1.6 m; a ground point 10 m ahead, 3 m right projects and locates back
    R = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)                 # rows: cam x (south), cam y (down), cam z (east)
    cams = pd.DataFrame(dict(camE=[0.0], camN=[0.0], f=[0.6], k1=[0.0], k2=[0.0], fisheye=[False], vehicle=[True],
                             R_sfm=[R]), index=['a'])
    Pw = np.array([10.0, -3.0, -1.6]); pc = R @ Pw; xn, yn = pc[0] / pc[2], pc[1] / pc[2]
    W, H = 1024, 768; u = (xn * 0.6 * W + W / 2) / W; v = (yn * 0.6 * W + H / 2) / H
    E, N, r = locate_px(cams, ['a'], [u], [v], [W], [H])
    assert abs(E[0] - 10) < 1e-6 and abs(N[0] + 3) < 1e-6 and abs(r[0] - math.hypot(10, 3)) < 1e-6, (E, N, r)
    return 'ok'


if __name__ == '__main__':
    print(_selftest())

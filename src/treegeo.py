"""treegeo — tree geolocation from street-level images (ground-plane method).

Pipeline per detection: trunk-base pixel (u, v) + camera (lat, lon, compass, pitch, f, cx, cy, h_cam)
  -> range D on the ground plane -> bearing -> tree lat/lon (local ENU) -> merge duplicates per road side
  -> gaps along the road -> plantable positions with uncertainty.

Conventions
- Image: u to the right, v downwards, pixels. Principal point (cx, cy) = image centre unless given.
- Mapillary perspective cameras: camera_parameters = [f_norm, k1, k2], with f_px = f_norm * max(W, H)
  (OpenSfM convention). Lens distortion is ignored at the trunk base (small near the image centre rows).
- pitch: camera tilt in degrees, positive = looking up. The horizon row is cy + f*tan(pitch).
- compass: heading of the optical axis, degrees clockwise from north.
"""
import math
import numpy as np

R_EARTH = 6371008.8  # mean Earth radius, m


# ---------- camera ----------
def focal_px_from_fov(width_px, hfov_deg):
    return (width_px / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def focal_px_from_mapillary(f_norm, width_px, height_px):
    return f_norm * max(width_px, height_px)


def horizon_row(cy, f_px, pitch_deg=0.0):
    return cy + f_px * math.tan(math.radians(pitch_deg))


# ---------- single detection ----------
def ground_range(v_base, f_px, h_cam, cy, pitch_deg=0.0):
    """Horizontal distance (m) from camera to the trunk base on a flat ground plane.
    Returns nan if the trunk base is at or above the horizon (not on the ground ahead)."""
    ang_below_axis = math.atan2(v_base - cy, f_px)          # ray angle below the optical axis
    depression = ang_below_axis - math.radians(pitch_deg)    # angle below the horizontal (camera tilted up by pitch)
    if depression <= 1e-6:
        return float('nan')
    return h_cam / math.tan(depression)


def bearing_to_pixel(u_base, f_px, cx, compass_deg):
    return (compass_deg + math.degrees(math.atan2(u_base - cx, f_px))) % 360.0


def side_of_road(u_base, cx):
    return 'right' if u_base >= cx else 'left'


def destination(lat, lon, bearing_deg, dist_m):
    """Local-tangent-plane offset (accurate to < 1 cm at < 100 m)."""
    b = math.radians(bearing_deg)
    dn, de = dist_m * math.cos(b), dist_m * math.sin(b)
    dlat = math.degrees(dn / R_EARTH)
    dlon = math.degrees(de / (R_EARTH * math.cos(math.radians(lat))))
    return lat + dlat, lon + dlon


def locate_tree(u_base, v_base, cam):
    """Exact ground-plane intersection of the pixel ray (pinhole camera tilted up by `pitch`).
    cam: dict(lat, lon, compass, f_px, cx, cy, h_cam, pitch=0). Returns dict or None (ray above horizon)."""
    xn = (u_base - cam['cx']) / cam['f_px']
    yn = (v_base - cam['cy']) / cam['f_px']
    p = math.radians(cam.get('pitch', 0.0)); c, s_ = math.cos(p), math.sin(p)
    fwd_dir = c + s_ * yn                    # horizontal forward component of the ray
    up_dir = s_ - c * yn                     # vertical component (negative = pointing down)
    if up_dir >= -1e-6:
        return None
    t = -cam['h_cam'] / up_dir
    fwd, right = t * fwd_dir, t * xn
    D = math.hypot(fwd, right)
    brg = (cam['compass'] + math.degrees(math.atan2(right, fwd))) % 360.0
    lat, lon = destination(cam['lat'], cam['lon'], brg, D)
    return dict(lat=lat, lon=lon, range_m=D, bearing=brg, side=side_of_road(u_base, cam['cx']))


# ---------- metres between points ----------
def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * math.asin(math.sqrt(a))


def to_enu(lat, lon, lat0, lon0):
    e = math.radians(lon - lon0) * R_EARTH * math.cos(math.radians(lat0))
    n = math.radians(lat - lat0) * R_EARTH
    return e, n


# ---------- merge duplicate detections (same tree seen in several frames) ----------
def merge_duplicates(points, radius_m=3.0):
    """points: list of dict(lat, lon, side). Single-linkage clustering within radius_m, per side.
    Returns list of dict(lat, lon, side, n_views)."""
    out = []
    for side in ('left', 'right'):
        P = [p for p in points if p['side'] == side]
        if not P:
            continue
        lat0, lon0 = P[0]['lat'], P[0]['lon']
        X = np.array([to_enu(p['lat'], p['lon'], lat0, lon0) for p in P])
        parent = list(range(len(P)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
        for i, j in zip(*np.where(np.triu(d <= radius_m, 1))):
            parent[find(i)] = find(j)
        groups = {}
        for i in range(len(P)):
            groups.setdefault(find(i), []).append(i)
        for idx in groups.values():
            out.append(dict(lat=float(np.mean([P[i]['lat'] for i in idx])),
                            lon=float(np.mean([P[i]['lon'] for i in idx])),
                            side=side, n_views=len(idx)))
    return out


# ---------- gaps along the road ----------
def chainage(lat, lon, road_latlon):
    """Distance along a road polyline (list of (lat, lon)) to the projection of a point, plus offset (m)."""
    lat0, lon0 = road_latlon[0]
    V = np.array([to_enu(a, b, lat0, lon0) for a, b in road_latlon])
    p = np.array(to_enu(lat, lon, lat0, lon0))
    best = (np.inf, 0.0)
    s0 = 0.0
    for a, b in zip(V[:-1], V[1:]):
        ab = b - a
        L = float(np.hypot(*ab))
        t = float(np.clip(np.dot(p - a, ab) / (L * L), 0, 1)) if L > 0 else 0.0
        q = a + t * ab
        off = float(np.hypot(*(p - q)))
        if off < best[0]:
            best = (off, s0 + t * L)
        s0 += L
    return best[1], best[0]


def plantable_positions(gap_m, spacing_m=12.0):
    """New trees that fit in a gap between two existing trees at the given spacing."""
    return max(0, int(math.floor(gap_m / spacing_m)) - 1)


def gaps_per_side(trees, road_latlon, spacing_m=12.0):
    """trees: merged trees (dict lat, lon, side). Returns list of dict(side, s_start, s_end, gap_m, n_plant)."""
    res = []
    for side in ('left', 'right'):
        s = sorted(chainage(t['lat'], t['lon'], road_latlon)[0] for t in trees if t['side'] == side)
        for a, b in zip(s[:-1], s[1:]):
            res.append(dict(side=side, s_start=a, s_end=b, gap_m=b - a, n_plant=plantable_positions(b - a, spacing_m)))
    return res


def prob_plantable(gap_m, sigma_tree_m, spacing_m=12.0, n_sim=20000, rng=None):
    """P(gap really fits >= 1 new tree), given independent along-road position error (sd sigma) on each end tree."""
    rng = rng or np.random.default_rng(0)
    g = gap_m + rng.normal(0, sigma_tree_m, n_sim) - rng.normal(0, sigma_tree_m, n_sim)
    return float(np.mean(np.floor(g / spacing_m) - 1 >= 1))


# ---------- equirectangular panoramas (Google Street View / Pasadena convention) ----------
def pano_pixel_to_geo(x, y, W, H, lat0, lon0, yaw_deg, h_cam=3.0, pitch_deg=0.0):
    """Ground-plane geolocation of a pixel in a full 360x180 equirectangular panorama.
    Column 0 points to (yaw + 180 deg); the horizon is at row H/2 (+ pitch)."""
    look = x * 2 * math.pi / W
    tilt = (H / 2 - y) * math.pi / H + math.radians(pitch_deg)
    if tilt >= -1e-2:
        return None
    z = -h_cam / math.tan(tilt)                      # horizontal range, m
    az = (look - math.pi + math.radians(yaw_deg))     # radians clockwise from north
    lat, lon = destination(lat0, lon0, math.degrees(az), z)
    return dict(lat=lat, lon=lon, range_m=z, bearing=math.degrees(az) % 360)

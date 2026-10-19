"""Canopy cover along the Dhaka–Chattogram route from the Meta/WRI 1 m canopy height map v2 (CC BY 4.0).

Rules (hw_prereg.md, amendment 3, part C): 100 m stretches along the route line; corridor = pixels whose ground distance
to the route line is <= 20 m (both sides combined) and whose nearest route point falls inside the stretch;
canopy cover = share of corridor pixels with height >= 3 m (>= 5 m reported as a sensitivity value).
Tiles are read remotely as cloud-optimised GeoTIFFs (only the blocks along the corridor are fetched).
This script only extracts the satellite values; the comparison with our map is run once, after the map is final.
"""
import json, math, glob, time
import numpy as np, pandas as pd, rasterio
from rasterio.env import Env
from rasterio.windows import from_bounds
from matplotlib.path import Path

R_MERC, R_EARTH, STEP, HALF_W, H3, H5 = 6378137.0, 6371008.8, 100.0, 20.0, 3, 5
BASE = 'https://dataforgood-fb-data.s3.amazonaws.com/forests/v2/global/dinov3_global_chm_v2_ml3/chm/'
line = np.array(json.load(open('n1_coverage.json'))['line'])           # (lon, lat)
lon, lat = line[:, 0], line[:, 1]
mx = np.radians(lon) * R_MERC; my = R_MERC * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
la1, la2, lo1, lo2 = map(np.radians, (lat[:-1], lat[1:], lon[:-1], lon[1:]))
seg = 2 * R_EARTH * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))
s = np.r_[0, np.cumsum(seg)]; L = s[-1]
tiles = []
with Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif', VSI_CACHE=True,
         VSI_CACHE_SIZE=768 * 1024 * 1024, GDAL_CACHEMAX=768, GDAL_HTTP_MULTIRANGE='YES', GDAL_HTTP_MERGE_CONSECUTIVE_RANGES='YES'):
    for q in json.load(open('chm_v2_tiles_needed.json')):
        tiles.append((q, rasterio.open('/vsicurl/' + BASE + q + '.tif')))
    # acquisition-date polygons (lon/lat or web-mercator, detected from coordinate range)
    polys = []
    for f in sorted(glob.glob('chm_meta/*.geojson')):
        for ft in json.load(open(f))['features']:
            g = ft['geometry']; rings = g['coordinates'] if g['type'] == 'Polygon' else [p[0] for p in g['coordinates']]
            if g['type'] == 'Polygon': rings = [rings[0]]
            for r in rings: polys.append((Path(np.asarray(r)[:, :2]), ft['properties'].get('acq_date')))
    merc_meta = abs(polys[0][0].vertices[0, 0]) > 360
    rows, t0 = [], time.time()
    for k, a in enumerate(np.arange(0, L, STEP)):
        b = min(a + STEP, L)
        if b - a < 50: break
        sel = np.where((s[1:] >= a - 30) & (s[:-1] <= b + 30))[0]                 # segments near this stretch
        ax_, ay_, bx_, by_ = mx[sel], my[sel], mx[sel + 1], my[sel + 1]
        latm = float(np.interp((a + b) / 2, s, lat)); lonm = float(np.interp((a + b) / 2, s, lon))
        k_m = 1 / math.cos(math.radians(latm))                                     # mercator units per ground metre
        wm = HALF_W * k_m
        x_lo, x_hi = min(ax_.min(), bx_.min()) - wm - 3, max(ax_.max(), bx_.max()) + wm + 3
        y_lo, y_hi = min(ay_.min(), by_.min()) - wm - 3, max(ay_.max(), by_.max()) + wm + 3
        vals = []
        for q, ds in tiles:
            l, bt, r, tp = ds.bounds
            if x_hi < l or x_lo > r or y_hi < bt or y_lo > tp: continue
            win = from_bounds(max(x_lo, l), max(y_lo, bt), min(x_hi, r), min(y_hi, tp), ds.transform).round_offsets().round_lengths()
            if win.width <= 0 or win.height <= 0: continue
            arr = ds.read(1, window=win)
            T = ds.window_transform(win)
            cols, rws = np.meshgrid(np.arange(arr.shape[1]) + 0.5, np.arange(arr.shape[0]) + 0.5)
            px, py = T.c + cols * T.a, T.f + rws * T.e
            P = np.stack([px.ravel(), py.ravel()], 1)
            dx, dy = bx_ - ax_, by_ - ay_; ll = dx * dx + dy * dy + 1e-9
            t = np.clip(((P[:, 0:1] - ax_) * dx + (P[:, 1:2] - ay_) * dy) / ll, 0, 1)
            d = np.hypot(P[:, 0:1] - (ax_ + t * dx), P[:, 1:2] - (ay_ + t * dy))
            j = d.argmin(1); dmin = d[np.arange(len(P)), j]
            sproj = s[sel][j] + t[np.arange(len(P)), j] * seg[sel][j]
            keep = (dmin <= wm) & (sproj >= a) & (sproj < b)
            vals.append(arr.ravel()[keep])
        v = np.concatenate(vals) if vals else np.array([], np.uint8)
        pt = (lonm, latm) if not merc_meta else (math.radians(lonm) * R_MERC, R_MERC * math.log(math.tan(math.pi / 4 + math.radians(latm) / 2)))
        acq = next((dte for pth, dte in polys if pth.contains_point(pt)), None)
        rows.append(dict(stretch=k, s_start_m=round(a, 1), s_end_m=round(b, 1), km=round((a + b) / 2000, 3), lon=round(lonm, 6), lat=round(latm, 6),
                         n_px=int(v.size), canopy_ge3=round(float((v >= H3).mean()), 4) if v.size else np.nan,
                         canopy_ge5=round(float((v >= H5).mean()), 4) if v.size else np.nan,
                         mean_h=round(float(v.mean()), 2) if v.size else np.nan, acq_date=acq))
        if k % 200 == 0: print(k, round(time.time() - t0), flush=True)
    for q, ds in tiles: ds.close()
D = pd.DataFrame(rows); D.to_csv('chm_corridor_100m.csv', index=False)
print('stretches', len(D), 'route km', round(L / 1000, 1), 'px per stretch median', int(D.n_px.median()),
      'canopy>=3 m median', round(D.canopy_ge3.median(), 3), 'acq dates', D.acq_date.str[:4].value_counts().sort_index().to_dict(), 'minutes', round((time.time() - t0) / 60, 1))

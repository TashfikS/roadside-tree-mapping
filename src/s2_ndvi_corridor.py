"""U6 (protocols/q1_upgrades_prereg.md): contemporary vegetation along the Dhaka-Chattogram route from Sentinel-2 L2A
(Microsoft Planetary Computer). Dry season 1 Dec - 28 Feb of 2023/24 and 2024/25; scene cloud cover < 20 %; pixels with
SCL class 4 (vegetation) or 5 (bare soil) only; per-pixel median NDVI over the scenes (>= 2 clear observations).
Pixels: 10 m grid (UTM 46N) within 20 m of the route line; along-route position from the same projection as the map.
Output: results/q1_upgrades/s2_ndvi_1km.csv (km, n_pix, share_ndvi_gt04, median_ndvi), s2_scenes.csv.
Run from a folder that contains n1_coverage.json and hw_geo.py (and hw_map.py for the projection).
"""
import json, math, time, threading, os, sys, concurrent.futures as cf
import numpy as np, pandas as pd, requests, rasterio
from rasterio.env import Env
from rasterio.warp import transform as wtransform
from rasterio.windows import Window
from scipy.spatial import cKDTree
OUT = sys.argv[1] if len(sys.argv) > 1 else '.'
line = np.array(json.load(open('n1_coverage.json'))['line'])
X, Y = wtransform('EPSG:4326', 'EPSG:32646', line[:, 0].tolist(), line[:, 1].tolist()); L = np.c_[X, Y]
seg = np.hypot(*np.diff(L, axis=0).T); s = np.r_[0, np.cumsum(seg)]
dd = np.arange(0, s[-1], 5.0); D = np.c_[np.interp(dd, s, L[:, 0]), np.interp(dd, s, L[:, 1])]
x0, y0 = math.floor(D[:, 0].min() / 10) * 10 - 30, math.floor(D[:, 1].min() / 10) * 10 - 30
x1, y1 = math.ceil(D[:, 0].max() / 10) * 10 + 30, math.ceil(D[:, 1].max() / 10) * 10 + 30
kd = cKDTree(D); cells = []
for yy in np.arange(y0 + 5, y1, 10.0):                       # pixel centres of the 10 m S2 grid
    xx = np.arange(x0 + 5, x1, 10.0); P = np.c_[xx, np.full_like(xx, yy)]
    dist, j = kd.query(P, distance_upper_bound=20.0); ok = np.isfinite(dist)
    if ok.any(): cells.append(P[ok])
C = pd.DataFrame(np.concatenate(cells), columns=['x', 'y'])
lon, lat = wtransform('EPSG:32646', 'EPSG:4326', C.x.tolist(), C.y.tolist()); C['lon'], C['lat'] = lon, lat
# along-route km with the map's own projection (hw_map.py)
g = dict(np=np, pd=pd, json=json, math=math, cKDTree=cKDTree); sys.path.insert(0, '.'); import hw_geo as hg; g['hg'] = hg
src = open('hw_map.py').read(); i = src.index("# route polyline"); j = src.index("# detections -> ground points"); exec(src[i:j], g)
E = hg.enu(C.lat.values, C.lon.values); sm, off = g['project'](E[:, 0], E[:, 1])
C['km'] = (sm // 1000).astype(int); C = C[np.abs(off) <= 20.5].reset_index(drop=True)
print('corridor pixels', len(C), flush=True)

bbox = [float(line[:, 0].min()) - 0.01, float(line[:, 1].min()) - 0.01, float(line[:, 0].max()) + 0.01, float(line[:, 1].max()) + 0.01]
feats = []
for a, b in (('2023-12-01', '2024-02-29'), ('2024-12-01', '2025-02-28')):
    body = dict(collections=['sentinel-2-l2a'], bbox=bbox, limit=250, query={'eo:cloud_cover': {'lt': 20}}, datetime=f'{a}T00:00:00Z/{b}T23:59:59Z')
    r = requests.post('https://planetarycomputer.microsoft.com/api/stac/v1/search', json=body, timeout=120).json()
    while True:
        feats += r.get('features', [])
        nxt = [l for l in r.get('links', []) if l.get('rel') == 'next']
        if not nxt: break
        n = nxt[0]; r = requests.post(n['href'], json=n.get('body', body), timeout=120).json() if n.get('method', 'GET') == 'POST' else requests.get(n['href'], timeout=120).json()
print('dry-season scenes', len(feats), flush=True)

_tok = {'t': None, 'exp': 0}; _lock = threading.Lock()
def token():
    with _lock:
        if time.time() > _tok['exp']:
            _tok['t'] = requests.get('https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a', timeout=60).json()['token']; _tok['exp'] = time.time() + 1800
        return _tok['t']

VAL = np.full((len(C), len(feats)), np.nan, np.float32)
def sample(ds, xs, ys):
    """Values of band ds at map coordinates (in ds.crs); reads only the blocks touched."""
    col, row = ~ds.transform * (xs, ys); col = np.floor(col).astype(int); row = np.floor(row).astype(int)
    inside = (col >= 0) & (row >= 0) & (col < ds.width) & (row < ds.height); out = np.full(len(xs), np.nan)
    blk = pd.DataFrame(dict(i=np.where(inside)[0], r=row[inside], c=col[inside])); B = 512
    for (br, bc), gb in blk.groupby([blk.r // B, blk.c // B]):
        w = Window(bc * B, br * B, min(B, ds.width - bc * B), min(B, ds.height - br * B)); a = ds.read(1, window=w)
        out[gb.i.values] = a[gb.r.values - br * B, gb.c.values - bc * B]
    return out
def scene(k):
    f = feats[k]
    for attempt in range(3):
        try:
            with Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif,.TIF', GDAL_HTTP_MAX_RETRY='4', GDAL_HTTP_RETRY_DELAY='2'):
                t = token(); op = lambda a: rasterio.open('/vsicurl/' + f['assets'][a]['href'] + '?' + t)
                with op('B04') as r4, op('B08') as r8, op('SCL') as sc:
                    if r4.crs.to_epsg() == 32646: xs, ys = C.x.values, C.y.values
                    else: xs, ys = map(np.asarray, wtransform('EPSG:4326', r4.crs, C.lon.tolist(), C.lat.tolist()))
                    red, nir, scl = sample(r4, xs, ys), sample(r8, xs, ys), sample(sc, xs, ys)
                    ok = np.isin(scl, (4, 5)) & (red > 0) & (nir > 0)
                    red, nir = (red - 1000) / 10000, (nir - 1000) / 10000        # processing baseline >= 04.00 offset
                    ndvi = np.where(ok, (nir - red) / np.maximum(nir + red, 1e-6), np.nan)
                    VAL[:, k] = ndvi.astype(np.float32); return k, int(np.isfinite(ndvi).sum())
        except Exception as e:
            err = repr(e)[:160]; time.sleep(5)
    return k, err
t0 = time.time(); log = []
with cf.ThreadPoolExecutor(6) as ex:
    for n, res in enumerate(ex.map(scene, range(len(feats)))):
        log.append(res)
        if n % 10 == 0: print(n, len(feats), round((time.time() - t0) / 60, 1), 'min', res, flush=True)
os.makedirs(OUT, exist_ok=True)
pd.DataFrame([dict(id=feats[k]['id'], date=feats[k]['properties']['datetime'][:10], cloud=feats[k]['properties']['eo:cloud_cover'],
                   baseline=feats[k]['properties'].get('s2:processing_baseline'), clear=v) for k, v in log]).to_csv(f'{OUT}/s2_scenes.csv', index=False)
nobs = np.isfinite(VAL).sum(1); comp = np.where(nobs >= 2, np.nanmedian(np.where(np.isfinite(VAL), VAL, np.nan), axis=1), np.nan)
C['ndvi'] = comp
K = C.groupby('km').agg(n_pix=('ndvi', lambda x: int(x.notna().sum())), share_ndvi_gt04=('ndvi', lambda x: float((x.dropna() > 0.4).mean()) if x.notna().any() else np.nan),
                        median_ndvi=('ndvi', 'median')).reset_index()
K.to_csv(f'{OUT}/s2_ndvi_1km.csv', index=False); print('done', len(K), int(K.n_pix.sum()), flush=True)

"""Hot-season land surface temperature along the Dhaka–Chattogram route (rules: hw_prereg.md, amendment 6).
Landsat 8/9 Collection 2 Level-2 ST_B10 from Microsoft Planetary Computer; 1 Mar–31 May 2022–2025; scene cloud < 40 %;
QA_PIXEL bits 0–4 (fill, dilated cloud, cirrus, cloud, cloud shadow) removed; per-pixel median of clear observations
(>= 3 required) on the 30 m UTM 46N grid; 1 km stretches = median of pixels within 50 m of the route line.
Reads only the 256 x 256 blocks that the corridor touches. Output: lst_corridor_1km.csv, lst_scenes.csv."""
import json, math, time, threading, concurrent.futures as cf
import numpy as np, pandas as pd, requests, rasterio
from rasterio.env import Env
from rasterio.warp import transform as wtransform
from rasterio.windows import Window
from scipy.spatial import cKDTree

line = np.array(json.load(open('n1_coverage.json'))['line'])
X, Y = wtransform('EPSG:4326', 'EPSG:32646', line[:, 0].tolist(), line[:, 1].tolist()); L = np.c_[X, Y]
seg = np.hypot(*np.diff(L, axis=0).T); s = np.r_[0, np.cumsum(seg)]
dd = np.arange(0, s[-1], 10.0); D = np.c_[np.interp(dd, s, L[:, 0]), np.interp(dd, s, L[:, 1])]
# 30 m grid centres (multiples of 30 m in UTM 46N, the Landsat C2 grid) within 50 m of the route
x0, y0 = math.floor(D[:, 0].min() / 30) * 30 - 90, math.floor(D[:, 1].min() / 30) * 30 - 90
x1, y1 = math.ceil(D[:, 0].max() / 30) * 30 + 90, math.ceil(D[:, 1].max() / 30) * 30 + 90
kd = cKDTree(D); cells = []
for yy in np.arange(y0, y1 + 1, 30.0):
    xx = np.arange(x0, x1 + 1, 30.0); P = np.c_[xx, np.full_like(xx, yy)]
    dist, j = kd.query(P, distance_upper_bound=50.0); ok = np.isfinite(dist)
    if ok.any(): cells.append(np.c_[P[ok], dd[j[ok]]])
C = np.concatenate(cells); C = pd.DataFrame(C, columns=['x', 'y', 's']); C['stretch'] = (C.s // 1000).astype(int)
print('corridor pixels', len(C), 'route km', round(s[-1] / 1000, 1), flush=True)

body = dict(collections=['landsat-c2-l2'], bbox=[float(line[:, 0].min()) - 0.01, float(line[:, 1].min()) - 0.01, float(line[:, 0].max()) + 0.01, float(line[:, 1].max()) + 0.01],
            limit=250, query={'eo:cloud_cover': {'lt': 40}, 'platform': {'in': ['landsat-8', 'landsat-9']}}, datetime='2022-03-01T00:00:00Z/2025-05-31T23:59:59Z')
feats, r = [], requests.post('https://planetarycomputer.microsoft.com/api/stac/v1/search', json=body, timeout=120).json()
while True:
    feats += r.get('features', [])
    nxt = [l for l in r.get('links', []) if l.get('rel') == 'next']
    if not nxt: break
    n = nxt[0]; r = requests.post(n['href'], json=n.get('body', body), timeout=120).json() if n.get('method', 'GET') == 'POST' else requests.get(n['href'], timeout=120).json()
feats = [f for f in feats if f['properties']['datetime'][5:7] in ('03', '04', '05')]
print('hot-season scenes', len(feats), flush=True)

_tok = {'t': None, 'exp': 0}; _lock = threading.Lock()
def token():
    with _lock:
        if time.time() > _tok['exp']:
            _tok['t'] = requests.get('https://planetarycomputer.microsoft.com/api/sas/v1/token/landsateuwest/landsat-c2', timeout=60).json()['token']
            _tok['exp'] = time.time() + 1800
        return _tok['t']

VAL = np.full((len(C), len(feats)), np.nan, np.float32)
def scene(k):
    f = feats[k]
    for attempt in range(3):
        try:
            with Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.TIF,.tif', GDAL_HTTP_MAX_RETRY='4', GDAL_HTTP_RETRY_DELAY='2'):
                t = token()
                with rasterio.open('/vsicurl/' + f['assets']['lwir11']['href'] + '?' + t) as st, rasterio.open('/vsicurl/' + f['assets']['qa_pixel']['href'] + '?' + t) as qa:
                    inv = ~st.transform
                    col, row = inv * (C.x.values, C.y.values); col = np.floor(col).astype(int); row = np.floor(row).astype(int)
                    inside = (col >= 0) & (row >= 0) & (col < st.width) & (row < st.height)
                    out = np.full(len(C), np.nan, np.float32)
                    blk = pd.DataFrame(dict(i=np.where(inside)[0], r=row[inside], c=col[inside]))
                    for (br, bc), gb in blk.groupby([blk.r // 256, blk.c // 256]):
                        w = Window(bc * 256, br * 256, min(256, st.width - bc * 256), min(256, st.height - br * 256))
                        a = st.read(1, window=w); q = qa.read(1, window=w)
                        rr, cc = gb.r.values - br * 256, gb.c.values - bc * 256
                        dn, qv = a[rr, cc].astype(np.float64), q[rr, cc]
                        clear = (dn > 0) & ((qv & 0b11111) == 0)
                        out[gb.i.values[clear]] = (dn[clear] * 0.00341802 + 149.0 - 273.15).astype(np.float32)
                    VAL[:, k] = out
                    return k, int(inside.sum()), int(np.isfinite(out).sum())
        except Exception as e:
            err = repr(e)[:160]; time.sleep(5)
    return k, -1, err

t0 = time.time(); log = []
with cf.ThreadPoolExecutor(6) as ex:
    for n, res in enumerate(ex.map(scene, range(len(feats)))):
        log.append(res)
        if n % 20 == 0: print(n, len(feats), round((time.time() - t0) / 60, 1), 'min', res, flush=True)
pd.DataFrame([dict(id=feats[k]['id'], date=feats[k]['properties']['datetime'][:10], cloud=feats[k]['properties']['eo:cloud_cover'],
                   inside=a, clear=b) for k, a, b in log]).to_csv('lst_scenes.csv', index=False)
nobs = np.isfinite(VAL).sum(1); comp = np.where(nobs >= 3, np.nanmedian(np.where(np.isfinite(VAL), VAL, np.nan), axis=1), np.nan)
C['lst_c'] = comp; C['n_obs'] = nobs
S = C.groupby('stretch').agg(km_start=('s', lambda x: float(x.min()) / 1000), n_px=('x', 'size'), n_px_valid=('lst_c', lambda x: int(x.notna().sum())),
                             lst_median_c=('lst_c', 'median'), obs_median=('n_obs', 'median')).reset_index()
S.to_csv('lst_corridor_1km.csv', index=False)
print('stretches', len(S), 'with LST', int(S.lst_median_c.notna().sum()), '| LST median', round(float(S.lst_median_c.median()), 1), 'C, p10/p90',
      np.round(S.lst_median_c.quantile([.1, .9]).values, 1), '| clear obs per pixel median', int(np.median(nobs)), '| failed scenes',
      sum(1 for _, a, _ in log if a == -1), '| minutes', round((time.time() - t0) / 60, 1))

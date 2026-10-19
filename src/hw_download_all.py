"""Download every non-panorama highway photo (1024 px thumbnails) to hw_all/; reuses copies already on disk."""
import os, shutil, time, urllib.request, concurrent.futures as cf
import pandas as pd
M = pd.read_parquet('hw_all_meta.parquet'); M = M[M.camera_type.isin(['perspective', 'fisheye']) & M.url1024.notna()]
os.makedirs('hw_all', exist_ok=True)
def get(r):
    p = f'hw_all/{r.id}.jpg'
    if os.path.exists(p) and os.path.getsize(p) > 0: return 1
    for q in (f'hw/img/{r.id}.jpg', f'hw_hz/{r.id}.jpg'):
        if os.path.exists(q) and os.path.getsize(q) > 0: shutil.copy(q, p); return 1
    for k in range(3):
        try:
            with urllib.request.urlopen(r.url1024, timeout=60) as f: open(p, 'wb').write(f.read()); return 1
        except Exception: time.sleep(3)
    return 0
t0 = time.time()
with cf.ThreadPoolExecutor(16) as ex: ok = list(ex.map(get, M.itertuples()))
print('photos', len(M), 'downloaded or present', sum(ok), 'minutes', round((time.time() - t0) / 60, 1), flush=True)

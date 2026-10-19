"""Map figures: study areas (fig_study_area), a Pasadena street excerpt with mapped trees and planting gaps
(fig_pasadena_map), and the Dhaka-Chattogram highway on a map (fig_highway_map).

Inputs : ../results (route, densities, temperature, mapped trees), ../data (basemap, field sites, highway photos),
         and the Pasadena Urban Trees benchmark (PASADENA_DIR) for streets, areas and the inventory.
"""
import os, sys, json, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'src')]
os.chdir(HERE)
os.environ.setdefault('GAP_AREA', '2'); os.environ.setdefault('GAP_ASSIGN_MAX', '12')
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import LineCollection
from figstyle import apply_figure_style
import pas_gaps as pg

apply_figure_style(sizes=(8, 7, 6)); mpl.rcParams['savefig.bbox'] = 'standard'
C_OURS, C_INV, C_GAP, C_TREE, C_HOT, C_LAND, C_SEA = '#1f5f99', '0.30', '#d99a00', '#2f7d4f', '#c4572a', '#f2f0eb', '#dde9f2'
SPLIT = {0: ('training', '#b9c7d6'), 1: ('validation', '#e2c68f'), 2: ('test', C_OURS)}
R, D = '../results', '../data'


def letter(fig, ax, L, dx=0.0, dy=0.008):
    b = ax.get_position(); fig.text(b.x0 + dx, b.y1 + dy, L, fontsize=9, fontweight='bold', ha='left', va='bottom')


def rings(geom):
    c = geom['coordinates']
    if geom['type'] == 'Polygon': return [c[0]]
    if geom['type'] == 'MultiPolygon': return [p[0] for p in c]
    return []


def lines(geom):
    c = geom['coordinates']
    return [c] if geom['type'] == 'LineString' else (c if geom['type'] == 'MultiLineString' else [])


BM = json.load(open(f'{D}/basemap_bangladesh_naturalearth.geojson'))['features']
lay = lambda n: [f for f in BM if f['properties']['layer'] == n]
CH = pd.read_csv(f'{R}/chm_corridor_100m.csv').sort_values('km')
k2ll = lambda km: (np.interp(km, CH.km, CH.lon), np.interp(km, CH.km, CH.lat))
HK = pd.read_csv(f'{R}/hw_km.csv'); HR = pd.read_csv(f'{R}/hw_heat_ranking.csv')
LS = pd.read_csv(f'{R}/lst_canopy_1km.csv')
scale = json.load(open(f'{R}/audit/audit_values.json'))['route_km']['map'] / 239.12


def basemap(ax, ext, rivers=True, admin=True):
    ax.set_facecolor(C_SEA)
    for f in lay('admin_0_countries'):
        bgd = f['properties']['ADM0_A3'] == 'BGD'
        for r in rings(f['geometry']):
            ax.add_patch(Polygon(r, closed=True, fc=C_LAND if bgd else '#e6e4df', ec='0.55', lw=0.5, zorder=1))
    if admin:
        for f in lay('admin_1_states_provinces'):
            for r in rings(f['geometry']): ax.add_patch(Polygon(r, closed=True, fill=False, ec='0.78', lw=0.35, zorder=2))
    if rivers:
        for f in lay('rivers_lake_centerlines'):
            for l in lines(f['geometry']): a = np.array(l); ax.plot(a[:, 0], a[:, 1], color='#a9c6dc', lw=0.6, zorder=2)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect(1 / math.cos(math.radians((ext[2] + ext[3]) / 2)), adjustable='datalim')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(True); s.set_color('0.6'); s.set_linewidth(0.6)


def scalebar(ax, x, y, km, lat, label=None, lw=1.2):
    dlon = km / (111.32 * math.cos(math.radians(lat)))
    ax.plot([x, x + dlon], [y, y], color='0.15', lw=lw, solid_capstyle='butt', zorder=8)
    ax.text(x + dlon / 2, y, (label or f'{km:g} km') + '\n', ha='center', va='center', fontsize=6, zorder=8)


def city(ax, name, dx=0.04, dy=0.0, ha='left', size=6.5):
    f = [f for f in lay('populated_places_simple') if f['properties']['name'] == name][0]; lo, la = f['geometry']['coordinates']
    ax.plot(lo, la, 's', ms=3.2, color='0.15', zorder=7)
    ax.text(lo + dx, la + dy, name, fontsize=size, ha=ha, va='center', color='0.15', zorder=7)
    return lo, la


# ============================================================== Figure: study areas
fig = plt.figure(figsize=(7.2, 4.3))
axA = fig.add_axes([0.01, 0.06, 0.47, 0.86]); basemap(axA, (88.0, 92.75, 20.6, 26.7))
lon, lat = CH.lon.values, CH.lat.values
cov = HK.set_index('km').covered_km.reindex(np.arange(0, 240)).fillna(0).values
seg_cov = cov[np.clip((CH.km.values).astype(int), 0, 239)] > 0
P = np.c_[lon, lat]; S = np.stack([P[:-1], P[1:]], 1)
axA.add_collection(LineCollection(S[~seg_cov[:-1]], colors='0.55', lw=2.6, zorder=5))
axA.add_collection(LineCollection(S[seg_cov[:-1]], colors=C_TREE, lw=3.0, zorder=6))
HP = pd.read_csv(f'{D}/highway_photos.csv', dtype={'id': str})
for sp_, mk, col in (('test', 'o', C_OURS),):
    q = HP[HP.split == sp_]; axA.scatter(q.lon, q.lat, s=1.5, marker=mk, color=col, lw=0, zorder=7)
for nm, dx, dy, ha in (('Dhaka', -0.06, 0.12, 'right'), ('Chattogram', 0.08, 0.0, 'left'), ('Comilla', 0.08, 0.0, 'left'),
                       ('Sylhet', 0.06, 0.0, 'left'), ('Rajshahi', 0.06, 0.0, 'left'), ('Khulna', 0.06, 0.0, 'left'),
                       ('Mymensingh', 0.06, 0.0, 'left'), ('Barishal', 0.06, 0.0, 'left')):
    city(axA, nm, dx, dy, ha)
FS = pd.read_csv(f'{D}/field_gps_vs_aruler.csv')
axA.scatter([FS.user_lon.mean()], [FS.user_lat.mean()], s=60, marker='*', color=C_HOT, ec='white', lw=0.5, zorder=9)
axA.text(FS.user_lon.mean() - 0.07, FS.user_lat.mean() - 0.18, 'field photos', fontsize=6.5, color=C_HOT, ha='right', va='center', zorder=9)
axA.text(90.05, 21.25, 'Bay of Bengal', fontsize=6.5, color='#5d7f99', style='italic', ha='center')
axA.text(88.15, 26.55, 'INDIA', fontsize=6.5, color='0.45', va='top'); axA.text(92.6, 21.0, 'MYANMAR', fontsize=6, color='0.45', ha='right', rotation=90)
leg = [mpl.lines.Line2D([], [], color=C_TREE, lw=2.4, label='Dhaka–Chattogram route, mapped'),
       mpl.lines.Line2D([], [], color='0.55', lw=2.0, label='route, no usable photos'),
       mpl.lines.Line2D([], [], marker='o', ls='', ms=3, color=C_OURS, label='highway test photos (150)'),
       mpl.lines.Line2D([], [], marker='*', ls='', ms=7, color=C_HOT, label='GPS-tagged field photos (Dhaka)'),
       mpl.lines.Line2D([], [], marker='s', ls='', ms=3, color='0.15', label='division headquarters / city')]
axA.legend(handles=leg, loc='upper right', fontsize=6, frameon=True, framealpha=0.92, edgecolor='0.8', borderpad=0.5, handlelength=1.8)
scalebar(axA, 88.25, 20.95, 100, 21.0)
axA.set_title('Bangladesh: highway application and field test', loc='left', fontsize=8, pad=4)
letter(fig, axA, 'a', dx=-0.005, dy=0.04)

# Pasadena
axB = fig.add_axes([0.52, 0.06, 0.47, 0.86])
dd = pg.dd; inv = pd.concat([pd.DataFrame(dd[s]) for s in ('train', 'val', 'test')], ignore_index=True)
axB.scatter(inv.lng, inv.lat, s=0.35, color='0.35', lw=0, alpha=0.7, rasterized=True, zorder=2)
for r in pg.rois:
    a, b, c, d = r['roi']; nm, col = SPLIT[r['train_val_test']]
    axB.add_patch(Rectangle((min(b, d), min(a, c)), abs(d - b), abs(c - a), fill=False, ec=col, lw=1.6 if r['train_val_test'] == 2 else 1.1, zorder=4))
pl = np.array([[float(v['Location']['lng']), float(v['Location']['lat'])] for v in pg.panos.values()])
x0, x1 = np.percentile(pl[:, 0], [0.2, 99.8]); y0, y1 = np.percentile(pl[:, 1], [0.2, 99.8])
padx, pady = (x1 - x0) * 0.04, (y1 - y0) * 0.04
axB.set_xlim(x0 - padx, x1 + padx); axB.set_ylim(y0 - pady * 4, y1 + pady); axB.set_aspect(1 / math.cos(math.radians(34.15)), adjustable='datalim', anchor='N')
axB.set_xticks([]); axB.set_yticks([]); axB.set_facecolor('white')
for s in axB.spines.values(): s.set_color('0.6'); s.set_linewidth(0.6)
leg = [mpl.patches.Patch(fill=False, ec=SPLIT[k][1], lw=1.4, label=f'{SPLIT[k][0]} areas') for k in (0, 1, 2)] + \
      [mpl.lines.Line2D([], [], marker='o', ls='', ms=2, color='0.55', label='city inventory records')]
axB.legend(handles=leg, loc='lower left', fontsize=6, frameon=True, framealpha=0.92, edgecolor='0.8')
scalebar(axB, x1 - padx - 2 / (111.32 * math.cos(math.radians(34.15))), y1 - pady * 0.2, 2, 34.15)
axB.set_title('Pasadena, California: benchmark areas', loc='left', fontsize=8, pad=4)
letter(fig, axB, 'b', dx=-0.005, dy=0.04)
fig.savefig('fig_study_area.png', dpi=300); fig.savefig('fig_study_area.pdf', dpi=300); plt.close(fig)

# ============================================================== Figure: Pasadena street excerpt
inv_t = pg.load_inventory(); trees = inv_t[~inv_t.cn.isin(pg.NONTREE)].reset_index(drop=True); vac = inv_t[inv_t.cn.isin(pg.VACANT)]
pred = pd.read_csv(f'{R}/pas_student_test_trees.csv'); pv2 = pred[pred.n_views >= 2].reset_index(drop=True)
ET, EP, EV = pg.enu(trees.lat, trees.lng), pg.enu(pv2.lat, pv2.lng), pg.enu(vac.lat, vac.lng)
IA, PA = pg.assign(ET), pg.assign(EP)
PANO = '91aRv1LLZuLyb9d-tb5r_g'; c0 = np.asarray(pg.xy[PANO]); W2, H2 = 330, 262
cx, cy = c0[0] + 170, c0[1] - 25
inwin = lambda E: (np.abs(E[:, 0] - cx) <= W2) & (np.abs(E[:, 1] - cy) <= H2)


def offset_line(P, s0, s1, sP, side, off):
    s = np.linspace(s0, s1, max(2, int((s1 - s0) / 2) + 2))
    x = np.interp(s, sP, P[:, 0]); y = np.interp(s, sP, P[:, 1]); dx, dy = np.gradient(x), np.gradient(y); n = np.hypot(dx, dy) + 1e-9
    return np.c_[x - side * off * dy / n, y + side * off * dx / n]


def runs(mask, g):
    out, st = [], None
    for k, mm in enumerate(np.r_[mask, False]):
        if mm and st is None: st = k
        if not mm and st is not None: out.append((g[st], g[k - 1])); st = None
    return out


Sp = 12.0
fig = plt.figure(figsize=(7.2, 6.0)); ax = fig.add_axes([0.01, 0.01, 0.98, 0.86])
gold, dark, nflag = [], [], 0
for ci, ch in enumerate(pg.chains):
    P = np.array([pg.xy[k] for k in ch])
    if not inwin(P).any(): continue
    sP = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]; L = sP[-1]
    ax.plot(P[:, 0], P[:, 1], color='0.86', lw=6.5, solid_capstyle='round', zorder=1)
    for sd in (1, -1):
        si = IA.s[(IA.chain == ci) & (IA.side == sd)].values; sp = PA.s[(PA.chain == ci) & (PA.side == sd)].values
        g, mi = pg.plantable_mask(si, L, Sp); _, mp = pg.plantable_mask(sp, L, Sp)
        for a, b in runs(mp, g): gold.append(offset_line(P, a, b, sP, sd, 9.0))
        for a, b in runs(mi, g): dark.append(offset_line(P, a, b, sP, sd, 12.5))
for Q in gold: ax.plot(Q[:, 0], Q[:, 1], color=C_GAP, lw=2.6, solid_capstyle='butt', zorder=3)
for Q in dark: ax.plot(Q[:, 0], Q[:, 1], color=C_INV, lw=0.9, zorder=3)
m = inwin(ET); ax.scatter(ET[m, 0], ET[m, 1], s=11, facecolor='none', edgecolor=C_INV, lw=0.6, zorder=4)
m = inwin(EP); ax.scatter(EP[m, 0], EP[m, 1], s=6, color=C_OURS, lw=0, zorder=5)
m = inwin(EV); ax.scatter(EV[m, 0], EV[m, 1], s=22, marker='^', color=C_TREE, ec='white', lw=0.4, zorder=6)
ax.set_xlim(cx - W2, cx + W2); ax.set_ylim(cy - H2, cy + H2); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_color('0.6'); s.set_linewidth(0.6)
ax.plot([cx - W2 + 20, cx - W2 + 120], [cy - H2 + 14] * 2, color='0.15', lw=1.4); ax.text(cx - W2 + 70, cy - H2 + 20, '100 m', ha='center', va='bottom', fontsize=6.5)
ax.annotate('', xy=(cx + W2 - 18, cy + H2 - 20), xytext=(cx + W2 - 18, cy + H2 - 60), arrowprops=dict(arrowstyle='-|>', lw=0.9, color='0.15'))
ax.text(cx + W2 - 18, cy + H2 - 66, 'N', ha='center', va='top', fontsize=7)
leg = [mpl.lines.Line2D([], [], marker='o', ls='', ms=3, color=C_OURS, label='mapped tree (seen in ≥ 2 panoramas)'),
       mpl.lines.Line2D([], [], marker='o', ls='', ms=4, mfc='none', mec=C_INV, label='city inventory tree'),
       mpl.lines.Line2D([], [], marker='^', ls='', ms=5, color=C_TREE, label="city record: vacant planting site"),
       mpl.lines.Line2D([], [], color=C_GAP, lw=2.6, label=f'empty kerb ≥ {Sp:.0f} m from our map'),
       mpl.lines.Line2D([], [], color=C_INV, lw=0.9, label=f'empty kerb ≥ {Sp:.0f} m from the inventory'),
       mpl.lines.Line2D([], [], color='0.86', lw=5, label='street (panorama network)')]
ax.legend(handles=leg, loc='lower left', fontsize=6.5, frameon=False, ncol=3, bbox_to_anchor=(0.0, 1.0), borderaxespad=0.3, columnspacing=1.2)
fig.text(0.01, 0.982, 'Pasadena test area (excerpt): mapped trees, city inventory and empty kerb length on each side of the street',
         fontsize=8, ha='left', va='center')
fig.savefig('fig_pasadena_map.png', dpi=300); fig.savefig('fig_pasadena_map.pdf', dpi=300); plt.close(fig)

# ============================================================== Figure: highway on the map
ext = (90.30, 91.92, 22.22, 23.82)
fig = plt.figure(figsize=(7.2, 4.0))
axs = [fig.add_axes([0.01, 0.02, 0.48, 0.86]), fig.add_axes([0.51, 0.02, 0.48, 0.86])]
kmseg = np.clip(CH.km.values[:-1].astype(int), 0, 239)
tpk = HK.set_index('km').trees_per_km.reindex(np.arange(240)).values
ls_km = np.full(240, np.nan); st = LS.set_index('stretch').lst_median_c
for s_, v_ in st.items():
    k_ = int((s_ + 0.5) * scale)
    if 0 <= k_ < 240: ls_km[k_] = v_
ls_seg = np.interp(CH.km.values[:-1], (LS.stretch.values + 0.5) * scale, LS.lst_median_c.values)
trunc = lambda n, a, b: mpl.colors.LinearSegmentedColormap.from_list(n + '_t', mpl.colormaps[n](np.linspace(a, b, 256)))
cm_t = trunc('YlGn', 0.25, 1.0); nt = mpl.colors.Normalize(0, 40)
cm_l = trunc('YlOrRd', 0.15, 1.0); nl = mpl.colors.Normalize(32, 40)
hot = HR[HR.heat_rank <= 10]
for ax, kind in zip(axs, ('trees', 'lst')):
    basemap(ax, ext, rivers=True, admin=True)
    if kind == 'trees':
        v = tpk[kmseg]; ok = ~np.isnan(v)
        ax.add_collection(LineCollection(S[~ok], colors='0.7', lw=2.0, zorder=5))
        ax.add_collection(LineCollection(S[ok], colors=cm_t(nt(v[ok])), lw=3.2, zorder=6))
        sm = mpl.cm.ScalarMappable(nt, cm_t); lab = 'Trees per km (as detected)'
    else:
        ax.add_collection(LineCollection(S, colors=cm_l(nl(ls_seg)), lw=3.2, zorder=6))
        sm = mpl.cm.ScalarMappable(nl, cm_l); lab = 'Hot-season surface temperature (°C)'
    hx, hy = k2ll(hot.km.values + 0.5)
    ax.scatter(hx, hy, s=42, facecolor='none', edgecolor='0.1', lw=0.9, zorder=8)
    for nm, dx, dy, ha in (('Dhaka', 0.05, 0.06, 'left'), ('Narayanganj', 0.0, -0.055, 'center'), ('Comilla', 0.05, 0.03, 'left'), ('Chattogram', -0.05, -0.05, 'right')):
        city(ax, nm, dx, dy, ha)
    b = ax.get_position(); cax = fig.add_axes([b.x0 + b.width * 0.06, b.y0 + b.height * 0.30, b.width * 0.38, 0.022])
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal', extend='both' if kind == 'lst' else 'max'); cb.set_label(lab, fontsize=6.5); cb.ax.tick_params(labelsize=6)
    scalebar(ax, ext[0] + 0.08, ext[2] + 0.10, 20, ext[2])
axs[0].set_title('Tree density (grey: no usable photos)', loc='left', fontsize=8, pad=4)
axs[1].set_title('Hot-season land surface temperature', loc='left', fontsize=8, pad=4)
for a_ in axs: a_.text(0.06, 0.40, '○  one of the 10 hottest mapped kilometres', transform=a_.transAxes, fontsize=6.5, va='bottom')
letter(fig, axs[0], 'a', dy=0.045); letter(fig, axs[1], 'b', dy=0.045)
fig.savefig('fig_highway_map.png', dpi=300); fig.savefig('fig_highway_map.pdf', dpi=300); plt.close(fig)
print('maps ok', len(gold), len(dark), int(seg_cov.mean() * 100), '% route segments covered')

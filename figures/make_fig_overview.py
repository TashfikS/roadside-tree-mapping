"""Figure 1: what goes in, what each step does, and what it is checked against (real data throughout).

Row 1  the three image sources (Bangladesh street-view training photos, Pasadena panoramas with the city inventory,
       crowd-sourced Mapillary photos of the Dhaka-Chattogram highway).
Row 2  the four steps on real examples: trunk-base point, ground-plane range, merging views, empty kerb length.
Row 3  the independent answer key used for each output, with the headline result (values from the result files).
Also writes graphical_abstract.png (same panels, one row).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'src')]
os.chdir(HERE)
from figstyle import apply_figure_style

import os, sys, json, math
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch, Arc
from PIL import Image

os.environ.setdefault('GAP_AREA', '2'); os.environ.setdefault('GAP_ASSIGN_MAX', '12')
import pas_gaps as pg  # noqa: E402

THESIS = os.environ.get('THESIS_DIR', 'data/external')
BD = os.environ.get('BD_DATASET_DIR', 'data/external/roadside-trees-of-bangladesh-street-view')
C_OURS, C_INV, C_GAP, C_TREE, C_KEY, C_BOX = '#1f5f99', '0.25', '#d99a00', '#2f7d4f', '#d4237a', '#ffd400'
mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7,
                     'xtick.labelsize': 6, 'ytick.labelsize': 6, 'savefig.dpi': 300, 'pdf.fonttype': 42})

# ---------------------------------------------------------------- data
TR = pd.read_csv('../data/bangladesh_test_trunk_review.csv'); TR['img'] = TR.key.str.rsplit('/', n=1).str[0]
AV = json.load(open('../results/audit/audit_values.json'))
HO = json.load(open('../results/hw_outputs.json'))
HT = json.load(open('../results/hw_test_results.json')) if os.path.exists('../results/hw_test_results.json') else {}


def bd_img(img):
    f, i = img.split('/')
    return Image.open(f'{BD}/{"Test" if f.startswith("BD_Test") else "Train"}/{f}/{i}.jpg').convert('RGB')


def fit(ax, im, cx, cy, frac=1.0):
    """Largest crop of im with the axes' aspect ratio, centred near (cx, cy) in pixels, scaled by frac."""
    fig = ax.figure; b = ax.get_position(); a = (b.width * fig.get_figwidth()) / (b.height * fig.get_figheight())
    W, H = im.size; h = min(H, W / a) * frac; w = h * a
    x0 = int(np.clip(cx - w / 2, 0, W - w)); y0 = int(np.clip(cy - h / 2, 0, H - h))
    return (x0, y0, int(x0 + w), int(y0 + h))


def show(ax, im, box=None):
    if box: im = im.crop(box)
    ax.imshow(im, aspect='auto'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    return box[0] if box else 0, box[1] if box else 0


def letter(fig, ax, L, dx=-0.012, dy=0.012):
    b = ax.get_position(); fig.text(b.x0 + dx, b.y1 + dy, L, fontsize=9, fontweight='bold', ha='right', va='bottom')


# Pasadena example street (chain 222, La Loma Rd, test area); mapped trees seen in >= 2 views (gap setting)
inv = pg.load_inventory(); trees = inv[~inv.cn.isin(pg.NONTREE)].reset_index(drop=True)
pred = pd.read_csv('../results/pas_student_test_trees.csv'); pv2 = pred[pred.n_views >= 2].reset_index(drop=True)
PANO = '91aRv1LLZuLyb9d-tb5r_g'   # La Loma Rd, Pasadena test area
CI = max((i for i, c in enumerate(pg.chains) if PANO in c), key=lambda i: pg.chain_len[i]); chain = pg.chains[CI]; Pc = np.array([pg.xy[k] for k in chain])
sP = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Pc, axis=0), axis=1))]; Lc = sP[-1]
IA = pg.assign(pg.enu(trees.lat, trees.lng)); PA = pg.assign(pg.enu(pv2.lat, pv2.lng))
Ti = pg.enu(trees.lat, trees.lng)[(IA.chain == CI).values]; Tp = pg.enu(pv2.lat, pv2.lng)[(PA.chain == CI).values]
nv = pv2.n_views.values[(PA.chain == CI).values]
c0 = Pc.mean(0); U, Sv, Vt = np.linalg.svd(Pc - c0); Rm = Vt  # rotate street to horizontal
rot = lambda X: (np.asarray(X) - c0) @ Rm.T
Pr, Tir, Tpr = rot(Pc), rot(Ti), rot(Tp)
if Pr[-1, 0] < Pr[0, 0]: Pr, Tir, Tpr = Pr * [-1, 1], Tir * [-1, 1], Tpr * [-1, 1]


def side_line(s0, s1, side, off):
    s = np.linspace(s0, s1, max(2, int((s1 - s0) / 2) + 2))
    x = np.interp(s, sP, Pc[:, 0]); y = np.interp(s, sP, Pc[:, 1])
    dx, dy = np.gradient(x), np.gradient(y); n = np.hypot(dx, dy) + 1e-9
    nx, ny = -dy / n, dx / n                                  # left normal of travel direction
    Q = np.c_[x + side * off * nx, y + side * off * ny]; Q = rot(Q)
    return Q * ([-1, 1] if rot(Pc)[-1, 0] < rot(Pc)[0, 0] else [1, 1])


def runs(mask, g):
    out, st = [], None
    for k, m in enumerate(np.r_[mask, False]):
        if m and st is None: st = k
        if not m and st is not None: out.append((g[st], g[k - 1])); st = None
    return out


S = 12.0
gaps = {}
for sd in (1, -1):
    si = IA.s[(IA.chain == CI) & (IA.side == sd)].values; sp = PA.s[(PA.chain == CI) & (PA.side == sd)].values
    g, mi = pg.plantable_mask(si, Lc, S); _, mp = pg.plantable_mask(sp, Lc, S)
    gaps[sd] = dict(pred=runs(mp, g), inv=runs(mi, g))

# ---------------------------------------------------------------- panel painters


def tile_bd(ax):
    img = 'BD_740/1514'; im = bd_img(img); W, H = im.size
    x0, y0 = show(ax, im, fit(ax, im, W / 2, 0.52 * H))
    for r in TR[TR.img == img].itertuples():
        ax.add_patch(Rectangle((r.x0b - x0, r.y0b - y0), r.x1b - r.x0b, r.y1b - r.y0b, fill=False, ec=C_BOX, lw=0.9))


def tile_pano(ax):
    k = PANO; yaw, *_ = pg_info(k)
    im = Image.open(f'{pg.D}/streetview/{k}_z2.jpg').convert('RGB'); W, H = im.size
    la, lo = float(pg.panos[k]['Location']['lat']), float(pg.panos[k]['Location']['lng'])
    d = pg.enu(trees.lat, trees.lng) - pg.enu(la, lo); D = np.hypot(d[:, 0], d[:, 1]); m = (D > 3) & (D < 25)
    beta = np.degrees(np.arctan2(d[:, 0], d[:, 1]))
    u = ((np.radians(beta - yaw) + math.pi) % (2 * math.pi)) / (2 * math.pi) * W; v = H / 2 + np.arctan(3.0 / D) * H / math.pi
    box = fit(ax, im, 0.5 * W, 0.40 * H, frac=0.80); x0, y0 = show(ax, im, box)
    m = m & (u >= box[0] + 8) & (u <= box[2] - 8) & (v >= box[1] + 8) & (v <= box[3] - 8)
    ax.scatter(u[m] - x0, v[m] - y0, s=55, facecolor='none', edgecolor='white', lw=2.2, zorder=3)
    ax.scatter(u[m] - x0, v[m] - y0, s=55, facecolor='none', edgecolor='black', lw=0.8, zorder=4)
    ax.set_xlim(-0.5, box[2] - box[0] - 0.5); ax.set_ylim(box[3] - box[1] - 0.5, -0.5)


def pg_info(k):
    import ast
    v = pg.panos[k]; pr = v['Projection'] if isinstance(v['Projection'], dict) else ast.literal_eval(v['Projection'])
    return float(pr['pano_yaw_deg']), float(pr.get('tilt_pitch_deg', 0) or 0)


def tile_hw(ax):
    i = '1000078231284036'; im = Image.open(os.path.join(os.environ.get('HIGHWAY_PHOTO_DIR', '../data/external/highway_photos'), f'{i}.jpg')).convert('RGB'); W, H = im.size
    x0, y0 = show(ax, im, fit(ax, im, W / 2, 0.58 * H))
    P = pd.read_csv('../results/hw_test_preds_hwb_s0.csv', dtype={'id': str}); P = P[(P.id == i) & (P.score >= 0.25)]
    for r in P.itertuples():
        ax.add_patch(Rectangle((r.x0 * W - x0, r.y0 * H - y0), (r.x1 - r.x0) * W, (r.y1 - r.y0) * H, fill=False, ec=C_BOX, lw=0.9))


def step_detect(ax):
    img = 'BD_740/1311'; im = bd_img(img); W, H = im.size
    r = TR[TR.img == img].sort_values('x0b').iloc[1]
    bw, bh = r.x1b - r.x0b, r.y1b - r.y0b
    x0, y0 = show(ax, im, fit(ax, im, (r.x0b + r.x1b) / 2, r.y0b + 0.62 * bh, frac=min(1.0, 1.32 * bh / H)))
    ax.add_patch(Rectangle((r.x0b - x0, r.y0b - y0), bw, bh, fill=False, ec=C_BOX, lw=1.0))
    ax.plot([r.bu - x0], [r.bv - y0], marker='x', ms=6, mew=1.6, color='white', zorder=4)
    ax.plot([r.bu - x0], [r.bv - y0], marker='x', ms=6, mew=0.6, color='black', zorder=5)
    ax.scatter([r.pu - x0], [r.pv - y0], s=26, color=C_KEY, edgecolor='white', lw=0.7, zorder=6)
    ax.text(0.03, 0.04, 'box bottom', transform=ax.transAxes, color='white', fontsize=6, va='bottom',
            bbox=dict(boxstyle='square,pad=0.15', fc='black', ec='none', alpha=0.55))
    ax.text(0.97, 0.04, 'trunk base', transform=ax.transAxes, color=C_KEY, fontsize=6, va='bottom', ha='right', fontweight='bold',
            bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none', alpha=0.85))


def step_locate(ax):
    h, xt = 3.0, 15.5
    ax.set_xlim(-1.6, 19.5); ax.set_ylim(-2.4, 12.6); ax.set_aspect('equal', adjustable='datalim'); ax.axis('off')
    ax.add_patch(Rectangle((-1.2, -0.45), 20.7, 0.45, color='0.85', lw=0))
    ax.plot([-1.2, 19.5], [0, 0], color='0.45', lw=0.8)
    ax.plot([0.6, 0.6], [0, h], color='0.3', lw=1.2); ax.add_patch(Rectangle((0.25, h - 0.25), 0.9, 0.55, color='0.2'))
    ax.add_patch(Rectangle((xt - 0.22, 0), 0.44, 3.3, color='#6b4f2a', lw=0))
    ax.add_patch(Circle((xt, 5.2), 2.3, color=C_TREE, lw=0, alpha=0.9))
    ax.plot([1.15, 12.8], [h + 0.02, h + 0.02], ls=(0, (3, 2)), color='0.55', lw=0.7)
    ax.text(12.6, h + 0.25, 'horizon', ha='right', va='bottom', fontsize=6, color='0.4')
    ax.annotate('', xy=(xt, 0.02), xytext=(1.15, h), arrowprops=dict(arrowstyle='-|>', color=C_OURS, lw=1.2, mutation_scale=7))
    ax.scatter([xt], [0.02], s=22, color=C_KEY, edgecolor='white', lw=0.6, zorder=5)
    ax.add_patch(Arc((1.15, h), 7.0, 7.0, theta1=-math.degrees(math.atan(h / (xt - 1.15))), theta2=0, color=C_OURS, lw=0.8))
    ax.text(4.9, h - 0.62, 'α', color=C_OURS, fontsize=7)
    ax.annotate('', xy=(-0.5, 0), xytext=(-0.5, h), arrowprops=dict(arrowstyle='<->', lw=0.7, color='0.2', shrinkA=0, shrinkB=0))
    ax.text(-0.75, h / 2, 'h', ha='right', va='center', fontsize=7, style='italic')
    ax.annotate('', xy=(0.6, -1.1), xytext=(xt, -1.1), arrowprops=dict(arrowstyle='<->', lw=0.7, color='0.2', shrinkA=0, shrinkB=0))
    ax.text((xt + 0.6) / 2, -1.25, 'D', ha='center', va='top', fontsize=7, style='italic')
    ax.text(9.0, 11.6, r'$D = h\,/\tan\alpha$', ha='center', va='center', fontsize=8, color=C_OURS)
    ax.text(9.0, 9.9, 'camera height known;\ntree height not needed', ha='center', va='center', fontsize=6, color='0.3', linespacing=1.1)


def street_axes(ax, pad=14):
    ax.set_aspect('equal', adjustable='datalim'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    xs = Pr[:, 0]; ax.set_xlim(xs.min() - 6, xs.max() + 6); ax.set_ylim(Pr[:, 1].min() - pad - 8, Pr[:, 1].max() + pad + 8)
    ax.plot(Pr[:, 0], Pr[:, 1], color='0.82', lw=7, solid_capstyle='round', zorder=0)


def step_merge(ax):
    street_axes(ax)
    ax.scatter(Pr[:, 0], Pr[:, 1], marker='s', s=5, color='0.45', zorder=2, lw=0)
    ax.scatter(Tir[:, 0], Tir[:, 1], s=14, facecolor='none', edgecolor=C_INV, lw=0.6, zorder=3)
    ax.scatter(Tpr[:, 0], Tpr[:, 1], s=8, color=C_OURS, zorder=4, lw=0)
    x1 = Pr[:, 0].max() - 52
    ax.plot([x1, x1 + 50], [Pr[:, 1].min() - 19] * 2, color='0.2', lw=1.0)
    ax.text(x1 + 25, Pr[:, 1].min() - 21, '50 m', ha='center', va='top', fontsize=6)


def step_gaps(ax):
    street_axes(ax)
    ax.scatter(Tir[:, 0], Tir[:, 1], s=14, facecolor='none', edgecolor=C_INV, lw=0.6, zorder=3)
    ax.scatter(Tpr[:, 0], Tpr[:, 1], s=8, color=C_OURS, zorder=4, lw=0)
    for sd in (1, -1):
        for a, b in gaps[sd]['pred']:
            Q = side_line(a, b, sd, 10.5); ax.plot(Q[:, 0], Q[:, 1], color=C_GAP, lw=2.6, solid_capstyle='butt', zorder=2)
        for a, b in gaps[sd]['inv']:
            Q = side_line(a, b, sd, 14.0); ax.plot(Q[:, 0], Q[:, 1], color=C_INV, lw=0.9, zorder=2)


def answer_band(fig, rect):
    ax = fig.add_axes(rect); ax.axis('off'); ax.set_xlim(0, 4); ax.set_ylim(0, 1)
    e2 = AV['e2_point']['retrained (final)']
    items = [('City tree inventory, Pasadena', f"map score AP@4 m {e2['AP4_official']:.2f}\nempty length 49 % truly empty\n(17 % by chance)", 'positions, gaps'),
             ('Field distances, Bangladesh', 'one phone photo: 2.3 m median error\n(1.4 m after scale correction);\nphone GPS: 7.0 m', '22 trees, ARuler'),
             ('Marked trees, highway photos', f"{HO['recall_used']*100:.0f} % found by the adapted detector\n(38 % before adaptation)\n→ trees per km", '356 trees, 83 photos'),
             ('Satellite canopy + Landsat heat', f"tree density vs canopy ρ = {HO['heat']['spearman_trees_canopy_1km']:.2f}\nhottest 10 km: {1.7:.1f} vs {22.5:.1f} trees/km\nin the coolest 10 km", '128 km stretches')]
    for k, (t, body, sub) in enumerate(items):
        x = k + 0.04
        ax.add_patch(FancyBboxPatch((x, 0.05), 0.92, 0.80, boxstyle='round,pad=0,rounding_size=0.04', fc='#f3f6fa', ec='#c9d3df', lw=0.6))
        ax.text(x + 0.46, 0.76, t, ha='center', va='top', fontsize=6.2, fontweight='bold', color='0.15')
        ax.text(x + 0.46, 0.52, body, ha='center', va='center', fontsize=6, color='0.15', linespacing=1.15)
        ax.text(x + 0.46, 0.12, sub, ha='center', va='bottom', fontsize=5.6, color='0.45', style='italic')
    return ax


def arrow(fig, x0, x1, y):
    fig.patches.append(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, arrowstyle='-|>', mutation_scale=9, lw=1.0, color='0.45'))


# ---------------------------------------------------------------- Figure 1
fig = plt.figure(figsize=(7.2, 5.6))
L, R_, gap = 0.035, 0.99, 0.02
w3 = (R_ - L - 2 * gap) / 3
row1 = [fig.add_axes([L + k * (w3 + gap), 0.655, w3, 0.255]) for k in range(3)]
tile_bd(row1[0]); tile_pano(row1[1]); tile_hw(row1[2])
heads1 = [('Bangladesh street photos', 'benchmark: 1,765 images, 5,311 boxed trees'),
          ('Pasadena panoramas', 'test bed: ○ city-inventory trees projected'),
          ('Dhaka–Chattogram highway', 'application: Mapillary photos, 240 km')]
for ax, (t, sub), Lt in zip(row1, heads1, 'abc'):
    ax.set_title(r'$\bf{' + Lt + r'}$  ' + t, loc='left', fontsize=7.2, pad=10)
    ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=6, color='0.35', va='bottom')
fig.text(L, 0.965, 'Data', fontsize=8, fontweight='bold', color=C_OURS, va='center')

w4 = (R_ - L - 3 * 0.035) / 4
row2 = [fig.add_axes([L + k * (w4 + 0.035), 0.255, w4, 0.31]) for k in range(4)]
step_detect(row2[0]); step_locate(row2[1]); step_merge(row2[2]); step_gaps(row2[3])
heads2 = [('Trunk base', 'one ground point per tree'),
          ('Range', 'from the known camera height'),
          ('Merge views into trees', '● mapped  ○ inventory  ▪ panorama'),
          ('Empty kerb', f'gold: ours, dark: inventory (S = {S:.0f} m)')]
for ax, (t, sub), Lt in zip(row2, heads2, 'defg'):
    ax.set_title(r'$\bf{' + Lt + r'}$  ' + t, loc='left', fontsize=7.2, pad=10)
    ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=5.6, color='0.35', va='bottom')
fig.text(L, 0.615, 'Pipeline (no tree height, no depth network)', fontsize=8, fontweight='bold', color=C_OURS, va='center')
for k in range(3):
    b0, b1 = row2[k].get_position(), row2[k + 1].get_position(); arrow(fig, b0.x1 + 0.004, b1.x0 - 0.004, b0.y0 + b0.height / 2)

fig.text(L + 0.01, 0.205, 'Each output is checked against an independent answer key', fontsize=8, fontweight='bold', color=C_OURS, va='center')
band = answer_band(fig, [L - 0.01, 0.01, R_ - L + 0.01, 0.175]); fig.text(L - 0.005, 0.205, 'h', fontsize=7.5, fontweight='bold', ha='right', va='center')
fig.savefig('fig_overview.png', dpi=300); fig.savefig('fig_overview.pdf'); plt.close(fig)

# ---------------------------------------------------------------- graphical abstract (5 x 13 cm, one row)
ga = plt.figure(figsize=(13 / 2.54 * 1.6, 5 / 2.54 * 1.25))
ws = 0.18; xs = [0.012 + k * (ws + 0.018) for k in range(5)]
axs = [ga.add_axes([x, 0.05, ws, 0.72]) for x in xs]
tile_hw(axs[0]); step_detect(axs[1]); step_locate(axs[2]); step_gaps(axs[3])
ax = axs[4]; ax.axis('off')
ax.text(0.5, 0.86, 'Checked against', ha='center', fontsize=7, fontweight='bold')
for j, t in enumerate(['10,662 surveyed records', '22 field-measured distances', '356 marked highway trees', 'satellite canopy + heat']):
    ax.text(0.5, 0.68 - j * 0.17, t, ha='center', fontsize=6.4, color='0.2',
            bbox=dict(boxstyle='round,pad=0.3', fc='#f3f6fa', ec='#c9d3df', lw=0.6))
for a, t in zip(axs[:4], ['Street photos', 'Trunk base', 'Range from camera height', 'Empty kerb length']):
    a.set_title(t, fontsize=7.5, loc='left', pad=4)
for k in range(4):
    b0, b1 = axs[k].get_position(), axs[k + 1].get_position(); arrow(ga, b0.x1 + 0.002, b1.x0 - 0.002, b0.y0 + b0.height / 2)
ga.text(0.012, 0.94, 'Where are roadside trees missing? Street photos → validated tree map → empty kerb length → where to plant first',
        fontsize=7.5, fontweight='bold', color=C_OURS)
ga.savefig('graphical_abstract.png', dpi=300); plt.close(ga)
print('ok', len(Tir), len(Tpr), {k: (len(v['pred']), len(v['inv'])) for k, v in gaps.items()})

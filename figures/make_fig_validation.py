import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'src')]
os.chdir(HERE)
from figstyle import apply_figure_style
import json
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from PIL import Image as _I

# skill:figure-style kernel.py (auto-injected on skill load)
# skill:figure-composer kernel.py (auto-injected on skill load)
# These helpers must be available; loading them as in the trace
import importlib.util, sys, os

def apply_figure_style(*, frame='open', font=None, sizes=(8, 7, 6), grid=False):
    """Set matplotlib rcParams for publication-grade output. Call once before plotting."""
    base = sizes[0]
    mpl.rcParams.update({
        'font.size': base,
        'axes.titlesize': base,
        'axes.labelsize': sizes[1],
        'xtick.labelsize': sizes[2],
        'ytick.labelsize': sizes[2],
        'legend.fontsize': sizes[2],
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.spines.top': frame != 'open',
        'axes.spines.right': frame != 'open',
        'axes.grid': grid,
        'font.family': font if font else mpl.rcParams['font.family'],
    })

def panel_px(outline, letter, dpi=300, gutter_mm=4):
    ncol = outline['ncol']
    width_mm = outline['width_mm']
    row_heights_mm = outline['row_heights_mm']
    gutter_px = int(round(gutter_mm / 25.4 * dpi))
    total_w_px = int(round(width_mm / 25.4 * dpi))
    col_w = (total_w_px - (ncol - 1) * gutter_px) / ncol
    panel = next(p for p in outline['panels'] if p['letter'] == letter)
    row = panel['row']
    colspan = panel['colspan']
    col = panel['col']
    w = int(round(col_w * colspan + gutter_px * (colspan - 1)))
    h = int(round(row_heights_mm[row] / 25.4 * dpi))
    return w, h

from PIL import Image, ImageDraw, ImageFont


def grid_geom(outline, dpi=300, gutter_mm=4):
    mm = dpi/25.4
    W = int(outline["width_mm"]*mm); ncol = outline["ncol"]; g = int(gutter_mm*mm)
    colw = (W - g*(ncol-1)) // ncol
    rowh = [int(h*mm) for h in outline["row_heights_mm"]]
    row_y = [sum(rowh[:i]) + g*i for i in range(len(rowh))]
    return W, ncol, colw, rowh, row_y, g


def panel_xy(outline, letter, dpi=300, gutter_mm=4):
    W, ncol, colw, rowh, row_y, g = grid_geom(outline, dpi, gutter_mm)
    p = next(q for q in outline["panels"] if q["letter"]==letter)
    return p["col"]*(colw+g), row_y[p["row"]]


def compose_figure(outline, panel_paths, out_path, dpi=300, gutter_mm=4,
                   letter_font="DejaVuSans-Bold.ttf", letter_pt=9, letter_case="lower"):
    W, ncol, colw, rowh, row_y, g = grid_geom(outline, dpi, gutter_mm)
    H = row_y[-1] + rowh[-1]
    canvas = Image.new("RGB",(W,H),"white"); draw = ImageDraw.Draw(canvas)
    try: ft = ImageFont.truetype(letter_font, int(letter_pt/72*dpi))
    except Exception: ft = ImageFont.load_default()
    for p in outline["panels"]:
        L = p["letter"]; w,h = panel_px(outline,L,dpi,gutter_mm); x,y = panel_xy(outline,L,dpi,gutter_mm)
        im = Image.open(panel_paths[L]).convert("RGBA")
        if im.size != (w,h): im = im.resize((w,h))
        canvas.paste(im,(x,y),im)
        stamp = L.lower() if letter_case == "lower" else L.upper()
        draw.text((x+int(1.5/25.4*dpi), y+int(1/25.4*dpi)), stamp, fill="black", font=ft)
    canvas.save(out_path); return out_path,(W,H)


def compose_crops(outline, dpi=300, gutter_mm=4, pad_px=4):
    ncol = outline['ncol']
    width_mm = outline['width_mm']
    row_heights_mm = outline['row_heights_mm']
    gutter_px = int(round(gutter_mm / 25.4 * dpi))
    total_w_px = int(round(width_mm / 25.4 * dpi))
    col_w = (total_w_px - (ncol - 1) * gutter_px) / ncol
    row_h = [int(round(h / 25.4 * dpi)) for h in row_heights_mm]
    row_tops = []
    y = 0
    for h in row_h:
        row_tops.append(y)
        y += h + gutter_px
    crops = {}
    for panel in outline['panels']:
        letter = panel['letter']
        row = panel['row']
        col = panel['col']
        colspan = panel['colspan']
        x0 = int(round(col * (col_w + gutter_px)))
        y0 = row_tops[row]
        w = int(round(col_w * colspan + gutter_px * (colspan - 1)))
        h = row_h[row]
        crops[letter] = (x0, y0, x0 + w, y0 + h)
    return crops

# Load data
GTp = pd.read_csv('../results/pas_test_gt.csv')
p = json.load(open('../results/pas_e7_result.json'))
ps = json.load(open('../results/pas_e7_scaled_result.json'))

lat0 = 34.15; R_ = 6371008.8

def en(la, lo):
    return np.c_[np.radians(np.asarray(lo) + 118.13) * R_ * math.cos(math.radians(lat0)),
                 np.radians(np.asarray(la) - lat0) * R_]

G_ = en(GTp.lat, GTp.lng)
kg_ = cKDTree(G_)

curves = {}
for nm, f in (('ours', '../results/pas_student_test_trees.csv'),
              ('depth', '../results/pas_e7_test_trees.csv')):
    Pp = pd.read_csv(f)
    X_ = en(Pp.lat, Pp.lng)
    d_m2s, _ = kg_.query(X_)
    d_s2m, _ = cKDTree(X_).query(G_)
    curves[nm] = dict(m2s=d_m2s, s2m=d_s2m, n=len(Pp))

nd = ~GTp['diff'].astype(bool).values
for nm, f in (('ours', '../results/pas_student_test_trees.csv'),
              ('depth', '../results/pas_e7_test_trees.csv')):
    Pp = pd.read_csv(f)
    X_ = en(Pp.lat, Pp.lng)
    d_s2m, _ = cKDTree(X_).query(G_[nd])
    curves[nm]['s2m_nd'] = d_s2m

GV = pd.read_csv('../results/pas_student_gap_validation_test.csv')
GV['Sn'] = GV.S.astype(str).str.extract(r'^(\d+)')[0].astype(int)
GV['base'] = GV.S.astype(str).str.contains('baseline')
o_ = GV[~GV.base].sort_values('Sn')
b_ = GV[GV.base].sort_values('Sn')
v2 = json.load(open('../results/vlm2_result.json'))['test']
B_ = pd.read_csv('../results/e7_bd_results.csv')
e_o = (B_['D_h1.5'] - B_.aruler_m).abs().median()
e_d = (B_.D_da - B_.aruler_m).abs().median()

C_OURS = '#1f5f99'; C_DEPTH = '#c4572a'; C_BASE = '0.55'; C_VLM = '#2f7d4f'

OUT_ = {"claim": "Each stage is checked against an independent answer key: map positions, distances and planting gaps all beat their baselines.",
        "width_mm": 180, "ncol": 12, "row_heights_mm": [62, 62],
        "panels": [{"letter": "a", "role": "primary", "row": 0, "col": 0, "colspan": 6, "chart_family": "ECDF", "message": "Pasadena positions", "data_vid": None, "ask": ""},
                   {"letter": "b", "role": "evidence", "row": 0, "col": 6, "colspan": 6, "chart_family": "dot plot", "message": "Pasadena AP@4 m", "data_vid": None, "ask": ""},
                   {"letter": "c", "role": "evidence", "row": 1, "col": 0, "colspan": 6, "chart_family": "scatter", "message": "Bangladesh distances", "data_vid": None, "ask": ""},
                   {"letter": "d", "role": "evidence", "row": 1, "col": 6, "colspan": 6, "chart_family": "dot plot", "message": "Planting gaps", "data_vid": None, "ask": ""}]}

AVF = json.load(open('../results/audit/audit_values.json')); _PT = json.load(open('../results/q1_upgrades/pas_table.json'))
rows = [('Ours, retrained detector', AVF['e2_point']['retrained (final)']['AP4_official'], C_OURS, 'o', _PT['ci95']['retrained_box']['official']),
        ('Ours, first detector', AVF['e2_point']['first detector, validation-chosen merge']['AP4_official'], C_OURS, 'o', _PT['ci95']['first_detector']['official']),
        (f"Depth model × {ps['k']:.2f}\n(fitted on surveyed trees)", ps['test']['AP_4m'], C_DEPTH, 'o', None),
        ('Depth model, as is', p['test_depth']['AP_4m'], C_DEPTH, 'o', None),
        ('Branson et al. 2018,\nstreet view only*', 0.581, C_BASE, 'D', None)]

paths_ = {}

def newfig(L):
    w, h = panel_px(OUT_, L)
    fig = plt.figure(figsize=(w / 300, h / 300), dpi=300)
    return fig, w, h

apply_figure_style(sizes=(8, 7, 6))
mpl.rcParams['savefig.bbox'] = 'standard'

# Panel a
fig, w, h = newfig('a')
ax = fig.add_axes([0.15, 0.16, 0.82, 0.74])
xs = np.linspace(0, 10, 201)
for nm, col in (('ours', C_OURS), ('depth', C_DEPTH)):
    c_ = curves[nm]
    ax.plot(xs, [(c_['m2s'] <= x).mean() * 100 for x in xs], color=col, lw=1.4)
    ax.plot(xs, [(c_['s2m_nd'] <= x).mean() * 100 for x in xs], color=col, lw=1.1, ls='--')
ax.axvline(4, color='0.6', lw=0.7, ls=':')
ax.text(4.15, 40, '4 m', fontsize=6, color='0.4')
ax.text(0.35, 62, 'ours', color=C_OURS, fontsize=7, fontweight='bold')
ax.text(8.8, 26, 'depth model', color=C_DEPTH, fontsize=7, fontweight='bold', ha='center')
ax.text(0.985, 0.02, 'solid: mapped → nearest record\ndashed: scored tree → nearest mapped',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=6, color='0.3')
ax.set_xlim(0, 10)
ax.set_ylim(0, 100)
ax.set_xlabel('Distance to nearest tree (m)')
ax.set_ylabel('Share within distance (%)')
ax.set_title('Pasadena test area: map positions', loc='left')
fig.savefig('vf_a.png', dpi=300, transparent=True)
paths_['a'] = 'vf_a.png'
plt.close('all')

# Panel b
apply_figure_style(sizes=(8, 7, 6))
mpl.rcParams['savefig.bbox'] = 'standard'
fig, w, h = newfig('b')
ax = fig.add_axes([0.40, 0.20, 0.53, 0.70])
for i, (lab, v, col, mk, ci) in enumerate(rows):
    ax.plot([0, v], [i, i], color='0.85', lw=1.0, zorder=1)
    if ci is not None:
        ax.plot(ci, [i, i], color=col, lw=1.3, zorder=2, solid_capstyle='butt')
    ax.scatter([v], [i], color=col, marker=mk, s=22, zorder=3)
    ax.text((ci[1] if ci is not None else v) + 0.028, i, f'{v:.2f}', va='center', fontsize=6.5)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows], fontsize=6.5)
ax.invert_yaxis()
ax.set_xlim(0, 0.8)
ax.set_xlabel('AP@4 m')
ax.set_title('Pasadena test area: map score', loc='left')
fig.text(0.03, 0.012, '* reported value; whiskers: 95 % spatial block-bootstrap interval', fontsize=5.5, color='0.35', va='bottom')
fig.savefig('vf_b.png', dpi=300, transparent=True)
paths_['b'] = 'vf_b.png'
plt.close('all')

# Panel c
apply_figure_style(sizes=(8, 7, 6))
mpl.rcParams['savefig.bbox'] = 'standard'
fig, w, h = newfig('c')
ax = fig.add_axes([0.15, 0.16, 0.82, 0.74])
lim = max(B_.aruler_m.max(), B_.D_da.max(), B_['D_h1.5'].max()) * 1.05
ax.plot([0, lim], [0, lim], color='0.6', lw=0.8, ls=':')
ax.text(lim * 0.80, lim * 0.86, '1:1', fontsize=6, color='0.45')
ax.scatter(B_.aruler_m, B_['D_h1.5'], color=C_OURS, s=14)
ax.scatter(B_.aruler_m, B_.D_da, color=C_DEPTH, s=14, marker='s')
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.text(0.04, 0.96, f'ours: median error {e_o:.1f} m', transform=ax.transAxes, color=C_OURS, fontsize=6.5, va='top')
ax.text(0.04, 0.88, f'depth model: {e_d:.1f} m', transform=ax.transAxes, color=C_DEPTH, fontsize=6.5, va='top')
ax.text(0.96, 0.04, f'n = {len(B_)} trees, no calibration', transform=ax.transAxes, ha='right', fontsize=6, color='0.35')
ax.set_xlabel('ARuler distance (m)')
ax.set_ylabel('Distance from one photo (m)')
ax.set_title('Bangladesh field photos: distance', loc='left')
fig.savefig('vf_c.png', dpi=300, transparent=True)
paths_['c'] = 'vf_c.png'
plt.close('all')

# Panel d
apply_figure_style(sizes=(8, 7, 6))
mpl.rcParams['savefig.bbox'] = 'standard'
fig, w, h = newfig('d')
ax = fig.add_axes([0.15, 0.16, 0.62, 0.74])
ax.plot(o_.Sn, o_.precision * 100, '-o', color=C_OURS, ms=4, lw=1.2)
ax.plot(b_.Sn, b_.precision * 100, '-o', color=C_BASE, ms=4, lw=1.0)
ax.plot(o_.Sn, o_.recall * 100, '--', color=C_OURS, lw=1.0, marker='^', ms=4)
for y_, t_, c_ in ((o_.precision.iloc[-1] * 100, 'flagged space\nreally free', C_OURS),
                   (o_.recall.iloc[-1] * 100, 'free space\nfound', C_OURS),
                   (b_.precision.iloc[-1] * 100, 'chance\n(no detector)', C_BASE)):
    ax.text(15.35, y_, t_, color=c_, fontsize=6, va='center')
ax.set_xlim(9.5, 15.2)
ax.set_xticks([10, 12, 15])
ax.set_ylim(0, 100)
ax.set_xlabel('Planting spacing S (m)')
ax.set_ylabel('Share of road length (%)')
ax.set_title('Pasadena test area: planting gaps', loc='left')
ax.text(0.02, 0.97,
        f"VLM second look (candidate spots, S = 12 m):\n{v2['unfiltered'] * 100:.1f} % → {v2['precision'] * 100:.1f} % really free",
        transform=ax.transAxes, fontsize=6, color=C_VLM, va='top')
fig.savefig('vf_d.png', dpi=300, transparent=True)
paths_['d'] = 'vf_d.png'
plt.close('all')

compose_figure(OUT_, paths_, 'fig_validation.png')
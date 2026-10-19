import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'src')]
os.chdir(HERE)
from figstyle import apply_figure_style
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# skill:figure-composer kernel.py (auto-injected on skill load)
def grid_geom(outline, dpi=300, gutter_mm=4):
    mm = dpi/25.4
    W = int(outline["width_mm"]*mm); ncol = outline["ncol"]; g = int(gutter_mm*mm)
    colw = (W - g*(ncol-1)) // ncol
    rowh = [int(h*mm) for h in outline["row_heights_mm"]]
    row_y = [sum(rowh[:i]) + g*i for i in range(len(rowh))]
    return W, ncol, colw, rowh, row_y, g

def panel_px(outline, letter, dpi=300, gutter_mm=4):
    W, ncol, colw, rowh, row_y, g = grid_geom(outline, dpi, gutter_mm)
    p = next(q for q in outline["panels"] if q["letter"]==letter)
    cs, rs, r = p["colspan"], p.get("rowspan",1), p["row"]
    return colw*cs + g*(cs-1), sum(rowh[r:r+rs]) + g*(rs-1)

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

# figure-style helpers (inline)
def apply_figure_style(sizes=(8,7,6)):
    mpl.rcParams.update({
        'font.size': sizes[2],
        'axes.titlesize': sizes[0],
        'axes.labelsize': sizes[1],
        'xtick.labelsize': sizes[2],
        'ytick.labelsize': sizes[2],
        'legend.fontsize': sizes[2],
        'figure.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'savefig.bbox': 'standard',
    })

# Load data
HR = pd.read_csv('../results/hw_heat_ranking.csv')
LSTf = pd.read_csv('../results/lst_corridor_1km.csv')
TR = pd.read_csv('../results/hw_test_results_runs.csv')
TJ = json.load(open('../results/hw_test_results.json'))
HO = json.load(open('../results/hw_outputs.json'))
KM = pd.read_csv('../results/hw_km.csv')

OUT2 = {"claim": "On the Bangladesh highway the adapted detector finds 59 % of trees, and its tree density tracks satellite canopy and is lowest where it is hottest.",
        "width_mm": 180, "ncol": 12, "row_heights_mm": [70, 58],
        "panels": [{"letter": "a", "role": "primary", "row": 0, "col": 0, "colspan": 12, "chart_family": "track", "message": "", "data_vid": None, "ask": ""},
                   {"letter": "b", "role": "evidence", "row": 1, "col": 0, "colspan": 5, "chart_family": "dot", "message": "", "data_vid": None, "ask": ""},
                   {"letter": "c", "role": "evidence", "row": 1, "col": 5, "colspan": 7, "chart_family": "scatter", "message": "", "data_vid": None, "ask": ""}]}

def nf(L, o=OUT2):
    w, h = panel_px(o, L)
    return plt.figure(figsize=(w/300, h/300), dpi=300)

plt.close('all')
apply_figure_style(sizes=(8, 7, 6))
mpl.rcParams['savefig.bbox'] = 'standard'

C_TREE = '#2f7d4f'
C_HOT = '#c4572a'

hot10 = set(HR[HR.heat_rank <= 10].km)
cov = HR.set_index('km')
AVF = json.load(open('../results/audit/audit_values.json'))
scale = AVF['route_km']['map'] / 239.12

# Panel a: two tracks
fig = nf('a')
a1 = fig.add_axes([0.10, 0.56, 0.88, 0.36])
a2 = fig.add_axes([0.10, 0.13, 0.88, 0.36], sharex=a1)
for k in range(0, 240):
    if k not in cov.index:
        a1.axvspan(k, k+1, color='0.93', lw=0, zorder=0)
a1.bar(cov.index+0.5, cov.trees_per_km, width=0.9, color=C_TREE, lw=0)
for k in hot10:
    for ax in (a1, a2):
        ax.axvspan(k, k+1, color=C_HOT, alpha=0.25, lw=0, zorder=0)
a1.set_ylabel('Trees per km\n(as detected)')
a1.set_ylim(0, cov.trees_per_km.max()*1.08)
a1.set_title('Dhaka–Chattogram highway: tree density and hot-season surface temperature', loc='left')
a1.text(133, 0.88*a1.get_ylim()[1], 'grey: not mapped\norange: 10 hottest km', ha='center', va='top', fontsize=6, color='0.35')
a2.plot((LSTf.stretch+0.5)*scale, LSTf.lst_median_c, color=C_HOT, lw=1.0)
a2.set_ylabel('Surface\ntemperature (°C)')
a2.set_xlabel('Distance along the highway from Dhaka (km)')
a2.set_xlim(0, 240)
plt.setp(a1.get_xticklabels(), visible=False)
fig.savefig('hf_a.png', dpi=300, transparent=True)
plt.close('all')

# Panel b: transfer test
apply_figure_style(sizes=(8, 7, 6))
mpl.rcParams['savefig.bbox'] = 'standard'
fig = nf('b')
ax = fig.add_axes([0.27, 0.18, 0.68, 0.70])
for i, (g_, col) in enumerate((('old', '0.55'), ('new', C_TREE))):
    v = TR[TR.group == g_].found_all.values * 100
    m = TJ[g_]['mean'] * 100
    lo, hi = [x*100 for x in TJ[g_]['ci95']]
    ax.plot([lo, hi], [i, i], color=col, lw=1.6, solid_capstyle='butt')
    ax.scatter(v, [i]*len(v), color=col, s=14, zorder=3, alpha=0.85)
    ax.plot([m, m], [i-0.18, i+0.18], color='k', lw=1.6)
    ax.text(max(hi, v.max())+3, i, f'{m:.1f} %', va='center', fontsize=6.5)
ax.axvspan(40, 70, color='0.92', zorder=0, lw=0)
ax.text(55, 1.42, 'trees per km\nallowed (40–70 %)', ha='center', fontsize=5.5, color='0.35')
ax.set_yticks([0, 1])
ax.set_yticklabels(['Original\nmodel', 'Highway\nmodel (B)'], fontsize=6.5)
ax.set_ylim(-0.6, 1.75)
ax.set_xlim(0, 100)
ax.set_xlabel('Clicked test trees found (%)')
ax.set_title('Highway test photos (356 trees)', loc='left')
fig.savefig('hf_b.png', dpi=300, transparent=True)
plt.close('all')

# Panel c: trees per km vs canopy
apply_figure_style(sizes=(8, 7, 6))
mpl.rcParams['savefig.bbox'] = 'standard'
fig = nf('c')
ax = fig.add_axes([0.13, 0.18, 0.84, 0.70])
ax.scatter(HR.canopy_pct, HR.trees_per_km, s=10, color=C_TREE, alpha=0.75, lw=0)
ax.set_xlabel('Satellite tree canopy cover within 20 m of the road (%)')
ax.set_ylabel('Trees per km (street photos, as detected)')
ax.text(0.97, 0.95, f"Spearman ρ = {HO['heat']['spearman_trees_canopy_1km']:.2f}\nn = {HO['heat']['n_stretches']} km stretches",
        transform=ax.transAxes, ha='right', va='top', fontsize=6.5)
ax.set_title('Street-photo tree density vs satellite canopy', loc='left')
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
fig.savefig('hf_c.png', dpi=300, transparent=True)
plt.close('all')

compose_figure(OUT2, {'a': 'hf_a.png', 'b': 'hf_b.png', 'c': 'hf_c.png'}, 'fig_highway.png')
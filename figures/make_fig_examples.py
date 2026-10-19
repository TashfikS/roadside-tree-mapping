"""Detection examples on test images (fig_examples) and trunk-base point accuracy (fig_trunk).

Inputs : ../data/trunk_pose_labels.json (human boxes), ../data/bangladesh_test_trunk_review.csv (reviewed points),
         ../results/test_preds_pose_s0.csv (main model, run s0, Bangladesh test split),
         ../results/hw_test_preds_hwb_s0.csv and ../data/highway_test_trunk_points.csv (highway test photos),
         images from BD_DATASET_DIR and HIGHWAY_PHOTO_DIR.
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'src')]
os.chdir(HERE)
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from figstyle import apply_figure_style

apply_figure_style(sizes=(8, 7, 6)); mpl.rcParams['savefig.bbox'] = 'standard'
C_OURS, C_BOX, C_HUM, C_KEY, C_REF, C_BB = '#1f5f99', '#ffd400', 'white', '#d4237a', '#39d353', '0.45'
BD = os.environ.get('BD_DATASET_DIR', '../data/external/roadside-trees-of-bangladesh-street-view')
HW = os.environ.get('HIGHWAY_PHOTO_DIR', '../data/external/highway_photos')
LB = json.load(open('../data/trunk_pose_labels.json'))
TR = pd.read_csv('../data/bangladesh_test_trunk_review.csv'); TR['img'] = TR.key.str.rsplit('/', n=1).str[0].str.replace('/', '_')
PP = pd.read_csv('../results/test_preds_pose_s0.csv'); PP = PP[PP.score >= 0.25]
HA = pd.read_csv('../data/highway_test_trunk_points.csv', dtype={'id': str})
HP = pd.read_csv('../results/hw_test_preds_hwb_s0.csv', dtype={'id': str}); HP = HP[HP.score >= 0.25]


def bd_img(key):
    f, i = key.rsplit('_', 1)
    return Image.open(f'{BD}/{"Test" if f.startswith("BD_Test") else "Train"}/{f}/{i}.jpg').convert('RGB')


def show(ax, im):
    ax.imshow(im, aspect='auto'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)


def bd_panel(ax, key, title):
    im = bd_img(key); W, H = im.size; show(ax, im)
    for l in LB[key]:
        _, xc, yc, w, h, *_ = map(float, l.split())
        ax.add_patch(Rectangle(((xc - w / 2) * W, (yc - h / 2) * H), w * W, h * H, fill=False, ec=C_HUM, lw=1.0, ls=(0, (3, 2))))
    for r in PP[PP.image == key + '.jpg'].itertuples():
        ax.add_patch(Rectangle((r.x0 * W, r.y0 * H), (r.x1 - r.x0) * W, (r.y1 - r.y0) * H, fill=False, ec=C_BOX, lw=1.1))
        ax.scatter([r.ku * W], [r.kv * H], s=16, color=C_KEY, ec='white', lw=0.5, zorder=5)
    t = TR[(TR.img == key) & (TR.vis > 0)]
    ax.scatter(t.hu, t.hv, s=30, facecolor='none', edgecolor=C_REF, lw=1.1, zorder=4)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_title(title, loc='left', fontsize=7, pad=3)


def hw_panel(ax, i, title):
    im = Image.open(f'{HW}/{i}.jpg').convert('RGB'); W, H = im.size; show(ax, im)
    for r in HP[HP.id == i].itertuples():
        ax.add_patch(Rectangle((r.x0 * W, r.y0 * H), (r.x1 - r.x0) * W, (r.y1 - r.y0) * H, fill=False, ec=C_BOX, lw=1.1))
    t = HA[(HA.id == i) & (HA.status == 'done')]
    ax.scatter(t.u * W, t.v * H, s=30, facecolor='none', edgecolor=C_REF, lw=1.1, zorder=4)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_title(title, loc='left', fontsize=7, pad=3)


# ============================================================== examples
fig = plt.figure(figsize=(7.2, 4.55))
cols, rows_y = 3, (0.535, 0.075)
w, h, gx = 0.318, 0.375, 0.023
axes = [fig.add_axes([0.01 + c * (w + gx), y, w, h]) for y in rows_y for c in range(cols)]
bd_panel(axes[0], 'BD_740_1307', 'Bangladesh: both trees found')
bd_panel(axes[1], 'BD_740_1301', 'Small, distant trees missed')
bd_panel(axes[2], 'BD_740_1083', 'Continuous canopy: boxes split differently')
bd_panel(axes[3], 'BD_740_1075', 'Right tree: trunk base hidden by vegetation')
hw_panel(axes[4], '2719212188464410', 'Highway photo: marked trees inside boxes')
hw_panel(axes[5], '1527988998774931', 'Highway photo: roadside row missed')
for ax, L in zip(axes, 'abcdef'):
    b = ax.get_position(); fig.text(b.x0, b.y1 + 0.045, L, fontsize=9, fontweight='bold', ha='left', va='bottom')
leg = [mpl.lines.Line2D([], [], color=C_HUM, lw=1.2, ls=(0, (3, 2)), label='human box'),
       mpl.lines.Line2D([], [], color=C_BOX, lw=1.4, label='detector box'),
       mpl.lines.Line2D([], [], marker='o', ls='', ms=4, color=C_KEY, label='predicted trunk base'),
       mpl.lines.Line2D([], [], marker='o', ls='', ms=5, mfc='none', mec=C_REF, mew=1.1, label='reviewed / marked trunk base')]
lg = fig.legend(handles=leg, loc='lower center', ncol=4, fontsize=6.5, frameon=True, facecolor='0.25', edgecolor='none',
                bbox_to_anchor=(0.5, 0.0), borderpad=0.4)
for t in lg.get_texts(): t.set_color('white')
fig.savefig('fig_examples.png', dpi=300); fig.savefig('fig_examples.pdf', dpi=300); plt.close(fig)

# ============================================================== trunk-base accuracy (run s0, matched test boxes)
v = TR[(TR.vis > 0) & TR.pu.notna()].copy()
v['ours'] = np.hypot(v.pu - v.hu, v.pv - v.hv); v['box'] = np.hypot(v.bu - v.hu, v.bv - v.hv)
v['ours_rel'] = 100 * v.ours / v.box_h; v['box_rel'] = 100 * v.box / v.box_h
T3 = pd.read_csv('../results/trunk_point_main_model_3runs.csv')
fig = plt.figure(figsize=(7.2, 2.75))
a1 = fig.add_axes([0.07, 0.17, 0.27, 0.68]); a2 = fig.add_axes([0.42, 0.17, 0.27, 0.68]); a3 = fig.add_axes([0.81, 0.17, 0.17, 0.68])
xs = np.linspace(0, 150, 301)
for col, c, lab in (('ours', C_OURS, 'predicted trunk base'), ('box', C_BB, 'bottom-centre of box')):
    a1.plot(xs, [(v[col] <= x).mean() * 100 for x in xs], color=c, lw=1.5 if col == 'ours' else 1.1)
a1.text(32, 78, 'predicted\ntrunk base', color=C_OURS, fontsize=6.5, fontweight='bold'); a1.text(78, 42, 'bottom-centre\nof box', color=C_BB, fontsize=6.5)
a1.set_xlim(0, 150); a1.set_ylim(0, 100); a1.set_xlabel('Distance to reviewed point (px)'); a1.set_ylabel('Share of test trees (%)')
a1.set_title(f'Closer to the reviewed point (n = {len(v)})', loc='left')
grp = [('all', v), ('visible base\n(proposal kept)', v[(v.vis == 2) & (v.accepted == 1)]), ('hidden base\n(annotator)', v[v.vis == 1])]
rng = np.random.default_rng(0)
for k, (nm, d) in enumerate(grp):
    for j, (col, c) in enumerate((('ours_rel', C_OURS), ('box_rel', C_BB))):
        x = k + (j - 0.5) * 0.36; y = d[col].clip(upper=60).values
        a2.scatter(x + rng.uniform(-0.1, 0.1, len(y)), y, s=3, color=c, alpha=0.35, lw=0, rasterized=True)
        a2.plot([x - 0.14, x + 0.14], [np.median(d[col])] * 2, color='black', lw=1.6)
    a2.text(k, 63, f'n = {len(d)}', ha='center', fontsize=6, color='0.35')
a2.set_xticks(range(3)); a2.set_xticklabels([g[0] for g in grp], fontsize=6); a2.set_xlim(-0.6, 2.6); a2.set_ylim(0, 66)
a2.set_ylabel('Distance / box height (%)'); a2.set_title('Also where the annotator placed it', loc='left')
for k, r in T3.iterrows():
    a3.plot([0, 1], [r.median_px, r.box_bottom_median_same_trees], color='0.8', lw=0.8, zorder=1)
    a3.scatter([0], [r.median_px], color=C_OURS, s=18, zorder=3); a3.scatter([1], [r.box_bottom_median_same_trees], color=C_BB, s=18, zorder=3)
a3.set_xticks([0, 1]); a3.set_xticklabels(['trunk\nbase', 'box\nbottom'], fontsize=6.5); a3.set_xlim(-0.4, 1.4); a3.set_ylim(0, 70)
a3.set_ylabel('Median distance (px)'); a3.set_title('Three training runs', loc='left')
for ax, L in zip((a1, a2, a3), 'abc'):
    b = ax.get_position(); fig.text(b.x0 - 0.055, b.y1 + 0.06, L, fontsize=9, fontweight='bold')
fig.text(0.42, 0.005, 'lower = closer to the reviewed point; values above 60 % drawn at 60 %', fontsize=6, color='0.4')
fig.savefig('fig_trunk.png', dpi=300); fig.savefig('fig_trunk.pdf', dpi=300); plt.close(fig)
print('examples ok', len(v), round(v.ours.median(), 1), round(v.box.median(), 1),
      {nm: (len(d), round(d.ours.median(), 1), round(d.box.median(), 1), round(d.ours_rel.median(), 1), round(d.box_rel.median(), 1)) for nm, d in grp})

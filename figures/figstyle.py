"""Shared matplotlib settings for the paper figures (font-size ladder, open frame, 300 dpi, embedded fonts)."""
import matplotlib as mpl


def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    base, secondary, tick = sizes
    boxed = frame == "boxed"
    rc = {"font.family": "sans-serif", "font.size": base, "axes.labelsize": base, "axes.titlesize": base,
          "legend.fontsize": secondary, "xtick.labelsize": tick, "ytick.labelsize": tick, "axes.linewidth": 0.6,
          "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 3, "ytick.major.size": 3,
          "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": boxed, "axes.spines.right": boxed,
          "axes.spines.left": frame != "none", "axes.spines.bottom": frame != "none", "axes.grid": bool(grid),
          "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
          "axes.titleweight": "normal", "axes.titlelocation": "left", "axes.labelweight": "normal",
          "lines.linewidth": 1.2, "patch.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42}
    if font:
        rc["font.sans-serif"] = [font, "DejaVu Sans"]
    mpl.rcParams.update(rc)

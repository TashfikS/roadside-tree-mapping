"""Crop rendering for the VLM second-look test (shared by the local check and the Kaggle job).

Panorama convention (same as the E2 geolocation, treegeo.pano_pixel_to_geo): equirectangular W x H, column 0 points to
(yaw + 180 deg), horizon at row H/2, camera height 3.0 m above flat ground.
"""
import math
import numpy as np

W, H, HCAM = 1664, 832, 3.0
PX_DEG = W / 360.0


def ground_to_pixel(cam_en, yaw_deg, pts_en, h=HCAM):
    """ENU ground points (N,2) -> panorama pixels (N,2) and horizontal range (N,)."""
    d = np.asarray(pts_en, float) - np.asarray(cam_en, float)
    z = np.hypot(d[:, 0], d[:, 1])
    bearing = np.degrees(np.arctan2(d[:, 0], d[:, 1]))              # clockwise from north
    x = (((bearing - yaw_deg + 180.0) % 360.0) / 360.0) * W
    y = H / 2 + np.degrees(np.arctan2(h, z)) * PX_DEG
    return np.stack([x, y], 1), z


def window(px, kind, x_c=None):
    """Crop window around overlay pixels. Returns x_c, half-width, y0, y1 (pixels) and overlay as (dx, y)."""
    if x_c is None:
        x_c = px[len(px) // 2, 0]
    dx = ((px[:, 0] - x_c + W / 2) % W) - W / 2
    if kind == 'line':
        hw = min(75 * PX_DEG, max(abs(dx).max() + 8 * PX_DEG, 30 * PX_DEG))
        y0, y1 = H / 2 - 40 * PX_DEG, min(H - 1, px[:, 1].max() + 6 * PX_DEG)
    else:                                                            # circle at one ground point
        hw = 30 * PX_DEG
        y0, y1 = H / 2 - 35 * PX_DEG, min(H - 1, px[:, 1].max() + 12 * PX_DEG)
    return float(x_c), float(hw), float(y0), float(y1), np.stack([dx, px[:, 1]], 1)


def render(img, x_c, hw, y0, y1, ov, kind, scale=2, color=(255, 0, 0)):
    """img: PIL panorama. Crops the window (wrapping at the seam), upsamples, draws the overlay. Returns a PIL image."""
    from PIL import Image, ImageDraw
    a = np.asarray(img.convert('RGB'))
    a = np.roll(a, int(round(W / 2 - x_c)), axis=1)
    xa, xb, ya, yb = int(round(W / 2 - hw)), int(round(W / 2 + hw)), int(round(y0)), int(round(y1))
    c = Image.fromarray(a[ya:yb, xa:xb]).resize(((xb - xa) * scale, (yb - ya) * scale), Image.LANCZOS)
    d = ImageDraw.Draw(c)
    P = [((u + W / 2 - xa) * scale, (v - ya) * scale) for u, v in ov]
    if kind == 'line':
        d.line(P, fill=color, width=3)
    else:
        u, v = P[0]; r = 9
        d.ellipse([u - r, v - r, u + r, v + r], outline=color, width=3)
    return c

"""Frame maps from Blender passes (numpy): camera, EXR/PNG reading, depth, normals, lines, segmentation, notan, tone, smoke.
Lines, notan and tone repeat the sketch25.views rules (edges by id/parts/depth jump/crease, thickness bands by distance,
notan by roles, air) — but on real Blender passes, with shadows from the light."""
import json
import math

import cv2
import numpy as np
import OpenEXR
from PIL import Image

from core.contract import classes as C


# ------------------------------------------------------------------------------------------------ camera
class Cam:
    """A camera from cameras.json (matrix_world: the columns are right, up, back), the focal length for 36 mm, shift as in Blender."""

    def __init__(self, spec):
        self.W, self.H = spec["resolution"]
        M = np.array(spec["matrix_world"], float)
        self.right, self.up, self.fwd, self.pos = M[:3, 0], M[:3, 1], -M[:3, 2], M[:3, 3]
        # as in Blender: sensor_fit decides which side of the frame the sensor is fitted to (AUTO — the larger one; VERTICAL — the sensor height
        # to the frame height); the shift is in fractions of the same side (in 9:16 it would otherwise be off by 233 px)
        fit = spec.get("sensor_fit", "AUTO")
        if fit == "HORIZONTAL": L, sensor = self.W, spec.get("sensor_width_mm", 36.0)
        elif fit == "VERTICAL": L, sensor = self.H, spec.get("sensor_height_mm", 24.0)
        else: L, sensor = max(self.W, self.H), spec.get("sensor_width_mm", 36.0)
        self.f = spec["lens_mm"] / sensor * L
        self.cx = self.W / 2 - spec.get("shift_x", 0.0) * L
        self.cy = self.H / 2 + spec.get("shift_y", 0.0) * L
        self.dx = ((np.arange(self.W) + 0.5 - self.cx) / self.f)[None, :].astype(np.float32)
        self.dy = (-(np.arange(self.H) + 0.5 - self.cy) / self.f)[:, None].astype(np.float32)
        self.dnorm = np.sqrt(1 + self.dx ** 2 + self.dy ** 2)           # ray length per unit of depth along the axis

    def ray_len(self, z):
        return z * self.dnorm

    def world(self, z):
        """World points from depth along the axis (H, W, 3)."""
        d = self.fwd[None, None, :] + self.dx[..., None] * self.right + self.dy[..., None] * self.up
        return self.pos + z[..., None] * d

    def ray_dirs(self):
        d = self.fwd[None, None, :] + self.dx[..., None] * self.right + self.dy[..., None] * self.up
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def project(self, P):
        P = np.asarray(P, float)
        d = P - self.pos
        x, y, z = d @ self.right, d @ self.up, d @ self.fwd
        zz = np.where(z > 1e-6, z, np.nan)
        return self.cx + self.f * x / zz, self.cy - self.f * y / zz, z

    def horizon_y(self):
        """The horizon height at the frame center, as a fraction of the frame height (along camera directions, z = 0)."""
        h = np.array([self.fwd[0], self.fwd[1], 0.0])
        if np.linalg.norm(h) < 1e-9:
            return None
        h /= np.linalg.norm(h)
        z = h @ self.fwd
        if z <= 1e-6:
            return None
        v = self.cy - self.f * (h @ self.up) / z
        return float(v / self.H)


# ------------------------------------------------------------------------------------------------ reading the passes
def read_id(path):
    a = np.array(Image.open(path).convert("RGB")).astype(np.int32)
    return a[..., 0] + 256 * a[..., 1] + 65536 * a[..., 2]


def read_exr(path):
    f = OpenEXR.File(path)
    out = {}
    for p in f.parts:
        n = p.name().split(".")[-1]
        ch = list(p.channels.values())
        if n == "Combined":
            px = np.array(ch[0].pixels)
            out["combined"] = px[..., :3] if px.ndim == 3 else np.stack([px] * 3, -1)
        elif n == "Depth":
            out["z"] = np.array(ch[0].pixels).astype(np.float32)
        elif n == "Normal":
            out["normal"] = np.stack([np.array(c.pixels) for c in ch], -1).astype(np.float32)
    return out


# ------------------------------------------------------------------------------------------------ depth, normals
def encode_depth(z, sky, window=None):
    """As in monocular depth estimators: nearer is lighter, 1/z; the sky is black (byte 0 only for emptiness). uint8.
    The 1/z window is the 2–98 percentiles of the frame or a given (lo, hi) for the whole shot, so the tone does not jump when the camera moves.
    Geometry uses bytes 1–255: distant objects (a city on the horizon) do not merge with the sky (28 % of the geometry used to land on 0)."""
    fin = ~sky & np.isfinite(z) & (z > 0)
    inv = np.where(fin, 1.0 / np.maximum(z, 1e-3), 0.0)
    if window: lo, hi = window
    elif fin.any(): lo, hi = np.percentile(inv[fin], 2), np.percentile(inv[fin], 98)
    else: lo, hi = 0.0, 1.0                                         # the frame has only sky: the depth is empty, the window is nominal
    d = np.clip((inv - lo) / (hi - lo + 1e-9), 0, 1)
    return np.where(fin, 1 + (d * 254 + 0.5).astype(np.uint8), 0).astype(np.uint8), (float(lo), float(hi))


def encode_normal(n_world, cam, sky):
    """Normals in the camera frame: R — right, G — up, B — toward the viewer; the sky is flat (127, 127, 255)."""
    n = np.stack([n_world @ cam.right, n_world @ cam.up, -(n_world @ cam.fwd)], -1)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
    rgb = ((n * 0.5 + 0.5) * 255 + 0.5).astype(np.uint8)
    rgb[sky] = (127, 127, 255)
    return rgb


# ------------------------------------------------------------------------------------------------ lines
BANDS = [(8, 2.6), (30, 2.0), (120, 1.5), (500, 1.1), (1e12, 0.8)]      # (up to how many meters, thickness in px) — as in sketch25.views


def edges(grp, part, t, nrm, g, delta, big_grp, fig_grp):
    """Edges by object group, body part, depth jump and normal crease (the sketch25.views.edges rules).
    For a building (not a figure) a jump and a crease give a line inside a single object too: ladder rungs, wall teeth, crater rims.
    g — |normal · ray|, delta — the pixel angle (rad): a depth jump is allowed for the tilt of a surface (the ground near the horizon)."""
    H, W = grp.shape
    E = np.zeros((H, W), bool)
    T = np.full((H, W), np.inf, np.float32)
    for sa, sb in (((slice(None), slice(0, W - 1)), (slice(None), slice(1, W))),
                   ((slice(0, H - 1), slice(None)), (slice(1, H), slice(None)))):
        ta, tb = t[sa], t[sb]
        tmin = np.minimum(ta, tb)
        gm = np.minimum(g[sa], g[sb])
        slope = 2.0 * tmin * delta * np.sqrt(np.clip(1 - gm * gm, 0, 1)) / np.maximum(gm, 0.02)      # the expected difference on a plane
        with np.errstate(invalid="ignore"):
            jump = np.abs(ta - tb) > np.maximum(0.03 * tmin, 0.12) + slope
        jump |= np.isinf(ta) != np.isinf(tb)
        crease = (nrm[sa] * nrm[sb]).sum(-1) < 0.9
        ga, gb, pa, pb = grp[sa], grp[sb], part[sa], part[sb]
        e = (ga != gb) | ((pa != pb) & (jump | crease)) | (~fig_grp[ga] & ~fig_grp[gb] & (jump | crease))
        e |= (ga == gb) & (pa != pb) & big_grp[ga]
        near_a = ta <= tb
        ea, eb = e & near_a, e & ~near_a
        E[sa] |= ea
        E[sb] |= eb
        T[sa] = np.where(ea, np.minimum(T[sa], ta), T[sa])
        T[sb] = np.where(eb, np.minimum(T[sb], tb), T[sb])
    return E, T


def line_cover(E, T):
    """Line coverage 0..1: thicker and darker for nearer ones."""
    acc = np.zeros(E.shape, np.float32)
    lo, s = 0.0, (max(E.shape) / 1280 if max(E.shape) >= 1920 else 1.0)   # from a 2K frame on — the thickness is relative to the frame, not in points
    for hi, w in BANDS:
        m = E & (T >= lo) & (T < hi)
        lo = hi
        if not m.any():
            continue
        k = max(1, int(round(w * s)))
        if k > 1:
            m = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
        strength = 1.0 if hi <= 30 else 0.85 if hi <= 120 else 0.65 if hi <= 500 else 0.45
        acc = np.maximum(acc, m.astype(np.float32) * strength)
    return np.clip(cv2.GaussianBlur(acc, (0, 0), 0.55 * s), 0, 1)


# ------------------------------------------------------------------------------------------------ segmentation, id colors
def seg_map(cls_idx, names):
    out = np.zeros(cls_idx.shape + (3,), np.uint8)
    for i, n in enumerate(names):
        out[cls_idx == i] = C.SEG_COLOR[n]
    return out


def id_palette(n, seed=3):
    import colorsys
    rng = np.random.default_rng(seed)
    pal = np.zeros((n + 1, 3), np.uint8)
    for i in range(1, n + 1):
        h = (i * 0.61803398875 + rng.uniform(0, 0.05)) % 1.0
        pal[i] = [int(c * 255) for c in colorsys.hsv_to_rgb(h, 0.75, 0.95)]
    return pal


# ------------------------------------------------------------------------------------------------ tone, air, sky, pit
def srgb(lin):
    lin = np.clip(lin, 0, None)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)


def sky_tone(cam, world):
    s = world.get("sky", {}) or {}
    top, hor = float(s.get("top", 0.78)), float(s.get("horizon", 0.94))
    el = np.degrees(np.arcsin(np.clip(cam.ray_dirs()[..., 2], -1, 1)))
    k = np.clip(el / 50.0, 0, 1) ** 0.7
    return (hor + (top - hor) * k).astype(np.float32)


def cavity(zw, sky, fig, depth=2.5):
    """How far below the ground a point is: 0 — at the level or above, toward 1 — deep (crater, trench). The sky and figures are left alone."""
    k = 1.0 - np.exp(-np.clip(-zw - 0.05, 0, None) / depth)
    k[sky | fig] = 0
    return k.astype(np.float32)


def figure_shade(grp, part, big_px, h_px_of_grp, mask):
    """Volume of the parts of large figures: lighter toward the middle of a part, darker toward its edge (a "pillow")."""
    out = np.ones(grp.shape, np.float32)
    if not mask.any():
        return out
    b = np.zeros(grp.shape, bool)
    b[:, 1:] |= (part[:, 1:] != part[:, :-1]) | (grp[:, 1:] != grp[:, :-1])
    b[1:, :] |= (part[1:, :] != part[:-1, :]) | (grp[1:, :] != grp[:-1, :])
    dt = cv2.distanceTransform((~b & mask).astype(np.uint8), cv2.DIST_L2, 3)
    k = np.maximum(0.045 * h_px_of_grp[grp], 1.0)
    return np.where(mask, 0.68 + 0.42 * np.clip(dt / k, 0, 1), 1.0).astype(np.float32)


def fx_layer(cam, sprites, tray, haze):
    """Smoke and fire: puff-patches, the far ones first (as in sketch25.raster.draw_sprite). Returns color, opacity, smoke density."""
    H, W = tray.shape
    fc, fa, fp = (np.zeros((H, W), np.float32) for _ in range(3))
    order = sorted(sprites, key=lambda s: -float(np.linalg.norm(np.array(s["pos"]) - cam.pos)))
    for s in order:
        c = np.array(s["pos"], float)
        u, v, z = cam.project(c[None])
        z = float(z[0])
        r = float(s["radius"])
        if z < 0.05 + r * 0.2 or not np.isfinite(u[0]):
            continue
        u, v = float(u[0]), float(v[0])
        rp = cam.f * r / z
        if rp < 0.7:
            continue
        x0, x1 = max(0, int(u - 2 * rp)), min(W, int(u + 2 * rp) + 1)
        y0, y1 = max(0, int(v - 2 * rp)), min(H, int(v + 2 * rp) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d2 = ((xx + 0.5 - u) ** 2 + (yy + 0.5 - v) ** 2) / (rp * rp)
        puff = s["kind"] == "smoke"
        if puff:
            q = np.clip((1.0 - np.sqrt(d2)) / 0.3, 0, 1)
            a = s["alpha"] * q * q * (3 - 2 * q)
        else:
            a = s["alpha"] * np.exp(-d2 * 1.6)
        tc = float(np.linalg.norm(c - cam.pos))
        a = a * np.clip((tray[y0:y1, x0:x1] - (tc - r)) / (2 * r), 0, 1)
        tone = s["tone"]
        if s["kind"] in ("smoke", "haze"):
            e = math.exp(-tc / haze[0])
            tone = tone * e + haze[1] * (1 - e)
        c_ = fc[y0:y1, x0:x1]
        a_ = fa[y0:y1, x0:x1]
        if puff:
            add = np.clip(a - a_, 0, None)
            c_ += tone * add
            a_ += add
            np.maximum(fp[y0:y1, x0:x1], a, out=fp[y0:y1, x0:x1])
        else:
            c_[:] = tone * a + c_ * (1 - a)
            a_[:] = a + a_ * (1 - a)
    return fc, fa, fp

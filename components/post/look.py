"""Grading a frame over the render (the "Post" component): operations on a finished frame (screen colour, sRGB) and its depth — without
Blender. The order runs from the physics of light to film: haze → painterly distance → DoF → aberration → bloom →
vignette → halation → palette → tone → sun glow → grain → dither → letterbox. Mixing and blurring of light is done in linear colour (sRGB → linear →
back); the palette works in screen colour, as the references do (palettes.yaml). The same path works for model images made from our maps.
  .venv/bin/python -m components.post.look <frame.png | camera folder> <output.png> [--preset trailer|anime|crush|clean] [--set operation.knob=value …]
      [--depth depth.exr|.npy] [--camera far.json]
"""
import json
import math
import os
import sys

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ("haze", "kuwahara", "dof", "ca", "bloom", "vignette", "halation", "palette", "grade", "para", "grain", "dither", "letterbox")
FAR = 1e9                                      # sky: the core's depth is 1e10


def srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x):
    x = np.clip(x, 0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def luma(x):
    return x @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def smoothstep(a, b, x):
    t = np.clip((x - a) / max(b - a, 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


# ── presets and palettes ────────────────────────────────────────────────────────────────────────────────────
def presets():
    return yaml.safe_load(open(os.path.join(HERE, "presets.yaml")))["presets"]


def default_preset():
    """The main preset (presets.yaml default — trailer): used when the format has not chosen its own."""
    return yaml.safe_load(open(os.path.join(HERE, "presets.yaml"))).get("default") or "trailer"


def palettes():
    p = os.path.join(HERE, "palettes.yaml")
    return (yaml.safe_load(open(p)) or {}).get("palettes", {}) if os.path.exists(p) else {}


def resolve(preset=None, overrides=None):
    """A preset (none — the main one) + adjustments {operation: {knob: value} | number | False} → {operation: knobs}; disabled ones are removed."""
    P = presets(); preset = preset or default_preset()
    if preset not in P:
        raise SystemExit(f"no preset '{preset}': {', '.join(P)}")
    ops = {k: (dict(v) if isinstance(v, dict) else v) for k, v in (P[preset] or {}).items()}
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(ops.get(k), dict): ops[k].update(v)
        else: ops[k] = v
    return {k: ({} if v is True else v) for k, v in ops.items() if v not in (None, False, 0, 0.0)}


# ── frame ───────────────────────────────────────────────────────────────────────────────────────────────────
class Frame:
    """A frame and what is known about it: img — sRGB 0..1 (H, W, 3); z — depth, m (sky — inf), or None; cam — lens and matrix;
    sun — the direction to the sun (world), index — the frame number (grain, dithering)."""

    def __init__(self, img, z=None, cam=None, sun=None, index=0, name=""):
        self.img, self.z, self.cam, self.sun, self.index, self.name = img, z, cam or {}, sun, index, name

    @property
    def size(self):
        return self.img.shape[1], self.img.shape[0]


def read_png(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None: raise SystemExit(f"cannot be read: {path}")
    im = im[..., :3][..., ::-1] if im.ndim == 3 else np.stack([im] * 3, -1)
    return (im.astype(np.float32) / (65535.0 if im.dtype == np.uint16 else 255.0))


def write_png(path, img):
    cv2.imwrite(path, (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)[..., ::-1])


def _depth_file(path):
    if path.endswith(".npy"): return np.load(path).astype(np.float32)
    from core.light import far
    _, _, z = far.read_rgba(path)
    return z


def frame_from_dir(cam_dir, index=0):
    """A camera folder of the "frames" output (lit.png, depth — raw/passes.exr or lit.exr, far.json) → Frame."""
    from core.light import far
    img = read_png(os.path.join(cam_dir, "lit.png"))
    z = None
    try: z = far.depth_of(cam_dir)
    except Exception: pass
    cam, sun = {}, None
    fj = os.path.join(cam_dir, "far.json")
    if os.path.exists(fj):
        F = json.load(open(fj)); cam = F.get("camera") or {}
        import glob
        run = os.path.dirname(os.path.dirname(os.path.abspath(cam_dir)))   # the light plan: the frame's path, otherwise the light step of the same run
        for p in [F.get("plan") or ""] + sorted(glob.glob(os.path.join(run, "*_light", "far_plan.json"))) + sorted(glob.glob(os.path.join(run, "*_light", "far_plan.json"))):
            if p and os.path.exists(p):
                sun = json.load(open(p)).get("sun", {}).get("dir"); break
    elif os.path.exists(os.path.join(cam_dir, "raw", "camera.json")):
        cam = json.load(open(os.path.join(cam_dir, "raw", "camera.json")))
    return Frame(img, _fit_depth(z, img), cam, sun, index, os.path.basename(cam_dir.rstrip("/")))


def _fit_depth(z, img):
    if z is None: return None
    z = np.where(np.isfinite(z) & (z > 0) & (z < FAR), z, np.inf).astype(np.float32)
    H, W = img.shape[:2]
    if z.shape != (H, W):
        fin = np.isfinite(z)
        z = cv2.resize(np.where(fin, z, 1e10).astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        z = np.where(z < FAR, z, np.inf).astype(np.float32)
    return z


# ── operations: (img sRGB, Frame, knobs, stats) → img sRGB ─────────────────────────────────────────────────
def _horizon_colour(lin, z):
    """The haze colour — the sky at the horizon: the lower third of the rows where the sky (depth inf) takes more than half; no sky —
    the median of the farthest 2 % of pixels."""
    sky = ~np.isfinite(z)
    rows = np.where(sky.mean(1) > 0.5)[0]
    if len(rows) >= 3:
        band = rows[-max(1, len(rows) // 3):]
        px = lin[band][sky[band]]
    else:
        zz = np.where(np.isfinite(z), z, -1)
        px = lin[zz >= np.percentile(zz, 98)]
    return np.median(px.reshape(-1, 3), 0) if len(px) else np.array([0.5, 0.5, 0.5], np.float32)


def op_haze(img, fr, p, st):
    """Aerial perspective by Koschmieder's law: haze share 1 − exp(−3.912·d/V); V — visibility, km; the sky is left alone."""
    if fr.z is None: st.setdefault("skipped", []).append("haze: no depth"); return img
    V = float(p.get("visibility_km", 8.0)) * 1000.0
    lin = srgb_to_lin(img)
    c = np.asarray(p["colour"], np.float32) if p.get("colour") else _horizon_colour(lin, fr.z)
    zf = np.where(np.isfinite(fr.z), fr.z, 0.0)
    f = (1 - np.exp(-3.912 * zf / V)) * float(p.get("strength", 1.0))
    st["haze_colour"] = [round(float(v), 4) for v in c]
    return lin_to_srgb(lin * (1 - f[..., None]) + c * f[..., None])


def kuwahara_filter(img, r):
    """Kuwahara over four (r+1)² pixel quadrants: the colour of the quadrant with the lowest brightness variance — a "painted" distance."""
    k = r + 1
    y = luma(img)
    means, vars_ = [], []
    for ax, ay in ((k - 1, k - 1), (0, k - 1), (k - 1, 0), (0, 0)):
        m = cv2.boxFilter(img, -1, (k, k), anchor=(ax, ay), normalize=True, borderType=cv2.BORDER_REFLECT)
        my = cv2.boxFilter(y, -1, (k, k), anchor=(ax, ay), normalize=True, borderType=cv2.BORDER_REFLECT)
        my2 = cv2.boxFilter(y * y, -1, (k, k), anchor=(ax, ay), normalize=True, borderType=cv2.BORDER_REFLECT)
        means.append(m); vars_.append(my2 - my * my)
    i = np.argmin(np.stack(vars_), 0)
    M = np.stack(means)
    return np.take_along_axis(M, i[None, ..., None].repeat(3, -1), 0)[0]


def op_kuwahara(img, fr, p, st):
    """Painterly distance (satsuei: a smoothing filter over the distance; here Kuwahara): beyond from_m — the filter, with a transition soft_m."""
    r = int(p.get("radius", 3))
    K = kuwahara_filter(img, r)
    if fr.z is None: return K
    a, s = float(p.get("from_m", 250.0)), float(p.get("soft_m", 150.0))
    m = smoothstep(a, a + s, np.where(np.isfinite(fr.z), fr.z, 1e9))[..., None]
    return img * (1 - m) + K * m


def _disc(r):
    n = int(math.ceil(r))
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    d = np.clip(r + 0.5 - np.hypot(xx, yy), 0, 1).astype(np.float32)
    return d / d.sum()


def _blur_disc(x, r):
    """Blur with a disc of radius r pixels; large discs — on a downscaled frame (by r/8 times), an 8-pixel disc."""
    if r < 0.5: return x
    if r <= 8: return cv2.filter2D(x, -1, _disc(r), borderType=cv2.BORDER_REFLECT)
    s = r / 8.0
    H, W = x.shape[:2]
    small = cv2.resize(x, (max(1, round(W / s)), max(1, round(H / s))), interpolation=cv2.INTER_AREA)
    small = cv2.filter2D(small, -1, _disc(8.0), borderType=cv2.BORDER_REFLECT)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)


def coc_px(fr, fstop, focus_m):
    """Radius of the circle of confusion, pixels, signed (closer than the focus — minus): c = f²/(N(s−f)) · |d−s|/d, relative to the sensor width."""
    f = float(fr.cam.get("lens_mm", 50.0)) / 1000.0
    sw = float(fr.cam.get("sensor_width_mm", 36.0)) / 1000.0
    s = max(float(focus_m), f * 1.01)
    d = np.where(np.isfinite(fr.z), fr.z, 1e7)
    c = f * f / (fstop * (s - f)) * np.abs(d - s) / d
    return np.sign(d - s) * 0.5 * c / sw * fr.size[0]


def op_dof(img, fr, p, st):
    """DoF by depth layers: the far part is blurred without bleeding in the near part, the near part spreads over it.
    Lens and sensor — the frame's camera; focus — a knob or the median depth of the middle of the frame. "Not a miniature" (Held et al. 2010):
    the f-number is not below 0.95 — otherwise it is tilt-shift; on purpose — allow_miniature."""
    if fr.z is None: st.setdefault("skipped", []).append("dof: no depth"); return img
    N = float(p.get("fstop", 2.8))
    if N < 0.95 and not p.get("allow_miniature"):
        st.setdefault("notes", []).append(f"dof: f/{N} → f/0.95 (not a miniature)"); N = 0.95
    H, W = fr.z.shape
    if p.get("focus_m") not in (None, "auto"):
        s = float(p["focus_m"])
    else:
        c = fr.z[int(H * 0.3):int(H * 0.7), int(W * 0.3):int(W * 0.7)]
        c = c[np.isfinite(c)]
        s = float(np.median(c)) if c.size else 100.0
    r = coc_px(fr, N, s)
    if p.get("miniature_scale"):                       # a deliberate "miniature": the world at 1/scale — depth and focus are divided
        k = float(p["miniature_scale"]); r = coc_px(Frame(img, fr.z / k, fr.cam), N, s / k)
    rmax = float(p.get("max_px", 40))
    r = np.clip(r, -rmax, rmax)
    st["dof_focus_m"], st["dof_max_px"] = round(s, 2), round(float(np.abs(r).max()), 1)
    lin = srgb_to_lin(img)
    levels = [0.0] + [v for v in (1, 2, 4, 8, 16, 32, 64) if v < rmax] + [rmax]
    far_m = (r >= 0).astype(np.float32)
    near_m = 1.0 - far_m
    rf = np.where(r >= 0, r, 0.0)
    rn = cv2.dilate(np.where(r < 0, -r, 0.0).astype(np.float32), np.ones((9, 9), np.uint8), iterations=max(1, int(rmax // 4)))

    def layered(mask, radius):
        out = np.zeros_like(lin); cover = np.zeros(mask.shape, np.float32)
        prev = None
        for i, R in enumerate(levels):
            C = _blur_disc(lin * mask[..., None], R); A = _blur_disc(mask, R)
            if prev is not None:
                R0 = levels[i - 1]
                t = np.clip((radius - R0) / max(R - R0, 1e-6), 0, 1)
                sel = (radius >= R0) & (radius <= R)
                tt = t[..., None]
                out = np.where(sel[..., None], prev[0] * (1 - tt) + C * tt, out)
                cover = np.where(sel, prev[1] * (1 - t) + A * t, cover)
            prev = (C, A)
        return out, cover

    Cf, Af = layered(far_m, rf)
    F = np.where(Af[..., None] > 1e-4, Cf / np.maximum(Af, 1e-4)[..., None], lin)
    if near_m.any():
        Cn, An = layered(near_m, rn)
        Nn = Cn / np.maximum(An, 1e-4)[..., None]
        a = np.clip(An * (rn > 0.5), 0, 1)[..., None]
        F = F * (1 - a) + Nn * a
    return lin_to_srgb(F)


def _highlights(lin, thr):
    y = luma(lin)
    return lin * smoothstep(thr, thr + 0.35, y)[..., None]


def op_bloom(img, fr, p, st):
    """Bloom of bright areas: three blur widths (0.4, 2, 8 % of the frame width), the share — strength."""
    lin = srgb_to_lin(img); W = fr.size[0]
    h = _highlights(lin, float(p.get("threshold", 0.7)))
    b = sum(w * cv2.GaussianBlur(h, (0, 0), s * W) for w, s in ((0.6, 0.004), (0.3, 0.02), (0.1, 0.08)))
    return lin_to_srgb(lin + float(p.get("strength", 0.3)) * b)


def op_halation(img, fr, p, st):
    """Film halation: light reflected by the base — a narrow red-orange halo around the brightest areas."""
    lin = srgb_to_lin(img); W = fr.size[0]
    h = luma(_highlights(lin, float(p.get("threshold", 0.8))))
    b = cv2.GaussianBlur(h, (0, 0), float(p.get("size", 0.006)) * W)
    tint = np.asarray(p.get("tint", [1.0, 0.35, 0.12]), np.float32)
    return lin_to_srgb(lin + float(p.get("strength", 0.15)) * b[..., None] * tint)


def op_ca(img, fr, p, st):
    """Chromatic aberration: red slightly wider, blue slightly narrower from the centre (the share amount)."""
    a = float(p.get("amount", 0.0015)); W, H = fr.size
    out = img.copy()
    for ch, k in ((0, 1 + a), (2, 1 - a)):
        M = np.float32([[1 / k, 0, (1 - 1 / k) * W / 2], [0, 1 / k, (1 - 1 / k) * H / 2]])
        out[..., ch] = cv2.warpAffine(img[..., ch], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out


def op_vignette(img, fr, p, st):
    """Vignette: a falloff toward the corners (r — a fraction of the half-diagonal) by the power power, in linear colour."""
    W, H = fr.size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)) / math.sqrt(2)
    k = 1 - float(p.get("strength", 0.25)) * r ** float(p.get("power", 2.2))
    return lin_to_srgb(srgb_to_lin(img) * k[..., None])


def mkl(mu_s, cov_s, mu_r, cov_r):
    """MKL palette transfer (Pitié, Kokaram 2007): a linear mapping that brings the source's mean and covariance to the reference.
    T = Σs^−½ (Σs^½ Σr Σs^½)^½ Σs^−½; x' = (x − μs)·T + μr. Verified against color-matcher (bench)."""
    def sqrtm(A, inv=False):
        w, U = np.linalg.eigh((A + A.T) / 2)
        w = np.clip(w, 1e-10, None)
        return (U * (w ** (-0.5 if inv else 0.5))) @ U.T
    Ss, Ssi = sqrtm(cov_s), sqrtm(cov_s, True)
    T = Ssi @ sqrtm(Ss @ cov_r @ Ss) @ Ssi
    return {"mu_s": np.asarray(mu_s, np.float64), "T": T, "mu_r": np.asarray(mu_r, np.float64)}


def stats_of(imgs, max_px=400_000):
    """Mean and covariance of the colour of a set of frames (screen colour 0..1) — one transform for the whole set, so the frames do not flicker."""
    px = np.concatenate([i.reshape(-1, 3) for i in imgs])
    if len(px) > max_px: px = px[np.random.default_rng(0).choice(len(px), max_px, replace=False)]
    return px.mean(0), np.cov(px.T)


def op_palette(img, fr, p, st):
    """Palette by a reference (palettes.yaml): MKL on the world; accents (fire, flashes — bright and saturated) are protected by the share protect."""
    pal = palettes().get(p.get("ref") or "")
    if not pal: st.setdefault("skipped", []).append(f"palette: no reference '{p.get('ref')}'"); return img
    tr = p.get("_transform")
    if tr is None:
        mu, cov = stats_of([img]); tr = mkl(mu, cov, pal["mean"], pal["cov"])
    x = img.reshape(-1, 3).astype(np.float64)
    y = ((x - tr["mu_s"]) @ tr["T"] + tr["mu_r"]).reshape(img.shape).astype(np.float32)
    hsv = cv2.cvtColor(np.clip(img, 0, 1), cv2.COLOR_RGB2HSV)
    acc = smoothstep(0.45, 0.75, hsv[..., 1]) * smoothstep(0.55, 0.85, hsv[..., 2])
    w = float(p.get("mix", 0.7)) * (1 - float(p.get("protect", 0.6)) * acc)
    return np.clip(img * (1 - w[..., None]) + y * w[..., None], 0, 1)


def op_grade(img, fr, p, st):
    """Tone: exposure (EV), contrast around middle grey 0.18, saturation, hue — in linear colour."""
    lin = srgb_to_lin(img) * (2.0 ** float(p.get("exposure", 0.0))) * np.asarray(p.get("tint", [1, 1, 1]), np.float32)
    c = float(p.get("contrast", 0.0))
    if c: lin = 0.18 * np.power(np.clip(lin, 0, None) / 0.18, 1 + c)
    s = float(p.get("saturation", 1.0))
    if s != 1.0:
        y = luma(lin)[..., None]; lin = y + (lin - y) * s
    return lin_to_srgb(lin)


def _sun_screen(fr):
    """The sun in frame coordinates (fractions of width and height, top to bottom; may be beyond the edge) or None."""
    if fr.sun is None or "matrix_world" not in fr.cam: return None
    M = np.asarray(fr.cam["matrix_world"], np.float64)[:3, :3]
    v = M.T @ np.asarray(fr.sun, np.float64)               # in camera axes: it looks along −z
    if v[2] >= -1e-3: return None
    f = float(fr.cam.get("lens_mm", 50.0)) / float(fr.cam.get("sensor_width_mm", 36.0))
    W, H = fr.size
    x = 0.5 + f * v[0] / -v[2]
    y = 0.5 - f * v[1] / -v[2] * W / H
    return x, y


def op_para(img, fr, p, st):
    """Sun glow (anime satsuei "para", パラ): a soft gradient from the sun (or from a frame corner) blended in screen mode — the colour and warmth of the light."""
    W, H = fr.size
    pos = _sun_screen(fr) or tuple(p.get("from", [0.85, -0.1]))
    st["para_from"] = [round(float(v), 2) for v in pos]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.hypot(xx / W - pos[0], (yy / H - pos[1]) * H / W)
    g = np.clip(1 - d / float(p.get("spread", 1.1)), 0, 1) ** 2 * float(p.get("strength", 0.2))
    col = np.asarray(p.get("colour", [1.0, 0.78, 0.5]), np.float32)
    return 1 - (1 - img) * (1 - g[..., None] * col)


def op_grain(img, fr, p, st):
    """Grain: brightness noise of size size pixels, stronger in the midtones; its own for each frame. Do not use it for Shorts —
    the platform smooths it out and the shadow detail pays for it."""
    W, H = fr.size; s = max(1.0, float(p.get("size", 1.5)))
    rng = np.random.default_rng(1000 + int(fr.index))
    n = rng.normal(0, 1, (max(1, round(H / s)), max(1, round(W / s)))).astype(np.float32)
    n = cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC)
    y = luma(img)
    return img + (float(p.get("strength", 0.03)) * n * 4 * y * (1 - y))[..., None]


def op_letterbox(img, fr, p, st):
    """Letterbox: black bars up to the aspect ratio aspect; a frame already no wider than that is left alone (vertical 9:16 — no letterbox)."""
    W, H = fr.size; a = float(p.get("aspect", 2.39))
    if H > W: st.setdefault("notes", []).append("letterbox: vertical frame — skipped"); return img
    if W / H >= a - 1e-3: st.setdefault("notes", []).append("letterbox: the frame is already wide enough — skipped"); return img
    h = int(round((H - W / a) / 2))
    out = img.copy(); out[:h] = 0; out[H - h:] = 0
    return out


def op_dither(img, fr, p, st):
    """Dithering before 8 bits: triangular noise of ±1 code value — no banding in dark gradients."""
    rng = np.random.default_rng(7000 + int(fr.index))
    t = (rng.random(img.shape[:2], np.float32) + rng.random(img.shape[:2], np.float32) - 1.0)[..., None]
    return img + float(p.get("amount", 1.0)) * t / 255.0


OPS = {k: globals()[f"op_{k}"] for k in ORDER}


def apply(fr, ops, stats=None):
    """The preset's operations in the ORDER order → sRGB 0..1; stats — notes (what was skipped and why)."""
    st = stats if stats is not None else {}
    img = fr.img.astype(np.float32)
    for k in ORDER:
        if k in ops:
            img = np.clip(OPS[k](img, fr, ops[k] if isinstance(ops[k], dict) else {"strength": ops[k]}, st), 0, 1).astype(np.float32)
    return img


def _parse_set(items):
    out = {}
    for it in items:
        k, v = it.split("=", 1)
        op, _, key = k.partition(".")
        val = yaml.safe_load(v)
        if key: out.setdefault(op, {})[key] = val
        else: out[op] = val
    return out


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--preset", default=None); ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--depth"); ap.add_argument("--camera")
    a = ap.parse_args(argv)
    if os.path.isdir(a.src):
        fr = frame_from_dir(a.src)
    else:
        img = read_png(a.src)
        cam = (json.load(open(a.camera)).get("camera") or json.load(open(a.camera))) if a.camera else {}
        fr = Frame(img, _fit_depth(_depth_file(a.depth), img) if a.depth else None, cam)
    st = {}
    write_png(a.out, apply(fr, resolve(a.preset, _parse_set(a.set)), st))
    print("__RESULT__" + json.dumps({"out": a.out, **st}, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])

"""A marker sketch for a closed model (sb): three tones in patches, figures as lumps, rare trembling construction lines.
The logic and the numbers come from sketch25/rough.py (marker), but the inputs are Blender frame maps rather than the old engine's raster.
The technique (from the old project): the model takes the layout and masses, and draws the form and pose itself."""
import cv2
import numpy as np

CROWD_PX = 24.0      # height in the frame, px: smaller means part of the crowd mass; from it up to FIG_BIG — a silhouette with a gesture
FIG_BIG = 70.0       # from it up the figure is drawn in parts and volume


def vnoise(h, w, scale, rng):
    gh, gw = max(2, int(h / scale) + 2), max(2, int(w / scale) + 2)
    g = rng.uniform(-1, 1, (gh, gw)).astype(np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)


def fbm(h, w, scale, rng, octaves=3):
    out, amp, tot = np.zeros((h, w), np.float32), 1.0, 0.0
    for i in range(octaves):
        out += amp * vnoise(h, w, max(2.0, scale / 2 ** i), rng)
        tot += amp
        amp *= 0.5
    return out / tot


def warp(img, amp, scale, rng):
    """Each pixel is displaced by a smooth field — lines and edges "tremble" as if drawn by hand."""
    h, w = img.shape[:2]
    dx, dy = fbm(h, w, scale, rng) * amp, fbm(h, w, scale, rng) * amp
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img.astype(np.float32), gx + dx, gy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def paper(h, w, rng, base=0.95):
    return np.clip(base + fbm(h, w, 160, rng) * 0.025 + rng.normal(0, 0.012, (h, w)).astype(np.float32), 0, 1)


def strokes(cov, rng, amp, keep, double=True):
    h, w = cov.shape
    a = warp(cov, amp, 45, rng)
    gaps = (fbm(h, w, 22, rng) > -keep).astype(np.float32)
    s = a * gaps
    if double:
        s = np.maximum(s, warp(cov, amp * 1.6, 80, rng) * 0.55 * (fbm(h, w, 30, rng) > -keep * 0.8))
    return np.clip(s, 0, 1)


def lumpy(fig, rng, grow=5, amp=7.0):
    """A figure silhouette → a lump: wider and with an uneven edge, without the exact outline of arms and weapons."""
    m = cv2.dilate(fig.astype(np.uint8), np.ones((grow, grow), np.uint8)).astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), grow * 0.6)
    return warp(m, amp, 18, rng) > 0.45


def toneline(tone, cov):
    return np.clip(tone * (1 - 0.85 * cov), 0, 1)


def make(tone, cov, fig, notan, big, small, mid, seed=1):
    """A marker storyboard. All inputs are (H, W): tone, cov, notan — float 0..1; fig, big, small, mid — bool."""
    rng = np.random.default_rng(seed)
    h, w = tone.shape
    n = cv2.GaussianBlur(notan.astype(np.float32), (0, 0), 2.5)
    n = warp(n, 8, 45, rng)
    q = np.where(n < 0.33, 0.28, np.where(n < 0.75, 0.62, 0.93)).astype(np.float32)   # three markers: dark, medium, paper
    q = cv2.GaussianBlur(q, (0, 0), 1.2) * (1 + 0.05 * fbm(h, w, 12, rng, octaves=2))
    # small and distant ones are a crowd mass: a dark band with a ragged top, the figures themselves visible inside
    lump = lumpy(small, rng, grow=3, amp=2.5)
    q = np.where(lump, np.minimum(q, 0.36 + 0.06 * fbm(h, w, 10, rng)), q)
    q = np.where(warp(small.astype(np.float32), 1.0, 20, rng) > 0.5, np.minimum(q, 0.22), q)
    # medium figures and vehicles are silhouettes without inflation: the gesture and the outline must read
    q = np.where(warp(mid.astype(np.float32), 1.5, 25, rng) > 0.5, np.minimum(q, 0.26 + 0.05 * fbm(h, w, 10, rng)), q)
    # large ones are drawn in parts and volume, as in the base sketch, with a slight tremble
    figt = warp(toneline(tone, cov), 1.5, 25, rng)
    bigw = warp(big.astype(np.float32), 1.5, 25, rng) > 0.5
    q = np.where(bigw, figt * 0.95, q)
    struct = strokes(cov * (~cv2.dilate(fig.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)), rng, 5, 0.1,
                     double=False) * (~bigw)
    img = q * paper(h, w, rng, 1.0)
    img *= 1 - 0.7 * struct
    return np.clip(img, 0, 1)

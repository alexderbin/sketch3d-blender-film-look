"""Gradient noise by seed (Perlin), fractal sums fbm and ridged — pure numpy, no third-party libraries.
Shared by the core (ground textures) and components (relief, meadow); moved here from components/site/terrain.py (30.10)
so the core does not import a component. Results are bit-identical to the former terrain.perlin/fbm/ridged."""
import numpy as np

_PERM = {}
_GRAD = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1], [1.41, 0], [-1.41, 0], [0, 1.41], [0, -1.41]], float)


def _perm(seed):
    if seed not in _PERM:
        p = np.random.default_rng(seed).permutation(256)
        _PERM[seed] = np.concatenate([p, p]).astype(np.int64)
    return _PERM[seed]


def perlin(x, y, seed=0):
    p = _perm(int(seed) & 0xFFFF)
    x0, y0 = np.floor(x), np.floor(y); fx, fy = x - x0, y - y0
    xi, yi = x0.astype(np.int64) & 255, y0.astype(np.int64) & 255
    u, v = fx ** 3 * (fx * (fx * 6 - 15) + 10), fy ** 3 * (fy * (fy * 6 - 15) + 10)
    a, b = p[xi] + yi, p[xi + 1] + yi
    g = lambda h, dx, dy: _GRAD[h & 7, 0] * dx + _GRAD[h & 7, 1] * dy
    n0 = g(p[a], fx, fy) * (1 - u) + g(p[b], fx - 1, fy) * u
    n1 = g(p[a + 1], fx, fy - 1) * (1 - u) + g(p[b + 1], fx - 1, fy - 1) * u
    return n0 * (1 - v) + n1 * v


def fbm(x, y, octaves=5, seed=0, gain=0.5):
    """A sum of octaves, roughly −1…1."""
    s, a, norm = np.zeros(np.shape(x)), 1.0, 0.0
    for i in range(octaves):
        f = 2.0 ** i
        s = s + a * perlin(x * f + 17.3 * i, y * f - 9.1 * i, seed + 31 * i); norm += a; a *= gain
    return s / norm * 1.35


def ridged(x, y, octaves=5, seed=0, gain=0.5):
    """Ridges (Musgrave multifractal): 0…1, sharp ridgelines."""
    s, a, w, norm = np.zeros(np.shape(x)), 1.0, np.ones(np.shape(x)), 0.0
    for i in range(octaves):
        f = 2.0 ** i
        n = np.clip(1 - np.abs(perlin(x * f + 5.7 * i, y * f + 3.1 * i, seed + 53 * i)) * 1.35, 0, 1) ** 2 * w
        w = np.clip(n * 2, 0, 1); s = s + a * n; norm += a; a *= gain
    return s / norm


def smoothstep(e0, e1, x):
    """Smooth threshold (Hermite): 0 up to e0, 1 after e1."""
    t = np.clip((x - e0) / np.where(np.abs(e1 - e0) < 1e-9, 1e-9, e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)

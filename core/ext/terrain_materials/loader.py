"""Infinigen terrain materials in our Blender 5.2: their modules (the infinigen/ folder, unmodified) + gin (pylib/) + stubs for heavy
imports that the materials do not call (trimesh, cv2, tqdm; matplotlib.colors — only rgb↔hsv, reimplemented via colorsys)
+ a shim for the 4.2 → 5.2 node sockets (bl52_compat).
In Blender: sys.path.insert(0, <this folder>); import loader as IG; IG.MATS["dirt"]().apply(obj, selection="w_dirt")."""
import colorsys, importlib, os, sys, types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
for p in (HERE, os.path.join(HERE, "pylib")):
    if p not in sys.path: sys.path.insert(0, p)


def _stub(name, **attrs):
    if name in sys.modules: return sys.modules[name]
    m = types.ModuleType(name); m.__dict__.update(attrs); m.__path__ = []; sys.modules[name] = m
    parent, _, leaf = name.rpartition(".")
    if parent in sys.modules: setattr(sys.modules[parent], leaf, m)
    return m


def _rgb_to_hsv(a):
    a = np.asarray(a, float); f = a.reshape(-1, 3); return np.array([colorsys.rgb_to_hsv(*c) for c in f]).reshape(a.shape)


def _hsv_to_rgb(a):
    a = np.asarray(a, float); f = a.reshape(-1, 3); return np.array([colorsys.hsv_to_rgb(*c) for c in f]).reshape(a.shape)


def _no(*a, **k): raise NotImplementedError("not needed by the terrain materials")


_stub("trimesh", Trimesh=_no)
_stub("cv2", resize=_no, INTER_LANCZOS4=4)
_stub("tqdm", tqdm=lambda it=None, *a, **k: it, trange=lambda *a, **k: range(*a))
_stub("matplotlib"); _stub("matplotlib.colors", rgb_to_hsv=_rgb_to_hsv, hsv_to_rgb=_hsv_to_rgb)

import bl52_compat  # noqa: E402
bl52_compat.install()

NAMES = ("mountain", "sandstone", "stone", "chunkyrock", "cobble_stone", "cracked_ground", "dirt", "soil", "mud", "sand", "ice")
MATS = {}
for n in NAMES:
    M = importlib.import_module(f"infinigen.assets.materials.terrain.{n}")
    MATS[n] = next(v for v in vars(M).values() if isinstance(v, type) and hasattr(v, "apply") and v.__module__ == M.__name__)
from infinigen.core.util.math import FixedSeed  # noqa: E402  (the seed of colors and pattern comes from the site)
from infinigen.core import surface  # noqa: E402
from infinigen.core.nodes.node_wrangler import NodeWrangler  # noqa: E402,F401

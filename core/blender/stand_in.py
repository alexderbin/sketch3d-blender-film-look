"""Sketch stand-ins (the sketch stage): primitives within the model's extents. The object's owner decides which primitive and what
size (from its own plan), the module decides how to build it: a mesh in meters, the bottom at z = 0, the axis +z, normals outward.
The tags are the object contract (core/contract/contract.py): sk_stand_in = 1, sk_kind — the class, sk_id — the model id (one
object per model) or none (a composite object of a class: a thousand trees are one object, since Workbench pays per object, not
per face). The color is the attribute Col (BYTE_COLOR, per face), drawn by the sketch frame (core/outputs/bl_sketch.py); the
material reads the same attribute, so Eevee sees the color too.
    crown = stand_in.lathe(stand_in.ellipse_profile(3.0, 4.0))              # a solid of revolution from the profile (r, z), bottom to top
    parts = [stand_in.place(crown, at=(x, y, z + 2.5), rgb=(0.10, 0.22, 0.07), smooth=True),
             stand_in.place(stand_in.cylinder(0.3, 2.6), at=(x, y, z), kind="trunk")]
    ob = stand_in.build("forest_stand_in", parts, coll, kind="tree")"""
import math

import bpy
import numpy as np

from core.contract import contract as C

KIND_RGB = {"tree": (0.10, 0.22, 0.07), "trunk": (0.20, 0.14, 0.08), "rock": (0.40, 0.39, 0.37), "building": (0.55, 0.50, 0.42),
            "roof": (0.36, 0.18, 0.12), "figure": (0.30, 0.30, 0.34), "fx": (0.95, 0.55, 0.15), "smoke": (0.26, 0.25, 0.24),
            "dust": (0.42, 0.36, 0.28), "ground": (0.30, 0.28, 0.20), "water": (0.10, 0.18, 0.25)}   # linear values; a class without its own color
GREY = (0.5, 0.5, 0.5)


def lathe(profile, n=12):
    """A solid of revolution around +z from the profile [(r, z), …] bottom to top; r = 0 is a pole (a single vertex). A cylinder, cone,
    crown, capsule, dome, spire — with one code. → (V, F)."""
    a = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    V, rings = [], []
    for r, z in profile:
        if r <= 0: rings.append([len(V)]); V.append((0.0, 0.0, z))
        else: rings.append(list(range(len(V), len(V) + n))); V += [(r * math.cos(t), r * math.sin(t), z) for t in a]
    F = []
    for lo, hi in zip(rings, rings[1:]):
        if len(lo) == 1 and len(hi) == 1: continue
        for i in range(n):
            j = (i + 1) % n
            if len(lo) == 1: F.append([lo[0], hi[j], hi[i]])                  # from the bottom pole — the underside faces down
            elif len(hi) == 1: F.append([lo[i], lo[j], hi[0]])                # to the top pole
            else: F.append([lo[i], lo[j], hi[j], hi[i]])
    return np.asarray(V, float), F


def cylinder(r, h, n=12):
    return lathe([(0, 0), (r, 0), (r, h), (0, h)], n)


def cone(r, h, n=12, r_top=0.0):
    return lathe([(0, 0), (r, 0), (r_top, h)] + ([(0, h)] if r_top > 0 else []), n)


def ellipse_profile(r, h, m=7, z0=0.0):
    """An ellipsoid profile (a crown, a puff of smoke, a dust dome): radius r, height h, the bottom at z0."""
    return [(r * math.sin(math.pi * j / m), z0 + h * (1 - math.cos(math.pi * j / m)) / 2) for j in range(m + 1)]


def ellipsoid(rx, ry, rz, n=12, m=7):
    """An ellipsoid of height 2·rz, the bottom at z = 0; ry ≠ rx — flattened along y."""
    V, F = lathe(ellipse_profile(1.0, 2 * rz, m), n)
    return V * np.array([rx, ry, 1.0]), F


def capsule(r, h, n=10, m=4):
    """A capsule (a crowd figure, a smoke column): radius r, full height h (hemispheres at the ends)."""
    r = min(r, h / 2)
    bot = [(r * math.sin(math.pi / 2 * j / m), r - r * math.cos(math.pi / 2 * j / m)) for j in range(m + 1)]
    top = [(r * math.cos(math.pi / 2 * j / m), h - r + r * math.sin(math.pi / 2 * j / m)) for j in range(m + 1)]
    return lathe(bot + top, n)


def box(w, d, h):
    x, y = w / 2, d / 2
    V = np.array([[-x, -y, 0], [x, -y, 0], [x, y, 0], [-x, y, 0], [-x, -y, h], [x, -y, h], [x, y, h], [-x, y, h]], float)
    return V, [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]


def prism(poly, h, roof_h=0.0):
    """A prism over a footprint [(x, y), …] counterclockwise, height h; roof_h > 0 — a hipped roof toward the footprint center (the roof of a convex footprint)."""
    p = np.asarray(poly, float); n = len(p)
    V = np.vstack([np.c_[p, np.zeros(n)], np.c_[p, np.full(n, h)]])
    F = [list(range(n - 1, -1, -1))] + [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    if roof_h > 0:
        V = np.vstack([V, [[*p.mean(0), h + roof_h]]]); F += [[n + i, n + (i + 1) % n, 2 * n] for i in range(n)]
    else: F.append(list(range(n, 2 * n)))
    return V, F


def place(shape, at=(0.0, 0.0, 0.0), yaw=0.0, scale=1.0, rgb=None, kind=None, smooth=False):
    """A primitive in the world: scale (a number or (sx, sy, sz)), rotation around z (rad), offset; the color is rgb, otherwise the color of the class kind."""
    V, F = shape
    c, s = math.cos(yaw), math.sin(yaw)
    V = (V * np.asarray(scale, float)) @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float).T + np.asarray(at, float)
    return {"V": V, "F": F, "rgb": rgb or KIND_RGB.get(kind), "smooth": smooth}


def material():
    """The shared stand-in material "stand_in": Base Color is the Col attribute (Eevee sees the same color as Workbench)."""
    m = bpy.data.materials.get("stand_in")
    if m is None:
        m = bpy.data.materials.new("stand_in"); m.use_nodes = True; nt = m.node_tree
        bs = next(x for x in nt.nodes if x.type == "BSDF_PRINCIPLED"); at = nt.nodes.new("ShaderNodeAttribute"); at.attribute_name = "Col"
        nt.links.new(at.outputs["Color"], bs.inputs["Base Color"]); bs.inputs["Roughness"].default_value = 0.9
        m.diffuse_color = (*GREY, 1.0)
    return m


def build(name, parts, coll=None, kind=None, sk_id=None):
    """A single object from place() parts: the mesh, the Col color per face, the parts' smoothness, the stand-in tags. → an object (in coll or in the scene)."""
    V, F, cols, smooth, off = [], [], [], [], 0
    base = KIND_RGB.get(kind, GREY)
    for p in parts:
        V.append(p["V"]); F += [[i + off for i in f] for f in p["F"]]; off += len(p["V"])
        cols += [p.get("rgb") or base] * len(p["F"]); smooth += [bool(p.get("smooth"))] * len(p["F"])
    me = bpy.data.meshes.new(name)
    me.from_pydata(np.vstack(V).tolist() if V else [], [], F); me.update()
    if F:
        lt = np.array([len(f) for f in F]); rgba = np.c_[np.asarray(cols, float), np.ones(len(cols))]
        ca = me.color_attributes.new("Col", "BYTE_COLOR", "CORNER"); ca.data.foreach_set("color", np.repeat(rgba, lt, axis=0).ravel())
        me.color_attributes.active_color = ca
        me.polygons.foreach_set("use_smooth", smooth)
    me.materials.append(material())
    ob = bpy.data.objects.new(name, me)
    (coll or bpy.context.scene.collection).objects.link(ob)
    ob[C.SK_STAND_IN] = 1
    if kind: ob[C.SK_KIND] = kind
    if sk_id is not None: ob[C.SK_ID] = sk_id
    return ob

"""Material surface — a texture of the substance (stone, mortar) on masonry and buildings, with no repetition (the blocks
and the difference in their tone are the masonry logic in components/constructions/masonry/bl_masonry.py; a building
pattern baked into the texture would conflict with it). Two ready-made routes, both on top of
the masonry stone tone (the Col attribute: a light/dark block is its decision; the surface only adds the substance
pattern, normalized to 1):
  A. procedural stone (the stone, sandstone, chunkyrock, mountain materials):
     3D noise, no repetition by construction; each block has its own noise shift (as if every block was hewn from a
     different place in the boulder).
  B. a surface scan (Poly Haven CC0, library/materials/surfaces.yaml → core/surface/surface_tex.py): each block gets its
     own piece of the scan: a shift and rotation by the stone number (a random offset per brick), the normal map is rotated
     by the same angle; the UV map — bl_matlib.plane_uv (meters on the face plane).
  Mortar — procedural sandy 3D noise (the mortar body is solid — there is nothing to cut).
Per-block randomness comes from an integer hash of the stone number in Python (face attribute sk_rnd, 3 numbers 0…1):
masonry numbers go up to 5·10^7, while the shader's float is exact only up to 1.7·10^7 — neighboring blocks would get
the same piece.
  import bl_surface as SF; SF.block_random(ob); m = SF.ig_stone("stone"); SF.apply_masonry(scene, {"wall": m, …})"""
import core.paths        # noqa: E402  core parts on sys.path
import json
import math
import os
import sys

import bpy
import numpy as np

ROOT = core.paths.ROOT
TEX = os.path.join(ROOT, "library", "textures", "surface")
RND = "sk_rnd"


def block_random(ob, attr="sk_stone", out=RND):
    """Three random numbers 0…1 per face from the block number (splitmix64 on int64 — exact for any numbers) → the face
    attribute out (FLOAT_VECTOR). No number — None."""
    me = ob.data
    a = me.attributes.get(attr)
    if a is None or a.domain != "FACE":
        return None
    ids = np.empty(len(me.polygons), np.int32); a.data.foreach_get("value", ids)
    z = ids.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    vals = []
    for k in range(3):
        x = z + np.uint64(k) * np.uint64(0xD1B54A32D192ED03)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
        vals.append((x >> np.uint64(40)).astype(np.float64) / float(1 << 24))
    v = np.stack(vals, 1).astype(np.float32)
    at = me.attributes.get(out) or me.attributes.new(out, "FLOAT_VECTOR", "FACE")
    at.data.foreach_set("vector", v.ravel())
    return out


class _N:
    """Small helpers for the nodes of a single node tree."""

    def __init__(self, nt):
        self.nt, self.L = nt, nt.links

    def new(self, kind, **inp):
        n = self.nt.nodes.new(kind)
        for k, v in inp.items():
            self.put(n.inputs[k], v)
        return n

    def put(self, sock, x):
        if hasattr(x, "is_output") or hasattr(x, "links") and hasattr(x, "node"):
            self.L.new(x, sock)
        elif hasattr(x, "outputs"):
            self.L.new(x.outputs[0], sock)
        else:
            sock.default_value = x

    def math(self, op, a, b=None):
        n = self.new("ShaderNodeMath"); n.operation = op; self.put(n.inputs[0], a)
        if b is not None:
            self.put(n.inputs[1], b)
        return n.outputs[0]

    def vmath(self, op, a, b=None, s=None):
        n = self.new("ShaderNodeVectorMath"); n.operation = op; self.put(n.inputs[0], a)
        if b is not None:
            self.put(n.inputs[1], b)
        if s is not None:
            self.put(n.inputs["Scale"], s)
        return n.outputs[0] if op != "DOT_PRODUCT" else n.outputs["Value"]

    def attr(self, name, out="Vector"):
        n = self.new("ShaderNodeAttribute"); n.attribute_type = "GEOMETRY"; n.attribute_name = name
        return n.outputs[out]

    def mix_rgb(self, kind, f, a, b, clamp=False):
        n = self.new("ShaderNodeMix"); n.data_type = "RGBA"; n.blend_type = kind; n.clamp_result = clamp
        ins = {x.identifier: x for x in n.inputs}
        self.put(ins["Factor_Float"], f); self.put(ins["A_Color"], a); self.put(ins["B_Color"], b)
        return next(x for x in n.outputs if x.identifier == "Result_Color")


def _tone(nt, bsdf, mean_rgb, strength=1.0, tone_attr="Col"):
    """Base Color → masonry tone × substance pattern: Col × lerp(1, color / mean color, strength) (the pattern is normalized to
    1: light and dark blocks stay as the masonry set them)."""
    N = _N(nt); s = bsdf.inputs["Base Color"]
    src = s.links[0].from_socket if s.is_linked else None
    if src is None:
        return
    m = [max(float(c), 1e-4) for c in mean_rgb]
    norm = N.mix_rgb("DIVIDE", 1.0, src, (m[0], m[1], m[2], 1.0))
    pat = N.mix_rgb("MIX", strength, (1.0, 1.0, 1.0, 1.0), norm)
    col = N.attr(tone_attr, "Color")
    N.put(s, N.mix_rgb("MULTIPLY", 1.0, col, pat))


def _relink_position(nt, scale, offset_amp, rnd=RND):
    """All inputs that took the position (Geometry → Position, Texture Coordinate → Object / Generated) now take
    P · scale + (rnd − 0.5) · offset_amp: each block gets its own piece of noise, the pattern scale is scale."""
    N = _N(nt)
    geo = N.new("ShaderNodeNewGeometry")
    p = N.vmath("SCALE", geo.outputs["Position"], s=scale)
    off = N.vmath("SCALE", N.vmath("SUBTRACT", N.attr(rnd), (0.5, 0.5, 0.5)), s=offset_amp)
    pos = N.vmath("ADD", p, off)
    k = 0
    for n in list(nt.nodes):
        if n == geo:
            continue
        outs = []
        if n.type == "NEW_GEOMETRY":
            outs = [n.outputs["Position"]]
        elif n.type == "TEX_COORD":
            outs = [n.outputs["Object"], n.outputs["Generated"]]
        for o in outs:
            for l in list(o.links):
                nt.links.new(pos, l.to_socket); k += 1
    return k


def _ig():
    p = os.path.join(core.paths.EXT, "terrain_materials")
    if p not in sys.path:
        core.paths.ext("terrain_materials")
    import loader as IG
    return IG


def ig_stone(kind="stone", seed=1, scale=2.5, offset_amp=200.0, strength=1.0, rough=None, name=None):
    """A: procedural stone from the terrain materials package (kind: stone | sandstone | chunkyrock | mountain) on masonry blocks. scale — how many times
    finer the package's relief pattern is (its own scale is meant for cliffs); rough — own roughness (None — the package's)."""
    IG = _ig()
    from core.surface import bl_terrain_surf as TS
    NodeWrangler = IG.NodeWrangler
    m = bpy.data.materials.new(name or f"sk_surf_{kind}"); nt = m.node_tree; nt.nodes.clear()
    with IG.FixedSeed(int(seed)):
        b = TS._shader_of(IG, kind)(NodeWrangler(nt))
    bsdf_out = b.outputs[0] if hasattr(b, "outputs") else b
    out = nt.nodes.new("ShaderNodeOutputMaterial"); nt.links.new(bsdf_out, out.inputs["Surface"])
    bsdf = bsdf_out.node if bsdf_out.node.type == "BSDF_PRINCIPLED" else next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    k = _relink_position(nt, scale, offset_amp)
    if rough is not None:
        for l in list(bsdf.inputs["Roughness"].links):
            nt.links.remove(l)
        bsdf.inputs["Roughness"].default_value = rough
    from core.surface import bl_matlib as ML
    mean = ML.albedo(m) or (0.3, 0.3, 0.3)
    _tone(nt, bsdf, mean, strength)
    m["sk_surface"] = {"path": "procedural", "kind": kind, "scale": scale, "position_inputs": k,
                       "medium": [round(x, 4) for x in mean]}
    return m


def scan_meta(sid):
    p = os.path.join(TEX, sid, "meta.json")
    if not os.path.exists(p):
        raise SystemExit(f"bl_surface: no surface {sid} — .venv/bin/python -m core.surface.surface_tex fetch")
    return json.load(open(p))


def _img(path, color):
    im = bpy.data.images.load(os.path.join(ROOT, path), check_existing=True)
    if not color:
        im.colorspace_settings.name = "Non-Color"
    return im


def scan_stone(sid, strength=1.0, rotate=True, bump=0.25, uv="sk_uv", name=None, rnd=RND):
    """B: a surface scan (Poly Haven) on blocks: square — its size in meters; per block — a shift (rnd.yz) and rotation
    (rnd.x · 2π) of the piece; the scan's normal is rotated by −angle (the tangents come from the uv map)."""
    mt = scan_meta(sid); size = float(mt["size_m"][0])
    m = bpy.data.materials.new(name or f"sk_scan_{sid}"); nt = m.node_tree; N = _N(nt)
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    r = N.new("ShaderNodeSeparateXYZ"); N.put(r.inputs[0], N.attr(rnd))
    ang = N.math("MULTIPLY", r.outputs["X"], 2 * math.pi) if rotate else 0.0
    uvn = N.new("ShaderNodeUVMap"); uvn.uv_map = uv
    v = N.vmath("SCALE", uvn.outputs["UV"], s=1.0 / size)
    rot = N.new("ShaderNodeVectorRotate"); rot.rotation_type = "Z_AXIS"; N.put(rot.inputs["Vector"], v); N.put(rot.inputs["Angle"], ang)
    sh = N.new("ShaderNodeCombineXYZ"); N.put(sh.inputs["X"], N.math("MULTIPLY", r.outputs["Y"], 37.0))
    N.put(sh.inputs["Y"], N.math("MULTIPLY", r.outputs["Z"], 37.0))
    vec = N.vmath("ADD", rot.outputs[0], sh.outputs[0])

    def tex(key, color):
        t = N.new("ShaderNodeTexImage"); t.image = _img(mt["maps"][key], color); N.put(t.inputs["Vector"], vec)
        return t
    dif, nor, rgh = tex("Diffuse", True), tex("nor_gl", False), tex("Rough", False)
    N.put(bsdf.inputs["Base Color"], dif.outputs["Color"])
    N.put(bsdf.inputs["Roughness"], rgh.outputs["Color"])
    n2 = N.vmath("SUBTRACT", N.vmath("SCALE", nor.outputs["Color"], s=2.0), (1.0, 1.0, 1.0))   # normal −1…1
    nr = N.new("ShaderNodeVectorRotate"); nr.rotation_type = "Z_AXIS"; N.put(nr.inputs["Vector"], n2)
    N.put(nr.inputs["Angle"], N.math("MULTIPLY", ang, -1.0) if rotate else 0.0)
    n01 = N.vmath("ADD", N.vmath("SCALE", nr.outputs[0], s=0.5), (0.5, 0.5, 0.5))
    nm = N.new("ShaderNodeNormalMap"); nm.uv_map = uv; N.put(nm.inputs["Color"], n01); nm.inputs["Strength"].default_value = 1.0
    if bump > 0 and "Displacement" in mt["maps"]:
        dsp = tex("Displacement", False)
        bp = N.new("ShaderNodeBump"); bp.inputs["Strength"].default_value = bump; bp.inputs["Distance"].default_value = 0.004
        N.put(bp.inputs["Height"], dsp.outputs["Color"]); N.put(bp.inputs["Normal"], nm.outputs["Normal"])
        N.put(bsdf.inputs["Normal"], bp.outputs["Normal"])
    else:
        N.put(bsdf.inputs["Normal"], nm.outputs["Normal"])
    px = np.empty(dif.image.size[0] * dif.image.size[1] * dif.image.channels, np.float32); dif.image.pixels.foreach_get(px)
    g = px.reshape(-1, dif.image.channels)[::7, :3]
    lin = np.where(g <= 0.04045, g / 12.92, ((g + 0.055) / 1.055) ** 2.4).mean(0)   # mean scan color (linear)
    _tone(nt, bsdf, lin, strength)
    m["sk_surface"] = {"path": "scan", "id": sid, "square_m": size, "turn": bool(rotate), "medium": [round(float(x), 4) for x in lin]}
    return m


def _hex(N, st, rot=0.5):
    """Stochastic hexagonal tiling (Mikkelsen, JCGT 2022, "Practical Real-Time Hex-Tiling"; the computation follows the
    mmikk/noddingSloth port, MIT): a triangular grid in skewed coordinates, three vertices → three pieces of the tile, each
    with its own shift and rotation (ours uses a vertex White Noise: fract(sin) loses precision on large coordinates on the
    GPU), the weights are barycentrics^7. → [(piece vector, weight^7, angle)] × 3."""
    S3 = 2.0 * math.sqrt(3.0)
    sep = N.new("ShaderNodeSeparateXYZ"); N.put(sep.inputs[0], st)
    x, y = N.math("MULTIPLY", sep.outputs["X"], S3), N.math("MULTIPLY", sep.outputs["Y"], S3)
    sx = N.math("SUBTRACT", x, N.math("MULTIPLY", y, 0.57735027)); sy = N.math("MULTIPLY", y, 1.15470054)
    bx, by = N.math("FLOOR", sx), N.math("FLOOR", sy)
    fx, fy = N.math("SUBTRACT", sx, bx), N.math("SUBTRACT", sy, by)
    z = N.math("SUBTRACT", N.math("SUBTRACT", 1.0, fx), fy)
    s = N.math("LESS_THAN", z, 1e-7); s2 = N.math("SUBTRACT", N.math("MULTIPLY", s, 2.0), 1.0); ns = N.math("SUBTRACT", 1.0, s)
    ws = [N.math("MULTIPLY", N.math("MULTIPLY", z, -1.0), s2),
          N.math("SUBTRACT", s, N.math("MULTIPLY", fy, s2)),
          N.math("SUBTRACT", s, N.math("MULTIPLY", fx, s2))]
    verts = [(N.math("ADD", bx, s), N.math("ADD", by, s)), (N.math("ADD", bx, s), N.math("ADD", by, ns)),
             (N.math("ADD", bx, ns), N.math("ADD", by, s))]
    out = []
    for (vx, vy), w in zip(verts, ws):
        cv = N.new("ShaderNodeCombineXYZ"); N.put(cv.inputs["X"], vx); N.put(cv.inputs["Y"], vy)
        wn = N.new("ShaderNodeTexWhiteNoise"); wn.noise_dimensions = "3D"; N.put(wn.inputs["Vector"], cv.outputs[0])
        rs = N.new("ShaderNodeSeparateColor"); N.put(rs.inputs[0], wn.outputs["Color"])
        ang = N.math("MULTIPLY", N.math("SUBTRACT", wn.outputs["Value"], 0.5), 2.0 * math.pi * rot)
        cen = N.new("ShaderNodeCombineXYZ")
        N.put(cen.inputs["X"], N.math("DIVIDE", N.math("ADD", vx, N.math("MULTIPLY", vy, 0.5)), S3))
        N.put(cen.inputs["Y"], N.math("DIVIDE", N.math("MULTIPLY", vy, 0.8660254), S3))
        rv = N.new("ShaderNodeVectorRotate"); rv.rotation_type = "Z_AXIS"
        N.put(rv.inputs["Vector"], st); N.put(rv.inputs["Center"], cen.outputs[0]); N.put(rv.inputs["Angle"], ang)
        off = N.new("ShaderNodeCombineXYZ"); N.put(off.inputs["X"], N.math("MULTIPLY", rs.outputs["Red"], 13.0))
        N.put(off.inputs["Y"], N.math("MULTIPLY", rs.outputs["Green"], 13.0))
        out.append((N.vmath("ADD", rv.outputs[0], off.outputs[0]), N.math("POWER", N.math("MAXIMUM", w, 0.0), 7.0), ang))
    return out


def _gain3(N, ws, r=0.75):
    """Mikkelsen weight contrast (Gain3, r = 0.75 → k = 2): x < 0.5 → 2x², otherwise 1 − 2(1 − x)²; then normalization."""
    g = []
    for w in ws:
        lo = N.math("MULTIPLY", N.math("POWER", w, 2.0), 2.0)
        hi = N.math("SUBTRACT", 1.0, N.math("MULTIPLY", N.math("POWER", N.math("SUBTRACT", 1.0, w), 2.0), 2.0))
        c = N.math("LESS_THAN", w, 0.5)
        g.append(N.math("ADD", N.math("MULTIPLY", c, lo), N.math("MULTIPLY", N.math("SUBTRACT", 1.0, c), hi)))
    tot = N.math("ADD", N.math("ADD", g[0], g[1]), g[2])
    return [N.math("DIVIDE", x, N.math("MAXIMUM", tot, 1e-6)) for x in g]


def hex_maps(N, st, maps, rot=0.5, falloff=0.6):
    """Maps {kind: (image, color?)} tiled with Mikkelsen's hex-tiling by the vector st (in tile squares). The weights are shared
    by all maps (by color brightness, falloff 0.6, Gain3 0.75); the piece's normal is rotated by −angle. → {kind: output}."""
    H = _hex(N, st, rot)
    smp = {k: [] for k in maps}
    for vec, _w, _a in H:
        for k, (im, _c) in maps.items():
            t = N.new("ShaderNodeTexImage"); t.image = im; N.put(t.inputs["Vector"], vec); smp[k].append(t.outputs["Color"])
    key = next((k for k in maps if maps[k][1]), next(iter(maps)))
    ws = []
    for (vec, w7, _a), c in zip(H, smp[key]):
        lum = N.new("ShaderNodeRGBToBW"); N.put(lum.inputs[0], c)
        ws.append(N.math("MULTIPLY", N.math("ADD", 1.0 - falloff, N.math("MULTIPLY", lum.outputs[0], falloff)), w7))
    tot = N.math("ADD", N.math("ADD", ws[0], ws[1]), ws[2])
    W = _gain3(N, [N.math("DIVIDE", x, N.math("MAXIMUM", tot, 1e-6)) for x in ws])
    out = {}
    for k, (im, color) in maps.items():
        parts = []
        for (vec, _w, ang), c, w in zip(H, smp[k], W):
            if k.startswith("nor"):
                n2 = N.vmath("SUBTRACT", N.vmath("SCALE", c, s=2.0), (1.0, 1.0, 1.0))
                rv = N.new("ShaderNodeVectorRotate"); rv.rotation_type = "Z_AXIS"; N.put(rv.inputs["Vector"], n2)
                N.put(rv.inputs["Angle"], N.math("MULTIPLY", ang, -1.0)); c = rv.outputs[0]
            parts.append(N.vmath("SCALE", c, s=w))
        acc = N.vmath("ADD", N.vmath("ADD", parts[0], parts[1]), parts[2])
        if k.startswith("nor"):
            acc = N.vmath("ADD", N.vmath("SCALE", N.vmath("NORMALIZE", acc), s=0.5), (0.5, 0.5, 0.5))
        out[k] = acc
    return out


def scan_plane(sid, strength=1.0, rot=0.5, uv="sk_uv", name=None, tone_attr="Col"):
    """A surface scan on a solid plane (plaster, the mortar body, a roof without tile geometry): Mikkelsen tiling by the uv map
    (meters) — no repetition; the tone comes from tone_attr (style / masonry color)."""
    mt = scan_meta(sid); size = float(mt["size_m"][0])
    m = bpy.data.materials.new(name or f"sk_plane_{sid}"); nt = m.node_tree; N = _N(nt)
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    uvn = N.new("ShaderNodeUVMap"); uvn.uv_map = uv
    st = N.vmath("SCALE", uvn.outputs["UV"], s=1.0 / size)
    maps = {"Diffuse": (_img(mt["maps"]["Diffuse"], True), True), "nor_gl": (_img(mt["maps"]["nor_gl"], False), False),
            "Rough": (_img(mt["maps"]["Rough"], False), False)}
    o = hex_maps(N, st, maps, rot)
    N.put(bsdf.inputs["Base Color"], o["Diffuse"]); N.put(bsdf.inputs["Roughness"], o["Rough"])
    nm = N.new("ShaderNodeNormalMap"); nm.uv_map = uv; N.put(nm.inputs["Color"], o["nor_gl"]); N.put(bsdf.inputs["Normal"], nm.outputs["Normal"])
    im = maps["Diffuse"][0]
    px = np.empty(im.size[0] * im.size[1] * im.channels, np.float32); im.pixels.foreach_get(px)
    g = px.reshape(-1, im.channels)[::7, :3]
    lin = np.where(g <= 0.04045, g / 12.92, ((g + 0.055) / 1.055) ** 2.4).mean(0)
    _tone(nt, bsdf, lin, strength, tone_attr)
    m["sk_surface"] = {"path": "scan, Mikkelsen tiling", "id": sid, "square_m": size}
    return m


def mortar(scale=1.0, strength=1.0, name="sk_surf_mortar"):
    """Mortar: sandy 3D noise (grain 2–4 mm, spots 5–10 cm), roughness 0.95, tone — the masonry's Col."""
    m = bpy.data.materials.new(name); nt = m.node_tree; N = _N(nt)
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    geo = N.new("ShaderNodeNewGeometry"); P = geo.outputs["Position"]
    spots = N.new("ShaderNodeTexNoise", Scale=12.0 * scale, Detail=4.0, Roughness=0.55); N.put(spots.inputs["Vector"], P)
    grain = N.new("ShaderNodeTexVoronoi", Scale=300.0 * scale); N.put(grain.inputs["Vector"], P)
    fine = N.new("ShaderNodeTexNoise", Scale=900.0 * scale, Detail=1.0); N.put(fine.inputs["Vector"], P)
    v = N.math("ADD", N.math("MULTIPLY", N.math("SUBTRACT", spots.outputs["Fac"], 0.5), 0.5 * strength),
               N.math("MULTIPLY", N.math("SUBTRACT", fine.outputs["Fac"], 0.5), 0.25 * strength))
    f = N.math("ADD", v, 1.0)
    col = N.attr("Col", "Color")
    cc = N.new("ShaderNodeCombineColor"); N.put(cc.inputs["Red"], f); N.put(cc.inputs["Green"], f); N.put(cc.inputs["Blue"], f)
    N.put(bsdf.inputs["Base Color"], N.mix_rgb("MULTIPLY", 1.0, col, cc.outputs[0]))
    bsdf.inputs["Roughness"].default_value = 0.95
    h = N.math("ADD", N.math("MULTIPLY", grain.outputs["Distance"], 0.6), N.math("MULTIPLY", fine.outputs["Fac"], 0.4))
    bp = N.new("ShaderNodeBump"); bp.inputs["Strength"].default_value = 0.35; bp.inputs["Distance"].default_value = 0.002
    N.put(bp.inputs["Height"], h); N.put(bsdf.inputs["Normal"], bp.outputs["Normal"])
    m["sk_surface"] = {"path": "mortar", "scale": scale}
    return m


ROLE_MATS = {"wall": "sk_wall", "finish": "sk_finish", "mortar": "sk_mortar", "rubble": "sk_rubble"}   # names of bl_masonry.mat_role


def apply_masonry(scene, mats, uv=True):
    """Replace the materials of the masonry roles (sk_wall, sk_finish, sk_mortar, sk_rubble) with the surfaces mats
    {role: material}; masonry objects get sk_rnd by stone number and a per-face UV map (for scans). → {role: object count}."""
    from core.surface import bl_matlib as ML
    by_name = {v: k for k, v in ROLE_MATS.items()}
    stat = {}
    for ob in scene.objects:
        if ob.type != "MESH":
            continue
        slots = [s.material.name if s.material else None for s in ob.material_slots]
        roles = [by_name.get((n or "").split(".")[0]) for n in slots]
        if not any(r in mats for r in roles):
            continue
        block_random(ob)
        if uv:
            ML.plane_uv(ob)
        for i, r in enumerate(roles):
            if r in mats:
                ob.material_slots[i].material = mats[r]; stat[r] = stat.get(r, 0) + 1
    return stat

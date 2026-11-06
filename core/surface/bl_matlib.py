"""Library materials in a Blender scene — BlenderKit materials as the base solution (download — core/surface/blenderkit.py, the set by
roles — library/materials/blenderkit.yaml → set.json). This module is only the glue:
  load      — a material from library/materials/blenderkit/<id>/<resolution>.blend into the scene (images packed in);
  plane_uv  — a UV map by the plane of the face in world meters (buildings have no UV maps of their own): masonry rows
              are horizontal and continuous, tiles run along the eave, along the pitch — true length; the shift along u is
              per object; box_uv — a "box" for rocks and stones (an uneven grid);
  measure   — the material's pattern from its maps (height, AO, color): the period of rows / columns by autocorrelation,
              otherwise the diameter of a spot; tile_for — the size of the UV square in meters by role (a stone row
              0.35 m, brick 0.075, tile 0.2, plank 0.2…): the asset metadata "textureSizeMeters" is almost always 1 m by default
              (otherwise stone comes out as brick);
  fit       — a material onto the UV map: the UV input (Texture Coordinate, UV Map, an empty image input) → sk_uv ÷ the size
              of the square (the author's own Mapping stays after it); normal maps — onto the same UV map; displacement —
              by relief along the normal;
  tint      — a tint toward the palette of the building style (a "color" mix: tone and saturation from the palette,
              brightness and pattern are its own);
  roles_attr — the role of a face by the sk_role label (masonry: wall, finish, roof, wood, metal, glass, mortar, rubble);
  roles     — by the vertex color "Col" (the style palette: stone, roof, plaster…) — for unlabeled meshes (bl_castle);
  assign    — material slots by role (not one material for everything: a shader on Metal is limited to 16 textures).
States — components/materials/bl_mat.py (the weather layer goes in front of every BSDF, including those of these materials).
  import bl_matlib as ML; S = ML.role_set(); m = ML.load(ML.pick("wall stone", 0, S)); ML.fit(m, tile_m=ML.tile_for(m, …)[0])
  ML.plane_uv(ob); ML.assign(ob, ML.roles_attr(ob) or ML.roles(ob, PAL), {"wall": m})"""
import core.paths        # noqa: E402  core parts on sys.path
import json
import zlib
import math
import os
import sys

import bpy
import numpy as np

ROOT = core.paths.ROOT
LIB = os.path.join(ROOT, "library", "materials", "blenderkit")
UV = "sk_uv"


def meta(asset):
    p = os.path.join(LIB, asset, "meta.json")
    if not os.path.exists(p):
        raise SystemExit(f"bl_matlib: no material {asset} — .venv/bin/python -m core.surface.blenderkit get {asset}")
    return json.load(open(p))


def _set():
    p = os.path.join(LIB, "set.json")
    if not os.path.exists(p):
        raise SystemExit("bl_matlib: no set — .venv/bin/python -m core.surface.blenderkit set")
    d = json.load(open(p))
    return d if "roles" in d else {"roles": d, "elements": {}}


def role_set():
    """The set by roles (library/materials/blenderkit.yaml → set.json on core/surface/blenderkit.py set): {role: [{id, name,
    tile_m, style, tiling?}]}."""
    return _set()["roles"]


def elements():
    """Pattern element size by role: {role: {axis, m} | {tile}}."""
    return _set()["elements"]


def pick(role, k=0, sets=None):
    """The k-th material of a role (cycling if there are fewer variants) → id."""
    lst = (sets or role_set())[role]
    return lst[k % len(lst)]["id"]


def entry(asset, sets=None):
    """A set entry by id (role, name, style, correction)."""
    for role, lst in (sets or role_set()).items():
        for e in lst:
            if e["id"] == asset:
                return {**e, "role": role}
    return {}


def dress(role, k=0, tint_rgb=None, tint_k=1.0, sets=None, shuffle=True):
    """A role's material ready for a building: load, scale by the role's element, sk_uv map, tint (a copy per palette).
    → the material (properties sk_tile_m, sk_scale — how the scale was obtained)."""
    sets = sets or role_set()
    asset = pick(role, k, sets)
    base = load(asset)
    e = entry(asset, sets)
    tile, how = tile_for(base, elements().get(role), e)
    key = f"{asset}|{tile:.4f}|{tuple(round(x, 3) for x in tint_rgb) if tint_rgb else ''}|{tint_k}|{int(shuffle)}"
    for x in bpy.data.materials:
        if x.get("sk_dress") == key:
            return x
    m = base.copy()                                                     # leave the original alone: a copy for the scale and tint
    m.use_fake_user = False
    r = measure(base)
    fit(m, tile_m=tile, rows=rows_of(r) if shuffle else None)
    m["sk_tile_m"], m["sk_scale"] = tile, how
    if tint_rgb is not None and tint_k > 0:
        a = albedo(base)
        tint(m, tint_rgb, tint_k, (0.2126 * a[0] + 0.7152 * a[1] + 0.0722 * a[2]) if a else r.get("brightness"))
    m["sk_dress"] = key
    return m


def load(asset, res="2K"):
    """A BlenderKit material into the scene (once per Blender session): properties sk_lib (id) and sk_tile_m (tile size, m)."""
    for m in bpy.data.materials:
        if m.get("sk_lib") == asset and not m.get("sk_fit"):              # the original (copies made for a building carry sk_fit)
            return m
    mt = meta(asset)
    d = os.path.join(LIB, asset)
    path = next((os.path.join(d, f) for f in (f"{res}.blend", "2K.blend", "source.blend") if os.path.exists(os.path.join(d, f))),
                os.path.join(ROOT, mt["blend"]))
    with bpy.data.libraries.load(path, link=False) as (src, dst):
        dst.materials = list(src.materials)
    mats = [m for m in dst.materials if m is not None]
    if not mats:
        raise SystemExit(f"bl_matlib: no materials in {path}")
    m = next((x for x in mats if x.name.lower().startswith(mt["name"].lower()[:12])), mats[0])
    for x in mats:
        if x is not m and x.users == 0:
            bpy.data.materials.remove(x)
    m["sk_lib"], m["sk_tile_m"], m["sk_name"] = asset, float(mt.get("textureSizeMeters") or 1.0), mt["name"]
    m.use_fake_user = True
    if any(n.type == "SHADERTORGB" for n in m.node_tree.nodes):          # toon trick: the color comes from the scene light (in our case — white)
        print(f"bl_matlib: {mt['name']} — 'Shader to RGB' (the color depends on the light, does not work in Cycles): do not put it in the set",
              flush=True)
        m["sk_toon"] = True
    for n in m.node_tree.nodes:                                          # pack the images (the render does not need the library file)
        if n.type == "TEX_IMAGE" and n.image is not None and not n.image.packed_file and n.image.source == "FILE":
            try:
                n.image.pack()
            except RuntimeError:
                pass
    return m


def _up(sock, seen=None):
    """The first image going up the links from an input (through Math, ColorRamp, Separate, etc.) → the image node."""
    seen = seen if seen is not None else set()
    for l in sock.links:
        n = l.from_node
        if n.name in seen:
            continue
        seen.add(n.name)
        if n.type == "TEX_IMAGE" and n.image is not None:
            return n
        for i in n.inputs:
            if i.is_linked:
                r = _up(i, seen)
                if r is not None:
                    return r
    return None


def _map_scale(node):
    """The author's Mapping scale at the image's Vector input (u, v); none — (1, 1)."""
    seen, stack = set(), [node.inputs["Vector"]]
    while stack:
        sk = stack.pop()
        for l in sk.links:
            n = l.from_node
            if n.name in seen:
                continue
            seen.add(n.name)
            if n.type == "MAPPING":
                sc = n.inputs["Scale"].default_value
                return float(sc[0]) or 1.0, float(sc[1]) or 1.0
            stack += [i for i in n.inputs if i.is_linked]
    return 1.0, 1.0


def maps(m):
    """Material maps by meaning: height (the height input of Displacement / Bump), color (Base Color), ao (by name) →
    {kind: (image, the author's Mapping scale (u, v))}."""
    nt, out = m.node_tree, {}
    for n in nt.nodes:
        key = {"DISPLACEMENT": "height", "BUMP": "height", "BSDF_PRINCIPLED": "color"}.get(n.type)
        if key and key not in out:
            t = _up(n.inputs["Height" if key == "height" else "Base Color"])
            if t is not None:
                out[key] = (t.image, _map_scale(t))
    for n in nt.nodes:
        if n.type == "TEX_IMAGE" and n.image is not None and "ao" not in out:
            nm = (n.image.name + " " + n.image.filepath).lower()
            if any(k in nm for k in ("_ao", "ambient", "occlusion", " ao")):
                out["ao"] = (n.image, _map_scale(n))
    return out


def gray(im, size=512):
    """Image → grayscale (size × size, sampled by stride — no interpolation), rows bottom to top (v grows upward)."""
    w, h = im.size
    px = np.empty(w * h * im.channels, np.float32); im.pixels.foreach_get(px); px = px.reshape(h, w, im.channels)
    sy, sx = max(1, h // size), max(1, w // size)
    px = px[::sy, ::sx]
    return px[..., :3].mean(2) if im.channels >= 3 else px[..., 0]


def _smooth(p, k=5):
    return np.convolve(np.pad(p, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def periods(g):
    """Pattern period along the axes (as a fraction of the image) from the autocorrelation R. Row v: the max profile over the u
    shifts of R[shift v, ·] — in a bond the neighboring row matches with a shift of half a stone, without max the period came
    out as two rows. The period is the first peak after the first trough, with a prominence of ≥ half of the strongest and
    ≥ 0.08. The spot is the first zero of R along the axis after a high-pass filter (σ 32 px at 512: slow tone variations
    overstated the size threefold) — the diameter of an element with no period (for spots of diameter d the autocorrelation
    drops to zero near d; for irregular masonry the error is up to two times — corrected in the set).
    → {u, v: [period | None, prominence, spot]}."""
    x = g - g.mean()
    f = np.fft.rfft2(x); R = np.fft.irfft2(f * np.conj(f), s=x.shape)
    R = R / R[0, 0] if R[0, 0] else R
    fy = np.fft.fftfreq(x.shape[0])[:, None]; fx = np.fft.rfftfreq(x.shape[1])[None, :]
    sig = 32.0 * max(x.shape) / 512                                     # σ of the blur, px (frequencies are in cycles per pixel)
    hp = f * (1 - np.exp(-2 * np.pi ** 2 * sig ** 2 * (fx ** 2 + fy ** 2)))
    Rh = np.fft.irfft2(hp * np.conj(hp), s=x.shape); Rh = Rh / Rh[0, 0] if Rh[0, 0] else Rh   # spot — without slow tone variations
    out = {}
    for ax, prof, line in (("v", R.max(1), Rh[:, 0]), ("u", R.max(0), Rh[0, :])):
        L = len(prof); n = L // 2
        p = _smooth(prof[:n + 3])[:n]
        z0 = next((i for i in range(1, n) if line[i] <= 0), n - 1)
        spot = round(z0 / L, 4)
        lo = next((i for i in range(2, n - 1) if p[i] <= p[i - 1] and p[i] <= p[i + 1]), None)
        if lo is None:
            out[ax] = [None, 0.0, spot]; continue
        pk = []
        for i in range(lo + 1, n - 1):
            if p[i] >= p[i - 1] and p[i] >= p[i + 1]:
                pk.append((i, float(p[i] - p[lo:i + 1].min())))
        best = max((q for _, q in pk), default=0.0)
        hit = next(((i, q) for i, q in pk if q >= max(0.08, 0.5 * best)), None)
        out[ax] = [round(hit[0] / L, 4), round(hit[1], 3), spot] if hit else [None, round(best, 3), spot]
    return out


def measure(m):
    """Material pattern as fractions of the UV square (the author's Mapping scale is accounted for): the period along the axes
    (the strongest prominence among the maps), the spot — the median over the maps and axes. Cache —
    library/materials/blenderkit/<id>/measure.json."""
    asset = m.get("sk_lib")
    cp = os.path.join(LIB, asset, "measure.json") if asset else None
    if cp and os.path.exists(cp):
        return json.load(open(cp))
    per, spots, grays, lum = {"u": [None, 0.0, ""], "v": [None, 0.0, ""]}, [], {}, None
    for kind, (im, sc) in maps(m).items():
        g = gray(im, 512); grays[kind] = (g, sc)
        r = periods(g)
        for ax, k in (("u", sc[0]), ("v", sc[1])):
            p, q, z = r[ax]
            spots.append(z / k)
            if p is not None and q >= 0.08 and q > per[ax][1]:
                per[ax] = [round(p / k, 5), q, kind]
        if kind == "color":                                             # mean brightness (linear) — toward the palette tone
            lin = g if im.is_float or im.colorspace_settings.name != "sRGB" else \
                np.where(g <= 0.04045, g / 12.92, ((g + 0.055) / 1.055) ** 2.4)
            lum = round(float(lin.mean()), 4)
    phase = {}
    for ax in ("u", "v"):                                               # joint phase: where the pattern has a trough (mortar is lower)
        p_uv, q, kind = per[ax]
        if p_uv and q >= 0.3 and kind in grays:
            g, sc = grays[kind]; k = sc[0] if ax == "u" else sc[1]
            prof = g.mean(1) if ax == "v" else g.mean(0)                # rows bottom to top: profile along v
            L = len(prof); P = p_uv * k * L                             # period in image pixels
            cand = np.arange(int(round(P)))
            score = [prof[(np.round(c + np.arange(int(L // P)) * P).astype(int)) % L].mean() for c in cand]
            phase[ax] = round(float(cand[int(np.argmin(score))]) / L / k, 5)
    seam = {}                                                           # tile joint: the most continuous mortar line in the tile
    kind = next((x for x in ("height", "ao", "color") if x in grays), None)
    if kind:
        g, sc = grays[kind]
        for ax, prof, k in (("v", g.mean(1), sc[1]), ("u", g.mean(0), sc[0])):
            prof = _smooth(prof, 3); i = int(np.argmin(prof))
            z = float((prof.mean() - prof[i]) / (prof.std() + 1e-6))     # trough depth in σ: a continuous joint is ≥ 1.5
            seam[ax] = [round(i / len(prof) / k, 5), round(1.0 / k, 5), round(z, 2)]
    out = {"u": per["u"], "v": per["v"], "spot": round(float(np.median(spots)), 5) if spots else None, "brightness": lum,
           "phase": phase, "joint": seam}
    if cp:
        json.dump(out, open(cp, "w"), ensure_ascii=False)
    return out


def albedo(m, size=64):
    """Mean Base Color of a material (linear) — by a snapshot: Base Color → emission on the UV 0…1 square under an ortho camera
    (all the author's nodes: AO, tints, procedural pattern). The mean brightness of a single color map was misleading: for a
    "dark" stone the map is light, and the nodes darken it. Cache — library/materials/blenderkit/<id>/albedo.json."""
    asset = m.get("sk_lib")
    cp = os.path.join(LIB, asset, "albedo.json") if asset else None
    if cp and os.path.exists(cp):
        return json.load(open(cp))["rgb"]
    import tempfile
    sc = bpy.data.scenes.new("sk_probe")
    sc.render.engine = "BLENDER_EEVEE"; sc.eevee.taa_render_samples = 4
    sc.render.resolution_x = sc.render.resolution_y = size; sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "OPEN_EXR"; sc.render.image_settings.color_depth = "32"
    me = bpy.data.meshes.new("sk_probe"); me.from_pydata([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [], [(0, 1, 2, 3)])
    uvl = me.uv_layers.new(name="UVMap"); uvl.data.foreach_set("uv", [0, 0, 1, 0, 1, 1, 0, 1])
    ob = bpy.data.objects.new("sk_probe", me); sc.collection.objects.link(ob)
    pm = m.copy(); nt = pm.node_tree
    p = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    outn = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
    if p is None or outn is None:
        bpy.data.materials.remove(pm); bpy.data.objects.remove(ob); bpy.data.meshes.remove(me); bpy.data.scenes.remove(sc)
        return None
    em = nt.nodes.new("ShaderNodeEmission")
    bc = p.inputs["Base Color"]
    if bc.is_linked:
        nt.links.new(bc.links[0].from_socket, em.inputs["Color"])
    else:
        em.inputs["Color"].default_value = bc.default_value
    for l in list(outn.inputs["Surface"].links) + list(outn.inputs["Displacement"].links):
        nt.links.remove(l)
    nt.links.new(em.outputs[0], outn.inputs["Surface"])
    me.materials.append(pm)
    cam = bpy.data.objects.new("sk_probe_cam", bpy.data.cameras.new("sk_probe_cam")); sc.collection.objects.link(cam)
    cam.data.type = "ORTHO"; cam.data.ortho_scale = 1.0; cam.location = (0.5, 0.5, 1.0); sc.camera = cam
    path = os.path.join(tempfile.gettempdir(), f"sk_probe_{os.getpid()}.exr")
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True, scene=sc.name)
    im = bpy.data.images.load(path); px = np.empty(size * size * 4, np.float32); im.pixels.foreach_get(px)
    rgb = [round(float(x), 4) for x in px.reshape(-1, 4)[:, :3].mean(0)]
    bpy.data.images.remove(im); os.remove(path)
    bpy.data.materials.remove(pm); bpy.data.objects.remove(ob); bpy.data.meshes.remove(me)
    bpy.data.objects.remove(cam); bpy.data.scenes.remove(sc)
    if cp:
        json.dump({"rgb": rgb}, open(cp, "w"))
    return rgb


def tile_for(m, spec=None, entry=None):
    """Size of the UV square in meters by role: spec {axis: v|u|any|spot, m: element size} or {tile: m}; the material correction
    entry {tiling: m}; a scan (realistic) with its own size ≠ 1 m — that size. → (meters, how obtained)."""
    spec, entry = spec or {}, entry or {}
    if entry.get("tiling"):
        return float(entry["tiling"]), "set correction"
    meta_t = float(m.get("sk_tile_m") or 1.0)
    if "tile" in spec or not spec:
        if entry.get("style") == "realistic" and abs(meta_t - 1.0) > 1e-3:
            return meta_t, "scan size"
        return float(spec.get("tile", meta_t)), "role tile"
    r, ax, size = measure(m), spec.get("axis", "any"), float(spec["m"])
    if ax == "any":
        ax = max(("u", "v"), key=lambda a: r[a][1] if r[a][0] else -1)
    if ax in ("u", "v") and r[ax][0]:
        return size / r[ax][0], f"period {ax} {r[ax][0]:.4f} ({1 / r[ax][0]:.1f} per square) → {size} m"
    if r.get("spot"):
        return size / r["spot"], f"spot {r['spot']:.4f} → {size} m"
    return meta_t, "no pattern — BlenderKit's size"


def plane_uv(ob, name=UV, seed=None):
    """UV map by the plane of the face in world meters. Wall: u — along the wall, v — height z (rows are horizontal and
    continuous across all faces, including the faces of round towers); pitch: u — along the eave, v — up the pitch at true
    length (tile rows run along the eave); nearly horizontal (|nz| > 0.95) — (x, y). A "box" stretched the pattern of slanted
    walls up to 1.41 times and laid the tile rows along the pitch. The u shift is random per object (identical walls do not
    repeat the pattern)."""
    me = ob.data
    nl = len(me.loops)
    if nl == 0:
        return None
    co = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    M4 = np.array(ob.matrix_world)
    w = co @ M4[:3, :3].T + M4[:3, 3]
    lv = np.empty(nl, np.int32); me.loops.foreach_get("vertex_index", lv)
    fn = np.empty(len(me.polygons) * 3, np.float32); me.polygons.foreach_get("normal", fn)
    N = fn.reshape(-1, 3) @ np.linalg.inv(M4[:3, :3])                    # normals to world (inverse transpose)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    T = np.cross(np.array([0.0, 0.0, 1.0]), N)
    flat = np.linalg.norm(T, axis=1) < 0.31                               # |nz| > 0,95
    T[flat] = (1.0, 0.0, 0.0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)
    B = np.cross(N, T)
    B[flat] = (0.0, 1.0, 0.0)
    lt = np.empty(len(me.polygons), np.int32); me.polygons.foreach_get("loop_total", lt)
    face_of = np.repeat(np.arange(len(me.polygons)), lt)
    p = w[lv]
    u = (p * T[face_of]).sum(1); v = (p * B[face_of]).sum(1)
    rng = np.random.default_rng(seed if seed is not None else zlib.crc32(ob.name.encode()))
    u += rng.random() * 7.0
    uv = me.uv_layers.get(name) or me.uv_layers.new(name=name)
    uv.data.foreach_set("uv", np.stack([u, v], 1).ravel().astype(np.float32))
    me.uv_layers.active = uv
    uv.active_render = True
    return uv


def box_uv(ob, name=UV, seed=None):
    """A "box" in world meters — for rocks and stones (an uneven mesh: a UV by the face plane would tear the pattern on every
    face): a face goes onto the plane of the main axis of its normal; the shift is random per object."""
    me = ob.data
    if not me.loops:
        return None
    co = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    M4 = np.array(ob.matrix_world); w = co @ M4[:3, :3].T + M4[:3, 3]
    lv = np.empty(len(me.loops), np.int32); me.loops.foreach_get("vertex_index", lv)
    fn = np.empty(len(me.polygons) * 3, np.float32); me.polygons.foreach_get("normal", fn)
    N = fn.reshape(-1, 3) @ np.linalg.inv(M4[:3, :3])
    lt = np.empty(len(me.polygons), np.int32); me.polygons.foreach_get("loop_total", lt)
    ax = np.abs(N).argmax(1)[np.repeat(np.arange(len(me.polygons)), lt)]
    p = w[lv]
    u = np.where(ax == 0, p[:, 1], p[:, 0]); v = np.where(ax == 2, p[:, 1], p[:, 2])
    off = np.random.default_rng(seed if seed is not None else zlib.crc32(ob.name.encode())).random(2) * 7.0
    uv = me.uv_layers.get(name) or me.uv_layers.new(name=name)
    uv.data.foreach_set("uv", np.stack([u + off[0], v + off[1]], 1).ravel().astype(np.float32))
    me.uv_layers.active = uv; uv.active_render = True
    return uv


def rows_of(r, min_q=0.3, min_z=1.5):
    """Rows for shifting (against repetition): an axis with a strong period (prominence ≥ min_q) and a found joint phase →
    (axis, period, phase) in fractions of the UV square. Irregular masonry with no period — strips of a whole tile along the
    most continuous joint of the tile (profile trough ≥ min_z σ; otherwise the cut would run through the stones) → the same
    form; neither one nor the other — None."""
    for ax in ("v", "u"):
        p, q, _ = r[ax]
        if p and q >= min_q and ax in (r.get("phase") or {}):
            return ax, p, r["phase"][ax]
    s = (r.get("joint") or {}).get("v")
    if s and s[2] >= min_z:
        return "v", s[1], s[0]
    return None


def _row_shuffle(nt, vec, rows):
    """Each row (a strip along the row axis between joints) gets its own random shift along the row: the tile repetition
    breaks up, while the rows and the bond within a row stay intact (the strip boundary runs along a mortar joint)."""
    ax, period, phase = rows
    sep = nt.nodes.new("ShaderNodeSeparateXYZ"); nt.links.new(vec, sep.inputs[0])
    a_row, a_along = ("Y", "X") if ax == "v" else ("X", "Y")
    m1 = nt.nodes.new("ShaderNodeMath"); m1.operation = "SUBTRACT"; nt.links.new(sep.outputs[a_row], m1.inputs[0])
    m1.inputs[1].default_value = phase
    m2 = nt.nodes.new("ShaderNodeMath"); m2.operation = "DIVIDE"; nt.links.new(m1.outputs[0], m2.inputs[0])
    m2.inputs[1].default_value = period
    fl = nt.nodes.new("ShaderNodeMath"); fl.operation = "FLOOR"; nt.links.new(m2.outputs[0], fl.inputs[0])
    wn = nt.nodes.new("ShaderNodeTexWhiteNoise"); wn.noise_dimensions = "1D"; nt.links.new(fl.outputs[0], wn.inputs["W"])
    sh = nt.nodes.new("ShaderNodeMath"); sh.operation = "MULTIPLY_ADD"; nt.links.new(wn.outputs["Value"], sh.inputs[0])
    sh.inputs[1].default_value = 7.0; nt.links.new(sep.outputs[a_along], sh.inputs[2])
    cb = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(sh.outputs[0], cb.inputs[a_along]); nt.links.new(sep.outputs[a_row], cb.inputs[a_row])
    return cb.outputs[0]


def fit(m, uv=UV, tile_m=None, rows=None):
    """Put a material onto the uv map (meters) with the tile size tile_m: everything that took UV now takes uv ÷ tile_m; rows —
    the row shift (rows_of)."""
    nt = m.node_tree
    k = 1.0 / float(tile_m or m.get("sk_tile_m", 1.0))
    src = nt.nodes.new("ShaderNodeUVMap"); src.uv_map = uv; src.label = "sk_uv"
    sc = nt.nodes.new("ShaderNodeVectorMath"); sc.operation = "SCALE"; sc.inputs["Scale"].default_value = k
    nt.links.new(src.outputs["UV"], sc.inputs[0])
    out = sc.outputs[0]
    if rows:
        out = _row_shuffle(nt, out, rows)
        m["sk_rows"] = {"axis": rows[0], "period": round(float(rows[1]), 5), "phase": round(float(rows[2]), 5)}
    for n in list(nt.nodes):
        if n.type == "TEX_COORD":
            for l in list(n.outputs["UV"].links) + list(n.outputs["Generated"].links):
                nt.links.new(out, l.to_socket)
        elif n.type == "UV_MAP" and n is not src:
            for l in list(n.outputs["UV"].links):
                nt.links.new(out, l.to_socket)
        elif n.type in ("TEX_IMAGE",) and not n.inputs["Vector"].is_linked:
            nt.links.new(out, n.inputs["Vector"])
        elif n.type == "NORMAL_MAP":
            n.uv_map = uv
    if m.displacement_method != "BUMP":                                  # displacing vertices on the coarse meshes of buildings tears the faces apart
        m.displacement_method = "BUMP"                                   # (each face has its own projection) → relief along the normal
    m["sk_fit"] = uv
    return m


def tint(m, rgb, amount=1.0, lum=None):
    """Toward the style palette: in front of the Base Color of every Principled — a "color" mix (the palette's tone and
    saturation, the brightness pattern is its own) and brightness: a multiplier "palette brightness / mean material
    brightness" (lum — albedo; 0.3–4). A half mix (0.5) left a violet roof and stone ranging from white to dark: the
    building's color is set by the style palette, the material supplies only the pattern."""
    nt = m.node_tree
    L_pal = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    kL = float(np.clip(L_pal / lum, 0.3, 4.0)) if lum else 1.0
    for p in [n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"]:
        s = p.inputs["Base Color"]
        mx = nt.nodes.new("ShaderNodeMix"); mx.data_type = "RGBA"; mx.blend_type = "COLOR"; mx.clamp_result = True
        ins = {x.identifier: x for x in mx.inputs}
        ins["Factor_Float"].default_value = amount
        ins["B_Color"].default_value = (*rgb[:3], 1.0)
        if s.is_linked:
            l = s.links[0]; nt.links.new(l.from_socket, ins["A_Color"]); nt.links.remove(l)
        else:
            ins["A_Color"].default_value = s.default_value
        out = next(x for x in mx.outputs if x.identifier == "Result_Color")
        if abs(kL - 1.0) > 0.02:
            ml = nt.nodes.new("ShaderNodeMix"); ml.data_type = "RGBA"; ml.blend_type = "MULTIPLY"; ml.clamp_result = False
            mi = {x.identifier: x for x in ml.inputs}
            mi["Factor_Float"].default_value = amount; mi["B_Color"].default_value = (kL, kL, kL, 1.0)
            nt.links.new(out, mi["A_Color"]); out = next(x for x in ml.outputs if x.identifier == "Result_Color")
        nt.links.new(out, s)
    m["sk_tint"], m["sk_tint_k"] = list(rgb[:3]), round(kL, 3)
    return m


SK_ROLE = {0: "wall", 1: "finish", 2: "roof", 3: "wood", 4: "metal", 5: "glass", 6: "mortar", 7: "rubble"}  # bl_masonry.ROLES
LIB_ROLE = {"wall": "wall stone", "finish": "wall stone", "roof": "roof", "wood": "wood", "metal": "metal",
            "mortar": "plaster", "rubble": "wall stone"}                # role of a building face → role in the set; glass is its own


def roles_attr(ob, attr="sk_role", names=SK_ROLE):
    """Role of a face by the building's label (sk_role from the architecture and masonry) → a list of names as long as the
    faces, None — no label."""
    a = ob.data.attributes.get(attr)
    if a is None or a.domain != "FACE":
        return None
    v = np.empty(len(ob.data.polygons), np.int32); a.data.foreach_get("value", v)
    return [names.get(int(x)) for x in v]


def roles(ob, palette, attr="Col", max_d=0.12):
    """Role of a face by vertex color: the nearest palette color {role: (r, g, b)} within max_d, otherwise None. → a list as
    long as the faces."""
    me = ob.data
    ca = me.color_attributes.get(attr)
    if ca is None:
        return [None] * len(me.polygons)
    a = np.empty(len(ca.data) * 4, np.float32); ca.data.foreach_get("color", a); a = a.reshape(-1, 4)[:, :3]
    if ca.domain == "POINT":
        lv = np.empty(len(me.loops), np.int32); me.loops.foreach_get("vertex_index", lv); a = a[lv]
    ls = np.empty(len(me.polygons), np.int32); me.polygons.foreach_get("loop_start", ls)
    fc = a[ls]                                                          # color of the first corner of the face (the face is painted uniformly)
    names = list(palette)
    P = np.array([palette[n][:3] for n in names], np.float32)
    d = np.linalg.norm(fc[:, None, :] - P[None, :, :], axis=2)
    j = d.argmin(1)
    return [names[i] if d[f, i] <= max_d else None for f, i in enumerate(j)]


def vcol_material(name="sk_vcol", attr="Col"):
    """Fallback material: vertex color (parts with no role — openings, ground)."""
    m = bpy.data.materials.get(name)
    if m is None:
        m = bpy.data.materials.new(name); nt = m.node_tree
        p = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
        a = nt.nodes.new("ShaderNodeAttribute"); a.attribute_type = "GEOMETRY"; a.attribute_name = attr
        nt.links.new(a.outputs["Color"], p.inputs["Base Color"]); p.inputs["Roughness"].default_value = 0.9
    return m


def assign(ob, face_roles, mats, fallback=None):
    """Material slots by role: mats {role: material}; a face with no role (or a role with no material) gets the fallback
    (vertex color)."""
    me = ob.data
    fb = fallback or vcol_material()
    used = [r for r in dict.fromkeys(face_roles) if r in mats]
    me.materials.clear()
    for r in used:
        me.materials.append(mats[r])
    me.materials.append(fb)
    idx = {r: i for i, r in enumerate(used)}
    mi = np.array([idx.get(r, len(used)) for r in face_roles], np.int32)
    me.polygons.foreach_set("material_index", mi)
    me.update()
    return {r: int((mi == i).sum()) for r, i in idx.items()}

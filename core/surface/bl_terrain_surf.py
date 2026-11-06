"""Ground shader (Eevee) built from ready-made parts: the layers are ambientCG photoscans (library/terrain/scans.yaml: color,
roughness, height; triplanar projection), procedural relief materials or baked — the color of the
baked site texture; natural layers are per pixel: slope from the normal, height (snow), spots from 3D noise; the plan
masks (road, platform, paving, bank, bed, gullies, cliff) are vertex attributes (relief_w.npz from terrain_tex.bake); the
layer on top is picked by scan height (HeightLerp: stones show through sand before the hollows). Tile repetition of a
scan: noise-driven offsets (Quilez, "texture repetition", technique 3), macro color spots, and the tile is larger in the
distance (distance blend). Geometry displacement — only where the attribute a_near is set (the bl_terrain_near patch).
The bake image is the active node: Workbench sees it. The biome recipe — <site folder>/surface.json.
Import in Blender: sys.path.insert(0, tools); import bl_terrain_surf as TS; TS.apply(ob, G, recipe, seed, snowline)."""
import core.paths        # noqa: E402  core parts on sys.path
import json, os, sys, time

import bpy
import numpy as np

ROOT = core.paths.ROOT
core.paths.ext("terrain_materials")
PLAN = ("bank", "bed", "pad", "pave", "road", "gully", "cliff")
DATA = ("moist", "expo", "foot")                                    # relief data (terrain_tex.variation): moisture, exposure, slope feet
MASKS = {"rock_from": 27, "rock_band": 11, "cliff_from": 43, "dirt_dry": 0.36, "talus_from": 16, "talus_to": 30, "jit": 4}
ORDER = ("dirt", "talus", "rock", "cliff", "snow", "bank", "bed", "pad", "pave", "road")   # on top of the base — one after another
IMG_BUDGET = 13                                                     # own images in the material: a shader on Metal has 16 textures, some of which are taken by
# Eevee; beyond that — a scan by color with no height, further — by tone
SEAM_M = 30.0                                                       # width of the uneven boundary between the near shader and the far one, m
SEAM_BAND = 90.0                                                    # the band beyond the edge of the near grid where the far shader pulls the color toward the map (a_macro), m
PRI = {"base": 0, "rock": 1, "cliff": 1, "dirt": 2, "talus": 3, "road": 4, "pave": 4, "pad": 4, "bank": 5, "bed": 5, "snow": 5}   # whose scan takes priority
COVER = ("soil", "litter", "needles", "moss", "dirt", "pebbles")   # site cover (components/grass, cover.npz): the visible ground under the grass —
# layers directly above the base; the scans come from the cover meta "scans" (soil — the site's dirt layer)
PACK = {"k1": ("moist", "expo", "foot", "cliff"), "k2": ("pad", "pave", "road", "gully"), "k3": ("bank", "bed", "c_pebbles", "")}
# material attributes — four at a time (RGBA): an Eevee mesh has few attributes, and the weather (components/materials) adds
# its own 5 (sk_g*, sk_w*, sk_wat) — the extra ones fall off from the second render on (the ground turns black in patches).
# The cover goes through COVER_IMG images on its own 0.5 m grid: the ground mesh is fine only in the middle of the site
# (±60 m, beyond that the step is 2.5 m) — 0.7 m trails and yards by vertices disappeared
COVER_IMG = {"cover_a": ("c_cov", "c_litter", "c_needles", "c_moss"), "cover_b": ("tint_r", "tint_g", "tint_b", "c_dirt")}   # tint — ÷2
CANOPY = {"blade": None, "k": 1.6, "near": (6.0, 16.0), "dens": (0.3, 0.65), "sheen": 0.7, "mottle": 0.45, "gain": 1.0}
# far canopy: grass that is no longer there as tufts in the frame is shown as a property of the ground (the base color):
# gaps by view angle e^(−k·density/cos θ) (from above the soil shows through the grass, at a grazing angle only stems),
# grazing sheen ((1 − cos θ)⁴ — the "velvet" of a field), tussocks and clumps by noise (fading out toward 180 m — no
# ripples). blade — grass color × the cover tint; empty — the mean of the base scan × tone fit (grass_colour: far grass is
# the same as it was — the color is not lost, the canopy adds the detail); near — from the camera the canopy shows under
# the tufts (6 → 16 m); dens — from which cover density. No new images (Metal's limit of 16 textures).
TINT_L, TINT_C, TINT_VEG = 0.9, 0.85, 0.35                         # fitting a scan to the site's tone: brightness, tint of minerals
# and of vegetation (0 — as captured, 1 — matched to the tone)
IG_NAMES = ("mountain", "sandstone", "stone", "chunkyrock", "cobble_stone", "cracked_ground", "dirt", "soil", "mud", "sand", "ice")


def _ring(nx, ny):
    """Vertices of the skirt of bl_relief.grid_mesh — the same edge traversal."""
    ii = np.arange(nx * ny).reshape(ny, nx)
    return np.concatenate([ii[0, :], ii[1:, -1], ii[-1, -2::-1], ii[-2:0:-1, 0]])


def _shader_of(IG, m):
    """The shader of a material from the terrain materials package (the module's shader_* function — the one the material's apply passes to add_material)."""
    import importlib, inspect
    M = importlib.import_module(IG.MATS[m].__module__); src = inspect.getsource(IG.MATS[m].apply)
    for nm, f in vars(M).items():
        if callable(f) and nm.startswith("shader") and nm in src: return f
    return next(f for nm, f in vars(M).items() if callable(f) and nm.startswith("shader"))


def scans():
    p = os.path.join(ROOT, "library", "textures", "terrain", "index.json")
    return json.load(open(p))["scans"] if os.path.exists(p) else {}


def attrs(ob, folder):
    """Plan masks (relief_w.npz) — vertex attributes a_<layer>; False — the wrong mesh."""
    W = np.load(os.path.join(folder, "relief_w.npz")); nx, ny = len(W["xs"]), len(W["ys"])
    me = ob.data; n = len(me.vertices); ring = _ring(nx, ny)
    if n not in (nx * ny, nx * ny + len(ring)): return False
    for k in PLAN + DATA:
        if f"a_{k}" in W.files:
            v = W[f"a_{k}"].astype(np.float32).ravel(); v = v if n == nx * ny else np.concatenate([v, v[ring]])
            me.attributes.new(f"a_{k}", "FLOAT", "POINT").data.foreach_set("value", v)
    if "var" in W.files:                                            # color dilution by relief data (terrain_tex.variation)
        v = W["var"].astype(np.float32).reshape(-1, 3); v = v if n == nx * ny else np.concatenate([v, v[ring]])
        me.attributes.new("a_var", "FLOAT_VECTOR", "POINT").data.foreach_set("vector", v.ravel())
    out = {**{k: float(W[f"a_{k}"].max()) for k in PLAN + DATA if f"a_{k}" in W.files}, "var": float("var" in W.files)}
    cov = {}
    cp = os.path.join(folder, "cover.npz")
    if os.path.exists(cp):                                          # cover (components/grass): density, ground, tint — at the mesh vertices
        C = np.load(cp); M = json.loads(str(C["meta"])); X, Y = np.meshgrid(W["xs"], W["ys"]); x0, y0, st = float(C["x0"]), float(C["y0"]), float(C["step"])
        fx, fy = (X.ravel() - x0) / st, (Y.ravel() - y0) / st
        def bil(A):
            A = np.asarray(A, np.float32); gx = np.clip(fx, 0, A.shape[1] - 1.001); gy = np.clip(fy, 0, A.shape[0] - 1.001); i, j = gx.astype(int), gy.astype(int)
            tx, ty = (gx - i)[:, None] if A.ndim == 3 else gx - i, (gy - j)[:, None] if A.ndim == 3 else gy - j
            v = A[j, i] * (1 - tx) * (1 - ty) + A[j, i + 1] * tx * (1 - ty) + A[j + 1, i] * (1 - tx) * ty + A[j + 1, i + 1] * tx * ty
            return v if n == nx * ny else np.concatenate([v, v[ring]])
        lw = {k: C["ground"][M["ground"].index(k)].astype(np.float32) / 255.0 for k in M["ground"]}
        z0 = np.zeros_like(C["density"], np.float32); ch = {"c_cov": C["density"].astype(np.float32) / 255.0, **{f"c_{k}": lw.get(k, z0) for k in COVER}}
        for k, A in ch.items(): out[k] = float(A.max())
        cov["c_pebbles"] = bil(ch["c_pebbles"]).astype(np.float32)
        out["c_scans"] = M.get("scans") or {}; cny, cnx = C["density"].shape; out["c_canopy"] = M.get("canopy")   # canopy — Grass knobs
        grid = dict(ch, tint_r=C["tint"][..., 0].astype(np.float32) / 2, tint_g=C["tint"][..., 1].astype(np.float32) / 2, tint_b=C["tint"][..., 2].astype(np.float32) / 2)
        for nm, ks in COVER_IMG.items():                            # cover images (row 0 is south, like the bottom of a Blender image)
            im = bpy.data.images.get(nm)
            if im is not None: bpy.data.images.remove(im)
            im = bpy.data.images.new(nm, cnx, cny, alpha=True); im.colorspace_settings.name = "Non-Color"; im.alpha_mode = "CHANNEL_PACKED"
            im.pixels.foreach_set(np.clip(np.stack([grid[k] for k in ks], -1), 0, 1).astype(np.float32).ravel()); im.pack()
        out["c_img"] = [x0 - st / 2, y0 - st / 2, cnx * st, cny * st]
        if "f_density" in C.files:                                  # the cover of the far belt (Grass, model far) — passed to the far shader as an image (canopy)
            fd = C["f_density"].astype(np.float32) / 255.0; ft = C["f_tint"].astype(np.float32) / 2; fny, fnx = fd.shape; fst = float(C["f_step"])
            im = bpy.data.images.get("cover_far")
            if im is not None: bpy.data.images.remove(im)
            im = bpy.data.images.new("cover_far", fnx, fny, alpha=True); im.colorspace_settings.name = "Non-Color"; im.alpha_mode = "CHANNEL_PACKED"
            im.pixels.foreach_set(np.clip(np.concatenate([fd[..., None], ft], -1), 0, 1).astype(np.float32).ravel()); im.pack()
            out["c_far"] = [float(C["f_x0"]) - fst / 2, float(C["f_y0"]) - fst / 2, fnx * fst, fny * fst]
    for nm, ks in PACK.items():                                     # four per color (RGBA) — the material needs fewer attributes
        cols = [cov[k] if k in cov else (W[f"a_{k}"].astype(np.float32).ravel() if k and f"a_{k}" in W.files else np.zeros(nx * ny, np.float32)) for k in ks]
        cols = [c if len(c) == n else np.concatenate([c, c[ring]]) for c in cols]
        me.attributes.new(f"a_{nm}", "FLOAT_COLOR", "POINT").data.foreach_set("color", np.stack(cols, -1).astype(np.float32).ravel())
    return out


def _gain(target, mean, veg=False):
    """Fitting a scan to the site's tone: brightness — almost fully; the tint of minerals (ground, stone, sand) — also, their
    color is the site's geology (otherwise a rust-colored rock and gray sand clash in patches); vegetation keeps its own color
    (with a full fit the grass became uniformly yellow)."""
    lum = np.array([0.2126, 0.7152, 0.0722]); t, m = np.asarray(target, float), np.maximum(np.asarray(mean, float), 1e-4)
    lt, lm = max(float(t @ lum), 1e-4), max(float(m @ lum), 1e-4)
    g = (lt / lm) ** TINT_L * np.power(np.clip((t / lt) / (m / lm), 0.2, 5.0), TINT_VEG if veg else TINT_C)
    return tuple(float(x) for x in np.clip(g, 0.25, 4.0)) + (1.0,)


def names(v):
    """Surfaces of a layer: one, or a {dry, wet} pair."""
    return [] if v is None else [v["dry"], v["wet"]] if isinstance(v, dict) else [v]


def vista_layers(recipe, SC, gain=None):
    """The far shader's layers (bl_terrain_vista) from the site recipe: scans of the base, ground, scree, rock, cliff, snow (a
    dry/wet pair — <layer>_wet); procedural and baked materials are skipped (the base is then the baked image)."""
    out = {}
    for lay in ("base", "dirt", "talus", "rock", "cliff", "snow"):
        for j, m in enumerate(names(recipe.get(lay))):
            if m in SC and "Color" in SC[m]:
                out[lay + ("_wet" if j else "")] = {"color": os.path.join(ROOT, SC[m]["Color"]), "size": float(SC[m]["size"]),
                                                     "mean": SC[m]["mean"], "gain": (gain or {}).get(m), "veg": bool(SC[m].get("veg")),
                                                     **({"rough": 0.62 + 0.38 * float(SC[m]["rough"])} if "rough" in SC[m] else {})}   # as for the near one
    return out


def macro(ob, G, recipe, SC, gain, canopy=None, soil=None):
    """The site's color map in vertices (attribute a_macro): the low frequencies of the ground — the shares of the bake layers
    (terrain_tex.layer_weights — the same rules as in the shaders) × the layer's color (mean of the scan × tone fit; if not
    a scan — the bake tone under the layer) × dilution. Shared by the near and far shaders: at the edge of the near grid the
    far shader brings its own low frequencies down to it, and only detail remains — no seam. → False | the number of vertices."""
    W = np.load(os.path.join(G.folder, "relief_w.npz")); nx, ny = len(W["xs"]), len(W["ys"]); me = ob.data; n = len(me.vertices); ring = _ring(nx, ny)
    if n not in (nx * ny, nx * ny + len(ring)) or "base" not in W.files or "var" not in W.files: return False
    lin = lambda c: np.asarray(c[:3], np.float32)
    def colour(lay):
        cs = [m for m in names(recipe.get(lay)) if m in SC and "mean" in SC[m]]
        if not cs: return lin(W[f"tone_{lay}"]) if f"tone_{lay}" in W.files else None
        cc = [lin(SC[m]["mean"]) * lin(gain.get(m, (1, 1, 1))) for m in cs]
        if len(cc) == 1: return cc[0]
        t = np.clip((W["a_moist"].astype(np.float32) - 0.44) / 0.14, 0, 1); t = t * t * (3 - 2 * t)      # dry/wet pair — by moisture
        return cc[0] * (1 - t[..., None]) + cc[1] * t[..., None]
    C = np.zeros((ny, nx, 3), np.float32); tot = np.zeros((ny, nx), np.float32)
    for lay in ("base",) + ORDER:
        if lay not in W.files: continue
        c = colour(lay); w = W[lay].astype(np.float32)
        if c is None: continue
        C += w[..., None] * (c if c.ndim == 3 else c[None, None, :]); tot += w
    C = C / np.maximum(tot, 1e-4)[..., None]
    if canopy and G.cover_meta():                                   # grass canopy (CANOPY): its mean is the same as in the shaders (a grazing view,
        X, Y = np.meshgrid(W["xs"], W["ys"]); dn = G.cover("density", X, Y); ti = G.cover("tint", X, Y)   # cos θ = 0,35)
        cb = np.asarray(canopy["blade"], np.float32) if canopy.get("blade") else colour("base")   # grass color — the site's base
        if cb is None: cb = np.array((0.18, 0.23, 0.05), np.float32)
        pv = np.exp(-float(canopy["k"]) * dn / 0.35)[..., None]; cm = ti * cb * float(canopy.get("gain", 1.0))
        cm = cm * (1 - pv) + np.asarray(soil, np.float32)[None, None, :3] * pv
        d0, d1 = canopy["dens"]; t = np.clip((dn - d0) / (d1 - d0), 0, 1); w = (t * t * (3 - 2 * t))[..., None]; C = C * (1 - w) + cm * w
    C = C * W["var"].astype(np.float32)
    v = C.reshape(-1, 3); v = v if n == nx * ny else np.concatenate([v, v[ring]])
    me.attributes.new("a_macro", "FLOAT_COLOR", "POINT").data.foreach_set("color", np.concatenate([v, np.ones((len(v), 1), np.float32)], 1).ravel())
    return C


def belt_attrs(ob, G, C=None):
    """Attributes of the belt mesh (bl_relief.build_belt) — by sampling relief_w at its vertices (indices of the shared grid's
    axes, bilinear): the same as for the shared grid (attrs, macro) — the far shader on the belt is the same. → the number of
    vertices."""
    W = np.load(os.path.join(G.folder, "relief_w.npz")); xs, ys = W["xs"], W["ys"]; me = ob.data; n = len(me.vertices)
    co = np.empty(n * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    fx = np.clip(np.interp(co[:, 0], xs, np.arange(len(xs))), 0, len(xs) - 1.001); fy = np.clip(np.interp(co[:, 1], ys, np.arange(len(ys))), 0, len(ys) - 1.001)
    i, j = fx.astype(int), fy.astype(int); tx, ty = fx - i, fy - j
    def bil(A):
        A = np.asarray(A, np.float32); w = [(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty]
        if A.ndim == 3: w = [x[:, None] for x in w]
        return A[j, i] * w[0] + A[j, i + 1] * w[1] + A[j + 1, i] * w[2] + A[j + 1, i + 1] * w[3]
    for nm, ks in PACK.items():
        if nm == "c_g1": continue
        cols = [bil(W[f"a_{k}"]) if f"a_{k}" in W.files else np.zeros(n, np.float32) for k in ks]
        me.attributes.new(f"a_{nm}", "FLOAT_COLOR", "POINT").data.foreach_set("color", np.stack(cols, -1).astype(np.float32).ravel())
    if "var" in W.files: me.attributes.new("a_var", "FLOAT_VECTOR", "POINT").data.foreach_set("vector", bil(W["var"]).astype(np.float32).ravel())
    if C is not None:
        me.attributes.new("a_macro", "FLOAT_COLOR", "POINT").data.foreach_set("color", np.concatenate([bil(C), np.ones((n, 1), np.float32)], 1).ravel())
    from core.surface.terrain_analysis import knife_crests             # knife-edge ridges: the slope of the sides (the far shader takes a slope no lower than this)
    K = knife_crests(G.Hf, float(G.fres)); r = float(G.fres)
    me.attributes.new("a_crest", "FLOAT", "POINT").data.foreach_set("value", G._bil(K, G.fx0, G.fy0, r, co[:, 0], co[:, 1]).astype(np.float32))
    return n


def gains(recipe, SC, W):
    """Fitting scans to the site's tone: the scan's tone → the site palette (the mean color of the baked texture under this
    scan's layers, weighted by coverage shares); for a pair the layer tone is shared by both — the tint is its own.
    → {scan: (r, g, b, 1)}."""
    tone = {}
    for lay in ("base",) + ORDER:
        for m in names(recipe.get(lay)):
            if m in SC and f"tone_{lay}" in W.files and lay in W.files:
                c_ = float(W[lay].astype(np.float32).sum()); t_ = tone.setdefault(m, [np.zeros(3), 0.0]); t_[0] += c_ * W[f"tone_{lay}"]; t_[1] += c_
    pairs = {n for v in recipe.values() if isinstance(v, dict) for n in names(v)}
    return {m: _gain(t_[0] / t_[1], SC[m]["mean"], SC[m].get("veg") or m in pairs) for m, t_ in tone.items() if t_[1] > 0 and "mean" in SC[m]}


def grass_colour(folder):
    """The site's grass color in the distance (linear): the mean of the base scan × the tone fit — what the shaders and the
    a_macro color map show. The far canopy and the meadow cards take it (one color — the tiers add the detail). A dry/wet
    pair — both. → (dry, wet) | None."""
    sj, pw = os.path.join(folder, "surface.json"), os.path.join(folder, "relief_w.npz")
    if not (os.path.exists(sj) and os.path.exists(pw)): return None
    rec = json.load(open(sj))["recipe"]; SC = scans(); g = gains(rec, SC, np.load(pw)); b = rec.get("base")
    col = lambda m: np.asarray(SC[m]["mean"][:3], float) * np.asarray(g.get(m, (1, 1, 1))[:3], float) if m in SC and "mean" in SC[m] else None
    if isinstance(b, dict): d, w = col(b.get("dry")), col(b.get("wet")); return (d, w if w is not None else d) if d is not None else None
    c = col(b) if isinstance(b, str) else None
    return (c, c) if c is not None else None


def apply(ob, G, recipe, seed=1, snowline=None, masks=None, far_cheap=True, canopy=None):
    """recipe — {layer: scan id | procedural material | baked | {dry, wet}}; masks — the biome's layer thresholds (surfaces.yaml
    → masks); far_cheap — the belt beyond the near grid with its own material: True — the far shader (bl_terrain_vista),
    "baked" — the baked image, False — the same shader as up close (expensive), "all" — the far shader on all the ground (the
    frame cost is the same as for an image); canopy — the far canopy (CANOPY; True or knob edits; needs a cover; None — by
    the cover meta: the canopy knobs from Grass's cover.yaml; False — none). Returns a summary."""
    t0 = time.time(); amax = attrs(ob, G.folder); recipe0 = recipe
    if canopy is None: cv_ = (amax or {}).get("c_canopy"); canopy = cv_ if cv_ and cv_.get("use", True) else None   # the knobs are in the cover meta
    if amax is False: return {"surf": "the grid does not match the masks"}
    mat0 = ob.data.materials[0] if ob.data.materials else None
    img = next((nd.image for nd in mat0.node_tree.nodes if nd.type == "TEX_IMAGE"), None) if mat0 else None
    mat = bpy.data.materials.new("terrain_surf"); mat.use_nodes = True; nt = mat.node_tree; nt.nodes.clear(); L = nt.links; N = nt.nodes.new
    SC = scans(); sd = int(seed); W = np.load(os.path.join(G.folder, "relief_w.npz"))
    gain = gains(recipe, SC, W)
    if amax.get("c_cov") is not None and "tone_dirt" in W.files:       # cover ground (leaf litter, needles, mud, pebbles) — matched to the site's soil tone:
        for k_, m in (amax.get("c_scans") or {}).items():               # the leaf-litter scan is autumn red-brown, which looks brown on summer ground; moss keeps its own green
            if m in SC and m not in gain and "mean" in SC[m] and k_ != "soil":
                gain[m] = _gain(W["tone_dirt"], SC[m]["mean"], k_ == "moss")

    def out_(x, i=0): return x.outputs[i] if hasattr(x, "outputs") else x
    def put(sock, x):
        if isinstance(x, (int, float)): sock.default_value = float(x)
        elif isinstance(x, tuple): sock.default_value = x
        else: L.new(out_(x), sock)
    def math_(op, a, b=None):
        m_ = N("ShaderNodeMath"); m_.operation = op; put(m_.inputs[0], a)
        if b is not None: put(m_.inputs[1], b)
        return m_.outputs[0]
    def vmath(op, a, b=None, s=None):
        m_ = N("ShaderNodeVectorMath"); m_.operation = op; put(m_.inputs[0], a)
        if b is not None: put(m_.inputs[1], b)
        if s is not None: put(m_.inputs["Scale"], s)
        return m_.outputs[0]
    def mix(kind, f, a, b):                                         # ShaderNodeMix: FLOAT | RGBA
        m_ = N("ShaderNodeMix"); m_.data_type = kind; ins = {s.identifier: s for s in m_.inputs}; t = "Float" if kind == "FLOAT" else "Color"
        put(ins["Factor_Float"], f); put(ins[f"A_{t}"], a); put(ins[f"B_{t}"], b)
        return next(s for s in m_.outputs if s.identifier == f"Result_{t}")
    def ramp(x, lo, hi):                                            # smooth threshold (smoothstep)
        mr = N("ShaderNodeMapRange"); mr.interpolation_type = "SMOOTHSTEP"; put(mr.inputs["Value"], x)
        mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = lo, hi; return mr.outputs[0]
    geo = N("ShaderNodeNewGeometry"); P = geo.outputs["Position"]
    def noise(scale, s_, detail=4.0):                               # 3D noise (non-repeating), 0…1; scale — m
        tx = N("ShaderNodeTexNoise"); tx.noise_dimensions = "4D"; tx.inputs["W"].default_value = s_ * 7.31
        tx.inputs["Scale"].default_value = 1.0 / scale; tx.inputs["Detail"].default_value = detail; L.new(P, tx.inputs["Vector"])
        return tx.outputs["Fac"]
    sepc = {}
    def cimg(nm):                                                   # cover image by the point's position (the cover grid is 0.5 m)
        if nm not in sepc:
            x0_, y0_, w_, h_ = amax["c_img"]; sx = N("ShaderNodeSeparateXYZ"); L.new(P, sx.inputs[0]); cv = N("ShaderNodeCombineXYZ")
            L.new(math_("DIVIDE", math_("SUBTRACT", sx.outputs["X"], x0_), w_), cv.inputs["X"]); L.new(math_("DIVIDE", math_("SUBTRACT", sx.outputs["Y"], y0_), h_), cv.inputs["Y"])
            t = N("ShaderNodeTexImage"); t.image = bpy.data.images[nm]; t.interpolation = "Linear"; t.extension = "EXTEND"; L.new(cv.outputs[0], t.inputs["Vector"])
            sp = N("ShaderNodeSeparateColor"); L.new(t.outputs["Color"], sp.inputs[0]); sepc[nm] = (sp, t)
        return sepc[nm]
    def attr(k):
        for nm, ks in COVER_IMG.items():
            if k in ks and amax.get("c_img"):
                sp, t = cimg(nm); i = ks.index(k); return t.outputs["Alpha"] if i == 3 else sp.outputs[i]
        for nm, ks in PACK.items():                                 # packed: a color component of RGBA (the material has few attributes)
            if k in ks:
                if nm not in sepc:
                    a = N("ShaderNodeAttribute"); a.attribute_name = f"a_{nm}"; sp = N("ShaderNodeSeparateColor"); L.new(a.outputs["Color"], sp.inputs[0]); sepc[nm] = (sp, a)
                sp, a = sepc[nm]; i = ks.index(k); return a.outputs["Alpha"] if i == 3 else sp.outputs[i]
        a = N("ShaderNodeAttribute"); a.attribute_name = f"a_{k}"; return a.outputs["Fac"]
    sep = N("ShaderNodeSeparateXYZ"); L.new(geo.outputs["Normal"], sep.inputs[0])
    slope = math_("DEGREES", math_("ARCCOSINE", sep.outputs["Z"]))  # slope per pixel, °
    pz = N("ShaderNodeSeparateXYZ"); L.new(P, pz.inputs[0])
    dist = N("ShaderNodeCameraData").outputs["View Distance"]
    far = ramp(dist, 25.0, 110.0)                                   # in the distance — a larger tile
    if amax.get("var"): va_ = N("ShaderNodeAttribute"); va_.attribute_name = "a_var"; var = va_.outputs["Color"]   # dilution by data
    else: var = math_("ADD", 0.86, math_("MULTIPLY", noise(23.0, sd + 11, 3.0), 0.28))   # old site: macro spots by noise
    K = {**MASKS, **(masks or {})}; J = float(K["jit"])
    jit = lambda x, s_, amp=2 * J: math_("ADD", x, math_("MULTIPLY", math_("SUBTRACT", noise(9.0, s_), 0.5), amp))   # noise only for the ragged edge
    expo = attr("expo") if amax.get("expo") is not None else 0.0; dry = math_("SUBTRACT", 1.0, attr("moist")) if amax.get("moist") is not None else 0.5
    rs = math_("ADD", jit(slope, sd + 4), math_("MULTIPLY", expo, 10.0))   # slope "for rock": exposed ground (ridges, brows) — rock starts earlier
    R0, C0 = float(K["rock_from"]), float(K["cliff_from"])
    fac = {"dirt": math_("MULTIPLY", math_("MAXIMUM", ramp(jit(dry, sd + 1, 0.12), 1 - float(K["dirt_dry"]), 1 - float(K["dirt_dry"]) + 0.18), attr("gully")),
                         math_("SUBTRACT", 1, ramp(slope, R0 - 2, R0 + 6))),   # dry exposed ground and gullies (by data, not noise patches)
           "talus": math_("MULTIPLY", math_("MAXIMUM", math_("MULTIPLY", attr("foot"), ramp(slope, 6, 14)),
                                            math_("MULTIPLY", ramp(jit(slope, sd + 2), K["talus_from"], K["talus_from"] + 8), ramp(noise(13.0, sd + 3), 0.45, 0.7))),
                          math_("SUBTRACT", 1, ramp(slope, K["talus_to"], K["talus_to"] + 8))),   # foot of cliffs, rare on slopes
           "rock": math_("MULTIPLY", ramp(rs, R0, R0 + K["rock_band"]), math_("SUBTRACT", 1, ramp(jit(slope, sd + 5), C0 + 1, C0 + 9))),
           "cliff": math_("MAXIMUM", ramp(jit(slope, sd + 6), C0, C0 + 9), attr("cliff")),
           "bank": attr("bank"), "bed": attr("bed"), "pad": math_("MULTIPLY", attr("pad"), 0.6), "pave": math_("MULTIPLY", attr("pave"), 0.9), "road": attr("road")}
    if snowline is not None:
        fac["snow"] = math_("MULTIPLY", ramp(jit(pz.outputs["Z"], sd + 7, 8.0), snowline - 6, snowline + 6), math_("SUBTRACT", 1, ramp(slope, 45, 60)))

    imgs = {}
    def image(path, color):
        if path not in imgs:
            im = bpy.data.images.load(os.path.join(ROOT, path), check_existing=True)
            if not color: im.colorspace_settings.name = "Non-Color"
            imgs[path] = im
        return imgs[path]
    def tex(path, vec, color=True, box=True):                        # triplanar projection (BOX) — cliffs without stretching; ground — from above
        t = N("ShaderNodeTexImage"); t.image = image(path, color); t.projection = "BOX" if box else "FLAT"; t.projection_blend = 0.3
        t.interpolation = "Linear"; L.new(vec, t.inputs["Vector"]); return t.outputs["Color"]
    cnp = {}
    def canopy_col(col, mean):                                      # far canopy (CANOPY) — with the base color, with no BSDF of its own (a shader branch
        if "w" not in cnp:                                          # — +14 % of the frame time): gaps by view angle, tussocks, grazing sheen
            def noise3(scale, s_):
                tx = N("ShaderNodeTexNoise"); tx.noise_dimensions = "3D"; tx.inputs["Scale"].default_value = 1.0 / scale
                tx.inputs["Detail"].default_value = 2.0; L.new(vmath("ADD", P, (s_ * 3.17, s_ * 1.91, 0.0)), tx.inputs["Vector"]); return tx.outputs["Fac"]
            dens = attr("c_cov"); mu = math_("MAXIMUM", math_("ABSOLUTE", vmath("DOT_PRODUCT", geo.outputs["Incoming"], geo.outputs["Normal"]).node.outputs["Value"]), 0.05)
            pv = math_("EXPONENT", math_("DIVIDE", math_("MULTIPLY", dens, -float(CN["k"])), mu))   # fraction of soil visible through the grass
            fade = math_("SUBTRACT", 1.0, ramp(dist, 70.0, 180.0)); mt = float(CN["mottle"])
            mot = math_("ADD", 1.0, math_("MULTIPLY", math_("ADD", math_("SUBTRACT", noise3(0.9, sd + 51), 0.5), math_("MULTIPLY", math_("SUBTRACT", noise3(5.0, sd + 52), 0.5), 0.6)),
                                                    math_("MULTIPLY", fade, mt)))
            gz = math_("MULTIPLY", math_("POWER", math_("SUBTRACT", 1.0, mu), 4.0), math_("MULTIPLY", dens, float(CN["sheen"])))   # grazing sheen ("velvet")
            cnp.update(pv=pv, s=math_("ADD", mot, gz), w=math_("MULTIPLY", ramp(dist, *CN["near"]), ramp(jit(dens, sd + 47, 0.1), *CN["dens"])))
        blade = CN["blade"] or tuple(mean[:3])                      # grass color — the mean of the base scan × the fit (the site's color is not lost)
        bl = vmath("SCALE", vmath("MULTIPLY", tint, tuple(float(c) * float(CN["gain"]) for c in blade)), s=cnp["s"])
        soil = tuple(float(c) for c in (W["tone_dirt"] if "tone_dirt" in W.files else (0.16, 0.15, 0.04)))
        c = mix("RGBA", 1.0, mix("RGBA", cnp["pv"], bl, soil + (1.0,)), var); c.node.blend_type = "MULTIPLY"   # × dilution — as in the far shader
        return mix("RGBA", cnp["w"], col, c)
    def scan(aid, k, box):
        """Scan: (BSDF, height 0…1, geometry displacement in m). The mode depends on the image count (mode): full — color, height,
        roughness; color — color only (no height or relief); tone — the scan's mean color."""
        S = SC[aid]; size = float(S["size"]); amp = (0.08 if aid.startswith("Rock") else 0.03) * size; md = mode.get(aid, "full")
        def finish(col):
            col = mix("RGBA", 1.0, col, var); col.node.blend_type = "MULTIPLY"
            if aid in gain: col = mix("RGBA", 1.0, col, gain[aid]); col.node.blend_type = "MULTIPLY"
            if cov_on and aid in base_names: col = mix("RGBA", 1.0, col, tint); col.node.blend_type = "MULTIPLY"   # the base grass — matched to the cover tone
            if cov_on and CN and aid in base_names: col = canopy_col(col, [float(S["mean"][i]) * float(gain.get(aid, (1, 1, 1, 1))[i]) for i in range(3)])   # far away — canopy
            return col
        if md not in ("full", "height"):
            if md == "tone": col = finish(tuple(S["mean"]) + (1.0,))
            else:
                v = vmath("SCALE", P, s=1.0 / size); v = vmath("ADD", v, (k * 0.37, k * 0.61, 0.0))
                idx = math_("MULTIPLY", noise(4.0 * size, sd + 20 + k, 2.0), 8.0); i = math_("FLOOR", idx); f = math_("FRACT", idx)
                def off_(j):
                    c = N("ShaderNodeCombineXYZ")
                    for a_, w_ in zip(c.inputs, (3.0, 7.0, 5.0)): L.new(math_("SINE", math_("MULTIPLY", j, w_)), a_)
                    return c.outputs[0]
                col = mix("RGBA", ramp(f, 0.2, 0.8), tex(S["Color"], vmath("ADD", v, off_(i)), True, box), tex(S["Color"], vmath("ADD", v, off_(math_("ADD", i, 1.0))), True, box))
                col = finish(mix("RGBA", math_("MULTIPLY", far, 0.55), col, tuple(S["mean"]) + (1.0,)))
            b = N("ShaderNodeBsdfPrincipled"); put(b.inputs["Base Color"], col); b.inputs["Roughness"].default_value = 0.88
            b.inputs["Specular IOR Level"].default_value = 0.35
            return b.outputs[0], None, 0.0
        tex_ = lambda path, vec, color=True: tex(path, vec, color, box)
        v = vmath("SCALE", P, s=1.0 / size); v = vmath("ADD", v, (k * 0.37, k * 0.61, 0.0))
        idx = math_("MULTIPLY", noise(4.0 * size, sd + 20 + k, 2.0), 8.0); i = math_("FLOOR", idx); f = math_("FRACT", idx)   # Quilez, technique 3
        def off(j):
            c = N("ShaderNodeCombineXYZ")
            for a_, w_ in zip(c.inputs, (3.0, 7.0, 5.0)): L.new(math_("SINE", math_("MULTIPLY", j, w_)), a_)
            return c.outputs[0]
        va, vb = vmath("ADD", v, off(i)), vmath("ADD", v, off(math_("ADD", i, 1.0)))
        ha, hb = tex_(S["Displacement"], va, False), tex_(S["Displacement"], vb, False)
        t = ramp(math_("SUBTRACT", f, math_("MULTIPLY", math_("SUBTRACT", ha, hb), 0.6)), 0.2, 0.8)
        col = mix("RGBA", t, tex_(S["Color"], va), tex_(S["Color"], vb))
        if aid.startswith("Rock"):                                  # far away: rock — the same tile, larger,
            col = mix("RGBA", math_("MULTIPLY", far, 0.7), col, tex_(S["Color"], vmath("ADD", vmath("SCALE", v, s=0.21), (0.13, 0.71, 0.0))))
        else: col = mix("RGBA", math_("MULTIPLY", far, 0.55), col, tuple(S["mean"]) + (1.0,))   # ground — toward the mean color (pebbles
        # with a larger tile gave giant "streaks" in the distance)
        col = finish(col)
        h = mix("FLOAT", t, ha, hb); r = mix("FLOAT", t, tex_(S["Roughness"], va, False), tex_(S["Roughness"], vb, False)) if md == "full" else 0.6
        bm = N("ShaderNodeBump"); put(bm.inputs["Height"], h); bm.inputs["Distance"].default_value = amp
        put(bm.inputs["Strength"], math_("SUBTRACT", 0.8, math_("MULTIPLY", far, 0.5)))
        b = N("ShaderNodeBsdfPrincipled"); put(b.inputs["Base Color"], col); put(b.inputs["Roughness"], math_("ADD", 0.62, math_("MULTIPLY", r, 0.38)))
        b.inputs["Specular IOR Level"].default_value = 0.35; L.new(bm.outputs["Normal"], b.inputs["Normal"])   # matte ground: on grass
        return b.outputs[0], h, amp                                 # there was a white sheen at a grazing angle
    bsdf = {}
    cov_on = amax.get("c_cov") is not None; base_names = set(names(recipe.get("base"))); CN = None
    if cov_on:                                                      # cover: the visible ground (1 − density) over the base; the base — tinted to the cover
        recipe = dict(recipe); cs = amax.get("c_scans") or {}
        for k in COVER: recipe["c_" + k] = recipe.get("dirt") if cs.get(k, "dirt") == "dirt" else cs[k]   # c_ — cover layers (its own dirt is mud)
        vis = ramp(jit(math_("SUBTRACT", 1.0, attr("c_cov")), sd + 41, 0.12), 0.30, 0.62)   # ground is visible where there is less than ~0.6 of grass
        fac["c_soil"] = vis
        for i_, k in enumerate(COVER[1:]):
            if amax.get(f"c_{k}", 0.0) > 0.02: fac["c_" + k] = math_("MULTIPLY", vis, ramp(jit(attr(f"c_{k}"), sd + 43 + i_, 0.14), 0.22, 0.55))
        fac["dirt"] = math_("MULTIPLY", attr("gully"), math_("SUBTRACT", 1, ramp(slope, R0 - 2, R0 + 6)))   # dry ground is already in the cover (density)
        cb = N("ShaderNodeCombineColor")
        for i_, k in enumerate(("tint_r", "tint_g", "tint_b")): L.new(math_("MULTIPLY", attr(k), 2.0), cb.inputs[i_])
        tint = cb.outputs[0]
        CN = {**CANOPY, **(canopy if isinstance(canopy, dict) else {})} if canopy else None

    def shader(m):
        key = json.dumps(m, sort_keys=True) if isinstance(m, dict) else m
        if key in bsdf: return bsdf[key]
        if m == "baked":
            tx = N("ShaderNodeTexImage"); tx.image = img; uv = N("ShaderNodeUVMap"); uv.uv_map = "ground"; L.new(uv.outputs[0], tx.inputs[0])
            b = N("ShaderNodeBsdfPrincipled"); L.new(tx.outputs[0], b.inputs["Base Color"]); b.inputs["Roughness"].default_value = 0.92
            bsdf[m] = (b.outputs[0], None, 0.0)
        elif isinstance(m, dict):                                   # pair by moisture: dry ↔ wet (the edges follow the scan height and noise)
            (sa, ha, aa), (sb, hb, ab) = shader(m["dry"]), shader(m["wet"])
            f = ramp(math_("ADD", attr("moist"), math_("MULTIPLY", math_("SUBTRACT", noise(7.0, sd + 31), 0.5), 0.16)), 0.44, 0.58)
            if hb is not None: f = ramp(math_("ADD", hb, math_("SUBTRACT", math_("MULTIPLY", f, 2.0), 1.0)), 0.3, 0.7)
            mx = N("ShaderNodeMixShader"); put(mx.inputs[0], f); L.new(sa, mx.inputs[1]); L.new(sb, mx.inputs[2])
            bsdf[key] = (mx.outputs[0], mix("FLOAT", f, ha if ha is not None else 0.5, hb if hb is not None else 0.5), mix("FLOAT", f, aa, ab))
        elif m in SC: bsdf[m] = scan(m, len(bsdf), box=any(m in names(recipe.get(l_)) for l_ in ("talus", "rock", "cliff")))
        elif m in IG_NAMES:
            import loader as IG
            NodeWrangler = IG.NodeWrangler
            with IG.FixedSeed(sd * 101 + len(bsdf)): bsdf[m] = (out_(_shader_of(IG, m)(NodeWrangler(nt), **({"obj": ob} if m == "mountain" else {}))), None, 0.0)
        else: raise KeyError(f"surface {m}: neither a scan (components/site/terrain_scans.py fetch) nor a procedural material from core/ext/terrain_materials")
        return bsdf[key]
    base = recipe.get("base", "baked")
    lays = [l for l in ((["c_" + k for k in COVER] + list(ORDER)) if cov_on else list(ORDER))
            if l in fac and l in recipe and not (l in PLAN and amax.get(l, 0.0) < 0.01)]
    mode, pics, order = {}, set(["baked"] if base == "baked" else []) | (set(COVER_IMG) if amax.get("c_img") else set()), []   # image budget (IMG_BUDGET) in three passes by importance:
    for lay in sorted(["base"] + lays, key=lambda l: PRI.get(l, 9)):    # 1) color — for every scan (yard soil, leaf litter, mud — not just tone),
        for m in names(recipe.get(lay) if lay != "base" else base):     # 2) height (relief, layer edges) — for the base, rock, ground…,
            if m in SC and m not in order: order.append(m)              # 3) roughness — if there is still room; if it did not fit — tone
    MAPS = {"color": ("Color",), "height": ("Color", "Displacement"), "full": ("Color", "Displacement", "Roughness")}
    for md in ("color", "height", "full"):
        for m in order:
            if md != "color" and m not in mode: continue
            need = {SC[m][q] for q in MAPS[md]} - pics
            if len(pics) + len(need) <= IMG_BUDGET: mode[m] = md; pics |= need
    for m in order: mode.setdefault(m, "tone")
    s_, h_, a_ = shader(base); used = [base]
    h_ = h_ if h_ is not None else 0.5; disp = math_("MULTIPLY", math_("SUBTRACT", h_, 0.5), a_)
    for lay in lays:
        s2, h2, a2 = shader(recipe[lay]); f = fac[lay]
        if h2 is not None:                                          # HeightLerp: natural — soft (sand flows onto the rock), man-made — sharp
            lo_, hi_ = (0.3, 0.7) if lay in ("pad", "pave", "road") else (0.12, 0.88)
            f = ramp(math_("ADD", h2, math_("SUBTRACT", math_("MULTIPLY", f, 2.0), 1.0)), lo_, hi_)
        mx = N("ShaderNodeMixShader"); put(mx.inputs[0], f); L.new(s_, mx.inputs[1]); L.new(s2, mx.inputs[2]); s_ = mx.outputs[0]
        disp = mix("FLOAT", f, disp, math_("MULTIPLY", math_("SUBTRACT", h2 if h2 is not None else 0.5, 0.5), a2))
        used.append(f"{lay}:{recipe[lay]}")
    out = N("ShaderNodeOutputMaterial"); L.new(s_, out.inputs["Surface"])
    aov = N("ShaderNodeOutputAOV"); aov.aov_name = "ground"; aov.inputs["Value"].default_value = 1.0   # ground mask of the frame (checks)
    dn = N("ShaderNodeDisplacement"); dn.inputs["Midlevel"].default_value = 0.0; dn.inputs["Scale"].default_value = 1.0
    put(dn.inputs["Height"], math_("MULTIPLY", disp, attr("near"))); L.new(dn.outputs[0], out.inputs["Displacement"])
    mat.displacement_method = "BOTH"                                # vertex displacement only on the patch near the camera (a_near), the rest — relief along the normal
    if img is not None:                                             # Workbench shows the material's active image
        nd = next((x for x in nt.nodes if x.type == "TEX_IMAGE" and x.image == img), None) or N("ShaderNodeTexImage")
        nd.image = img; nt.nodes.active = nd
    ob.data.materials.clear(); ob.data.materials.append(mat)
    nfar = 0
    if img is not None and far_cheap:                               # the belt beyond the near grid — its own light material (Eevee evaluates all branches
        if far_cheap == "baked":                                    # of the tree, a share of 0 does not help): the far shader (bl_terrain_vista) or, on request,
            mf = bpy.data.materials.new("terrain_far"); mf.use_nodes = True; ft = mf.node_tree; ft.nodes.clear()   # the baked image
            tx = ft.nodes.new("ShaderNodeTexImage"); tx.image = img; uv = ft.nodes.new("ShaderNodeUVMap"); uv.uv_map = "ground"
            ft.links.new(uv.outputs[0], tx.inputs[0]); bf = ft.nodes.new("ShaderNodeBsdfPrincipled"); ft.links.new(tx.outputs[0], bf.inputs["Base Color"])
            bf.inputs["Roughness"].default_value = 0.92; bf.inputs["Specular IOR Level"].default_value = 0.35
            o2 = ft.nodes.new("ShaderNodeOutputMaterial"); ft.links.new(bf.outputs[0], o2.inputs["Surface"]); ft.nodes.active = tx
            a2 = ft.nodes.new("ShaderNodeOutputAOV"); a2.aov_name = "ground"; a2.inputs["Value"].default_value = 1.0
        else:
            from core.surface import bl_terrain_vista as TV
            nrm, dat = os.path.join(G.folder, "relief_nrm.png"), os.path.join(G.folder, "relief_far.png")   # site terrain_tex: normals and data of the belt
            soil_ = tuple(float(c) for c in (W["tone_dirt"] if "tone_dirt" in W.files else (0.16, 0.15, 0.04)))
            C = macro(ob, G, recipe0, SC, gain, CN, soil_); seam = (*G.near_bounds(), SEAM_BAND) if C is not False and far_cheap != "all" else None   # at the edge —
            # toward the color map (all the ground is under the far shader — there is no near one, nothing to blend to)
            cvf = {"image": bpy.data.images["cover_far"], "box": amax["c_far"], "knobs": CN, "soil": soil_} if CN and amax.get("c_far") else None   # canopy — on the belt as well
            belt = bpy.data.objects.get(ob.name + "_belt")            # the belt has its own mesh (bl_relief.build_belt): the same far shader,
            mf = TV.material("terrain_far", vista_layers(recipe0, SC, gain), sd, snowline, K, None if belt is not None else nrm, img, True,
                             G.loc.get("strata"), dat, seam, cover=cvf)   # with a belt, "terrain" has only the faces of the site: normals — from its geometry
            # (the belt map at 4.7 m on the steep edge of the site drew smooth "streams" of light)
            if belt is not None:                                    # normals — from the geometry (its step is finer than the normal map)
                belt_attrs(belt, G, C if C is not False else None)
                mb = TV.material("terrain_belt", vista_layers(recipe0, SC, gain), sd, snowline, K, None, img, True, G.loc.get("strata"), dat, seam, cover=cvf)
                belt.data.materials.clear(); belt.data.materials.append(mb)
        ob.data.materials.append(mf); x0, y0, x1, y1 = G.near_bounds()
        c = np.empty(len(ob.data.polygons) * 3); ob.data.polygons.foreach_get("center", c); c = c.reshape(-1, 3)
        edge = np.minimum.reduce([c[:, 0] - x0, x1 - c[:, 0], c[:, 1] - y0, y1 - c[:, 1]])   # to the edge of the near grid (inside > 0)
        from core.surface.noise import fbm                          # the boundary between materials is patchy in the last 0–30 m of the near grid (like
        band = SEAM_M * np.clip(fbm(c[:, 0] / 45.0, c[:, 1] / 45.0, 3, sd + 97) * 0.9 + 0.5, 0, 1)   # an LOD transition in engines): no straight line —
        far_ = (edge < band).astype(np.int32); nfar = int(far_.sum())  # the shader difference shows as a variegated meadow
        if far_cheap == "all":                                      # all the ground is under the far shader: there is no near shader in the frame
            far_ = np.ones_like(far_); nfar = len(far_)
        ob.data.polygons.foreach_set("material_index", far_)
    return {"surf_far_faces": nfar, "surf_far": (mf["sk_vista"].to_dict() if "sk_vista" in mf else "baked") if nfar else None, "surf": used, "surf_images": len(pics), "surf_canopy": bool(canopy and cov_on), "surf_modes": {m: md for m, md in mode.items() if md != "full"}, "surf_gain": {m: [round(x, 2) for x in g[:3]] for m, g in gain.items()}, "s_surf": round(time.time() - t0, 2)}

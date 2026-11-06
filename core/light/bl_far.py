"""Far terrain and sky in Blender, from the far-terrain plan (core/light/far.py resolve → out/far/<name>/plan.json): the far_sky world —
a Multiple Scattering sky model driven by the sun, at night by the moon (night tint), the sun disk, the moon with a phase set by the sun direction,
stars; the disk, moon and stars are visible to camera rays only (the scene's light comes from a lamp, otherwise the sun would be counted twice); the sun
and moon lamps take color and strength from the plan (the far_sky_table.json calibration sits next to it). A module for core/outputs/bl_color.py: attach(sc, plan) — world, lamps,
the AgX view with the plan's exposure; per_camera(sc, spec) — star size per camera pixel; render_sky(sc, path) — the sky as seen by a camera.
blender -b --factory-startup -P core/blender/run.py -- core/light/bl_far.py -- <plan.json> <folder> [size=64]   → a sky cube (6 EXR faces) and a __RESULT__ line
with the "one sun" check: the brightest spot of the sky at the horizon lies at the azimuth of the sun lamp."""
import re
import json, math, os, sys
import core.paths as _core_paths      # the code is here, the core — core/<part> through kernel
import bpy
from mathutils import Vector


def sun_dir(az, el):
    a, e = math.radians(az), math.radians(el)
    return Vector((math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)))


class _G:
    """Shorthand for world nodes: a new node with inputs, links, vector and scalar arithmetic."""

    def __init__(self, nt):
        self.nt, self.N, self.link = nt, nt.nodes, nt.links.new

    def node(self, t, **kw):
        n = self.N.new(t)
        for k, v in kw.items():
            if k == "op": n.operation = v
            elif k == "clamp": n.use_clamp = v
            elif k == "name": n.name = n.label = v
            else: n.inputs[k].default_value = v
        return n

    def _in(self, n, i, v):
        if hasattr(v, "is_output") or hasattr(v, "links"): self.link(v, n.inputs[i])
        else: n.inputs[i].default_value = v

    def math(self, op, a, b=0.0, c=None, clamp=False, name=None):
        n = self.node("ShaderNodeMath", op=op, clamp=clamp)
        if name: n.name = n.label = name
        self._in(n, 0, a); self._in(n, 1, b)
        if c is not None: self._in(n, 2, c)
        return n.outputs[0]

    def vec(self, op, a, b=None, s=None):
        n = self.node("ShaderNodeVectorMath", op=op); self._in(n, 0, a)
        if b is not None: self._in(n, 1, b)
        if s is not None: self._in(n, 3, s)                                # SCALE: the "Scale" input
        return n.outputs["Value" if op in ("DOT_PRODUCT", "LENGTH", "DISTANCE") else "Vector"]

    def const(self, v):
        n = self.node("ShaderNodeCombineXYZ"); n.inputs[0].default_value, n.inputs[1].default_value, n.inputs[2].default_value = v
        return n.outputs[0]

    def xyz(self, v):
        n = self.node("ShaderNodeSeparateXYZ"); self.link(v, n.inputs[0]); return n.outputs


def _sky(g, az, el, sky, D=None):
    n = g.node("ShaderNodeTexSky"); n.sky_type = sky["model"]; n.sun_disc = False
    n.sun_elevation, n.sun_rotation = math.radians(el), math.radians(90.0 - az)   # the node's rotation is clockwise from +y (measured)
    n.air_density, n.aerosol_density, n.ozone_density = sky["air"], sky["aerosol"], sky["ozone"]
    if D is not None: g.link(D, n.inputs["Vector"])                     # in a material the direction is explicit (in the world it is the default)
    return n.outputs[0]


def _sky_color(g, P, D, dz=None, world=False):
    """The plan's sky in direction D (without the disk, moon, stars): the sun's sky, at night — the moon's with the night tint and airglow, overcast — CIE gray,
    weather saturation and tint. The world and the effects (the upward haze color) take the same thing."""
    sk = P["sky"]; Dv = None if world else D
    sky = g.vec("SCALE", _sky(g, P["sun"]["az"], P["sun"]["el"], sk, Dv), s=sk.get("k", 1.0))
    if sk["w_night"] > 0:                                                     # the moon's sky — the same model with the moon in place of the sun, night tint
        night = _sky(g, P["moon"]["az"], max(P["moon"]["el"], -6.0), sk, Dv)
        k = sk["k_night"] * sk["w_night"] * (1.0 if P["moon"]["el"] > -6 else 0.3)
        night = g.vec("MULTIPLY", night, g.const([k * t for t in sk["night_tint"]]))
        floor = g.const([0.0008 * sk["w_night"], 0.0010 * sk["w_night"], 0.0020 * sk["w_night"]])   # night sky glow: without the moon it is not a black hole
        sky = g.vec("ADD", g.vec("ADD", sky, night), floor)
    if dz is None: dz = g.xyz(D)[2]
    if sk.get("over", 0) > 0:                                                  # overcast: CIE gray sky, (1 + 2·sin h)/3 of the zenith
        grad = g.math("MULTIPLY", g.math("ADD", g.math("MULTIPLY", g.math("MAXIMUM", dz, 0.0), 2.0), 1.0), sk["over_L"] / 3.0)
        mx = g.node("ShaderNodeMix"); mx.data_type = "RGBA"; mx.inputs[0].default_value = sk["over"]
        g.link(sky, mx.inputs[6]); g.link(g.vec("SCALE", g.const(sk["over_rgb"]), s=grad), mx.inputs[7]); sky = mx.outputs[2]
    hs = g.node("ShaderNodeHueSaturation", Saturation=sk["sat"], Value=1.0, Fac=1.0); g.link(sky, hs.inputs["Color"])
    return g.vec("MULTIPLY", hs.outputs[0], g.const(sk["tint"]))


def build_world(P, name="far_sky"):
    w = bpy.data.worlds.new(name); nt = w.node_tree; g = _G(nt)
    for n in list(nt.nodes): nt.nodes.remove(n)
    S, M, sk = Vector(P["sun"]["dir"]), Vector(P["moon"]["dir"]), P["sky"]
    D = g.node("ShaderNodeTexCoord").outputs["Generated"]                     # view direction
    cam = g.node("ShaderNodeLightPath").outputs["Is Camera Ray"]
    dz = g.xyz(D)[2]
    sky = _sky_color(g, P, D, dz, world=True)
    # --- stars: Voronoi points on a sphere; size set per camera pixel (per_camera); they fade near the horizon (air mass)
    stars = None
    if sk["stars"] > 0:
        scale = 380.0
        vor = g.node("ShaderNodeTexVoronoi"); vor.voronoi_dimensions = "3D"; vor.feature = "F1"; vor.inputs["Scale"].default_value = 1.0
        g.link(g.vec("SCALE", D, s=scale), vor.inputs["Vector"])
        rgb = g.node("ShaderNodeSeparateColor"); g.link(vor.outputs["Color"], rgb.inputs[0])
        size = g.node("ShaderNodeValue", name="far_star_size"); size.outputs[0].default_value = 0.4
        core = g.math("POWER", g.math("SUBTRACT", 1.0, g.math("DIVIDE", vor.outputs["Distance"], size.outputs[0]), clamp=True), 2.0)
        there = g.math("GREATER_THAN", rgb.outputs[0], 0.965)                   # ~3.5 % of cells are a star
        bright = g.math("ADD", g.math("MULTIPLY", g.math("POWER", rgb.outputs[1], 6.0), 6.0), 0.15)   # many dim ones, few bright
        fade = g.math("MULTIPLY", g.math("SUBTRACT", g.math("DIVIDE", dz, 0.14), 0.05), 1.0, clamp=True)
        v = g.math("MULTIPLY", g.math("MULTIPLY", g.math("MULTIPLY", core, there), bright), g.math("MULTIPLY", fade, sk["stars"] * 0.02))
        warm = g.node("ShaderNodeMix"); warm.data_type = "RGBA"
        g.link(rgb.outputs[2], warm.inputs[0]); warm.inputs[6].default_value = (1.0, 0.82, 0.66, 1); warm.inputs[7].default_value = (0.72, 0.84, 1.0, 1)
        stars = g.vec("SCALE", warm.outputs[2], s=v)
    # --- moon: a disk with a sphere normal, the lit part and the terminator set by the sun direction
    moon_rgb, moon_mask = None, None
    mo = P["moon"]
    if mo["el"] > -1.0 and (sk["w_night"] > 0 or P["sun"]["el"] < 25):
        a = math.radians(mo["size_deg"] / 2); up = Vector((0, 0, 1)) if abs(M.z) < 0.99 else Vector((1, 0, 0))
        e1 = M.cross(up).normalized(); e2 = e1.cross(M).normalized()
        x = g.math("DIVIDE", g.vec("DOT_PRODUCT", D, g.const(list(e1))), math.sin(a)); y = g.math("DIVIDE", g.vec("DOT_PRODUCT", D, g.const(list(e2))), math.sin(a))
        r2 = g.math("ADD", g.math("MULTIPLY", x, x), g.math("MULTIPLY", y, y))
        front = g.math("GREATER_THAN", g.vec("DOT_PRODUCT", D, g.const(list(M))), 0.0)
        edge = g.math("MULTIPLY", g.math("DIVIDE", g.math("SUBTRACT", 1.0, r2), 0.04), 1.0, clamp=True)   # a soft edge a pixel or two wide
        moon_mask = g.math("MULTIPLY", edge, front)
        z = g.math("SQRT", g.math("MAXIMUM", g.math("SUBTRACT", 1.0, r2), 0.0))
        n = g.vec("ADD", g.vec("ADD", g.vec("SCALE", g.const(list(e1)), s=x), g.vec("SCALE", g.const(list(e2)), s=y)), g.vec("SCALE", g.const(list(M)), s=g.math("MULTIPLY", z, -1.0)))
        lam = g.math("MAXIMUM", g.vec("DOT_PRODUCT", n, g.const(list(S))), 0.0)
        nz = g.node("ShaderNodeTexNoise", Scale=2.5, Detail=6.0); xy = g.node("ShaderNodeCombineXYZ"); g.link(x, xy.inputs[0]); g.link(y, xy.inputs[1]); g.link(xy.outputs[0], nz.inputs["Vector"])
        alb = g.math("ADD", g.math("MULTIPLY", nz.outputs["Fac"], 0.55), 0.6)       # seas and continents as patches
        lit = g.math("ADD", g.math("MULTIPLY", lam, 2.2), 0.012)                   # earthshine — the dark part is faintly visible
        moon_rgb = g.vec("SCALE", g.const([1.0, 0.97, 0.92]), s=g.math("MULTIPLY", g.math("MULTIPLY", alb, lit), moon_mask))
    # --- sun disk: brightness = the lamp's light per solid angle of the disk (in the frame a white spot; the halo comes from the haze)
    sun_rgb = None
    if P["sun"]["el"] > -1.0 and max(P["sun"]["lamp_rgb"]) > 0:
        r = math.radians(P["sun"]["angle_deg"] / 2); rad = max(P["sun"]["lamp_rgb"]) / (math.pi * r * r)
        c = g.vec("DOT_PRODUCT", D, g.const(list(S)))
        disc = g.math("MULTIPLY", g.math("DIVIDE", g.math("SUBTRACT", c, math.cos(r * 1.25)), math.cos(r) - math.cos(r * 1.25)), 1.0, clamp=True)
        lr = P["sun"]["lamp_rgb"]; m_ = max(lr)
        sun_rgb = g.vec("SCALE", g.const([v / m_ for v in lr]), s=g.math("MULTIPLY", disc, rad))
    extra = None
    for part in (stars if moon_mask is None or stars is None else g.vec("SCALE", stars, s=g.math("SUBTRACT", 1.0, moon_mask)), moon_rgb, sun_rgb):
        if part is not None: extra = part if extra is None else g.vec("ADD", extra, part)
    col = sky if extra is None else g.vec("ADD", sky, g.vec("SCALE", extra, s=cam))   # disk, moon, stars — camera only
    img = sk.get("image")
    if img:                                                                   # sky as an image (far.sky_image_spec): the camera sees the image (rotation,
        tex = g.node("ShaderNodeTexEnvironment"); tex.image = bpy.data.images.load(img["path"], check_existing=True)   # calibration),
        mp = g.node("ShaderNodeMapping", name="far_sky_yaw"); mp.vector_type = "VECTOR"                  # the lighting uses the node's model
        mp.inputs["Rotation"].default_value = (0.0, 0.0, -math.radians(img["yaw"]))                       # the image is rotated by yaw — look toward −yaw
        g.link(D, mp.inputs["Vector"]); g.link(mp.outputs[0], tex.inputs["Vector"])
        mx = g.node("ShaderNodeMix"); mx.data_type = "RGBA"; g.link(cam, mx.inputs[0]); g.link(sky, mx.inputs[6])
        g.link(g.vec("SCALE", tex.outputs["Color"], s=img["gain"]), mx.inputs[7]); col = mx.outputs[2]   # disk, moon, stars — on the image
    bg = g.node("ShaderNodeBackground", Strength=1.0); g.link(col, bg.inputs["Color"])
    out = g.node("ShaderNodeOutputWorld"); g.link(bg.outputs[0], out.inputs["Surface"])
    w.sun_threshold = 1e9                                                     # the sun comes from the lamp — do not pull it out of the sky
    return w


def add_lamps(sc, P):
    made = []
    for key, name in (("sun", "far_sun"), ("moon", "far_moon")):
        L_ = P[key]; rgb = L_["lamp_rgb"]; m = max(rgb) if rgb else 0.0
        if m <= 0 or (key == "moon" and not L_.get("on")): continue
        ld = bpy.data.lights.new(name, "SUN"); ld.energy = m; ld.color = [c / m for c in rgb]
        if os.environ.get("FAR_NO_SUN_SHADOW") or (P.get("sun") or {}).get("shadow") is False:   # no luminary shadow: a knob of the "light" step
            ld.use_shadow = False                                               # shadows (plan: sun.shadow) or debugging
        ld.angle = math.radians(P["sun"]["angle_deg"] if key == "sun" else 0.6)
        ob = bpy.data.objects.new(name, ld); sc.collection.objects.link(ob)
        ob.rotation_euler = sun_dir(L_["az"], L_["el"]).to_track_quat("Z", "Y").to_euler(); made.append(ob)
    return made


def attach(sc, P):
    """World, lamps and the AgX view of the plan into the open scene (the caller removes the old lamps)."""
    sc.world = build_world(P)
    lamps = add_lamps(sc, P)
    sc.view_settings.view_transform = "AgX"; sc.view_settings.look = "None"; sc.view_settings.exposure = P["light"]["ev"]
    from core import hooks; hooks.call("light", sc, P)                # extension point: light through foliage (Plants), etc.
    return lamps


def per_camera(sc, spec):
    """A star is about 1.3 pixels of this camera (no flicker and no "snow" at 4K)."""
    W, H = spec["resolution"]; fit = spec.get("sensor_fit", "AUTO")
    L = H if fit == "VERTICAL" else W if fit == "HORIZONTAL" else max(W, H)
    f = spec["lens_mm"] / spec.get("sensor_width_mm" if fit != "VERTICAL" else "sensor_height_mm", 36.0) * L
    n = sc.world.node_tree.nodes.get("far_star_size") if sc.world else None
    if n: n.outputs[0].default_value = 1.3 / f * 380.0


def render_sky(sc, path):
    """The sky seen by the scene camera without objects — a linear EXR (the backdrop of the frame and of the previs)."""
    hidden = [o for o in sc.objects if o.type != "CAMERA" and not o.hide_render]
    for o in hidden: o.hide_render = True
    eng, ft, fmt, med = sc.render.engine, sc.render.film_transparent, sc.render.image_settings.file_format, sc.render.image_settings.media_type
    smp = sc.eevee.taa_render_samples
    sc.render.engine = "BLENDER_EEVEE"; sc.render.film_transparent = False; sc.eevee.taa_render_samples = 16
    sc.render.image_settings.media_type = "IMAGE"; sc.render.image_settings.file_format = "OPEN_EXR"; sc.render.image_settings.color_depth = "32"
    sc.render.filepath = path; bpy.ops.render.render(write_still=True)
    for o in hidden: o.hide_render = False
    sc.render.engine, sc.render.film_transparent, sc.eevee.taa_render_samples = eng, ft, smp
    sc.render.image_settings.media_type = med; sc.render.image_settings.file_format = fmt


def ensure_plan(scene_dir, path, kv=()):
    """The scene's far-terrain plan (core/light/far.py resolve in the .venv): the map and color passes use the same relief (cached by description)."""
    import subprocess
    root = _core_paths.ROOT
    r = subprocess.run([os.path.join(root, ".venv/bin/python"), "-m", "core.light.far", "resolve", scene_dir, *kv, f"out={path}"],
                       capture_output=True, text=True)
    if r.returncode: raise SystemExit("far.py resolve: " + r.stderr[-600:])
    P = json.load(open(path)); P["path"] = path; return P


def attach_terrain(sc, P, ground_far=None):
    """The plan's far terrain into the scene: two meshes (near, up to ~6 km, and far) with a top-down color texture — both Workbench
    (the active image) and Eevee see it. The scene's flat far ground drops a meter lower — it stays as a patch and does not compete with the mountains."""
    import numpy as np
    T = P.get("terrain")
    if not T: return []
    D = np.load(os.path.join(T["dir"], "mesh.npz")); obs = []
    for name in T.get("parts", ["near", "far"]):
        v, f, uv = D[f"{name}_v"], D[f"{name}_f"], D[f"{name}_uv"]
        me = bpy.data.meshes.new(f"far_terrain_{name}"); me.vertices.add(len(v)); me.vertices.foreach_set("co", v.ravel())
        me.loops.add(f.size); me.loops.foreach_set("vertex_index", f.ravel())
        me.polygons.add(len(f)); me.polygons.foreach_set("loop_start", np.arange(0, f.size, 4, dtype=np.int32)); me.polygons.foreach_set("loop_total", np.full(len(f), 4, np.int32))
        me.update(); me.shade_smooth()
        uvl = me.uv_layers.new(name="UVMap"); uvl.data.foreach_set("uv", uv[f.ravel()].ravel())
        img = bpy.data.images.load(os.path.join(T["dir"], f"{name}.png")); m = None
        if name == "near":                                          # up to ~6 km — the far shader with the same set as the site belt (the tone at the seam
            from core.surface import bl_terrain_vista as TV            # matches); snow — a fraction of the mountain height by biome, as for the image
            from core.light.far import BIOME
            dat = os.path.join(T["dir"], "near_data.png")            # the site's mountains (real ones): the site's snow and belt heights from the data map
            sl = T["snow_z"] if T.get("heights") else T["z0"] + BIOME.get(T["biome"], BIOME["temperate"])["snowline"] * T["height"]
            m = TV.horizon(f"far_terrain_{name}", img, sl, data=dat if T.get("heights") and os.path.exists(dat) else None)
        if m is None:                                               # farther out and with no site — a top-down color image (the haze eats detail in the distance)
            m = bpy.data.materials.new(f"far_terrain_{name}"); nt = m.node_tree; bs = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
            tx = nt.nodes.new("ShaderNodeTexImage"); tx.image = img; tx.extension = "EXTEND"
            nt.links.new(tx.outputs["Color"], bs.inputs["Base Color"]); bs.inputs["Roughness"].default_value = 0.95; nt.nodes.active = tx
        m.diffuse_color = (0.45, 0.43, 0.40, 1.0); me.materials.append(m)
        ob = bpy.data.objects.new(f"far_terrain_{name}", me); sc.collection.objects.link(ob); ob["sketch_kind"] = "ridge"; obs.append(ob)
    if ground_far is not None: ground_far.location.z -= 1.0
    return obs


def _air_fog(g, P, col, sk):
    """The plan's hazes in an effect shader (cloud, smoke, fire) — all layers, as in core/light/far.py fog_terms, by distance to the camera:
    air — the sky color in the view direction (a cloud dissolves into the sky behind it); the ground layer (fog, war smoke) — its own color lit by
    the sun and moon (Henyey–Greenstein halo + multiple scattering) and by the sky. The layers are mixed by thickness fractions."""
    geo = g.node("ShaderNodeNewGeometry"); cdat = g.node("ShaderNodeCameraData")
    I, Pos, D = geo.outputs["Incoming"], geo.outputs["Position"], cdat.outputs["View Distance"]
    zc = g.xyz(g.vec("ADD", Pos, g.vec("SCALE", I, s=D)))[2]                  # camera = point + toward-camera × distance
    v = g.math("MULTIPLY", g.xyz(I)[2], -1.0)
    skyc = _horizon_tex(g, P, I)                                               # the sky at the horizon by azimuth — the same table as for the frame haze
    lights, sky_E = _lights(P)
    tau, Lsum = None, None
    for Ly in P["fog"]["layers"]:
        H = Ly["H"]; k = g.math("DIVIDE", g.math("MULTIPLY", v, D), H)
        ks = g.math("ADD", k, g.math("MULTIPLY", g.math("COMPARE", k, 0.0, 1e-4), 1e-4))   # k ≈ 0 → 1e-4 (the limit (1 − e^−k)/k → 1)
        f = g.math("DIVIDE", g.math("SUBTRACT", 1.0, g.math("EXPONENT", g.math("MULTIPLY", g.math("MAXIMUM", ks, -40.0), -1.0))), ks)
        rho = g.math("MULTIPLY", g.math("EXPONENT", g.math("MINIMUM", g.math("DIVIDE", g.math("SUBTRACT", Ly["z0"], zc), H), 3.0)), Ly["a"])
        t_ = g.math("MULTIPLY", g.math("MULTIPLY", rho, D), f)
        if Ly["src"] == "horizon":                                             # haze light: at the horizon — the horizon sky by azimuth, upward —
            sky_up = _sky_color(g, P, g.vec("SCALE", I, s=-1.0), g.math("MULTIPLY", g.xyz(I)[2], -1.0))   # smoothly toward the sky in that direction
            wv = g.math("MULTIPLY", g.math("DIVIDE", g.math("SUBTRACT", v, 0.02), 0.3), 1.0, clamp=True)   # (the blue overhead comes from Rayleigh scattering above the layer)
            mxv = g.node("ShaderNodeMix"); mxv.data_type = "RGBA"; g.link(wv, mxv.inputs[0]); g.link(skyc, mxv.inputs[6]); g.link(sky_up, mxv.inputs[7])
            L = mxv.outputs[2]
        else:
            alb = Ly["albedo"]; L = g.vec("ADD", g.vec("SCALE", g.const([e / math.pi * 0.5 * a for e, a in zip(sky_E, alb)]), s=1.0), g.vec("SCALE", skyc, s=0.5))
            for Ld, E in lights:
                ph = g.math("ADD", g.math("MULTIPLY", _hg(g, I, Ld, Ly["g"]), 0.55), 0.45 * max(Ld.z, 0.0) / math.pi)
                L = g.vec("ADD", L, g.vec("SCALE", g.const([e * a for e, a in zip(E, alb)]), s=ph))
        tau = t_ if tau is None else g.math("ADD", tau, t_)
        Lsum = g.vec("SCALE", L, s=t_) if Lsum is None else g.vec("ADD", Lsum, g.vec("SCALE", L, s=t_))
    T = g.math("EXPONENT", g.math("MULTIPLY", tau, -1.0))
    Lin = g.vec("SCALE", Lsum, s=g.math("DIVIDE", g.math("SUBTRACT", 1.0, T), g.math("MAXIMUM", tau, 1e-9)))
    gm = (P.get("fires") or {}).get("glow_map")
    if gm and os.path.exists(gm):                                            # fire glow — from the direction map (far.py glow_map)
        im = bpy.data.images.get(os.path.basename(gm)) or bpy.data.images.load(gm); im.colorspace_settings.name = "Linear Rec.709"
        d = g.vec("SCALE", I, s=-1.0); dx, dy, dz = g.xyz(d)
        u = g.math("ADD", g.math("DIVIDE", g.math("ARCTAN2", dy, dx), 2 * math.pi), 0.5)
        vv = g.math("ADD", g.math("DIVIDE", g.math("ARCSINE", dz), math.pi), 0.5)
        uv = g.node("ShaderNodeCombineXYZ"); g.link(u, uv.inputs[0]); g.link(vv, uv.inputs[1])
        tx = g.node("ShaderNodeTexImage"); tx.image = im; tx.interpolation = "Linear"; tx.extension = "EXTEND"; g.link(uv.outputs[0], tx.inputs["Vector"])
        Lin = g.vec("ADD", Lin, g.vec("SCALE", tx.outputs["Color"], s=g.math("SUBTRACT", 1.0, T)))
    return g.vec("ADD", g.vec("SCALE", col, s=T), Lin)


def _horizon_tex(g, P, I):
    """The sky color at the horizon by view azimuth (the render_horizon table as a 180×1 image, linear, wrapping around) — as in far.py fog_terms."""
    from core.light import far
    hp = os.path.splitext(P["path"])[0] + "_horizon.json"; name = "far_hz_" + P["hash"]
    im = bpy.data.images.get(name)
    if im is None:
        H = json.load(open(hp)); n = len(H["rgb"])
        im = bpy.data.images.new(name, n, 1, float_buffer=True); im.colorspace_settings.name = "Linear Rec.709"
        im.pixels.foreach_set([c for rgb in far.horizon_rgb(H).tolist() for c in (*rgb, 1.0)]); im.pack()   # without luminary disks
    d = g.vec("SCALE", I, s=-1.0); dx, dy, _ = g.xyz(d)
    u = g.math("ADD", g.math("DIVIDE", g.math("ARCTAN2", dy, dx), 2 * math.pi), 1.0)       # azimuth 0…360° → 0…1 (wrapping around)
    uv = g.node("ShaderNodeCombineXYZ"); g.link(g.math("FRACT", u), uv.inputs[0]); uv.inputs[1].default_value = 0.5
    tx = g.node("ShaderNodeTexImage"); tx.image = im; tx.interpolation = "Linear"; tx.extension = "REPEAT"; g.link(uv.outputs[0], tx.inputs["Vector"])
    return tx.outputs["Color"]


def _lights(P):
    """Luminaries for effects: (direction, RGB illuminance) of the sun and moon from the plan; the sky — the illuminance of a horizontal surface."""
    out = []
    for key in ("sun", "moon"):
        if max(P[key]["lamp_rgb"]) > 0: out.append((Vector(P[key]["dir"]), P[key]["lamp_rgb"]))
    sky_E = [a + b for a, b in zip(P["light"]["sky_e_rgb"], P["light"].get("night_sky_e_rgb", [0, 0, 0]))]
    return out, sky_E


def _effect_material(name, P, build):
    """Effect material: build(g) → (color, alpha); emission independent of the scene's light, air haze, transparency."""
    m = bpy.data.materials.new(name); nt = m.node_tree; g = _G(nt)
    for n in list(nt.nodes): nt.nodes.remove(n)
    col, alpha = build(g)
    if not os.environ.get("FAR_FX_NOFOG"): col = _air_fog(g, P, col, P["sky"])         # debugging: FAR_FX_NOFOG=1 — effects without haze
    em = g.node("ShaderNodeEmission", Strength=1.0); g.link(col, em.inputs["Color"])
    tr = g.node("ShaderNodeBsdfTransparent"); mx = g.node("ShaderNodeMixShader"); g.link(alpha, mx.inputs[0])
    g.link(tr.outputs[0], mx.inputs[1]); g.link(em.outputs[0], mx.inputs[2])
    out = g.node("ShaderNodeOutputMaterial"); g.link(mx.outputs[0], out.inputs["Surface"])
    m.diffuse_color = (0.9, 0.92, 0.95, 0.6)
    return m


def _deck_material(P, d, i):
    """A cloud layer — a slab: the thickness is noise patches (coverage), the sun's light is taken at the layer height (a plan table; after
    sunset — from below, the underside glows). Single scattering — an integral over the thickness with a phase function (the silver lining of thin spots near the sun),
    multiple scattering — a two-stream approximation (R, T = 1/(1 + ¾(1 − g)τ)): the thick belly is gray, the top white; with a low sun —
    the shadow of a neighboring patch toward the sun; sky from above, ground from below."""
    import numpy as np
    alt = d["alt_km"] * 1000.0; A = P["light"]["sun_alt"]; hh = np.linspace(0.0, A["h_max"], len(A["rgb"]))
    E_sun = [float(np.interp(alt, hh, [r[c] for r in A["rgb"]])) for c in range(3)]
    s = 1.0 / (d["scale_km"] * 1000.0); t0 = 0.72 - 0.45 * d["cover"]; KD = 4.0 * d["density"]; om = 0.98
    lights = ([(Vector(P["sun"]["dir"]), E_sun)] if max(E_sun) > 0 else []) + \
             ([(Vector(P["moon"]["dir"]), P["moon"]["lamp_rgb"])] if P["moon"]["on"] and max(P["moon"]["lamp_rgb"]) > 0 else [])
    lt = P["light"]; sky_E = [a_ + b_ for a_, b_ in zip(lt["sky_e_rgb"], lt.get("night_sky_e_rgb", [0, 0, 0]))]; gnd = lt["ground_up_rgb"]

    def build(g):
        geo = g.node("ShaderNodeNewGeometry"); Pos, I = geo.outputs["Position"], geo.outputs["Incoming"]
        dens = lambda off: _deck_dens(g, Pos, d, i, off)
        tau0 = g.math("MULTIPLY", dens((0.0, 0.0)), KD)
        iz = g.xyz(I)[2]; vz = g.math("MAXIMUM", g.math("ABSOLUTE", iz), 0.06); up = g.math("GREATER_THAN", iz, 0.0)   # camera above the layer
        alpha = g.math("SUBTRACT", 1.0, g.math("EXPONENT", g.math("DIVIDE", g.math("MULTIPLY", tau0, -1.0), vz)))
        T2 = g.math("DIVIDE", 1.0, g.math("ADD", 1.0, g.math("MULTIPLY", tau0, 0.75 * (1 - 0.85)))); R2 = g.math("SUBTRACT", 1.0, T2)
        acc = None
        def add(v):
            nonlocal acc
            acc = v if acc is None else g.vec("ADD", acc, v)
        for Ld, E in lights:
            above = Ld.z > 0; Lze = max(abs(Ld.z), 0.2)
            same = up if above else g.math("SUBTRACT", 1.0, up)                         # light from the side where the camera is
            Ts = None
            if above and Ld.z < 0.5:                                                       # low sun from above: patch shadows toward the sun
                hx, hy = Vector((Ld.x, Ld.y)).normalized() if Vector((Ld.x, Ld.y)).length > 1e-6 else Vector((0.0, 0.0))
                ts = g.math("ADD", dens((hx * 600.0, hy * 600.0)), dens((hx * 1800.0, hy * 1800.0)))
                Ts = g.math("EXPONENT", g.math("MULTIPLY", ts, -0.6 * KD))
            mu = g.math("MULTIPLY", g.vec("DOT_PRODUCT", I, g.const(list(Ld))), -1.0)
            ph = g.math("ADD", g.math("MULTIPLY", _hg4(g, mu, 0.7), 0.75), g.math("MULTIPLY", _hg4(g, mu, -0.25), 0.25))
            iv = g.math("DIVIDE", 1.0, vz)
            k1 = g.math("ADD", iv, 1.0 / Lze); k2 = g.math("SUBTRACT", iv, 1.0 / Lze)
            k2 = g.math("ADD", k2, g.math("MULTIPLY", g.math("COMPARE", k2, 0.0, 1e-4), 1e-4))
            S_same = g.math("DIVIDE", g.math("SUBTRACT", 1.0, g.math("EXPONENT", g.math("MULTIPLY", g.math("MULTIPLY", tau0, k1), -1.0))), g.math("MULTIPLY", k1, vz))
            S_opp = g.math("MULTIPLY", g.math("EXPONENT", g.math("MULTIPLY", tau0, -1.0 / Lze)),
                           g.math("DIVIDE", g.math("SUBTRACT", 1.0, g.math("EXPONENT", g.math("MULTIPLY", g.math("MULTIPLY", tau0, k2), -1.0))), g.math("MULTIPLY", k2, vz)))
            S1 = g.math("ADD", g.math("MULTIPLY", S_same, same), g.math("MULTIPLY", S_opp, g.math("SUBTRACT", 1.0, same)))
            Ein = max(abs(Ld.z), 0.15)                                                     # illuminance of the face (from the side — through the patch edges)
            ms = g.math("ADD", g.math("MULTIPLY", R2, same),
                        g.math("MULTIPLY", g.math("MAXIMUM", g.math("SUBTRACT", T2, g.math("EXPONENT", g.math("MULTIPLY", tau0, -1.0 / Lze))), 0.0), g.math("SUBTRACT", 1.0, same)))
            v = g.math("ADD", g.math("MULTIPLY", g.math("MULTIPLY", S1, ph), om / (4 * math.pi)), g.math("MULTIPLY", ms, 0.6 * Ein / math.pi))
            if Ts is not None: v = g.math("MULTIPLY", v, Ts)
            add(g.vec("SCALE", g.const(list(E)), s=v))
        Tdir = g.math("EXPONENT", g.math("MULTIPLY", tau0, -2.0))                            # the sky shows through a thin spot anyway
        sky_k = g.math("ADD", g.math("MULTIPLY", R2, up), g.math("MULTIPLY", g.math("MAXIMUM", g.math("SUBTRACT", T2, Tdir), 0.0), g.math("SUBTRACT", 1.0, up)))
        add(g.vec("SCALE", g.const([v / math.pi for v in sky_E]), s=sky_k))
        gnd_k = g.math("ADD", g.math("MULTIPLY", R2, g.math("SUBTRACT", 1.0, up)), g.math("MULTIPLY", g.math("MAXIMUM", g.math("SUBTRACT", T2, Tdir), 0.0), up))
        add(g.vec("SCALE", g.const(list(gnd)), s=gnd_k))
        U = d.get("under")
        if U and os.path.exists(U["path"]):                                  # fire and city from below (far.py under_map): the layer's underside above the fires
            im = bpy.data.images.get(os.path.basename(U["path"])) or bpy.data.images.load(U["path"]); im.colorspace_settings.name = "Linear Rec.709"
            px, py, _ = g.xyz(Pos); R_ = U["R"]
            uv = g.node("ShaderNodeCombineXYZ")
            g.link(g.math("DIVIDE", g.math("ADD", g.math("SUBTRACT", px, U["center"][0]), R_), 2 * R_), uv.inputs[0])
            g.link(g.math("DIVIDE", g.math("ADD", g.math("SUBTRACT", py, U["center"][1]), R_), 2 * R_), uv.inputs[1])
            tx = g.node("ShaderNodeTexImage"); tx.image = im; tx.interpolation = "Linear"; tx.extension = "EXTEND"; g.link(uv.outputs[0], tx.inputs["Vector"])
            add(g.vec("SCALE", tx.outputs["Color"], s=g.math("DIVIDE", gnd_k, math.pi)))
        add(g.vec("SCALE", _flash_E(g, Pos), s=g.math("DIVIDE", gnd_k, math.pi)))      # the frame's flash from below (take, fly-around)
        return g.vec("SCALE", acc, s=g.math("DIVIDE", 1.0, g.math("MAXIMUM", alpha, 1e-3))), alpha
    m = _effect_material(f"far_deck_{i}", P, build); m.surface_render_method = "BLENDED"; return m


def _deck_dens(g, Pos, d, i, off=(0.0, 0.0), detail=6.0):
    """Layer coverage at a point (0…1): noise patches in world x, y, offset off (m) — the same for the layer and its shadow (coarser for the shadow)."""
    s = 1.0 / (d["scale_km"] * 1000.0); t0 = 0.72 - 0.45 * d["cover"]
    nz = g.node("ShaderNodeTexNoise", Scale=1.0, Detail=detail, Roughness=0.55); nz.noise_dimensions = "3D"
    dr = g.node("ShaderNodeCombineXYZ", name="far_drift")              # wind drift per frame (set_frame): the layer's pattern moves, the mesh stays
    p_ = g.vec("SUBTRACT", g.vec("ADD", Pos, g.const([off[0], off[1], 0.0])), dr.outputs[0])
    g.link(g.vec("ADD", g.vec("MULTIPLY", p_, g.const([s, s, 0.0])), g.const([0.0, 0.0, 17.0 + i * 3.1])), nz.inputs["Vector"])
    return g.math("MULTIPLY", g.math("DIVIDE", g.math("SUBTRACT", nz.outputs["Fac"], t0), 0.12), 1.0, clamp=True)


def _flash_E(g, Pos):
    """Light of the frame's flash from below onto a point (set_frame: position and strength × color of the frame's brightest flash): I · rgb · max(dz, 0) / d³."""
    fp = g.node("ShaderNodeCombineXYZ", name="far_flash_pos"); fe = g.node("ShaderNodeCombineXYZ", name="far_flash_E")
    v = g.vec("SUBTRACT", Pos, fp.outputs[0]); d = g.math("MAXIMUM", g.vec("LENGTH", v), 50.0)
    return g.vec("SCALE", fe.outputs[0], s=g.math("DIVIDE", g.math("MAXIMUM", g.xyz(v)[2], 0.0), g.math("POWER", d, 3.0)))


def _shadow_material(name, alpha_of):
    """Shadow caster material: opacity alpha_of(g) — for the shadow only (the object is invisible to the camera): an Eevee transparent shadow."""
    m = bpy.data.materials.new(name); nt = m.node_tree; g = _G(nt)
    for n in list(nt.nodes): nt.nodes.remove(n)
    tr, df, mx = g.node("ShaderNodeBsdfTransparent"), g.node("ShaderNodeBsdfDiffuse"), g.node("ShaderNodeMixShader")
    g.link(alpha_of(g), mx.inputs[0]); g.link(tr.outputs[0], mx.inputs[1]); g.link(df.outputs[0], mx.inputs[2])
    g.link(mx.outputs[0], g.node("ShaderNodeOutputMaterial").inputs["Surface"])
    m.surface_render_method = "DITHERED"; m.use_transparent_shadow = True; return m


def _casts(P):
    """Cloud and smoke shadows come only from a sun above the horizon: from the moon they are not visible in a night frame, and transparent shadows cost
    30 s of 54 per assault frame (1280 × 720)."""
    return P["sun"]["el"] > 0.0 and max(P["sun"]["lamp_rgb"]) > 0


def _shadow_only(ob, on=True):
    for a in ("visible_camera", "visible_diffuse", "visible_glossy", "visible_transmission", "visible_volume_scatter"): setattr(ob, a, False)
    ob.visible_shadow = on and not os.environ.get("FAR_NO_CLOUD_SHADOW"); ob["far_shadow"] = 1; return ob     # debugging: no cloud shadows


def _hg4(g, mu, gg):
    """Henyey–Greenstein phase × 4π (isotropic = 1) from the cosine of the scattering angle mu (a socket)."""
    den = g.math("POWER", g.math("SUBTRACT", 1 + gg * gg, g.math("MULTIPLY", mu, 2 * gg)), 1.5)
    return g.math("DIVIDE", 1 - gg * gg, den)


def _attr(g, name):
    n = g.node("ShaderNodeAttribute"); n.attribute_type = "INSTANCER"; n.attribute_name = name; return n.outputs["Fac"]


def _alt_tex(g, P, z):
    """Sun light at height z above the ground (the plan table light.sun_alt as an N×1 image, linear): clouds and smoke higher up are brighter
    and yellower, and after sunset they still glow."""
    A = P["light"]["sun_alt"]; n = len(A["rgb"]); name = "far_alt_" + P["hash"]
    im = bpy.data.images.get(name)
    if im is None:
        im = bpy.data.images.new(name, n, 1, float_buffer=True); im.colorspace_settings.name = "Linear Rec.709"
        im.pixels.foreach_set([c for rgb in A["rgb"] for c in (*rgb, 1.0)]); im.pack()
    u = g.math("MULTIPLY", g.math("DIVIDE", z, A["h_max"]), 1.0, clamp=True)
    u = g.math("DIVIDE", g.math("ADD", g.math("MULTIPLY", u, n - 1), 0.5), n)             # centers of the end pixels
    uv = g.node("ShaderNodeCombineXYZ"); g.link(u, uv.inputs[0]); uv.inputs[1].default_value = 0.5
    tx = g.node("ShaderNodeTexImage"); tx.image = im; tx.interpolation = "Linear"; tx.extension = "EXTEND"; g.link(uv.outputs[0], tx.inputs["Vector"])
    return tx.outputs["Color"]


def _fx_lights(P, z_top=16000.0):
    """Luminaries for clouds and smoke: the sun — while it shines even at the upper height (after sunset — on the clouds), the moon — when it is in the sky."""
    out = []
    if P["sun"]["el"] + math.degrees(math.sqrt(2 * z_top / (6371000.0 * 7 / 6))) > 0 and max(max(r) for r in P["light"]["sun_alt"]["rgb"]) > 0:
        out.append(dict(dir=Vector(P["sun"]["dir"]), E="alt", attr="psun"))
    if P["moon"]["on"] and max(P["moon"]["lamp_rgb"]) > 0:
        out.append(dict(dir=Vector(P["moon"]["dir"]), E=P["moon"]["lamp_rgb"], attr="pmoon"))
    return out


def _hg(g, I, Ld, gg):
    """Henyey–Greenstein phase from the angle between the view (−I) and the luminary, times 4π."""
    c = g.math("MULTIPLY", g.vec("DOT_PRODUCT", I, g.const(list(Ld))), -1.0)
    den = g.math("POWER", g.math("SUBTRACT", 1 + gg * gg, g.math("MULTIPLY", c, 2 * gg)), 1.5)
    return g.math("DIVIDE", (1 - gg * gg) / (4 * math.pi), den)


OCT, A_OCT, B_OCT, C_OCT = 2, 0.5, 0.5, 0.5                       # octaves of single+ scattering (Wrenninge 2013): weight, thickness, phase


def _puff_core(g, P, omega=0.999, kappa=2.5, glow=None, rho=0.85, n_s=4, edge_band=(0.35, 0.65), dome=0.55, detail=2.0, under=False):
    """A puff — a volume sphere on a sprite. Single scattering — an integral along the view ray through the sphere (n_s samples): toward the
    luminary — the chord to the sphere's edge plus the thickness of neighboring puffs (attributes psun / pmoon / pup / pdown, far.py puff_light); the phase is two
    Henyey–Greenstein lobes (0.7 and −0.25) and an octave (Wrenninge): the thin edge against the sun glows. Multiple scattering (the cloud body:
    without it a cloud is six times darker than a white wall) — diffuse along the normal at the visible point of the sphere: the side facing the luminary is bright, light
    wraps around (wrap 0.4), is attenuated through neighbors; the sky from above and the ground bounce from below likewise; rho is its albedo.
    The sun uses its own color at the puff height; glow is fire from below (smoke, attribute pglow). The edge is almost sharp (the alpha transition only within the
    edge_band strip): a wide semi-transparent edge at 16 samples crumbles into dithering grain, while a sharp edge looks like anime clouds.
    Returns (color without premultiplication, alpha)."""
    geo = g.node("ShaderNodeNewGeometry"); Pos, I = geo.outputs["Position"], geo.outputs["Incoming"]
    oi = g.node("ShaderNodeObjectInfo"); rad = _attr(g, "prad")
    rv = g.vec("SCALE", g.vec("SUBTRACT", Pos, oi.outputs["Location"]), s=g.math("DIVIDE", 1.0, rad))
    r2 = g.vec("DOT_PRODUCT", rv, rv)
    nz = g.node("ShaderNodeTexNoise", Scale=1.7, Detail=detail, Roughness=0.5); nz.noise_dimensions = "4D"   # the edge in large bumps
    tm = g.node("ShaderNodeValue", name="far_time"); tm.outputs[0].default_value = 0.0   # frame time (set_frame): the puff's edge boils
    g.link(rv, nz.inputs["Vector"]); g.link(g.math("ADD", g.math("MULTIPLY", oi.outputs["Random"], 100.0), g.math("MULTIPLY", tm.outputs[0], 0.25)), nz.inputs["W"])
    edge = g.math("ADD", 0.66, g.math("MULTIPLY", g.math("SUBTRACT", nz.outputs["Fac"], 0.5), 0.8))
    km = g.math("MULTIPLY", g.math("MULTIPLY", g.math("DIVIDE", g.math("SUBTRACT", edge, r2), 0.1), 1.0, clamp=True), kappa)
    h = g.math("SQRT", g.math("MAXIMUM", g.math("SUBTRACT", 1.0, r2), 0.0))
    alpha = g.math("SUBTRACT", 1.0, g.math("EXPONENT", g.math("MULTIPLY", g.math("MULTIPLY", km, h), -2.0)))
    n = g.vec("NORMALIZE", g.vec("ADD", rv, g.vec("SCALE", I, s=h)))                     # normal at the visible point of the sphere
    if dome > 0:                                                             # and of the whole cloud's dome: one shadow per cloud, not one per sphere
        pc = g.node("ShaderNodeCombineXYZ")
        for j_, k_ in enumerate(("pcx", "pcy", "pcz")): g.link(_attr(g, k_), pc.inputs[j_])
        nd = g.vec("NORMALIZE", g.vec("SUBTRACT", Pos, pc.outputs[0]))
        n = g.vec("NORMALIZE", g.vec("ADD", g.vec("SCALE", n, s=1.0 - dome), g.vec("SCALE", nd, s=dome)))
    nzz = g.xyz(n)[2]
    zc = g.math("SUBTRACT", g.xyz(oi.outputs["Location"])[2], P["fog"]["layers"][0]["z0"])
    lt = P["light"]; sky_E = [a_ + b_ for a_, b_ in zip(lt["sky_e_rgb"], lt.get("night_sky_e_rgb", [0, 0, 0]))]
    Ls, ms = [], None                                                        # luminaries for single scattering; multiple scattering is added immediately

    def add_ms(v):
        nonlocal ms
        ms = v if ms is None else g.vec("ADD", ms, v)
    for Lt in _fx_lights(P):
        E = _alt_tex(g, P, zc) if Lt["E"] == "alt" else g.const(list(Lt["E"]))
        mu = g.math("MULTIPLY", g.vec("DOT_PRODUCT", I, g.const(list(Lt["dir"]))), -1.0)
        ph = [g.math("ADD", g.math("MULTIPLY", _hg4(g, mu, 0.7 * C_OCT ** j), 0.75), g.math("MULTIPLY", _hg4(g, mu, -0.25 * C_OCT ** j), 0.25))
              for j in range(OCT)]
        tc = _attr(g, Lt["attr"]); Ls.append((Lt["dir"], E, tc, ph))
        wrap = g.math("POWER", g.math("MULTIPLY", g.math("DIVIDE", g.math("ADD", g.vec("DOT_PRODUCT", n, g.const(list(Lt["dir"]))), 0.5), 1.5), 1.0, clamp=True), 1.3)
        add_ms(g.vec("SCALE", E, s=g.math("MULTIPLY", g.math("MULTIPLY", wrap, _t2(g, tc)), rho / math.pi)))
    up_t, dn_t = _t2(g, _attr(g, "pup")), _t2(g, _attr(g, "pdown"))
    up_w, dn_w = g.math("MULTIPLY", g.math("ADD", nzz, 1.0), 0.5), g.math("MULTIPLY", g.math("SUBTRACT", 1.0, nzz), 0.5)
    add_ms(g.vec("SCALE", g.const([v * rho / math.pi for v in sky_E]), s=g.math("MULTIPLY", up_w, up_t)))      # sky onto the face: E(1 + n_z)/2
    add_ms(g.vec("SCALE", g.const([v * rho for v in lt["ground_up_rgb"]]), s=g.math("MULTIPLY", dn_w, dn_t)))   # ground from below: πL(1 − n_z)/2
    if under:                                                                # fire and city from below (pfire — illuminance at the puff's bottom, far.py under_puffs)
        pf = g.node("ShaderNodeAttribute"); pf.attribute_type = "INSTANCER"; pf.attribute_name = "pfire"
        add_ms(g.vec("SCALE", pf.outputs["Vector"], s=g.math("MULTIPLY", g.math("MULTIPLY", dn_w, dn_t), rho / math.pi)))
    add_ms(g.vec("SCALE", _flash_E(g, Pos), s=g.math("MULTIPLY", g.math("MULTIPLY", dn_w, dn_t), rho / math.pi)))   # the frame's flash from below
    if glow is not None:                                                     # fire from below (smoke): falloff along the column, through the lower puffs
        add_ms(g.vec("SCALE", g.const([v * rho / math.pi for v in glow]), s=g.math("MULTIPLY", g.math("MULTIPLY", dn_w, dn_t), _attr(g, "pglow"))))
    ss = None
    for k in range(n_s):
        sk = (k + 0.5) / n_s
        q = g.vec("ADD", rv, g.vec("SCALE", I, s=g.math("MULTIPLY", h, 1.0 - 2.0 * sk)))       # sample on the ray (toward the camera — +I)
        qq = g.vec("DOT_PRODUCT", q, q)
        w = g.math("MULTIPLY", g.math("EXPONENT", g.math("MULTIPLY", g.math("MULTIPLY", km, h), -2.0 * sk)), g.math("MULTIPLY", km, g.math("MULTIPLY", h, 2.0 / n_s)))
        for Ld, E, tc, ph in Ls:
            qL = g.vec("DOT_PRODUCT", q, g.const(list(Ld)))
            ch = g.math("ADD", g.math("MULTIPLY", qL, -1.0), g.math("SQRT", g.math("MAXIMUM", g.math("ADD", g.math("MULTIPLY", qL, qL), g.math("SUBTRACT", 1.0, qq)), 0.0)))
            tau = g.math("ADD", g.math("MULTIPLY", km, ch), tc)
            sc = None
            for j, pj in enumerate(ph):
                t_ = g.math("MULTIPLY", g.math("MULTIPLY", g.math("EXPONENT", g.math("MULTIPLY", tau, -(B_OCT ** j))), (A_OCT * omega) ** j), pj)
                sc = t_ if sc is None else g.math("ADD", sc, t_)
            term = g.vec("SCALE", E, s=g.math("MULTIPLY", sc, w))
            ss = term if ss is None else g.vec("ADD", ss, term)
    mr = g.node("ShaderNodeMapRange"); mr.interpolation_type = "SMOOTHSTEP"; g.link(alpha, mr.inputs["Value"])
    mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = edge_band
    a_out = mr.outputs["Result"]                                             # single scattering is added to the coverage (it does not flare at the edge)
    col = ms if ss is None else g.vec("ADD", ms, g.vec("SCALE", ss, s=g.math("DIVIDE", omega / (4 * math.pi), g.math("MAXIMUM", g.math("MAXIMUM", a_out, alpha), 1e-3))))
    return col, a_out


def _t2(g, tau, gg=0.85):
    """Scattered light through the thickness τ — the two-stream approximation 1/(1 + ¾(1 − g)τ) (inside a cloud the light does not die off exponentially)."""
    return g.math("DIVIDE", 1.0, g.math("ADD", 1.0, g.math("MULTIPLY", tau, 0.75 * (1 - gg))))


def _puff_material(P):
    """A cumulus cloud puff: _puff_core, albedo 0.98, the plan's thickness κ."""
    under = bool((P["clouds"].get("cumulus") or {}).get("under"))
    m = _effect_material("far_puff", P, lambda g: _puff_core(g, P, 0.999, P["clouds"].get("kappa", 2.5), under=under)); m.surface_render_method = "DITHERED"; return m


def attach_clouds(sc, P):
    """The plan's clouds into the far_fx collection: layers are disks up to 120 km at their own height (with Earth curvature), cumulus are puff sprites
    built with geometry nodes, turned toward the active camera. In the lit frame the same Eevee render draws them (the ordering against the mountains resolves itself),
    in the Workbench previs — the effects layer (render_fx)."""
    import numpy as np
    C = P.get("clouds") or {}
    coll = bpy.data.collections.get("far_fx") or bpy.data.collections.new("far_fx")
    if coll.name not in sc.collection.children: sc.collection.children.link(coll)
    cx, cy = P["envelope"]["center"]; z0 = P["fog"]["layers"][0]["z0"]; obs = []
    for i, d in enumerate(C.get("decks", [])):
        R, nphi = 120000.0, 256; rr = np.concatenate([[0.0], 200.0 * (R / 200.0) ** np.linspace(0, 1, 64)])
        ph = np.arange(nphi) * 2 * math.pi / nphi
        X = cx + rr[:, None] * np.cos(ph)[None, :]; Y = cy + rr[:, None] * np.sin(ph)[None, :]
        Z = z0 + d["alt_km"] * 1000.0 - (rr[:, None] ** 2) / (2 * 7432833.0) + 0 * ph[None, :]
        V = np.stack([X, Y, Z], -1).reshape(-1, 3).astype(np.float32); ii = np.arange(len(rr) * nphi).reshape(len(rr), nphi); nx = np.roll(ii, -1, 1)
        F = np.stack([ii[:-1], nx[:-1], nx[1:], ii[1:]], -1).reshape(-1, 4).astype(np.int32)
        me = bpy.data.meshes.new(f"far_deck_{i}"); me.from_pydata(V.tolist(), [], F.tolist()); me.materials.append(_deck_material(P, d, i))
        ob = bpy.data.objects.new(f"far_deck_{i}", me); coll.objects.link(ob); ob.visible_shadow = False; obs.append(ob)
        Lz = max(max(P["sun"]["dir"][2], P["moon"]["dir"][2] if P["moon"]["on"] else -1.0), 0.2); KD = 4.0 * d["density"]
        sm = _shadow_material(f"far_deck_shadow_{i}", lambda g, d=d, i=i: g.math("SUBTRACT", 1.0, g.math("EXPONENT", g.math(
            "MULTIPLY", _deck_dens(g, g.node("ShaderNodeNewGeometry").outputs["Position"], d, i, detail=2.0), -KD / Lz))))   # direct light through the thickness
        Rs = P["envelope"]["radius"] + 15000.0; k_ = int(np.searchsorted(rr, Rs)) + 1   # the shadow is a ~15 km disk around the envelope (farther away it is lost in the haze; a
        me2 = bpy.data.meshes.new(f"far_deck_shadow_{i}")                              # 120 km disk costs 8.7 s of 11 per 1280 × 720 frame)
        me2.from_pydata(V.reshape(len(rr), nphi, 3)[:k_].reshape(-1, 3).tolist(), [], [f for f in F.tolist() if max(f) < k_ * nphi]); me2.materials.append(sm)
        o2 = bpy.data.objects.new(f"far_deck_shadow_{i}", me2); coll.objects.link(o2); obs.append(_shadow_only(o2, _casts(P)))   # cloud shadow on the ground
    cu = C.get("cumulus")
    if cu and os.path.exists(cu["npz"]):
        D = np.load(cu["npz"])
        if len(D["pos"]):
            Lp = np.load(cu["light"]) if cu.get("light") and os.path.exists(cu["light"]) else {}
            at = {"prad": D["rad"], "pbase": D["base"], "ptop": D["top"], **{k: Lp[k] for k in ("psun", "pmoon", "pup", "pdown", "pcx", "pcy", "pcz") if k in Lp}}
            if cu.get("under") and os.path.exists(cu["under"]): at["pfire"] = np.load(cu["under"])["pfire"]
            obs.append(_billboards("far_cumulus", coll, D["pos"], at, _puff_material(P)))
            obs.append(_shadow_discs("far_cumulus_shadow", coll, D["pos"], D["rad"], 0.85, _casts(P)))
    return obs


def _shadow_discs(name, coll, pos, rad, opacity, on=True):
    """Puff shadows on the ground: horizontal soft disks at the puff centers (the shadow of a sphere from a low sun is the same elongated
    ellipse), for shadow only; overlapping ones are darker (transparent shadows multiply)."""
    import numpy as np
    me = bpy.data.meshes.new(name); me.vertices.add(len(pos)); me.vertices.foreach_set("co", np.asarray(pos, np.float32).ravel())
    a = me.attributes.new("prad", "FLOAT", "POINT"); a.data.foreach_set("value", np.asarray(rad, np.float32)); me.update()
    ob = bpy.data.objects.new(name, me); coll.objects.link(ob)
    def alpha(g):
        rv = g.vec("SCALE", g.vec("SUBTRACT", g.node("ShaderNodeNewGeometry").outputs["Position"], g.node("ShaderNodeObjectInfo").outputs["Location"]),
                   s=g.math("DIVIDE", 1.0, _attr(g, "prad")))
        return g.math("MULTIPLY", g.math("MULTIPLY", g.math("SUBTRACT", 1.0, g.vec("DOT_PRODUCT", rv, rv)), 8.0, clamp=True), opacity)   # narrow edge: a transparent shadow is stochastic (grain), the penumbra comes from the sun disk
    mat = _shadow_material(name, alpha)
    ng = bpy.data.node_groups.new(name + "_gn", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"); ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N = ng.nodes; Lk = ng.links.new; gi, go = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
    ci = N.new("GeometryNodeMeshCircle"); ci.fill_type = "TRIANGLE_FAN"; ci.inputs["Vertices"].default_value = 16; ci.inputs["Radius"].default_value = 1.0
    sm = N.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = mat; Lk(ci.outputs["Mesh"], sm.inputs["Geometry"])
    ra = N.new("GeometryNodeInputNamedAttribute"); ra.data_type = "FLOAT"; ra.inputs["Name"].default_value = "prad"
    ip = N.new("GeometryNodeInstanceOnPoints"); Lk(gi.outputs[0], ip.inputs["Points"]); Lk(sm.outputs[0], ip.inputs["Instance"])
    Lk(ra.outputs["Attribute"], ip.inputs["Scale"]); Lk(ip.outputs["Instances"], go.inputs[0])
    ob.modifiers.new(name, "NODES").node_group = ng
    return _shadow_only(ob, on)


def _billboards(name, coll, pos, attrs, mat, upright=False):
    """Sprites via geometry nodes at points pos: a 2×2 m plane × prad, facing the active camera (upright — rotating only about the vertical:
    flames stand upright); the point attributes go to the shader (INSTANCER)."""
    import numpy as np
    me = bpy.data.meshes.new(name); me.vertices.add(len(pos)); me.vertices.foreach_set("co", np.asarray(pos, np.float32).ravel())
    for k, arr in attrs.items():
        arr = np.asarray(arr, np.float32)
        if arr.ndim == 2: a = me.attributes.new(k, "FLOAT_VECTOR", "POINT"); a.data.foreach_set("vector", arr.ravel())
        else: a = me.attributes.new(k, "FLOAT", "POINT"); a.data.foreach_set("value", arr)
    me.update(); ob = bpy.data.objects.new(name, me); coll.objects.link(ob); ob.visible_shadow = False
    ng = bpy.data.node_groups.new(name + "_gn", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N = ng.nodes; Lk = ng.links.new; gi, go = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
    grid = N.new("GeometryNodeMeshGrid"); grid.inputs["Size X"].default_value = grid.inputs["Size Y"].default_value = 2.0
    grid.inputs["Vertices X"].default_value = grid.inputs["Vertices Y"].default_value = 2
    geo = grid.outputs["Mesh"]
    if upright:                                                              # card in the XZ plane, normal along Y
        tr = N.new("GeometryNodeTransform"); tr.inputs["Rotation"].default_value = (math.radians(90), 0, 0); Lk(geo, tr.inputs["Geometry"]); geo = tr.outputs["Geometry"]
    sm = N.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = mat; Lk(geo, sm.inputs["Geometry"])
    cam = N.new("GeometryNodeInputActiveCamera"); oi = N.new("GeometryNodeObjectInfo"); oi.transform_space = "ORIGINAL"; Lk(cam.outputs[0], oi.inputs["Object"])
    pos_ = N.new("GeometryNodeInputPosition"); sub = N.new("ShaderNodeVectorMath"); sub.operation = "SUBTRACT"
    Lk(oi.outputs["Location"], sub.inputs[0]); Lk(pos_.outputs[0], sub.inputs[1]); vec = sub.outputs[0]
    if upright:
        flat = N.new("ShaderNodeVectorMath"); flat.operation = "MULTIPLY"; flat.inputs[1].default_value = (1.0, 1.0, 0.0); Lk(vec, flat.inputs[0]); vec = flat.outputs[0]
    al = N.new("FunctionNodeAlignRotationToVector"); al.axis = "Y" if upright else "Z"; Lk(vec, al.inputs["Vector"])
    ra = N.new("GeometryNodeInputNamedAttribute"); ra.data_type = "FLOAT"; ra.inputs["Name"].default_value = "prad"
    ip = N.new("GeometryNodeInstanceOnPoints"); Lk(gi.outputs[0], ip.inputs["Points"]); Lk(sm.outputs[0], ip.inputs["Instance"])
    Lk(al.outputs["Rotation"], ip.inputs["Rotation"]); Lk(ra.outputs["Attribute"], ip.inputs["Scale"]); Lk(ip.outputs["Instances"], go.inputs[0])
    md = ob.modifiers.new(name, "NODES"); md.node_group = ng
    return ob


def _smoke_material(P, albedo, fire_rgb):
    """A smoke puff: a volume sphere (_puff_core) with the column's albedo (dark smoke absorbs), the plan's smoke thickness κ; from below — fire light
    (attribute pglow — falloff along the column) through the lower puffs; the tint is brown. The thickness's reflection is 0.6 of the albedo (for soot R∞ is a few percent:
    with ×1.6 the columns in the sun came out as beige cotton wool)."""
    def build(g):
        col, alpha = _puff_core(g, P, albedo, P["fires"].get("kappa", 1.5), glow=[60.0 * P["fires"].get("light_k", 1.0) * c for c in fire_rgb],
                                rho=albedo * 0.6, edge_band=(0.15, 0.75), dome=0.3, detail=5.0, under=bool(P["fires"].get("under")))
        return g.vec("MULTIPLY", col, g.const([1.0, 0.93, 0.86])), alpha
    m = _effect_material("far_smoke", P, build); m.surface_render_method = "DITHERED"; return m


def _flame_material(P):
    """A tongue of flame on an upright card: a drop with a ragged edge (noise over time), toward the edge from whitish yellow to red; self-luminous."""
    def build(g):
        geo = g.node("ShaderNodeNewGeometry"); oi = g.node("ShaderNodeObjectInfo")
        ra = g.node("ShaderNodeAttribute"); ra.attribute_type = "INSTANCER"; ra.attribute_name = "prad"
        rel = g.vec("SCALE", g.vec("SUBTRACT", geo.outputs["Position"], oi.outputs["Location"]), s=g.math("DIVIDE", 1.0, ra.outputs["Fac"]))
        x, y, zz = g.xyz(rel); h = g.math("SQRT", g.math("ADD", g.math("MULTIPLY", x, x), g.math("MULTIPLY", y, y)))
        t = g.math("MULTIPLY", g.math("ADD", zz, 1.0), 0.5, clamp=True)                 # 0 — bottom, 1 — top of the card
        nz = g.node("ShaderNodeTexNoise", Scale=3.0, Detail=4.0); nz.noise_dimensions = "4D"; g.link(rel, nz.inputs["Vector"])
        tm = g.node("ShaderNodeValue"); tm.outputs[0].default_value = 0.0
        fc = tm.outputs[0].driver_add("default_value"); fc.driver.expression = "frame * 0.15"
        g.link(g.math("ADD", tm.outputs[0], g.math("MULTIPLY", oi.outputs["Random"], 50.0)), nz.inputs["W"])
        width = g.math("MULTIPLY", g.math("SUBTRACT", 1.0, g.math("POWER", t, 1.4)), 0.62)
        d = g.math("ADD", h, g.math("MULTIPLY", g.math("SUBTRACT", nz.outputs["Fac"], 0.5), 0.45))
        alpha = g.math("MULTIPLY", g.math("DIVIDE", g.math("SUBTRACT", width, d), 0.12), 1.0, clamp=True)
        core = g.math("MULTIPLY", g.math("SUBTRACT", 1.0, g.math("DIVIDE", d, g.math("ADD", width, 1e-3))), 1.0, clamp=True)
        rm = g.node("ShaderNodeMix"); rm.data_type = "RGBA"; g.link(core, rm.inputs[0])
        rm.inputs[6].default_value = (1.0, 0.18, 0.03, 1); rm.inputs[7].default_value = (1.0, 0.78, 0.38, 1)
        return g.vec("SCALE", rm.outputs[2], s=g.math("ADD", 120.0, g.math("MULTIPLY", core, 480.0))), alpha
    m = _effect_material("far_flame", P, build); m.surface_render_method = "DITHERED"; return m


def attach_fires(sc, P):
    """The plan's burning sites and frame flashes (attach_flashes) into the far_fx collection: smoke columns and flames as sprites; fires closer than 1.5 km to the envelope get a point
    lamp (light on the surroundings); distant ones light only through the glow in the haze (core/light/far.py fog_terms)."""
    import numpy as np
    Fz = P.get("fires") or {}
    if not Fz.get("list") or not Fz.get("npz"): return attach_flashes(sc, P)
    D = np.load(Fz["npz"]); coll = bpy.data.collections.get("far_fx") or bpy.data.collections.new("far_fx")
    if coll.name not in sc.collection.children: sc.collection.children.link(coll)
    fire_rgb = (1.0, 0.42, 0.12); alb = float(np.mean([f["albedo"] for f in Fz["list"]]))
    Lp = np.load(Fz["light"]) if Fz.get("light") and os.path.exists(Fz["light"]) else {}
    obs = [_billboards("far_smoke", coll, D["pos"], {"prad": D["rad"], "pbase": D["base"], "ptop": D["top"], "pglow": D["glow"],
                                                     **{k: Lp[k] for k in ("psun", "pmoon", "pup", "pdown", "pcx", "pcy", "pcz") if k in Lp},
                                                     **({"pfire": np.load(Fz["under"])["pfire"]} if Fz.get("under") and os.path.exists(Fz["under"]) else {})},
                       _smoke_material(P, alb, fire_rgb)),
           _billboards("far_flame", coll, D["fpos"], {"prad": D["frad"]}, _flame_material(P), upright=True),
           _shadow_discs("far_smoke_shadow", coll, D["pos"], D["rad"], 0.7, _casts(P))]
    E = np.array(P["envelope"]["points"])[:, :2]
    for i, f in enumerate(Fz["list"]):
        x, y, z = f["at"]
        if np.min(np.hypot(E[:, 0] - x, E[:, 1] - y)) > 1500: continue
        ld = bpy.data.lights.new(f"far_fire_{i}", "POINT"); ld.energy = 2.0e6 * f["power"] * Fz.get("light_k", 1.0); ld.color = fire_rgb; ld.shadow_soft_size = f["size"] * 0.5
        ld.use_shadow = False                                            # shadows of fires 1–9 km away are not visible in the frame, and 11 such lamps cost 17 s of 54
        lo = bpy.data.objects.new(f"far_fire_{i}", ld); coll.objects.link(lo); lo.location = (x, y, z + f["size"]); obs.append(lo)
    return obs + attach_flashes(sc, P)


def attach_flashes(sc, P):
    """Frame flashes (far.py flash_sources): an explosion is a fireball (a glowing soft sphere sprite), lightning is a channel drawn as a polyline ribbon from the cloud to the ground;
    those near the envelope (3 km) get a point lamp (strength as for a fire of the same glow); the glow in the haze and the light on the underside of clouds
    come from the plan's sources (frame post-processing, light-from-below maps)."""
    import numpy as np
    F = P.get("flashes") or []
    if not F: return []
    coll = bpy.data.collections.get("far_fx") or bpy.data.collections.new("far_fx")
    if coll.name not in sc.collection.children: sc.collection.children.link(coll)
    E = np.array(P["envelope"]["points"])[:, :2]; obs = []
    for i, f in enumerate(F):
        x, y, z = f["at"]; rgb = f["rgb"]; env = f["env"]
        if f["kind"] == "explosion":
            def build(g, rgb=rgb, env=env):
                geo = g.node("ShaderNodeNewGeometry"); oi = g.node("ShaderNodeObjectInfo")
                rv = g.vec("SCALE", g.vec("SUBTRACT", geo.outputs["Position"], oi.outputs["Location"]), s=g.math("DIVIDE", 1.0, _attr(g, "prad")))
                r2 = g.vec("DOT_PRODUCT", rv, rv); core = g.math("POWER", g.math("SUBTRACT", 1.0, r2, clamp=True), 1.5)
                return g.vec("SCALE", g.const(list(rgb)), s=g.math("MULTIPLY", core, 4000.0 * env)), g.math("MULTIPLY", g.math("SUBTRACT", 1.0, r2), 3.0, clamp=True)
            m = _effect_material(f"far_flash_{i}", P, build); m.surface_render_method = "DITHERED"
            obs.append(_billboards(f"far_flash_{i}", coll, [(x, y, z)], {"prad": [f["size"] * (1.6 - 0.6 * env)]}, m))
        else:                                                                # lightning: a random polyline from the cloud down, as a ribbon toward the envelope center
            rng = np.random.default_rng(f["seed"]); n = 24; zz = np.linspace(z, f["ground"], n)
            xy = np.cumsum(rng.normal(0, (z - f["ground"]) / n * 0.45, (n, 2)), 0); xy -= xy[0]; P_ = np.c_[x + xy[:, 0], y + xy[:, 1], zz]
            c = np.array([*P["envelope"]["center"], P_[:, 2].mean()]); side = np.cross(P_[-1] - P_[0], c - P_.mean(0)); side = side / max(np.linalg.norm(side), 1e-6) * 6.0
            V = np.concatenate([P_ - side, P_ + side]); Fc = [(k, k + 1, n + k + 1, n + k) for k in range(n - 1)]
            me = bpy.data.meshes.new(f"far_bolt_{i}"); me.from_pydata(V.tolist(), [], Fc)
            m = _effect_material(f"far_bolt_{i}", P, lambda g, rgb=rgb, env=env: (g.const([c_ * 1.0e6 * env for c_ in rgb]), g.math("ADD", 0.0, 1.0)))   # the channel is the brightest thing: visible through the storm
            me.materials.append(m); ob = bpy.data.objects.new(f"far_bolt_{i}", me); coll.objects.link(ob); ob.visible_shadow = False; obs.append(ob)
        if np.min(np.hypot(E[:, 0] - x, E[:, 1] - y)) < 3000:                # lamp: the flash's light on the surroundings
            ld = bpy.data.lights.new(f"far_flash_l{i}", "POINT"); ld.energy = 2.0 * f["I"]; ld.color = rgb; ld.shadow_soft_size = f["size"]
            lo = bpy.data.objects.new(f"far_flash_l{i}", ld); coll.objects.link(lo); lo.location = (x, y, z if f["kind"] == "explosion" else (z + f["ground"]) / 2); obs.append(lo)
    return obs


def _city_material(P):
    """Walls and roofs of the far city: the kit color (the instance's kit attribute), windows — a grid in the instance's meters on the walls, some glow warm at night
    (the kit's windows fraction, brightness by the plan's night); scene lighting as for the terrain (Principled)."""
    kits = P["city"]["kits"] or [dict(wall=(0.3, 0.28, 0.27), roof=(0.22, 0.21, 0.21), win=(4.0, 3.2), windows=0.35)]
    m = bpy.data.materials.new("far_city"); nt = m.node_tree; g = _G(nt); bs = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    kt = g.node("ShaderNodeAttribute"); kt.attribute_type = "INSTANCER"; kt.attribute_name = "kit"
    sv = g.node("ShaderNodeAttribute"); sv.attribute_type = "INSTANCER"; sv.attribute_name = "sclv"
    tc = g.node("ShaderNodeTexCoord"); geo = g.node("ShaderNodeNewGeometry")
    loc = g.vec("MULTIPLY", tc.outputs["Object"], sv.outputs["Vector"])                        # instance meters
    lx, ly, lz = g.xyz(loc); nz_ = g.math("ABSOLUTE", g.xyz(geo.outputs["Normal"])[2])
    wall, roof, frac = None, None, None
    for i, k in enumerate(kits):                                                                # the set — by kit number
        sel = g.math("COMPARE", kt.outputs["Fac"], float(i), 0.5)
        wall = g.vec("SCALE", g.const(list(k["wall"])), s=sel) if wall is None else g.vec("ADD", wall, g.vec("SCALE", g.const(list(k["wall"])), s=sel))
        roof = g.vec("SCALE", g.const(list(k["roof"])), s=sel) if roof is None else g.vec("ADD", roof, g.vec("SCALE", g.const(list(k["roof"])), s=sel))
        frac = g.math("MULTIPLY", sel, k.get("windows", 0.35)) if frac is None else g.math("ADD", frac, g.math("MULTIPLY", sel, k.get("windows", 0.35)))
    up = g.math("GREATER_THAN", nz_, 0.7)
    mx = g.node("ShaderNodeMix"); mx.data_type = "RGBA"; g.link(up, mx.inputs[0]); g.link(wall, mx.inputs[6]); g.link(roof, mx.inputs[7])
    g.link(mx.outputs[2], bs.inputs["Base Color"]); bs.inputs["Roughness"].default_value = 0.9
    wh, ww = kits[0]["win"]
    row = g.math("DIVIDE", lz, wh); col_ = g.math("DIVIDE", g.math("ADD", lx, ly), ww)
    cell = g.node("ShaderNodeCombineXYZ"); g.link(g.math("FLOOR", row), cell.inputs[0]); g.link(g.math("FLOOR", col_), cell.inputs[1])
    oi = g.node("ShaderNodeObjectInfo"); g.link(g.math("MULTIPLY", oi.outputs["Random"], 997.0), cell.inputs[2])
    wn = g.node("ShaderNodeTexWhiteNoise"); wn.noise_dimensions = "3D"; g.link(cell.outputs[0], wn.inputs["Vector"])
    on = g.math("LESS_THAN", wn.outputs["Value"], frac)
    fr_r, fr_c = g.math("FRACT", row), g.math("FRACT", col_)
    inwin = g.math("MULTIPLY", g.math("MULTIPLY", g.math("GREATER_THAN", fr_r, 0.3), g.math("LESS_THAN", fr_r, 0.75)),
                   g.math("MULTIPLY", g.math("GREATER_THAN", fr_c, 0.2), g.math("LESS_THAN", fr_c, 0.7)))
    lit = g.math("MULTIPLY", g.math("MULTIPLY", inwin, on), g.math("LESS_THAN", nz_, 0.3))
    k_night = P["city"].get("win_k", max(P["sky"]["w_night"], 0.15 * (1 - smooth_el(P["sun"]["el"]))))   # plan: far.py win_k
    g.link(g.vec("SCALE", g.const([1.0, 0.72, 0.42]), s=g.math("MULTIPLY", lit, 3.5 * k_night)), bs.inputs["Emission Color"])
    bs.inputs["Emission Strength"].default_value = 1.0
    m.diffuse_color = (*kits[0]["wall"], 1.0)
    return m


def smooth_el(el):
    """0 at night → 1 by day (windows go out toward noon)."""
    t = max(0.0, min(1.0, (el + 6.0) / 16.0)); return t * t * (3 - 2 * t)


def attach_city(sc, P):
    """The plan's far city: prototypes (a block, a three-tier tower, a spire — unit size, base at 0) in an excluded collection, instances via geometry nodes at points
    (index, rotation, size). Both color and maps see it (class tower). Returns (object, prototypes)."""
    import numpy as np
    C = P.get("city") or {}
    if not C.get("npz") or not os.path.exists(C["npz"]): return None, []
    D = np.load(C["npz"])
    if not len(D["pos"]): return None, []
    mat = _city_material(P)
    pc = bpy.data.collections.new("far_city_protos"); sc.collection.children.link(pc)
    def box(x0, y0, z0, x1, y1, z1):
        v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        return v, [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    protos = []
    for name, parts in (("far_city_0", [box(-0.5, -0.5, 0, 0.5, 0.5, 1)]),
                        ("far_city_1", [box(-0.5, -0.5, 0, 0.5, 0.5, 0.55), box(-0.38, -0.38, 0.55, 0.38, 0.38, 0.85), box(-0.24, -0.24, 0.85, 0.24, 0.24, 1.0)]),
                        ("far_city_2", None)):
        if parts is None:                                                        # spire: an octagon tapering to the tip
            n = 8; V = [(0.5 * math.cos(2 * math.pi * i / n), 0.5 * math.sin(2 * math.pi * i / n), 0.0) for i in range(n)]
            V += [(0.3 * math.cos(2 * math.pi * i / n), 0.3 * math.sin(2 * math.pi * i / n), 0.7) for i in range(n)] + [(0, 0, 1.0)]
            Fc = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)] + [(n + i, n + (i + 1) % n, 2 * n) for i in range(n)]
        else:
            V, Fc = [], []
            for v_, f_ in parts: Fc += [tuple(i + len(V) for i in f) for f in f_]; V += v_
        me = bpy.data.meshes.new(name); me.from_pydata(V, [], Fc); me.materials.append(mat)
        ob = bpy.data.objects.new(name, me); pc.objects.link(ob); protos.append(ob)
    sc.view_layers[0].layer_collection.children[pc.name].exclude = True
    me = bpy.data.meshes.new("far_city"); me.vertices.add(len(D["pos"])); me.vertices.foreach_set("co", D["pos"].ravel())
    for k, dt, arr in (("pidx", "INT", D["idx"]), ("yaw", "FLOAT", D["yaw"]), ("sclv", "FLOAT_VECTOR", D["scl"]), ("kit", "FLOAT", D["kit"].astype(np.float32))):
        a = me.attributes.new(k, dt, "POINT"); a.data.foreach_set("vector" if dt == "FLOAT_VECTOR" else "value", np.asarray(arr).ravel())
    me.update(); city = bpy.data.objects.new("far_city", me); sc.collection.objects.link(city); city["sketch_kind"] = "tower"
    ng = bpy.data.node_groups.new("far_city_gn", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"); ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N = ng.nodes; Lk = ng.links.new; gi, go = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
    ci = N.new("GeometryNodeCollectionInfo"); ci.transform_space = "ORIGINAL"; ci.inputs["Collection"].default_value = pc
    ci.inputs["Separate Children"].default_value = True; ci.inputs["Reset Children"].default_value = True
    ip = N.new("GeometryNodeInstanceOnPoints"); Lk(gi.outputs[0], ip.inputs["Points"]); Lk(ci.outputs["Instances"], ip.inputs["Instance"])
    ip.inputs["Pick Instance"].default_value = True
    na = lambda nm, dt: (lambda n_: (setattr(n_, "data_type", dt), n_.inputs["Name"].__setattr__("default_value", nm), n_)[2])(N.new("GeometryNodeInputNamedAttribute"))
    Lk(na("pidx", "INT").outputs["Attribute"], ip.inputs["Instance Index"])
    cx = N.new("ShaderNodeCombineXYZ"); Lk(na("yaw", "FLOAT").outputs["Attribute"], cx.inputs[2])
    er = N.new("FunctionNodeEulerToRotation"); Lk(cx.outputs[0], er.inputs["Euler"]); Lk(er.outputs["Rotation"], ip.inputs["Rotation"])
    Lk(na("sclv", "FLOAT_VECTOR").outputs["Attribute"], ip.inputs["Scale"]); Lk(ip.outputs["Instances"], go.inputs[0])
    md = city.modifiers.new("far_city", "NODES"); md.node_group = ng
    return city, protos


WIND_MS = 10.0                                                       # cloud drift, m/s (at cloud height; the direction is the plan's wind)


def set_frame(sc, P, frame):
    """Frame `frame` (zero-based, rate P["fps"]; take, fly-around): the wind drifts the clouds (the layer pattern by shifting the noise, cumulus by shifting
    the object with its shadow), the puffs' edges boil, the frame's brightest flash (far.flashes_at: the description's flashes and the take's events) lights
    the clouds from below; the glow and the fire flicker of the frame are applied in post-processing (far.py post by frame from far.json)."""
    from core.light import far
    t = frame / float(P.get("fps", 12.0)); w = P.get("fires", {}).get("wind", [0.35, 0.12]); n = math.hypot(*w) or 1.0
    dx, dy = w[0] / n * WIND_MS * t, w[1] / n * WIND_MS * t
    F = far.flashes_at(P, frame); f = max(F, key=lambda f_: f_["I"]) if F else None
    for m in bpy.data.materials:
        if not m.node_tree: continue
        for nd in m.node_tree.nodes:
            if nd.name.startswith("far_drift"): nd.inputs[0].default_value, nd.inputs[1].default_value = dx, dy
            elif nd.name.startswith("far_time"): nd.outputs[0].default_value = t
            elif nd.name.startswith("far_flash_pos"): nd.inputs[0].default_value, nd.inputs[1].default_value, nd.inputs[2].default_value = f["at"] if f else (0, 0, -1e6)
            elif nd.name.startswith("far_flash_E"):
                for j in range(3): nd.inputs[j].default_value = f["I"] * f["rgb"][j] if f else 0.0
    for name in ("far_cumulus", "far_cumulus_shadow"):
        ob = bpy.data.objects.get(name)
        if ob is not None: ob.location = (dx, dy, 0.0)
    img = P["sky"].get("image")                                              # image sky: cloud drift is a rotation of the image (drift °/s)
    if img and img.get("drift") and sc.world and sc.world.node_tree.nodes.get("far_sky_yaw"):
        sc.world.node_tree.nodes["far_sky_yaw"].inputs["Rotation"].default_value = (0.0, 0.0, -math.radians(img["yaw"] + img["drift"] * t))


def render_fx(sc, path):
    """Effects layer for the Workbench previs: the clouds (the far_fx collection) in Eevee, everything else a holdout, transparent background → RGBA EXR."""
    fx = bpy.data.collections.get("far_fx")
    if fx is None or not fx.objects: return False
    mine = set(fx.objects); other = [o for o in sc.objects if o not in mine and o.type in ("MESH", "CURVE", "CURVES", "POINTCLOUD", "VOLUME")]
    old = [(o, o.is_holdout) for o in other]; hid = [(o, o.hide_render) for o in mine]
    for o in other: o.is_holdout = True
    for o in mine: o.hide_render = False
    eng, ft, fmt, med, smp = sc.render.engine, sc.render.film_transparent, sc.render.image_settings.file_format, sc.render.image_settings.media_type, sc.eevee.taa_render_samples
    sc.render.engine = "BLENDER_EEVEE"; sc.render.film_transparent = True; sc.eevee.taa_render_samples = 24
    sc.render.image_settings.media_type = "IMAGE"; sc.render.image_settings.file_format = "OPEN_EXR"; sc.render.image_settings.color_depth = "32"
    sc.render.filepath = path; bpy.ops.render.render(write_still=True)
    for o, h in old: o.is_holdout = h
    for o, h in hid: o.hide_render = h
    sc.render.engine, sc.render.film_transparent, sc.eevee.taa_render_samples = eng, ft, smp
    sc.render.image_settings.media_type = med; sc.render.image_settings.file_format = fmt
    return True


# ---------------------------------------------------------------- a scene with its own light under the node (the battle scene, components/battle/bl_battle.py)
def scene_light(sc, P, k_ev=1.0):
    """Scene light under the node: the scene's lamps (fires, square glow, gate light, blasts, muzzles) stay, with strength × the plan's light_k (at night —
    as placed, by day — weaker than the sky, like the node's fires) × k_ev = 2^(scene EV − node EV): the author placed the lamps for their own exposure, while the node
    exposes the sky, haze and moon darker at night — what the lamps light stays as intended, the night around it stays night. The scene's sun
    lamps (its moon) are removed — the node's luminaries and sky take over. The scene's smoke cubes (fx_smoke_*) are hidden: the smoke comes from the node's fires at the same
    places (the scene's far.yaml). Returns the number of scene lamps."""
    k = float(P["fires"].get("light_k", 1.0)) * k_ev; n = 0
    for o in [o for o in sc.objects if o.type == "LIGHT" and not o.name.startswith("far_")]:
        if o.data.type == "SUN": bpy.data.objects.remove(o); continue
        _scale_energy(o.data, k); n += 1
    for o in sc.objects:
        if o.name.startswith("fx_smoke_"): o.hide_render = True
    return n


def _scale_energy(ld, k):
    """Lamp strength × k — for a constant one, a flickering one (driver) and a flash (keys)."""
    if abs(k - 1.0) < 1e-9: return
    ad = ld.animation_data
    for d in (ad.drivers if ad else []):
        if d.data_path == "energy": d.driver.expression = f"({d.driver.expression})*{k:.6g}"; return
    if ad and ad.action:
        from bpy_extras import anim_utils
        cb = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
        for fc in (cb.fcurves if cb else []):
            if fc.data_path == "energy":
                for kp in fc.keyframe_points: kp.co[1] *= k; kp.handle_left[1] *= k; kp.handle_right[1] *= k
                return
    ld.energy *= k


def scene_emission(k):
    """Scene emission (windows, fire, beams, shield, gate gap, blast cores, tracers) — adapted to the node's exposure: the scene author set the emission
    strengths for their own exposure (the battle scene — −0.3), the node exposes by its own light; ×k = 2^(scene EV − node EV) keeps the look
    of glowing things as intended. Lamps are physics, their strength is unchanged (scene_light). The node's materials (far_*) are untouched."""
    if abs(k - 1.0) < 1e-6: return 0
    n = 0
    for m in bpy.data.materials:
        if m.name.startswith("far_") or not m.node_tree or m.get("far_emit"): continue
        nt = m.node_tree; hit = False
        for nd in list(nt.nodes):
            s = nd.inputs.get("Strength") if nd.type == "EMISSION" else nd.inputs.get("Emission Strength") if nd.type == "BSDF_PRINCIPLED" else None
            if s is None: continue
            if s.links:                                                 # strength driven by nodes (windows, beam falloff) — by a multiplication inserted into the link
                mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"; mul.inputs[1].default_value = k
                nt.links.new(s.links[0].from_socket, mul.inputs[0]); nt.links.new(mul.outputs[0], s); hit = True
            elif s.default_value > 0: s.default_value *= k; hit = True
        if hit: m["far_emit"] = 1; n += 1
    return n


def _fx_mats(o):
    """Effect materials: slots and the Set Material nodes of its node modifiers (GN tracers)."""
    ms = [s.material for s in o.material_slots if s.material]
    def walk(ng, seen):
        if ng is None or ng.name in seen: return
        seen.add(ng.name)
        for n in ng.nodes:
            if n.type == "SET_MATERIAL" and n.inputs["Material"].default_value: ms.append(n.inputs["Material"].default_value)
            elif n.type == "GROUP": walk(n.node_tree, seen)
    for md in o.modifiers:
        if md.type == "NODES": walk(md.node_group, set())
    return list(dict.fromkeys(ms))


def scene_fx(sc):
    """Scene effects as a separate layer over the haze (the sketch_fx tag): semi-transparent ones (BLENDED — beams, shield, blast cores and puffs,
    shells, muzzles, anti-aircraft guns) and node-based tracers write no depth, so the post-processing haze by depth cannot be computed behind them. Fire and the gate
    gap (DITHERED — they write depth) stay in the frame: their haze uses their own depth."""
    out = []
    for o in sc.objects:
        if not o.get("sketch_fx") or o.type != "MESH" or o.hide_render or o.name.startswith("fx_smoke_"): continue
        if any(m.surface_render_method == "BLENDED" for m in _fx_mats(o)) or any(md.type == "NODES" for md in o.modifiers): out.append(o)
    return out


def fx_fog(objs, P):
    """Haze for the layer's effects is done in the shader (post-processing does not see them): the surface is mixed with a transparent one by the transmittance to the camera
    T = exp(−Σ a·e^(−(z̄ − z0)/H)·d) over the plan's haze layers; d is the distance to the camera, z̄ the mid-ray height."""
    L = P.get("fog", {}).get("layers", []); done = set()
    for o in objs:
        for m in _fx_mats(o):
            if m.name in done or m.get("far_fog") or not m.node_tree or not L: continue
            done.add(m.name); nt = m.node_tree; g = _G(nt)
            out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
            if out is None or not out.inputs["Surface"].links: continue
            src = out.inputs["Surface"].links[0].from_socket
            d = g.node("ShaderNodeCameraData").outputs["View Distance"]; geo = g.node("ShaderNodeNewGeometry")
            zb = g.math("ADD", g.xyz(geo.outputs["Position"])[2], g.math("MULTIPLY", g.math("MULTIPLY", g.xyz(geo.outputs["Incoming"])[2], d), 0.5))
            tau = None
            for Ly in L:
                r = g.math("MULTIPLY", g.math("EXPONENT", g.math("DIVIDE", g.math("SUBTRACT", float(Ly["z0"]), zb), float(Ly["H"]))), float(Ly["a"]))
                t_ = g.math("MULTIPLY", r, d); tau = t_ if tau is None else g.math("ADD", tau, t_)
            mx = g.node("ShaderNodeMixShader"); g.link(g.math("EXPONENT", g.math("MULTIPLY", tau, -1.0)), mx.inputs[0])
            g.link(g.node("ShaderNodeBsdfTransparent").outputs[0], mx.inputs[1]); g.link(src, mx.inputs[2]); g.link(mx.outputs[0], out.inputs["Surface"])
            m["far_fog"] = 1
    return len(done)


def _fx_sources(objs):
    """Instance sources of effects (tracer prototypes: Collection Info and Object Info in their nodes) — they must not be held out: the holdout carries over
    to the instances and the tracers vanish from the layer."""
    src = set()
    def walk(ng, seen):
        if ng is None or ng.name in seen: return
        seen.add(ng.name)
        for n in ng.nodes:
            if n.type == "COLLECTION_INFO" and n.inputs["Collection"].default_value: src.update(n.inputs["Collection"].default_value.all_objects)
            elif n.type == "OBJECT_INFO" and n.inputs["Object"].default_value: src.add(n.inputs["Object"].default_value)
            elif n.type == "GROUP": walk(n.node_tree, seen)
    for o in objs:
        for md in o.modifiers:
            if md.type == "NODES": walk(md.node_group, set())
    return src


def _world_fill(w, rgb, E):
    """The world for lighting rays (not camera rays) plus an even light of illuminance E in color rgb (normalized to luminance 1); the sky in the frame is unchanged."""
    if E <= 0 or w is None or not w.node_tree: return 0.0
    nt = w.node_tree; g = _G(nt); out = next((n for n in nt.nodes if n.type == "OUTPUT_WORLD"), None)
    bg = out.inputs["Surface"].links[0].from_node if out and out.inputs["Surface"].links else None
    if bg is None or bg.type != "BACKGROUND" or not bg.inputs["Color"].links: return 0.0
    src = bg.inputs["Color"].links[0].from_socket; y = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    fill = g.vec("SCALE", g.const([v / y * E / math.pi for v in rgb]), s=g.math("SUBTRACT", 1.0, g.node("ShaderNodeLightPath").outputs["Is Camera Ray"]))
    g.link(g.vec("ADD", src, fill), bg.inputs["Color"])
    return E


def glow_fill(sc, P):
    """Glow fill (a scene with its own light): at night the node's exposure assumes that the glow of fires and windows lights the objects (½ e_glow),
    while the node's lamps are far away — without the fill, sides not facing the scene's lamps went black (in the battle scene its world played this role)."""
    return _world_fill(sc.world, (1.0, 0.42, 0.12), 0.5 * float(P["light"].get("e_glow", 0.0)))


def moon_fill(sc, P):
    """Moonlit fill at night (for a scene with its own light — battle; places without lamps are read through the night exposure night_ev: the fill turned a
    pass into daytime and flattened the relief; in film this is night shooting, in games — night ambient light): an even cool light
    E = π / 2^ev · 2^night_fill — an object in shadow sits that many stops below middle gray (anime −3.5, realism −4.5: forms stay readable,
    night stays night; −2.5 washed out the far city and the assault walls — "fog by day"); scaled by the plan's night fraction."""
    w = float(P["sky"].get("w_night", 0.0))
    return _world_fill(sc.world, P["sky"].get("night_tint", (0.55, 0.68, 1.0)),
                       math.pi / 2.0 ** P["light"]["ev"] * 2.0 ** float(P["sky"].get("night_fill", -3.5)) * w) if w > 0 else 0.0


def in_view(sc, objs):
    """Effects whose box falls within the active camera's frame (box corners in frame coordinates; behind the camera — no)."""
    from bpy_extras.object_utils import world_to_camera_view
    out = []
    for o in objs:
        P_ = [world_to_camera_view(sc, sc.camera, o.matrix_world @ Vector(c)) for c in o.bound_box]
        if all(p.z <= 0 for p in P_) or all(p.x < 0 for p in P_) or all(p.x > 1 for p in P_) or all(p.y < 0 for p in P_) or all(p.y > 1 for p in P_): continue
        out.append(o)
    return out


def render_scene_fx(sc, objs, path, shadows=False, casters=True):
    """The scene's effects layer (scene_fx.exr): only the effects, everything else is a holdout, transparent background → RGBA EXR (premultiplied);
    far.py post lays it over the frame with haze: lin·(1 − a) + rgb. In the lit frame these effects are hidden (bl_color, fly-around).
    No effects in the frame — no pass (a statue close-up, ~10 s per frame). shadows — lamp shadows in the layer (a take's volumetric explosions
    need the smoke shape: components/fx/bl_fx.py); without them — as with battle effects. casters=False — held-out objects cast no shadows
    (trial: the pass is 58–68 % cheaper, smoke loses the canopy shadow, the smoke's own shadow stays; final — True). → whether a pass was made."""
    if os.path.exists(path): os.remove(path)                            # a trace of the previous run in the same folder
    if not in_view(sc, objs): return False
    mine = set(objs) | _fx_sources(objs)
    other = [o for o in sc.objects if o not in mine and o.type in ("MESH", "CURVE", "CURVES", "POINTCLOUD", "VOLUME") and not o.hide_render]
    far_ = [o for o in other if o.name.startswith("far_")]           # the node's far terrain (far_deck clouds, mountains, city) is behind any effect: in the layer it is
    other = [o for o in other if not o.name.startswith("far_")]        # hidden rather than held out — the EEVEE holdout ate the volume in front of it (clouds
    old = [(o, o.is_holdout) for o in other]; hid = [(o, o.hide_render) for o in objs + far_]   # at 1.2 km "covered" the smoke at 13 m)
    cast = [(o, o.visible_shadow) for o in other] if not casters else []   # shadows of the held-out objects are the main cost of the pass
    for o, _ in cast: o.visible_shadow = False
    for o in other: o.is_holdout = True
    for o in far_: o.hide_render = True
    for o in objs: o.hide_render = False
    im = sc.render.image_settings; keep = (im.media_type, im.file_format, im.color_depth, im.color_mode, sc.render.film_transparent, sc.eevee.taa_render_samples)
    # motion blur as for the frame: fly-around tracers are blurred streaks, like the battle light (without it — short dashes)
    shadows = shadows or any(o.get("fx_shadows") for o in objs)        # (an object flag — this is how bl_color calls it, without changing the call)
    shd = [(L, L.use_shadow) for L in bpy.data.lights] if not shadows else []   # effects glow on their own: the layer needs no shadows (the shadow pool
    for L, _ in shd: L.use_shadow = False                               # of the battle scene is overfull anyway), samples — 12 (the tracers are thin)
    sc.render.film_transparent = True; im.media_type, im.file_format, im.color_depth = "IMAGE", "OPEN_EXR", "32"
    im.color_mode = "RGBA"                                              # the frame before this writes a PNG in RGB: without alpha post took a = 1 — the frame went black
    sc.eevee.taa_render_samples = min(sc.eevee.taa_render_samples, 12)
    sc.render.filepath = path; bpy.ops.render.render(write_still=True)
    for o, h in old: o.is_holdout = h
    for o, v in cast: o.visible_shadow = v
    for o, h in hid: o.hide_render = h
    for L, u in shd: L.use_shadow = u
    im.media_type, im.file_format, im.color_depth, im.color_mode, sc.render.film_transparent, sc.eevee.taa_render_samples = keep
    return True


def node_scene(sc, P, build_dir=None, fps=24.0):
    """The node in an open scene with its own light (a fly-around of the battle scene: formats/battle_trailer/bl_trailer.py far=): the world and the horizon table, the far terrain,
    city, clouds, burning sites, luminaries and the AgX view; the scene's lamps stay under the node, the smoke is the node's, effects go in a layer (hidden in the frame); the build's
    blasts (<build>/data/events.json) become node events by frame. The plan with events is written in place (far.py post reads it).
    Returns the layer's effects."""
    from core.light import far
    sc.world = build_world(P); render_horizon(sc, P, os.path.splitext(P["path"])[0] + "_horizon.json")
    attach_terrain(sc, P, None); attach_city(sc, P); attach_clouds(sc, P); attach_fires(sc, P)
    ev_s = sc.view_settings.exposure; attach(sc, P); k_ev = 2.0 ** (ev_s - P["light"]["ev"])
    scene_light(sc, P, k_ev); scene_emission(k_ev); glow_fill(sc, P); moon_fill(sc, P)
    sfx = scene_fx(sc); fx_fog(sfx, P)
    for o in sfx: o.hide_render = True
    ev = os.path.join(build_dir or "", "data", "events.json")
    if build_dir and os.path.exists(ev): P["fps"] = float(fps); P["events"] = far.battle_events(ev, float(fps), P["fires"].get("light_k", 1.0))
    json.dump({k: v for k, v in P.items() if k != "path"}, open(P["path"] + ".tmp", "w")); os.replace(P["path"] + ".tmp", P["path"])
    return sfx


def node_frame(sc, P, sfx, d, frame):
    """A node frame through the scene camera (fly-around; frame is zero-based): the camera sky, the lit frame (lit.exr — color and depth), the scene's effects layer
    and far.json; far.py post composites them (haze by depth, sky, effects on top, AgX) → lit.png."""
    cam = sc.camera; os.makedirs(d, exist_ok=True)
    fit_clip(cam, P); set_frame(sc, P, frame); spec = cam_spec(cam, sc); per_camera(sc, spec)
    render_sky(sc, os.path.join(d, "sky.exr"))
    F_ = {"plan": P["path"], "hash": P["hash"], "ev": P["light"]["ev"], "mode": "lit", "camera": spec, "frame": frame, "scene_fx": bool(sfx)}
    im = sc.render.image_settings; vl = sc.view_layers[0]; keep = (im.media_type, im.file_format, im.color_depth, sc.render.film_transparent)
    sc.render.film_transparent = True; vl.use_pass_z = True; im.media_type, im.file_format, im.color_depth = "MULTI_LAYER_IMAGE", "OPEN_EXR_MULTILAYER", "32"
    sc.render.filepath = os.path.join(d, "lit.exr"); bpy.ops.render.render(write_still=True); vl.use_pass_z = False
    im.media_type, im.file_format, im.color_depth, sc.render.film_transparent = keep
    F_["scene_fx"] = bool(sfx) and render_scene_fx(sc, sfx, os.path.join(d, "scene_fx.exr"))
    json.dump(F_, open(os.path.join(d, "far.json"), "w"))


def two_pass(sc, render, p, far_obs):
    """Workbench previs with the far terrain in two passes: the near part with shadows, the far part (the plan's relief and city) without shadows (the shadow volumes of a huge
    mesh at a far clip of ~100 km produce streaks across the near ground; a 1 m clip removes them but cuts close-ups nearer than a meter);
    the near part over the far part — the far terrain starts at the edge of the site belt, and from a camera inside the belt it is always farther away (a camera outside — a fly-around —
    would composite by depth). The film is transparent (the sky is the backdrop). render(sc, path) — render to a file (bl_scene.render_to)."""
    import numpy as np
    d = sc.display.shading; keep = [o for o in sc.objects if not o.hide_render and o not in far_obs]
    for o in far_obs: o.hide_render = True
    render(sc, p + ".near.png")
    for o in far_obs: o.hide_render = False
    for o in keep: o.hide_render = True
    sh_ = d.show_shadows; d.show_shadows = False; render(sc, p + ".far.png"); d.show_shadows = sh_
    for o in keep: o.hide_render = False
    px = []
    for q in (p + ".near.png", p + ".far.png"):
        im = bpy.data.images.load(q); im.colorspace_settings.name = "Non-Color"   # as is, without the sRGB → linear conversion
        a = np.empty(im.size[0] * im.size[1] * 4, np.float32); im.pixels.foreach_get(a); px.append(a.reshape(-1, 4))
        W_, H_ = im.size; bpy.data.images.remove(im); os.remove(q)
    n, f = px; out = n.copy(); out[:, 3] = n[:, 3] + f[:, 3] * (1 - n[:, 3])                   # PNG has straight alpha: near over far
    out[:, :3] = (n[:, :3] * n[:, 3:4] + f[:, :3] * f[:, 3:4] * (1 - n[:, 3:4])) / np.maximum(out[:, 3:4], 1e-6)
    im = bpy.data.images.new("two_pass", W_, H_, alpha=True); im.colorspace_settings.name = "Non-Color"
    im.pixels.foreach_set(out.ravel()); im.filepath_raw = p; im.file_format = "PNG"; im.save()
    bpy.data.images.remove(im)


def cam_spec(cam, sc):
    """Camera description (as in cameras.json, maps.Cam) taken from the camera object at the current frame — fly-around, shots by markers."""
    c = cam.data
    return {"name": cam.name, "resolution": [sc.render.resolution_x * sc.render.resolution_percentage // 100, sc.render.resolution_y * sc.render.resolution_percentage // 100],
            "matrix_world": [list(r) for r in cam.matrix_world], "lens_mm": c.lens, "sensor_fit": c.sensor_fit, "sensor_width_mm": c.sensor_width,
            "sensor_height_mm": c.sensor_height, "shift_x": c.shift_x, "shift_y": c.shift_y, "clip": [c.clip_start, c.clip_end]}


def fit_clip(cam, P):
    """Clipping for the far terrain: far — the camera horizon √(2hR) + 60 km (mountains from beyond the horizon), no farther than the skirt; near —
    no less than far / 400,000: Workbench shadows (shadow volumes) at 0.05 m / 200 km produce ray-like streaks on the ground (measured)."""
    T = P.get("terrain") if P else None
    if not T or cam is None: return
    h = max(cam.matrix_world.translation.z - T["z0"], 1.0)
    far_ = min(T.get("skirt_km", T["to_km"]) * 1000.0 * 1.25, max(40000.0, math.sqrt(2 * h * 7432833.0) + 60000.0))
    cam.data.clip_end = max(cam.data.clip_end, far_); cam.data.clip_start = max(cam.data.clip_start, cam.data.clip_end / 4.0e5)


def render_horizon(sc, P, path, N=64, el=1.5):
    """The sky at the horizon by azimuth (el° up, 2° step) — the color of the air haze (core/light/far.py fog_terms): a distant mountain dissolves
    exactly into the color of the sky behind it. 16 views at 22.5° with no tilt, from each the middle quarter of the frame, in every column the row
    exactly at elevation el (at the frame edge the axis row lies lower: at sunset this produced streaks in the haze); linear color; cached by the plan hash."""
    import numpy as np, shutil
    if os.path.exists(path) and json.load(open(path)).get("hash") == P["hash"]: return
    cache = os.path.join(_core_paths.ROOT, "out", "far", P.get("scene", "_"), f"horizon_{P['hash']}.json")
    if os.path.exists(cache): shutil.copy(cache, path); return          # shared cache by the plan hash: 16 sky renders — once per plan, not per run
    cam0, eng, rx, ry, pct = sc.camera, sc.render.engine, sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage
    cd = bpy.data.cameras.new("far_hz"); cd.lens, cd.sensor_width, cd.sensor_fit = 18.0, 36.0, "HORIZONTAL"
    cam = bpy.data.objects.new("far_hz", cd); sc.collection.objects.link(cam); sc.camera = cam
    sc.render.resolution_x = sc.render.resolution_y = N; sc.render.resolution_percentage = 100
    acc = {}; f = N / 2.0; te = math.tan(math.radians(el))
    tmp = os.path.splitext(path)[0] + "_tmp.exr"
    for k in range(16):
        yaw = k * 22.5; cam.rotation_euler = (math.radians(90), 0, math.radians(yaw - 90))
        render_sky(sc, tmp)
        im = bpy.data.images.load(tmp); px = np.array(im.pixels[:]).reshape(N, N, 4)[:, :, :3]; bpy.data.images.remove(im)
        for j in range(N):
            xn = (N / 2 - j - 0.5) / f
            if abs(xn) > 0.22: continue                                     # the middle quarter: ±12.4°
            y = N / 2 - 0.5 + te * math.sqrt(1 + xn * xn) * f; r0 = int(math.floor(y)); w = y - r0   # the row (from the bottom) at elevation el
            c = px[r0, j] * (1 - w) + px[min(r0 + 1, N - 1), j] * w
            acc.setdefault(round((yaw + math.degrees(math.atan(xn))) % 360), []).append(c)
    os.remove(tmp); bpy.data.objects.remove(cam); sc.camera = cam0
    sc.render.engine, sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = eng, rx, ry, pct
    azs = np.array(sorted(acc)); rgb = np.array([np.mean(acc[a], 0) for a in azs])
    grid = np.arange(0, 360, 2.0)
    out = np.stack([np.interp(grid, azs, rgb[:, i], period=360) for i in range(3)], -1)
    json.dump({"hash": P["hash"], "el": el, "az": grid.tolist(), "rgb": out.round(6).tolist()}, open(path, "w"))
    os.makedirs(os.path.dirname(cache), exist_ok=True); shutil.copy(path, cache)


def coverage(scene_dir, P, out, N=480):
    """Coverage check: from the envelope points (center, extreme points, cameras) — 8 views around the horizon (90°), flat Workbench: the sky is
    black, everything else white; a hole is sky below the true horizon (dip √(2h/R) for the height above the far-terrain level, margin 0.1°;
    a pixel is 0.19°: with 96 points a 1° gap at the edge of the far terrain was not caught)."""
    import numpy as np
    from core.blender import bl_scene as B
    sc, statics, members, objs, W, kit = B.load_scene(scene_dir)
    attach_terrain(sc, P, bpy.data.objects.get("ground_far"))
    B.flat_workbench(sc, bg=(0.0, 0.0, 0.0), color_type="SINGLE"); sc.display.shading.single_color = (1, 1, 1)
    cd = bpy.data.cameras.new("probe"); cd.lens, cd.sensor_width, cd.sensor_fit = 18.0, 36.0, "HORIZONTAL"; cd.clip_start, cd.clip_end = 0.5, 40000.0
    cam = bpy.data.objects.new("probe", cd); sc.collection.objects.link(cam); sc.camera = cam; fit_clip(cam, P)
    sc.render.resolution_x, sc.render.resolution_y = N, N // 2
    E = np.array(P["envelope"]["points"]); z0 = P["fog"]["layers"][0]["z0"] if P.get("fog") else 0.0
    probes = [E.mean(0)] + [E[np.argmax(E[:, 0])], E[np.argmin(E[:, 0])], E[np.argmax(E[:, 1])], E[np.argmin(E[:, 1])], E[np.argmax(E[:, 2])]]
    probes = [p for i, p in enumerate(probes) if not any(np.allclose(p, q) for q in probes[:i])]
    f = N / 2.0; ys = (np.arange(N // 2) + 0.5 - N / 4.0)[:, None] * np.ones((1, N)); xs = (np.arange(N) + 0.5 - N / 2.0)[None, :] * np.ones((N // 2, 1))
    el = np.degrees(np.arctan2(-ys, np.hypot(xs, f)))                         # elevation of the pixel's ray above the horizon (camera without tilt)
    rows, worst = [], 0
    for p in probes:
        dip = math.degrees(math.sqrt(max(2 * (p[2] - z0), 0.0) / 7432833.0)) + 0.1
        holes = 0
        for k in range(8):
            cam.location = (float(p[0]), float(p[1]), float(p[2])); cam.rotation_euler = (math.radians(90), 0, math.radians(k * 45 - 90))
            path = os.path.join(out, "cov.png"); sc.render.filepath = path; bpy.ops.render.render(write_still=True)
            im = bpy.data.images.load(path); px = np.array(im.pixels[:]).reshape(N // 2, N, 4)[::-1, :, 0]; bpy.data.images.remove(im)
            holes += int(((px < 0.5) & (el < -dip)).sum())
        rows.append({"at": [round(float(v), 1) for v in p], "dip_deg": round(dip, 2), "sky_below": holes}); worst = max(worst, holes)
    return {"probes": rows, "holes": worst}


if __name__ == "__main__" and sys.argv[sys.argv.index("--") + 1] == "check":
    a = sys.argv[sys.argv.index("--") + 2:]; os.makedirs(a[2], exist_ok=True)
    r = coverage(a[0], json.load(open(a[1])), a[2])
    print("__RESULT__" + json.dumps(r, ensure_ascii=False)); sys.exit(0)

if __name__ == "__main__":
    import numpy as np
    a = sys.argv[sys.argv.index("--") + 1:]; P = json.load(open(a[0])); out = a[1]; opt = dict(x.split("=", 1) for x in a[2:])
    N = int(opt.get("size", 64)); os.makedirs(out, exist_ok=True)
    sc = bpy.context.scene
    for o in list(bpy.data.objects): bpy.data.objects.remove(o)
    attach(sc, P)
    cd = bpy.data.cameras.new("cube"); cd.lens, cd.sensor_width, cd.sensor_fit = 18.0, 36.0, "HORIZONTAL"
    cam = bpy.data.objects.new("cube", cd); sc.collection.objects.link(cam); sc.camera = cam
    sc.render.resolution_x = sc.render.resolution_y = N; sc.render.engine = "BLENDER_EEVEE"; sc.eevee.taa_render_samples = 1
    lum = {}
    for k in range(8):                                                       # belt at the horizon: 8 views at 45° steps, 4° up
        yaw = k * 45.0; cam.rotation_euler = (math.radians(94), 0, math.radians(yaw - 90))
        p = os.path.join(out, f"hz_{k}.exr"); sc.render.filepath = p; sc.render.image_settings.file_format = "OPEN_EXR"
        bpy.ops.render.render(write_still=True)
        im = bpy.data.images.load(p); px = np.array(im.pixels[:]).reshape(N, N, 4); bpy.data.images.remove(im)
        L_ = px[:, :, :3] @ np.array([0.2126, 0.7152, 0.0722])
        for j in range(N):                                                   # azimuth of the frame column
            az = (yaw + math.degrees(math.atan((N / 2 - j - 0.5) / (N / 2)))) % 360
            lum[round(az)] = max(lum.get(round(az), 0), float(L_[N // 2 - 4: N // 2 + 4, j].mean()))
    best = max(lum, key=lum.get); d_az = (best - P["sun"]["az"] + 180) % 360 - 180
    print("__RESULT__" + json.dumps({"brightest_az": best, "sun_az": P["sun"]["az"], "d_az": round(d_az, 1),
                                     "one_sun": bool(P["sun"]["el"] < -2 or abs(d_az) <= 6)}, ensure_ascii=False))

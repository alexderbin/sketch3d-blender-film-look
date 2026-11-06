"""Far ground shader (Eevee): what is seen beyond ~150 m — the belt beyond the site's near grid (bl_terrain_surf) and the
light component's relief out to the horizon (core/light/bl_far). A light material for distant LODs: a color
map + detail; rock farther away uses a larger tile:
- one BSDF: layers are mixed by color, not by mixing BSDFs (the near shader has a BSDF per layer; far away that is an
  extra cost);
- slope — from the pixel's normal: the belt mesh's normal map (4.7 m step) on top of the coarse mesh (cells up to 25 m) —
  rock lies on real steeps and washouts, and the light follows them too; no map — the geometry normal;
- layers and thresholds — the same as in the near shader (base, dry ground and gullies, scree, rock, cliff, snow), the
  same scans (triplanar projection — no stretching on steeps); ground and grass go toward the scan's mean color with
  distance (a 1.5–3 m tile far away — ripples), rock — with the tile ×4.8 and ×16 (Quilez: a noise-driven tile shift —
  the repetition is not noticeable);
- dilution — by the relief data (a_var: moisture, exposure, foot slopes, spots 40–250 m) and by 9–35 m noise per pixel;
- strata on rock and cliff — tone bands by height (the site's strata pack: step, dip), as in the baked texture;
- mountains (there is a snow line and the belt map carries the height above the valley floor): height belts and rock type
  (_zones) — forest as crowns below the line, avalanche chutes, alpine meadow, rock by a threshold of its own height,
  streaks, scree under the walls, snow on ledges and in couloirs;
- far grass canopy — an optional cover layer: the belt cover (Grass) by world position, gaps by view angle, clumps,
  grazing sheen; after the soil, before forest and stone; without cover the shader is unchanged.
material(...) → a Blender material."""
import json

import bpy

MASKS = {"rock_from": 27, "rock_band": 11, "cliff_from": 43, "dirt_dry": 0.36, "talus_from": 16, "talus_to": 30, "jit": 4}   # same as bl_terrain_surf
VISTA = {"rock": (0.21, 0.062), "cliff": (0.21, 0.062), "talus": (0.4, None)}   # tile coordinate multiplier: far away the rock is larger (two scales)
ROUGH = {"snow": 0.55, "rock": 0.82, "cliff": 0.8}                  # everything else — 0.92 (matte ground)
ALPINE = (1.06, 1.03, 0.88, 1.0)                                     # alpine meadow — lighter and drier (so the distance does not turn yellow)


class _Kit:
    """Shader nodes through short calls (as in bl_terrain_surf.apply)."""

    def __init__(self, nt):
        self.nt, self.L, self.cache = nt, nt.links, {}

    def node(self, kind):
        return self.nt.nodes.new(kind)

    def out(self, x, i=0):
        return x.outputs[i] if hasattr(x, "outputs") else x

    def put(self, sock, x):
        if isinstance(x, (int, float)): sock.default_value = float(x)
        elif isinstance(x, tuple): sock.default_value = x
        else: self.L.new(self.out(x), sock)

    def math(self, op, a, b=None):
        m = self.node("ShaderNodeMath"); m.operation = op; self.put(m.inputs[0], a)
        if b is not None: self.put(m.inputs[1], b)
        return m.outputs[0]

    def vmath(self, op, a, b=None, s=None):
        m = self.node("ShaderNodeVectorMath"); m.operation = op; self.put(m.inputs[0], a)
        if b is not None: self.put(m.inputs[1], b)
        if s is not None: self.put(m.inputs["Scale"], s)
        return m.outputs[0]

    def mix(self, f, a, b, blend="MIX"):                             # color (RGBA)
        m = self.node("ShaderNodeMix"); m.data_type = "RGBA"; m.blend_type = blend; ins = {s.identifier: s for s in m.inputs}
        self.put(ins["Factor_Float"], f); self.put(ins["A_Color"], a); self.put(ins["B_Color"], b)
        return next(s for s in m.outputs if s.identifier == "Result_Color")

    def mixf(self, f, a, b):
        m = self.node("ShaderNodeMix"); m.data_type = "FLOAT"; ins = {s.identifier: s for s in m.inputs}
        self.put(ins["Factor_Float"], f); self.put(ins["A_Float"], a); self.put(ins["B_Float"], b)
        return next(s for s in m.outputs if s.identifier == "Result_Float")

    def ramp(self, x, lo, hi):                                       # smooth threshold (smoothstep)
        m = self.node("ShaderNodeMapRange"); m.interpolation_type = "SMOOTHSTEP"; self.put(m.inputs["Value"], x)
        self.put(m.inputs["From Min"], lo); self.put(m.inputs["From Max"], hi); return m.outputs[0]

    def noise(self, vec, scale, seed, detail=2.0):                   # 3D noise 0…1; scale — m; seed — a point shift (3D is cheaper than 4D)
        key = (id(vec), round(scale, 3), seed, detail)                # identical noise — one node (noise is the most expensive part of a shader)
        if key in self.cache: return self.cache[key]
        t = self.node("ShaderNodeTexNoise"); t.noise_dimensions = "3D"
        t.inputs["Scale"].default_value = 1.0 / scale; t.inputs["Detail"].default_value = detail
        self.put(t.inputs["Vector"], self.vmath("ADD", vec, (seed * 13.7, seed * 7.3, seed * 3.1)))
        self.cache[key] = t.outputs["Fac"]; return self.cache[key]

    def attr(self, name, out="Fac"):
        a = self.node("ShaderNodeAttribute"); a.attribute_name = name; return a.outputs[out]


def material(name, layers, seed=1, snowline=None, masks=None, nrm=None, active=None, attrs=True, strata=None, data=None, seam=None, uv_name="ground",
             cover=None):
    """layers — {layer: {"color": image path, "size": tile m, "mean": (r, g, b), "gain": (r, g, b, 1) | None, "veg": bool}}:
    base (none — the color of the baked image active by UV ground), base_wet, dirt, dirt_wet, talus, rock, cliff, snow —
    whichever exist.
    masks — layer thresholds (MASKS + surface.json → masks); nrm — path of the normal map (object-space, UV ground);
    active — the bake image (the active node: Workbench shows it); attrs — the mesh has relief_w attributes (a_k1, a_k2,
    a_var — bl_terrain_surf.attrs);
    strata — the site's strata pack (step, dip); data — belt data maps (R ridges, G thalwegs, B foot slopes —
    site terrain_tex.far_maps, UV ground):
    present — layers are built from them (rock on steep and convex ground, scree in washouts and under cliffs, soil on
    ridges), absent — from attributes;
    seam — (x0, y0, x1, y1, band m): at the edge of the near grid the low frequencies of color are brought toward the site's
    color map (attribute a_macro, which the near shader also holds) — color × map / its own means; the detail remains, no
    seam (macro + detail);
    cover — the far grass canopy (Grass): {"image": the belt cover image (R density, G B A — tint ÷ 2) by world position,
    "box": (x0, y0, width, height), "knobs": CANOPY knobs (blade empty — the site's base color), "soil": soil tone} — as a
    layer
    after the soil, before forest, scree, rock and snow: gaps by view angle e^(−k·density/cos θ), clumps, grazing sheen; in
    the mountains — matched to the alpine meadow; None — the shader is unchanged. → a material."""
    import os
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear(); g = _Kit(nt); sd = int(seed)
    K = {**MASKS, **(masks or {})}; J = float(K["jit"]); R0, C0 = float(K["rock_from"]), float(K["cliff_from"])
    geo = g.node("ShaderNodeNewGeometry"); P = geo.outputs["Position"]
    dist = g.node("ShaderNodeCameraData").outputs["View Distance"]
    uv = g.node("ShaderNodeUVMap"); uv.uv_map = uv_name
    N_ = geo.outputs["Normal"]; nmap = None
    if nrm and os.path.exists(nrm):                                  # belt mesh normals on top of the coarse geometry
        ti = g.node("ShaderNodeTexImage"); ti.image = bpy.data.images.load(nrm, check_existing=True); ti.image.colorspace_settings.name = "Non-Color"
        ti.interpolation = "Linear"; ti.extension = "EXTEND"; g.L.new(uv.outputs[0], ti.inputs["Vector"])
        nmap = g.node("ShaderNodeNormalMap"); nmap.space = "OBJECT"; g.L.new(ti.outputs["Color"], nmap.inputs["Color"]); N_ = nmap.outputs["Normal"]
    n0 = g.attr("sk_n0", "Vector"); ln = g.node("ShaderNodeVectorMath"); ln.operation = "LENGTH"; g.L.new(n0, ln.inputs[0])
    nm = g.node("ShaderNodeMix"); nm.data_type = "VECTOR"; ins = {s_.identifier: s_ for s_ in nm.inputs}   # layers — by the normal before the rocks are displaced
    g.L.new(g.ramp(ln.outputs["Value"], 0.3, 0.6), ins["Factor_Float"]); g.L.new(N_, ins["A_Vector"])      # (the belt's Geometry Nodes: sk_n0) —
    g.L.new(g.vmath("NORMALIZE", n0), ins["B_Vector"]); NM = next(o for o in nm.outputs if o.identifier == "Result_Vector")   # otherwise the snow comes out in squares
    nz = g.node("ShaderNodeSeparateXYZ"); g.L.new(NM, nz.inputs[0])
    slope = g.math("DEGREES", g.math("ARCCOSINE", g.math("MINIMUM", nz.outputs["Z"], 1.0)))
    if attrs:                                                        # knife-edge ridge: the belt mesh rounds off the top — the slope of the sides
        slope = g.math("MAXIMUM", slope, g.attr("a_crest"))          # (bl_terrain_surf.belt_attrs; the near mesh has no such attribute — 0)
    pz = g.node("ShaderNodeSeparateXYZ"); g.L.new(P, pz.inputs[0]); Z = pz.outputs["Z"]
    jn = g.math("SUBTRACT", g.noise(P, 9.0, sd + 4, 1.0), 0.5)       # uneven layer edge — one noise for all thresholds
    jit = lambda x, s_, amp=2 * J: g.math("ADD", x, g.math("MULTIPLY", jn, amp))
    if attrs:
        k1 = g.node("ShaderNodeSeparateColor"); g.L.new(g.attr("a_k1", "Color"), k1.inputs[0])   # moisture, exposure, slope feet, cliff
        k1a = g.attr("a_k1", "Alpha"); moist, expo, foot = k1.outputs[0], k1.outputs[1], k1.outputs[2]
        gully = g.attr("a_k2", "Alpha"); var = g.attr("a_var", "Color")
    else:                                                            # the light component's far field: there is no site data — moisture and spots by noise
        moist = g.math("ADD", 0.35, g.math("MULTIPLY", g.noise(P, 260.0, sd + 51, 2.0), 0.3)); expo = 0.0; foot = 0.0; k1a = 0.0; gully = 0.0
        var = g.math("ADD", 0.88, g.math("MULTIPLY", g.noise(P, 180.0, sd + 11, 3.0), 0.24))
    occ = 1.0; zon = None
    if data and os.path.exists(data):                               # data maps of the belt (ridges, thalwegs, foot slopes — an analysis of its mesh)
        td = g.node("ShaderNodeTexImage"); td.image = bpy.data.images.load(data, check_existing=True); td.image.colorspace_settings.name = "Non-Color"
        td.image.alpha_mode = "CHANNEL_PACKED"                       # alpha is data (height above the floor), not transparency
        td.interpolation = "Linear"; td.extension = "EXTEND"; g.L.new(uv.outputs[0], td.inputs["Vector"])
        dd = g.node("ShaderNodeSeparateColor"); g.L.new(td.outputs["Color"], dd.inputs[0]); ridge, thal, base_ = dd.outputs[0], dd.outputs[1], dd.outputs[2]
        wet_s = g.math("MULTIPLY", g.math("MULTIPLY", thal, g.ramp(slope, 6, 15)), g.math("SUBTRACT", 1.0, g.ramp(slope, 30, 40)))   # a washout — on a slope:
        # on flat ground the catchment accumulates everywhere (the belt's plain went dark — a seam with the near mesh)
        moist0 = moist; moist = g.math("MINIMUM", g.math("MAXIMUM", g.math("SUBTRACT", g.math("ADD", moist, g.math("MULTIPLY", wet_s, 0.35)), g.math("MULTIPLY", ridge, 0.2)), 0.0), 1.0)
        patch = lambda s_, sc_: g.math("SUBTRACT", g.noise(P, sc_, s_, 2.0), 0.5)
        rock_ = g.math("MAXIMUM", g.math("MULTIPLY", g.ramp(jit(slope, sd + 4), 30, 40), g.ramp(g.math("ADD", ridge, g.math("MULTIPLY", patch(sd + 70, 25.0), 0.7)), 0.2, 0.5)),
                       g.ramp(jit(slope, sd + 5), 40, 48))           # rock: steep and convex (ridges, shoulders), very steep — everywhere
        fac = {"rock": g.math("MULTIPLY", rock_, g.math("SUBTRACT", 1.0, g.math("MULTIPLY", wet_s, 0.6))),   # in washouts — scree, not rock
               "cliff": g.math("MAXIMUM", g.ramp(jit(slope, sd + 6), C0 + 4, C0 + 12), k1a),
               "talus": g.math("MULTIPLY", g.math("MULTIPLY", g.math("MAXIMUM", g.math("MULTIPLY", base_, g.ramp(slope, 6, 14)), g.math("MULTIPLY", thal, g.ramp(slope, 18, 30))),
                                                    g.math("SUBTRACT", 1.0, g.ramp(slope, 38, 46))), g.ramp(g.noise(P, 13.0, sd + 3), 0.3, 0.6)),
               "dirt": g.math("MAXIMUM", g.math("MULTIPLY", g.math("MAXIMUM", g.ramp(jit(g.math("SUBTRACT", 1.0, moist0), sd + 1, 0.12), 1 - float(K["dirt_dry"]),
                                                                                1 - float(K["dirt_dry"]) + 0.18), gully),
                                                g.math("SUBTRACT", 1, g.ramp(slope, R0 - 2, R0 + 6))),   # dry patches and small gullies — as in the near one (the seam),
                              g.math("MULTIPLY", g.math("MULTIPLY", ridge, g.ramp(slope, 14, 26)), g.ramp(g.math("ADD", patch(sd + 71, 40.0), 0.5), 0.45, 0.7)))}   # and thin soil on the ridges
        occ = g.math("MULTIPLY", g.math("SUBTRACT", 1.0, g.math("MULTIPLY", wet_s, 0.2)), g.math("ADD", 1.0, g.math("MULTIPLY", ridge, 0.08)))   # washouts darker, ridges lighter
        k_strata = 0.14
        if (snowline is not None or K.get("treeline") is not None) and _rgba(data):   # mountains: height belts and rock type (a map with the height above the floor);
            zon = _zones(g, P, NM, slope, Z, ridge, thal, base_, td.outputs["Alpha"],   # without snow — by the site's treeline (masks.treeline:
                         float(snowline) if snowline is not None else 1e5, K.get("treeline"), sd)   # mountains in a forest biome — site refresh)
            fac.update({k: zon[k] for k in ("talus", "rock", "snow", "cliff")}); k_strata = 0.22   # cliff — by the belt rules: the cliff flag
            # of the site's baked breakdown (a_k1.a) painted the edge of the site with the cliff scan —
            # a strip along the site boundary across the ridges ("a stream along the summits")
    else:
        dry = g.math("SUBTRACT", 1.0, moist)
        fac = {"dirt": g.math("MULTIPLY", g.math("MAXIMUM", g.ramp(jit(dry, sd + 1, 0.12), 1 - float(K["dirt_dry"]), 1 - float(K["dirt_dry"]) + 0.18), gully),
                              g.math("SUBTRACT", 1, g.ramp(slope, R0 - 2, R0 + 6))),
               "talus": g.math("MULTIPLY", g.math("MAXIMUM", g.math("MULTIPLY", foot, g.ramp(slope, 6, 14)),
                                                  g.math("MULTIPLY", g.ramp(jit(slope, sd + 2), K["talus_from"], K["talus_from"] + 8), g.ramp(g.noise(P, 13.0, sd + 3), 0.45, 0.7))),
                               g.math("SUBTRACT", 1, g.ramp(slope, K["talus_to"], K["talus_to"] + 8))),
               "rock": g.math("MULTIPLY", g.ramp(g.math("ADD", jit(slope, sd + 4), g.math("MULTIPLY", expo, 10.0)), R0, R0 + K["rock_band"]),
                              g.math("SUBTRACT", 1, g.ramp(jit(slope, sd + 5), C0 + 1, C0 + 9))),
               "cliff": g.math("MAXIMUM", g.ramp(jit(slope, sd + 6), C0, C0 + 9), k1a)}
        k_strata = 0.2
    if snowline is not None and zon is None:
        fac["snow"] = g.math("MULTIPLY", g.ramp(jit(Z, sd + 7, 8.0), snowline - 6, snowline + 6), g.math("SUBTRACT", 1, g.ramp(slope, 45, 60)))

    far = g.ramp(dist, 150.0, 1200.0)                                 # 0 at the edge of the near grid → 1 at a kilometer
    St = strata or {}; step_ = 3.0 * float(St.get("step", 2.6)); dip = float(St.get("dip", 0.08))
    sep = g.node("ShaderNodeSeparateXYZ"); g.L.new(P, sep.inputs[0])
    zp = g.math("ADD", g.math("ADD", Z, g.math("MULTIPLY", g.math("ADD", g.math("MULTIPLY", sep.outputs["X"], 0.6), g.math("MULTIPLY", sep.outputs["Y"], 0.8)), dip)),
                g.math("MULTIPLY", g.math("SUBTRACT", g.noise(P, 60.0, sd + 41, 3.0), 0.5), 9.0))   # the edge of a stratum is uneven: not straight horizontal lines
    band = g.node("ShaderNodeTexNoise"); band.noise_dimensions = "1D"; band.inputs["Detail"].default_value = 1.0; band.inputs["Scale"].default_value = 1.0
    g.L.new(g.math("DIVIDE", zp, step_), band.inputs["W"])            # each stratum has its own tone: bands by height with the dip of the pack
    strata_t = g.math("ADD", 1.0 - k_strata, g.math("MULTIPLY", band.outputs["Fac"], 2 * k_strata))
    imgs = {}

    def tex(path, vec):
        t = g.node("ShaderNodeTexImage")
        if path not in imgs: imgs[path] = bpy.data.images.load(path, check_existing=True)
        t.image = imgs[path]; t.projection = "BOX"; t.projection_blend = 0.3; t.interpolation = "Linear"; g.L.new(vec, t.inputs["Vector"])
        return t.outputs["Color"]

    def tiled(path, size, k, s_):                                    # tile without repetition: two noise-driven shifts, a smooth edge (Quilez, technique 3)
        v = g.vmath("SCALE", P, s=k / size); idx = g.math("MULTIPLY", g.noise(P, 40.0, sd + 20, 1.0), 8.0); i = g.math("FLOOR", idx)
        f = g.math("FRACT", idx)
        def off(j):
            c = g.node("ShaderNodeCombineXYZ")
            for a_, w_ in zip(c.inputs, (3.0, 7.0, 5.0)): g.L.new(g.math("SINE", g.math("MULTIPLY", j, w_)), a_)
            return c.outputs[0]
        return g.mix(g.ramp(f, 0.2, 0.8), tex(path, g.vmath("ADD", v, off(i))), tex(path, g.vmath("ADD", v, off(g.math("ADD", i, 1.0)))))

    cols = {}
    def colour(lay):
        S = layers[lay]; kind = lay.split("_")[0]; key = (S["color"], kind in VISTA)
        if key in cols: return cols[key]
        mean = tuple(S["mean"][:3]) + (1.0,); size = float(S["size"])
        if kind in VISTA:                                            # rock and scree: a larger tile, a second scale far away
            k1_, k2_ = VISTA[kind]; c = tiled(S["color"], size, k1_, sd + 20 + len(cols))
            if k2_: c = g.mix(g.math("MULTIPLY", g.ramp(g.noise(P, 220.0, sd + 23 + len(cols), 2.0), 0.35, 0.65), g.math("ADD", 0.35, g.math("MULTIPLY", far, 0.3))),
                              c, tex(S["color"], g.vmath("SCALE", P, s=k2_ / size)))
            c = g.mix(0.15, c, mean)                                 # the rock pattern — almost in full: far away it is the detail
        else:                                                        # ground and grass: the tile as up close (the seam with the near mesh), far away — toward the mean
            c = g.mix(g.math("ADD", 0.25 if zon is not None else 0.55, g.math("MULTIPLY", far, 0.6 if zon is not None else 0.3)),   # mountains: up close
                      tex(S["color"], g.vmath("SCALE", P, s=1.0 / size)), mean)   # more texture (the far shader's meadow was smooth near the camera)
            br = g.math("ADD", 1.0, g.math("MULTIPLY", g.math("SUBTRACT", g.noise(P, 70.0, sd + 61, 3.0), 0.5), g.math("MULTIPLY", g.ramp(dist, 250.0, 900.0), 0.24)))
            # grass-stand patches of 9–80 m — far away (there the scan pattern is gone); at the edge of the near
            # grid — none, as in the near shader (otherwise the variegation gives away the seam)
            c = g.mix(1.0, c, _grey(g, br), "MULTIPLY")
        if S.get("gain"): c = g.mix(1.0, c, tuple(S["gain"][:3]) + (1.0,), "MULTIPLY")
        cols[key] = c; return c

    def with_wet(lay):                                               # dry / wet pair — by the site's moisture
        c = colour(lay)
        if lay + "_wet" in layers: c = g.mix(g.ramp(g.math("ADD", moist, g.math("MULTIPLY", jn, 0.16)), 0.44, 0.58), c, colour(lay + "_wet"))
        return c

    def mean_of(lay):                                                # the layer's mean color (as in colour: fit, scree lighter)
        S = layers[lay]; m_ = [float(S["mean"][i]) * float((S.get("gain") or (1, 1, 1))[i]) * (1.08 if lay.startswith("talus") else 1.0) for i in range(3)]
        return tuple(m_) + (1.0,)

    def mean_wet(lay):
        c = mean_of(lay)
        if lay + "_wet" in layers: c = g.mix(g.ramp(moist, 0.44, 0.58), c, mean_of(lay + "_wet"))
        return c

    def rough_of(lay):                                               # layer roughness — as in the near shader (the mean of the scan map):
        r = float(layers[lay].get("rough", ROUGH.get(lay.split("_")[0], 0.92)))   # otherwise at a grazing angle far away there is less highlight —
        if lay + "_wet" in layers: return g.mixf(g.ramp(moist, 0.44, 0.58), r, float(layers[lay + "_wet"].get("rough", r)))   # a seam at the edge
        return r

    cv = None
    if cover:                                                        # far grass canopy (Grass): the belt cover by world position
        bx0, by0, bw, bh = (float(v) for v in cover["box"]); CN = cover["knobs"]
        cvec = g.node("ShaderNodeCombineXYZ"); g.L.new(g.math("DIVIDE", g.math("SUBTRACT", sep.outputs["X"], bx0), bw), cvec.inputs["X"])
        g.L.new(g.math("DIVIDE", g.math("SUBTRACT", sep.outputs["Y"], by0), bh), cvec.inputs["Y"])
        ci = g.node("ShaderNodeTexImage"); ci.image = cover["image"]; ci.interpolation = "Linear"; ci.extension = "EXTEND"; g.L.new(cvec.outputs[0], ci.inputs["Vector"])
        cs = g.node("ShaderNodeSeparateColor"); g.L.new(ci.outputs["Color"], cs.inputs[0]); dens = cs.outputs[0]
        tint = g.node("ShaderNodeCombineColor")
        for i_, o_ in enumerate((cs.outputs[1], cs.outputs[2], ci.outputs["Alpha"])): g.L.new(g.math("MULTIPLY", o_, 2.0), tint.inputs[i_])
        mu = g.math("MAXIMUM", g.math("ABSOLUTE", g.vmath("DOT_PRODUCT", geo.outputs["Incoming"], NM).node.outputs["Value"]), 0.05)
        bc = tuple(float(c) for c in CN["blade"]) if CN.get("blade") else (mean_wet("base") if "base" in layers else (0.18, 0.23, 0.05))   # grass color —
        blade = g.vmath("SCALE", g.vmath("MULTIPLY", tint.outputs[0], bc), s=float(CN.get("gain", 1.0)))   # the site's base (as in a_macro)
        pv = lambda mu_: g.math("EXPONENT", g.math("DIVIDE", g.math("MULTIPLY", dens, -float(CN["k"])), mu_))   # fraction of soil through the grass
        mot = g.math("ADD", 1.0, g.math("ADD", g.math("MULTIPLY", g.math("SUBTRACT", g.noise(P, 20.0, sd + 91, 2.0), 0.5), 0.35),   # clumps of 20 m and 6 m
                                         g.math("MULTIPLY", g.math("SUBTRACT", g.noise(P, 6.0, sd + 92, 2.0), 0.5),
                                                g.math("MULTIPLY", g.math("SUBTRACT", 1.0, g.ramp(dist, 600.0, 1500.0)), 0.3))))   # the small ones fade out toward 1.5 km
        gz = g.math("MULTIPLY", g.math("POWER", g.math("SUBTRACT", 1.0, mu), 4.0), g.math("MULTIPLY", dens, float(CN["sheen"])))   # grazing sheen
        soil = tuple(float(c) for c in cover["soil"][:3]) + (1.0,)
        cc = g.mix(pv(mu), g.vmath("SCALE", blade, s=g.math("ADD", mot, gz)), soil); cmean = g.mix(pv(0.35), blade, soil)   # the mean — as in a_macro
        if zon is not None: cc = g.mix(zon["alpine"], cc, ALPINE, "MULTIPLY"); cmean = g.mix(zon["alpine"], cmean, ALPINE, "MULTIPLY")
        gz_ = g.node("ShaderNodeSeparateXYZ"); g.L.new(geo.outputs["Normal"], gz_.inputs[0])   # rock walls (the bl_crags displacement): gentle by NM,
        steep_ = g.ramp(g.math("DEGREES", g.math("ARCCOSINE", g.math("MINIMUM", gz_.outputs["Z"], 1.0))), 32.0, 48.0)   # steep by geometry: no grass
        cv = (g.math("MULTIPLY", g.ramp(g.math("ADD", dens, g.math("MULTIPLY", jn, 0.1)), *CN["dens"]), g.math("SUBTRACT", 1.0, steep_)), cc, cmean)
    if "base" in layers: col = with_wet("base"); low = mean_wet("base")
    else: ta = g.node("ShaderNodeTexImage"); ta.image = active; g.L.new(uv.outputs[0], ta.inputs["Vector"]); col = low = ta.outputs["Color"]
    rough = rough_of("base") if "base" in layers else 0.92
    if zon is not None:                                              # above the forest — alpine meadow: lighter and yellower (dry short grass)
        col = g.mix(zon["alpine"], col, ALPINE, "MULTIPLY")
        near_c = g.ramp(dist, 300.0, 700.0)                          # crowns (4.5 m cells) — from 300–700 m: closer they read as scales
        zon["crown"] = g.math("ADD", 0.5, g.math("MULTIPLY", g.math("SUBTRACT", zon["crown"], 0.5), near_c))   # (up close a real forest is needed)
        zon["tone"] = g.math("ADD", 0.5, g.math("MULTIPLY", g.math("SUBTRACT", zon["tone"], 0.5), near_c))
        canopy, cmean = _canopy(g, zon["crown"], dist, zon["n2"], zon["tone"], tuple(K.get("canopy") or (0.032, 0.052, 0.026)))
    for lay in ("dirt", "forest", "talus", "rock", "cliff", "snow"):
        if lay == "forest" and cv is not None:                       # grass canopy — after the soil, before forest and stone
            col = g.mix(cv[0], col, cv[1])
            if seam: low = g.mix(cv[0], low, cv[2])
        if lay == "forest":                                          # forest below its line — a canopy of crowns (the forest of the Plants component exists only at the site)
            if zon is not None:
                col = g.mix(zon["forest"], col, canopy); rough = g.mixf(zon["forest"], rough, 0.85)
                if seam: low = g.mix(zon["forest"], low, cmean)
            continue
        if lay not in layers or lay not in fac: continue
        c = with_wet(lay)
        if lay == "cliff": c = g.mix(1.0, c, _grey(g, strata_t), "MULTIPLY")   # strata only on the cliff (on the slope they read as contour lines)
        if zon is not None and lay in ("rock", "cliff"): c = g.mix(1.0, c, _grey(g, zon["rtone"]), "MULTIPLY")   # streaks, lichen patches
        if lay == "talus": c = g.mix(1.0, c, (1.08, 1.08, 1.08, 1.0), "MULTIPLY")   # scree lighter (fresh chips)
        col = g.mix(fac[lay], col, c); rough = g.mixf(fac[lay], rough, rough_of(lay))
        if seam: low = g.mix(fac[lay], low, mean_wet(lay))
    col = g.mix(1.0, col, var, "MULTIPLY")                            # dilution by relief data
    if occ != 1.0: col = g.mix(1.0, col, _grey(g, occ), "MULTIPLY")
    if seam:                                                         # at the edge of the near grid: low frequencies — the site's color map
        x0, y0, x1, y1, band_ = seam
        out_ = g.math("MAXIMUM", g.math("MAXIMUM", g.math("SUBTRACT", x0, sep.outputs["X"]), g.math("SUBTRACT", sep.outputs["X"], x1)),
                      g.math("MAXIMUM", g.math("SUBTRACT", y0, sep.outputs["Y"]), g.math("SUBTRACT", sep.outputs["Y"], y1)))   # beyond the edge, m
        s_ = g.math("SUBTRACT", 1.0, g.ramp(out_, 0.25 * band_, band_))
        low = g.mix(1.0, low, var, "MULTIPLY"); mac = g.attr("a_macro", "Color")
        ratio = g.vmath("DIVIDE", mac, g.vmath("MAXIMUM", low, (0.002, 0.002, 0.002)))
        col = g.mix(1.0, col, g.mix(s_, (1.0, 1.0, 1.0, 1.0), ratio), "MULTIPLY")
    b = g.node("ShaderNodeBsdfPrincipled"); g.put(b.inputs["Base Color"], col); g.put(b.inputs["Roughness"], rough)
    b.inputs["Specular IOR Level"].default_value = 0.35
    rw = g.math("MINIMUM", g.math("MAXIMUM", g.math("MAXIMUM", fac.get("rock", 0.0), fac.get("cliff", 0.0)), g.math("MULTIPLY", fac.get("talus", 0.0), 0.6)), 1.0)
    if zon is not None:
        rw = g.math("MAXIMUM", rw, g.math("MULTIPLY", fac.get("snow", 0.0), 0.4))   # snow has its own relief (ripples, drifts): without it there is a gradient far away
        rw = g.math("MAXIMUM", rw, g.math("MULTIPLY", g.ramp(slope, 25.0, 40.0), 0.5))   # a steep meadow is not smooth either: between rough walls
        # (rock walls) a smooth gully floor read as a "stream"
    rn = g.node("ShaderNodeTexNoise"); rn.noise_type = "RIDGED_MULTIFRACTAL"; rn.noise_dimensions = "3D"; rn.inputs["Scale"].default_value = 1.0 / 6.0
    rn.inputs["Detail"].default_value = 2.0; g.L.new(P, rn.inputs["Vector"])   # relief detail of 2–10 m on top of the mesh normals: rock edges,
    fine = g.math("MULTIPLY", g.noise(P, 2.2, sd + 81, 2.0), g.math("SUBTRACT", 1.0, g.ramp(dist, 800.0, 2000.0)))   # scree ripples, grass tussocks;
    hgt = g.mixf(rw, 0.0,                                            # the fine detail fades out toward 2 km (otherwise it shimmers); grass — without it (tussocks darkened the meadow: a seam)
                 g.math("ADD", g.math("MULTIPLY", rn.outputs["Fac"], 0.65), g.math("MULTIPLY", fine, 0.35)))
    if zon is not None:                                              # crowns — canopy bumps: light and shadow on forest edges and slopes
        hgt = g.math("ADD", hgt, g.math("MULTIPLY", g.math("MULTIPLY", zon["forest"], g.math("SUBTRACT", zon["crown"], 0.5)), 0.35))   # mean 0:
        # no bump step at the forest edge (otherwise the light would outline the forest)
    bm = g.node("ShaderNodeBump"); g.put(bm.inputs["Height"], hgt); bm.inputs["Distance"].default_value = 1.5
    g.put(bm.inputs["Strength"], g.math("SUBTRACT", 1.0, g.math("MULTIPLY", g.ramp(dist, 1500.0, 4000.0), 0.6)))
    if nmap is not None: g.L.new(nmap.outputs["Normal"], bm.inputs["Normal"])
    g.L.new(bm.outputs["Normal"], b.inputs["Normal"])
    o = g.node("ShaderNodeOutputMaterial"); g.L.new(b.outputs[0], o.inputs["Surface"])
    a = g.node("ShaderNodeOutputAOV"); a.aov_name = "ground"; a.inputs["Value"].default_value = 1.0   # ground mask of the frame (checks)
    if active is not None:                                           # Workbench shows the active image
        ta = next((x for x in nt.nodes if x.type == "TEX_IMAGE" and x.image == active), None)
        if ta is None: ta = g.node("ShaderNodeTexImage"); ta.image = active; g.L.new(uv.outputs[0], ta.inputs["Vector"])
        nt.nodes.active = ta
    m["sk_vista"] = {"layers": sorted(layers), "n_pictures": len(imgs) + (nmap is not None) + (occ != 1.0) + (cv is not None), "normals": nmap is not None,
                     "data": occ != 1.0, "canopy": cv is not None}
    m["sk_vista_spec"] = json.dumps({"layers": layers, "seed": seed, "masks": masks, "strata": strata}, default=float)   # for the Light component's mountains (horizon)
    return m


def horizon(name, active=None, snowline=None, uv_name="UVMap", data=None):
    """Far shader for the mountains beyond the belt (core/light/bl_far.attach_terrain): the same set of scans and tone fit as
    the site belt (the terrain_far material — its spec): slope from the mesh, moisture and spots by noise, snow by mountain
    height. The tone matches at the junction of the belt and the mountains. data — the far data map (the site's real
    mountains: core/light/far._data_map) — the same height belts as the belt. No site belt (a scene without a site) — None,
    the mountains stay an image."""
    src = bpy.data.materials.get("terrain_far")
    if src is None or "sk_vista_spec" not in src: return None
    S = json.loads(src["sk_vista_spec"])
    return material(name, S["layers"], S.get("seed", 1), snowline, S.get("masks"), None, active, False, S.get("strata"), data, None, uv_name)


def _rgba(path):
    """The PNG has an alpha channel (color type 6 in the IHDR header) — a belt map of the new kind (with the height above the
    valley floor)."""
    try:
        with open(path, "rb") as f: h = f.read(26)
        return h[:8] == b"\x89PNG\r\n\x1a\n" and h[25] in (4, 6)
    except OSError: return False


def _zones(g, P, N_, slope, Z, ridge, thal, foot, hrel, snow, tree, sd):
    """Mountain belts and rock type. Soil holds up to a threshold, which drops with height (thin soils above the forest; Körner
    2003) and on convex ground high above the floor; walls steeper than 50° — always rock; scree — under the walls and in
    couloirs above the forest; forest — below the line (lower on north slopes), denser in gullies, without avalanche chutes
    (McClung, Schaerer); snow — on ledges and in couloirs (lower there), lower on north slopes, none on the walls. tree —
    the treeline, m (none — 0.43 of the snow line; a site right at the snow — the line is below 150 m above the floor, a
    pass — 300 m below the snow: in nature there are hundreds of meters between forest and snow)."""
    T = float(tree) if tree is not None else (0.43 * snow if snow >= 150.0 else snow - 300.0)
    sub = lambda a, b: g.math("SUBTRACT", a, b); mul = lambda a, b: g.math("MULTIPLY", a, b); add = lambda a, b: g.math("ADD", a, b)
    inv = lambda a: sub(1.0, a); clamp = lambda a: g.math("MINIMUM", g.math("MAXIMUM", a, 0.0), 1.0)
    ns = g.node("ShaderNodeSeparateXYZ"); g.L.new(N_, ns.inputs[0])
    north = g.math("MINIMUM", g.math("MAXIMUM", mul(ns.outputs["Y"], 2.0), -1.0), 1.0)   # +1 — the slope faces north (y+)
    n1, n2 = sub(g.noise(P, 35.0, sd + 90, 2.0), 0.5), sub(g.noise(P, 120.0, sd + 91, 2.0), 0.5)
    zs = add(sub(snow, mul(north, 0.08 * snow)), mul(n2, 50.0)); zt = add(sub(T, mul(north, 0.12 * T)), mul(n2, 70.0))
    thr = add(sub(sub(42.0, mul(g.ramp(Z, 0.6 * T, snow), 12.0)), mul(mul(ridge, g.ramp(hrel, 0.3, 0.7)), 6.0)), mul(n1, 6.0))
    rock = g.ramp(slope, sub(thr, 3.0), add(thr, 3.0))
    cliff = g.ramp(add(slope, mul(n1, 6.0)), 50.0, 58.0)
    gully = mul(g.ramp(thal, 0.3, 0.7), g.ramp(slope, 25.0, 35.0))
    foot_s = mul(mul(mul(foot, 1.6), g.ramp(slope, 20.0, 30.0)), inv(g.ramp(slope, 40.0, 46.0)))
    scree = mul(mul(clamp(g.math("MAXIMUM", foot_s, mul(mul(gully, rock), 0.8))), g.math("MAXIMUM", g.ramp(Z, sub(zt, 80.0), add(zt, 40.0)), 0.7)),
                g.ramp(g.noise(P, 13.0, sd + 3), 0.3, 0.6))                # scree under the walls — at any height (below the treeline the forest covers it)
    fell = mul(mul(g.ramp(Z, add(zt, mul(sub(zs, zt), 0.45)), add(zt, mul(sub(zs, zt), 0.7))), inv(rock)),   # stony debris above the meadows:
               g.ramp(g.noise(P, 60.0, sd + 94, 2.0), 0.3, 0.55))       # gentle tops and saddles — not a smooth meadow (far away it looks like a "river"); meadow patches
    chute = mul(mul(mul(g.ramp(thal, 0.3, 0.6), g.ramp(slope, 3.0, 10.0)), g.math("MAXIMUM", g.math("MAXIMUM", g.ramp(Z, sub(zt, 40.0), add(zt, 40.0)), g.ramp(foot, 0.2, 0.5)), g.ramp(slope, 28.0, 36.0))),   # a couloir floor above
                g.ramp(g.noise(P, 30.0, sd + 95, 2.0), 0.25, 0.5))    # the forest and near the walls — scree and pebbles, not meadow (the smooth floor of a narrow valley
    # between rocks read as a "river"; the 4.7 m height map smooths the floor)
    scree = g.math("MAXIMUM", g.math("MAXIMUM", scree, fell), chute)
    gully_s = mul(mul(g.ramp(thal, 0.5, 0.8), g.ramp(slope, 25.0, 35.0)), g.ramp(g.noise(P, 20.0, sd + 96, 2.0), 0.35, 0.6))   # snow in a couloir —
    snow_f = g.math("MAXIMUM", mul(g.ramp(Z, sub(zs, 15.0), add(zs, 15.0)), inv(g.ramp(slope, 38.0, 48.0))),   # in patches and narrower (a solid strip
                    mul(mul(g.ramp(Z, sub(mul(zs, 0.85), 15.0), add(mul(zs, 0.85), 15.0)), gully_s), inv(g.ramp(slope, 46.0, 54.0))))   # read as a "river")
    ledge = g.attr("sk_ledge", "Fac")                                 # strata ledges (the rock walls of site bl_crags): snow above 0.7, meadow below
    snow_l = mul(ledge, g.ramp(Z, sub(mul(zs, 0.7), 15.0), add(mul(zs, 0.7), 15.0)))
    snow_f = g.math("MAXIMUM", snow_f, snow_l); green = mul(sub(ledge, snow_l), 0.8)
    aval = mul(mul(g.ramp(thal, 0.45, 0.95), g.ramp(slope, 22.0, 30.0)), g.ramp(hrel, 0.08, 0.2))   # avalanche chutes through the forest
    dens = clamp(add(sub(add(0.75, mul(g.ramp(thal, 0.0, 0.45), 0.5)), mul(ridge, 0.35)), mul(n2, 0.8)))
    forest = mul(mul(inv(g.ramp(Z, sub(zt, 25.0), add(zt, 25.0))), inv(g.ramp(slope, 34.0, 40.0))), dens)
    forest = mul(mul(mul(forest, inv(mul(aval, 0.9))), inv(rock)), inv(scree))
    forest = g.ramp(add(forest, mul(sub(g.noise(P, 25.0, sd + 92, 2.0), 0.5), 0.6)), 0.35, 0.6)   # ragged forest edges, clearings and clumps
    vor = g.node("ShaderNodeTexVoronoi"); vor.voronoi_dimensions = "3D"; vor.inputs["Scale"].default_value = 1.0 / 4.5; g.L.new(P, vor.inputs["Vector"])
    crown = inv(g.ramp(vor.outputs["Distance"], 0.15, 0.85))          # a 4–5 m crown: a light top, a dark gap
    tone = g.node("ShaderNodeSeparateColor"); g.L.new(vor.outputs["Color"], tone.inputs[0])   # each crown has its own tone
    sx = g.node("ShaderNodeSeparateXYZ"); g.L.new(P, sx.inputs[0]); sv = g.node("ShaderNodeCombineXYZ")
    for i_, k_ in enumerate((0.25, 0.25, 1 / 30.0)): g.L.new(mul(sx.outputs[i_], k_), sv.inputs[i_])
    streak = g.node("ShaderNodeTexNoise"); streak.noise_dimensions = "3D"; streak.inputs["Scale"].default_value = 1.0; streak.inputs["Detail"].default_value = 2.0
    g.L.new(sv.outputs[0], streak.inputs["Vector"])                    # streaks along the fall line of the wall: 4 m across, 30 m along
    rtone = mul(sub(1.0, mul(g.ramp(streak.outputs["Fac"], 0.45, 0.75), 0.3)), add(0.95, mul(sub(g.noise(P, 70.0, sd + 93, 2.0), 0.5), 0.3)))
    alpine = mul(g.ramp(Z, sub(zt, 30.0), add(zt, 60.0)), g.ramp(hrel, 0.1, 0.3))   # alpine meadow — on the slopes; the floor — the site's meadow (seam)
    rock, cliff = mul(rock, inv(green)), mul(cliff, inv(green))       # on a ledge — meadow under the rock
    return {"rock": rock, "cliff": cliff, "talus": scree, "snow": snow_f, "forest": forest, "alpine": alpine,
            "crown": crown, "tone": tone.outputs[0], "rtone": mul(rtone, 0.88), "n2": n2}


def _canopy(g, crown, dist, n2, tone, base=(0.032, 0.052, 0.026)):
    """Canopy color: a conifer forest (linear fractions), crowns lighter at the top and each of its own tone, toward 1.8 km —
    toward the mean (the crown pattern is finer than a pixel)."""
    br = g.math("MULTIPLY", g.math("ADD", 0.7, g.math("MULTIPLY", crown, 0.35)), g.math("ADD", 0.82, g.math("MULTIPLY", tone, 0.36)))
    br = g.mixf(g.ramp(dist, 600.0, 1800.0), br, 0.88)
    br = g.math("MULTIPLY", br, g.math("ADD", 1.0, g.math("MULTIPLY", n2, 0.5)))   # groves lighter and darker (species, age)
    c = g.mix(1.0, base + (1.0,), _grey(g, br), "MULTIPLY")
    return c, tuple(x * 0.85 for x in base) + (1.0,)


def _grey(g, x):
    """A number → a color (x, x, x) for multiplying a color."""
    c = g.node("ShaderNodeCombineColor"); g.put(c.inputs[0], x); g.put(c.inputs[1], x); g.put(c.inputs[2], x); return c.outputs[0]

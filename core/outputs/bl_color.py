"""Frame color: a library scene (a .blend with textures and team colors) through our cameras → out/<name>/cam<K>/color.png. A marker
sketch is a gray mass without outlines; color plus our outline is previs.png (core/outputs/previs.py).
blender -b --factory-startup -P core/blender/run.py -- core/outputs/bl_color.py -- scenes/<name> out/<name> <K1,K2,…|all> [ground=0.52,0.47,0.40] [sky=0.80,0.84,0.90]
  [light=<node time dawn … night>|overcast|blaze|far|scene] [hdri=<Poly Haven id>] [fire=x,y,z;…] [fire_power=1500] [fog=0.02] [samples=32]
  (node time — as in light=far time=…; overcast — a day with overcast weather, blaze — a fire night: the node's weather= is set automatically; PRESET presets — only with hdri=)
  [weather=rain|after_rain|snow|fog|dust_storm|after_battle|mud] — the whole frame: figures, world, ground and the far sky;
  [rain=0–1] — raining now: puddle ripples, streaks and splashes (components/materials/bl_rain.py), for rain and storm it is automatic; [reflections=off] — no ray tracing
  [materials=1|0] [state=wet:0.6,dirt:0.3,…] [state_world=…] [state_ground=…] [closeup=1] [simplify=auto|off|lite] — the common material
  and states (components/materials/bl_mat.py): with light= it turns on by itself if the scene has materials.yaml (components/materials/mat_plan.py); command-line keys
  override the description; materials=0 — turn off; states: wet, dirt, dust, rust, wear, blood, burn, snow (0–1)
light= — a lit frame (Eevee) in lit.png: the preset's sun and sky, a Poly Haven HDRI sky, fire — for
burning objects of the scene automatic and by coordinates, haze as a volume; light=scene — the scene's own light, sky and smoke as placed
(sketch_fx objects — fires, searchlights; without light they are hidden); previs with light — core/outputs/previs.py --lit.
light=far [time=… weather=… style=… sky_image=<image 2:1>[:rotation] far=<plan.json>] — light and sky of the far-terrain plan (core/light/far.py, core/light/bl_far.py): Eevee writes
the geometry on a transparent background (lit.exr) and the camera sky (sky.exr), core/light/far.py post composites them into lit.png (AgX, the plan's exposure). For a scene
with a far-terrain description (scenes/<name>/far.yaml) the previs also gets the plan's sky: color.png and albedo.png — with a transparent sky, sky.exr next to them.
The source is scenes/<name>/source.json (written by the bridge bl_export_scene.py); plan edits (edits.json, core/model/scene_edit.py) are repeated on
the .blend objects; terrain (it has no UVs) — with the map's ground texture by its splat map (cache out/ground/),
without a map or with "ground=r,g,b" — a flat fill; parts appended later (append_parts) — from scene.obj, colored by kind.
The render is Workbench with textures, a fraction of a second per camera."""
import core.paths        # noqa: E402  core parts on sys.path
import json, math, os, subprocess, sys
ROOT = core.paths.ROOT
import bpy
import numpy as np
from mathutils import Matrix, Vector
from core.blender import bl_scene as B

a = sys.argv[sys.argv.index("--") + 1:]
SD, OUT, KS = os.path.abspath(a[0]), os.path.abspath(a[1]), a[2]
opt = dict(kv.split("=", 1) for kv in a[3:])
col = lambda s, d: tuple(float(x) for x in opt.get(s, d).split(","))
GROUND, SKY = col("ground", "0.52,0.47,0.40"), col("sky", "0.80,0.84,0.90")
src = json.load(open(os.path.join(SD, "source.json")))["blend"]
cams = json.load(open(os.path.join(SD, "cameras.json")))
ks = list(range(1, len(cams) + 1)) if KS == "all" else [int(k) for k in KS.split(",") if k]
bpy.ops.wm.open_mainfile(filepath=src, load_ui=False)
sc = bpy.context.scene
BATTLE = os.path.exists(os.path.join(SD, "battle.yaml")) or os.path.exists(os.path.join(os.path.dirname(src), "battle_job.json"))   # battle scene
# (components/battle/bl_battle.py; for fly-around and 4K views the battle build is in source.json): its lamps stay under the node (bl_far.scene_light)
bj_ = os.path.join(os.path.dirname(src), "battle_job.json")        # the battle scene's frame rate comes from its plan (24): the clip importer sets the file to
if os.path.exists(bj_): sc.render.fps, sc.render.fps_base = int(json.load(open(bj_))["spec"].get("anim_cfg", {}).get("fps", 24)), 1.0   # 30 fps
for o in [o for o in sc.objects if o.type == "CAMERA" or (o.type == "LIGHT" and opt.get("light") != "scene" and not (BATTLE and opt.get("light")))]:
    bpy.data.objects.remove(o)
for m in sc.timeline_markers: m.camera = None                      # a marker with a camera switches the render camera (eternity_gate.blend: shot1_…)
if not opt.get("light"):                                             # the scene's light, smoke, beams and shield (sketch_fx, components/battle/bl_battle.py) show as patches in Workbench
    for o in sc.objects:
        if o.get("sketch_fx"): o.hide_render = True
ep = os.path.join(SD, "edits.json")
for e in (json.load(open(ep)) if os.path.exists(ep) else []):        # plan edits — the same as in scene.obj
    ob = bpy.data.objects.get(e["id"])
    if ob is None: print("edit: no object", e["id"]); continue
    if e["op"] == "turn":
        p = Vector((e["pivot"][0], e["pivot"][1], 0.0))
        ob.matrix_world = Matrix.Translation(p) @ Matrix.Rotation(math.radians(e["deg"]), 4, "Z") @ Matrix.Translation(-p) @ ob.matrix_world
    else:
        ob.matrix_world = Matrix.Translation(Vector(e["d"])) @ ob.matrix_world
PART = {"tree": (0.30, 0.22, 0.15), "banner": (0.50, 0.09, 0.07), "rock": (0.36, 0.35, 0.33)}   # bark, cloth, boulder (bl_boulder); ridge and the rest use the ground color
base = set(json.load(open(os.path.join(SD, "source.json"))).get("ids", []))
add = {e["id"]: e["kind"] for e in json.load(open(os.path.join(SD, "world.json")))["reg"] if base and e["id"] not in base}
if add:                                                             # parts added by append_parts (trees, ridges, banners): they are not in the .blend — taken from scene.obj
    for pid, (V, F) in B.appended_parts(SD, base).items():          # through the core's shared reader (the maps from the .blend use it as well — bl_scene_blend)
        me = bpy.data.meshes.new(pid); me.from_pydata(V, [], F); me.shade_smooth()
        m = bpy.data.materials.new(pid); m.diffuse_color = (*PART.get(add[pid], GROUND), 1.0); me.materials.append(m)
        sc.collection.objects.link(bpy.data.objects.new(pid, me))
gj = os.path.join(ROOT, "out", "ground", os.path.splitext(os.path.basename(src))[0] + ".json")   # map ground the game's way (ground_tex.py)
jb = os.path.join(os.path.dirname(src), "job.json")
if os.path.exists(jb) and (not os.path.exists(gj) or os.path.getmtime(gj) < max(os.path.getmtime(jb), os.path.getmtime(core.paths.script("ground_tex.py")))):
    subprocess.run([os.path.join(ROOT, ".venv/bin/python"), *core.paths.python_target(core.paths.script("ground_tex.py")), src], capture_output=True, cwd=ROOT)
ter = bpy.data.objects.get("terrain")
G = json.load(open(gj)) if os.path.exists(gj) else {}
if G.get("cells") and ter is not None and ter.type == "MESH" and "ground" not in opt:   # ground= — a flat fill instead of a texture
    me = ter.data
    co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)   # vertex = to_bl(cell) of bl_piece
    u = (G["center"][0] + G["hw"] - co[:, 0] - G["gx0"]) / G["cells"]; v = 1 - (G["center"][1] + G["hh"] - co[:, 1] - G["gz0"]) / G["cells"]
    li = np.empty(len(me.loops), np.int32); me.loops.foreach_get("vertex_index", li)
    uv = me.uv_layers.new(name="ground"); uv.data.foreach_set("uv", np.stack([u[li], v[li]], 1).ravel().astype(np.float32))
    tm = bpy.data.materials.new("ground_tex"); tm.use_nodes = True; nt = tm.node_tree
    tx = nt.nodes.new("ShaderNodeTexImage"); tx.image = bpy.data.images.load(gj[:-5] + ".png")
    nt.links.new(tx.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"]); nt.nodes.active = tx   # Workbench takes the active image
    me.materials.clear(); me.materials.append(tm)
gm = bpy.data.materials.new("ground_color"); gm.diffuse_color = (*GROUND, 1.0)
if ter is not None and ter.type == "MESH" and ter.data.materials and ter.data.materials[0].name.startswith("ground_tex") and "mean_linear" in G: gm.diffuse_color = (*G["mean_linear"], 1.0)   # the far ground matches the texture's tone
far = B.add_far_ground(list(sc.objects))                            # as in the pipeline passes: no sky at the seam between a map piece and a ridge
for ob in list(sc.objects):                                         # terrain without a material and the far ground take the ground color
    if ob.type == "MESH" and (ob is far or ob.name.split(".")[0] in ("terrain", "ground")) and not ob.data.materials: ob.data.materials.append(gm)
if opt.get("light") and opt.get("near", "1") != "0" and ter is not None and os.path.exists(os.path.join(SD, "surface.json")) \
        and os.path.exists(os.path.join(SD, "relief_w.npz")):          # site (terrain.py scene): a ground patch at the frame's cameras
    from core.surface import bl_terrain_near as TN
    from core.surface.ground import Ground
    from core import hooks                                          # small details at the cameras (grass, pebbles) — the frame_near extension point: Site
    G_ = Ground(SD)
    tc_ = [B.make_camera(cams[k - 1], f"near_cam{k}", sc) for k in ks]
    print("ground nearby:", TN.build(ter, G_, tc_, sc=sc), *[f"{c}: {r}" for c, r in hooks.call("frame_near", sc, SD, G_, tc_)], flush=True)
    for o in tc_: bpy.data.objects.remove(o)
from core import hooks                                              # materials and weather of the frame — the frame_materials extension point (Materials:
for _c, MPLAN in hooks.call("frame_materials", sc, SD, [cams[k - 1] for k in ks], opt, OUT, strict=True):   # materials.yaml, command-line keys,
    if not MPLAN: continue                                          # weather=rain|snow|… is a preset; weather=clear|overcast|… affects only the far sky)
    if MPLAN.get("sky") and opt.get("weather") in (None, MPLAN.get("weather")):
        opt["weather"] = MPLAN["sky"]                               # the far sky follows the materials' weather unless set explicitly
    if not opt.get("light"):                                         # Workbench previs: rain streaks (sketch_fx) are not drawn
        for o in sc.objects:
            if o.get("sketch_fx"): o.hide_render = True
if "frame" in opt: sc.frame_set(int(opt["frame"]))                   # clip frame: drops, ripples, state keys (the far clouds follow the same key)
LIGHT = opt.get("light")                                            # color with light (Eevee): preset + sky + fire → lit.png
from core.light.far import node_light                                          # a node time (far.TIME) or a preset — the time (and weather) of the scene's light node
if node_light(LIGHT) and not opt.get("hdri"):                       # (one light-and-sky logic: far.py); the run node from frames.py — far=<node.json>
    opt.setdefault("time", node_light(LIGHT)[0]); LIGHT, w_ = "far", node_light(LIGHT)[1]   # (far here means the far ground); the weather — only
    if w_: opt.setdefault("weather", w_)                             # "overcast" and "blaze", otherwise the scene's own
FAR, SFX = None, []                                                 # the light-and-sky node exists for every scene (far.yaml or from world.json): the frame
if LIGHT != "scene" and not opt.get("hdri"):                        # with light and the previs use its sky; light=scene and HDRI bring their own
    from core.light import bl_far
    fp = opt.get("far") or os.path.join(OUT, "far_plan.json")
    if not opt.get("far"):                                         # the plan comes from the site description and the command-line keys (time, weather, style)
        kv = [f"{k}={opt[k]}" for k in ("time", "weather", "style", "frame", "sky_image") if k in opt]
        r_ = subprocess.run([os.path.join(ROOT, ".venv/bin/python"), "-m", "core.light.far", "resolve", SD, *kv, f"out={fp}"], capture_output=True, text=True)
        if r_.returncode: raise SystemExit("far.py resolve: " + r_.stderr[-600:])
    FAR = json.load(open(fp)); FAR["path"] = fp
    sc.world = bl_far.build_world(FAR)                              # the plan's sky first: the effects' haze needs the horizon table
    bl_far.render_horizon(sc, FAR, os.path.splitext(fp)[0] + "_horizon.json")
    bl_far.attach_terrain(sc, FAR, far)                              # the plan's far terrain: mountains up to 38 km instead of flat ground
    bl_far.attach_city(sc, FAR)                                      # far city: instances, windows at night
    for o in bl_far.attach_clouds(sc, FAR) + bl_far.attach_fires(sc, FAR):   # clouds, smoke and fire: in a lit frame they are drawn by the same render;
        if o.type != "LIGHT": o.hide_render = LIGHT != "far"            # in the Workbench previs
                                                                    # they are absent (it would draw planes) — only the render_fx effects layer
PRESET = {"day": (55, 140, (1.0, 0.97, 0.92), 4.0, 1.0, (0.55, 0.65, 0.85)), "sunset": (8, 250, (1.0, 0.50, 0.25), 6.0, 0.45, (0.80, 0.52, 0.40)),
          "overcast": (40, 140, (1.0, 1.0, 1.0), 0.0, 1.4, (0.62, 0.64, 0.68)), "night": (35, 200, (0.55, 0.65, 1.0), 0.9, 0.18, (0.03, 0.04, 0.08)),
          "blaze": (20, 200, (0.55, 0.65, 1.0), 0.45, 0.10, (0.06, 0.03, 0.02))}   # elevation°, azimuth°, sun color, strength, sky strength, sky color; at night the moon: forms are readable even without fire
FIRE = ("burning", "fire", "brazier", "torch", "flame", "candle")


def scene_fires(k=1.0):
    """Scene fires as lamps: the points "fire=x,y,z;…" and burning objects (barrels, braziers, torches) — each is a fire by itself. Returns the number of fires.
    k — the node's night correction (far.py fire_k): the fire is under its own exposure, the moonlit surroundings stay readable."""
    fires = [Vector(map(float, f.split(","))) for f in opt.get("fire", "").split(";") if f]
    for ob in list(sc.objects):
        if ob.type == "MESH" and any(t in ob.name.lower() for t in FIRE):
            bb = [ob.matrix_world @ Vector(c) for c in ob.bound_box]; fires.append(Vector((sum(q.x for q in bb) / 8, sum(q.y for q in bb) / 8, max(q.z for q in bb) + 0.4)))
    for i, fp in enumerate(fires):
        fl = bpy.data.lights.new(f"fire{i}", "POINT"); fl.energy, fl.color, fl.shadow_soft_size = float(opt.get("fire_power", 1500)) * k, (1.0, 0.45, 0.12), 0.3
        fo = bpy.data.objects.new(f"fire{i}", fl); sc.collection.objects.link(fo); fo.location = fp
    return len(fires)


def base(m, rgb):
    """The material color for Eevee too: in Blender 5 new materials always have a node tree, and Eevee does not see the viewport color."""
    m.diffuse_color = (*rgb[:3], 1.0)
    bs = next((n for n in (m.node_tree.nodes if m.node_tree else []) if n.type == "BSDF_PRINCIPLED"), None)
    if bs and not bs.inputs["Base Color"].is_linked: bs.inputs["Base Color"].default_value = (*rgb[:3], 1.0)


if LIGHT:
    for m in bpy.data.materials:
        if m.name.startswith(("ground_color", *PART)) or m.name in add: base(m, m.diffuse_color)
if LIGHT == "scene":                                                # the scene's own light, sky, smoke and Eevee settings as placed (components/battle/bl_battle.py)
    sc.render.engine = "BLENDER_EEVEE"
    if "samples" in opt: sc.eevee.taa_render_samples = int(opt["samples"])
    print(f"scene light: lamps {sum(o.type == 'LIGHT' for o in sc.objects)}, effects {sum(bool(o.get('sketch_fx')) for o in sc.objects)}", flush=True)
elif LIGHT == "far":                                               # light and sky of the far-terrain plan: the far_sky world, sun and moon lamps, AgX with the plan's exposure
    sc.render.engine = "BLENDER_EEVEE"; sc.eevee.taa_render_samples = int(opt.get("samples", 16))   # 16 for sheets, the reference and the take; 32 by key
    ev_s = sc.view_settings.exposure; bl_far.attach(sc, FAR)         # (the scene's exposure comes before the node: the author set the emission for it)
    if BATTLE:                                                      # battle scene: its lamps stay under the node, its smoke is the node's, effects go as a layer on top
        k_ev = 2.0 ** (ev_s - FAR["light"]["ev"]); nf = bl_far.scene_light(sc, FAR, k_ev)   # hazes (render_colour); lamps and emission —
        bl_far.scene_emission(k_ev); bl_far.glow_fill(sc, FAR); bl_far.moon_fill(sc, FAR)   # adapted to the node's exposure; the glow and
        # moon fills — only for a scene with its own light: for the others the node's sky provides the fill (with fills the shadow at sites became equal to the moonlit side,
        # and the mountain pass with 65 fires turned into daytime — 0.55 brightness)
        SFX = bl_far.scene_fx(sc); bl_far.fx_fog(SFX, FAR)          # adapted to the node's exposure, the build's blasts —
        for o in SFX: o.hide_render = True                          # as node flashes by frame (glow in the haze, light on the underside of clouds)
        bd_ = os.path.dirname(src); ev_ = os.path.join(bd_, "data", "events.json"); bj_ = os.path.join(bd_, "battle_job.json")
        if os.path.exists(ev_) and not FAR.get("events"):
            from core.light.far import battle_events
            FAR["fps"] = float((json.load(open(bj_))["spec"].get("anim_cfg", {}) if os.path.exists(bj_) else {}).get("fps", 24))
            FAR["events"] = battle_events(ev_, FAR["fps"], FAR["fires"].get("light_k", 1.0))
            json.dump({k: v for k, v in FAR.items() if k != "path"}, open(FAR["path"] + ".tmp", "w")); os.replace(FAR["path"] + ".tmp", FAR["path"])
    else: nf = scene_fires(FAR["light"].get("fire_k", 1.0))          # the scene's fires as lamps, as with presets; at night adapted to the fire exposure
    print(f"light far: sun {FAR['sun']['az']}°/{FAR['sun']['el']}°, moon {'present' if FAR['moon']['on'] else 'no'}, exposure {FAR['light']['ev']}, fires {nf}", flush=True)
elif LIGHT:
    # a node time (sunset …) with an HDRI — the old path's preset with the same time
    p_ = PRESET.get(LIGHT)                                          # day, sunset, night are TIME names; any other time is treated as day
    el_, az_, sun_c, sun_s, sky_s, sky_c = p_ or PRESET["day"]
    sc.render.engine = "BLENDER_EEVEE"; sc.eevee.taa_render_samples = int(opt.get("samples", 32))
    w = bpy.data.worlds.new("lit"); sc.world = w; nt = w.node_tree; bg = next(n for n in nt.nodes if n.type == "BACKGROUND")
    hdri = opt.get("hdri")
    if hdri:                                                        # Poly Haven — via the Blender helpers (the sky = the scene's world)
        try:
            core.paths.ext("bl_helpers"); import bl_helpers
            r_ = bl_helpers.polyhaven(hdri, "hdris", opt.get("hdri_res", "1k")); w = sc.world; hdri = None if (isinstance(r_, dict) and r_.get("error")) else hdri
        except Exception as e: print("hdri:", e); hdri = None
    if not hdri:
        bg.inputs["Color"].default_value = (*sky_c, 1.0); bg.inputs["Strength"].default_value = sky_s
    else:                                                           # the HDRI is light only; in the frame — the preset's sky (a foreign city beyond the horizon is not wanted)
        wn = sc.world.node_tree; b_ = next((n for n in wn.nodes if n.type == "BACKGROUND"), None); out_ = next(n for n in wn.nodes if n.type == "OUTPUT_WORLD")
        if b_:
            b_.inputs["Strength"].default_value = sky_s
            bg2 = wn.nodes.new("ShaderNodeBackground"); bg2.inputs["Color"].default_value = (*sky_c, 1.0); bg2.inputs["Strength"].default_value = 1.0
            lp = wn.nodes.new("ShaderNodeLightPath"); mx = wn.nodes.new("ShaderNodeMixShader")
            wn.links.new(lp.outputs["Is Camera Ray"], mx.inputs[0]); wn.links.new(b_.outputs[0], mx.inputs[1]); wn.links.new(bg2.outputs[0], mx.inputs[2])
            wn.links.new(mx.outputs[0], out_.inputs["Surface"])
    if sun_s > 0:
        sl = bpy.data.lights.new("sun", "SUN"); sl.energy, sl.color, sl.angle = sun_s, sun_c, math.radians(1.5)
        so = bpy.data.objects.new("sun", sl); sc.collection.objects.link(so)
        so.rotation_euler = (math.radians(90 - el_), 0, math.radians(az_ + 90))
    nf = scene_fires()
    if opt.get("fog"):                                            # haze: a world volume (Eevee), the density comes from the key
        vs = nt.nodes.new("ShaderNodeVolumeScatter"); vs.inputs["Density"].default_value = float(opt["fog"]); nt.links.new(vs.outputs[0], nt.nodes["World Output"].inputs["Volume"])
    print(f"light {LIGHT}: sun {sun_s}, fires {nf}, sky {'HDRI ' + opt['hdri'] if hdri else 'color'}", flush=True)
d = sc.display.shading
sc.render.engine = "BLENDER_WORKBENCH" if not LIGHT else sc.render.engine; d.light, d.color_type, d.show_cavity = "STUDIO", "TEXTURE", True
d.background_type, d.background_color = "VIEWPORT", SKY
sc.display.shading.show_shadows = True
if FAR and LIGHT != "far": sc.world = bl_far.build_world(FAR)      # the Workbench previs does not draw the world — the world is needed only for the camera sky
if FAR: bl_far.render_horizon(sc, FAR, os.path.splitext(FAR["path"])[0] + "_horizon.json")   # sky at the horizon — the haze color (far.py)


def render_colour(k, spec, out_dir, frame=None, meter=True):
    """Frame color through camera spec into out_dir: with light — lit.png, otherwise the studio color.png and a flat albedo.png. Shared with the take (bl_take.py).
    With a far-terrain plan — also sky.exr (the camera sky) and far.json; light=far — lit.exr (geometry on a transparent background), lit.png is composited by finish_far.
    frame — a clip frame (zero-based; take): the node per frame — cloud drift, flashes, fire flicker (bl_far.set_frame, far.py post).
    meter — the exposure of a static frame from the glow the camera sees (far.py meter_ev); for battle, take and panorama faces — the node's."""
    cam = B.make_camera(spec, f"cam{k}", sc); sc.camera = cam
    if FAR: bl_far.fit_clip(cam, FAR)                               # the far terrain extends to a skirt at 160 km
    if FAR and frame is not None: bl_far.set_frame(sc, FAR, frame)
    W, H = spec["resolution"]; sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = W, H, 100
    p = os.path.join(out_dir, "color.png"); os.makedirs(out_dir, exist_ok=True)
    if not FAR:                                                     # without a far-terrain plan, traces of a previous run with far terrain do not apply
        for f in ("far.json", "fog_t.png"):
            if os.path.exists(os.path.join(out_dir, f)): os.remove(os.path.join(out_dir, f))
    if FAR:
        bl_far.per_camera(sc, spec); bl_far.render_sky(sc, os.path.join(out_dir, "sky.exr"))
        fx = LIGHT != "far" and bl_far.render_fx(sc, os.path.join(out_dir, "fx.exr"))   # previs: clouds as an Eevee layer (the geometry is held out)
        json.dump({"plan": FAR["path"], "hash": FAR["hash"], "ev": FAR["light"]["ev"], "mode": "lit" if LIGHT == "far" else "previs",
                   "sky_bg": list(SKY), "camera": spec, "fx": bool(fx), "frame": frame, "scene_fx": bool(SFX) and LIGHT == "far",
                   "meter": bool(meter and LIGHT == "far" and not BATTLE and frame is None)},
                  open(os.path.join(out_dir, "far.json"), "w"))
        sc.render.film_transparent = True; sc.render.image_settings.color_mode = "RGBA"   # the sky is the backdrop (far.py post, previs.py)
    if LIGHT == "far":                                             # multilayer: color and depth (the hazes work without a map bundle too)
        im = sc.render.image_settings; vl = sc.view_layers[0]; vl.use_pass_z = True
        im.media_type, im.file_format, im.color_depth = "MULTI_LAYER_IMAGE", "OPEN_EXR_MULTILAYER", "32"
        B.render_to(sc, os.path.join(out_dir, "lit.exr")); vl.use_pass_z = False
        im.media_type, im.file_format, im.color_depth = "IMAGE", "PNG", "8"
        if SFX: bl_far.render_scene_fx(sc, SFX, os.path.join(out_dir, "scene_fx.exr"))   # the battle scene's effects — as a layer over the haze
        return
    if LIGHT: B.render_to(sc, os.path.join(out_dir, "lit.png")); return   # the lit frame has its own file, the approved previs does not change
    far_obs = [o for o in sc.objects if o.name.startswith(("far_terrain", "far_city")) and not o.hide_render]
    for light, q in (("STUDIO", p), ("FLAT", os.path.join(out_dir, "albedo.png"))):   # studio shading gives volume; a flat color —
        d.light = light                                                              # with a low camera the floor does not go black
        bl_far.two_pass(sc, B.render_to, q, far_obs) if far_obs and d.show_shadows else B.render_to(sc, q)   # the far terrain has no Workbench shadows




def finish_far(dirs):
    """The far-terrain plan's light: lit.exr + sky.exr → lit.png (core/light/far.py post, AgX and the plan's exposure)."""
    if LIGHT != "far" or not dirs: return
    r_ = subprocess.run([os.path.join(ROOT, ".venv/bin/python"), "-m", "core.light.far", "post", *dirs], capture_output=True, text=True)
    if r_.returncode: raise SystemExit("far.py post: " + r_.stderr[-600:])


if __name__ == "__main__":                                          # the take (components/time/bl_take.py) uses the scene preparation and render_colour
    done = []
    for k in ks:
        render_colour(k, cams[k - 1], os.path.join(OUT, f"cam{k}"), frame=int(opt["frame"]) - 1 if "frame" in opt else None); done.append(k)   # frame= —
        # the node for this frame (zero-based): flashes of the battle scene's blasts, cloud drift
    finish_far([os.path.join(OUT, f"cam{k}") for k in done])
    print("__RESULT__" + json.dumps({"color": done, "blend": os.path.basename(src)}, ensure_ascii=False))

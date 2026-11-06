"""Map passes straight from the .blend of a large scene at key frames: GN crowds, heroes
with clips, vehicles with keys, turrets move by the scene's own code — at frame t the same passes as in core/outputs/bl_pack.py are captured
(previs color, part numbers, body parts, depth and normals, tone) and files are written for core/outputs/pack.py and core/outputs/previs.py. The OBJ bridge
is not needed: crowd instances and beehive cities are not exported through it, and the scene itself in OBJ is 300 MB.
blender -b --factory-startup <scene.blend> -P core/blender/run.py -- core/outputs/bl_pack_blend.py -- <output folder> <frames: 1,7,13 | a-b/step> [res=1280x720]
    [camera=<name> | shot=<n_shots.json file>:<name> — the camera from a shot description (solver components/camera/battle_cam.py); otherwise — by markers]
→ <output>/f<frame>/raw/{id.png, part.png, passes.exr, beauty.exr, objects.json, camera.json}, f<frame>/{color.png, albedo.png, joints.json}
A part is a root with the sketch_kind tag (its whole branch gets one number); a GN instance creator (crowd, beehive city, spires) is its own part:
Workbench paints the instance with the creator's color.
Effects (sketch_fx) and the scene's lamps are hidden: the passes' light is the sun and sky, as in bl_pack. The reply is __RESULT__ with the time.
far=<node.json> — the light-and-sky node (battle_frames.py puts it in the keys folder): far mountains and city in the passes, the sun and sky from the node,
the previs on a transparent background with sky.exr and far.json (previs.py lays the node's sky and haze by depth underneath)."""
import json, math, os, sys, time
import bpy
from mathutils import Vector
from core.blender import bl_scene as B
from core.blender import bl_kit as CR
from core.contract import classes as CL
from core.assets.rig_joints import joints_of, rig_kind

t0 = time.time(); a = sys.argv[sys.argv.index("--") + 1:]; OUT = os.path.abspath(a[0]); opt = dict(x.split("=", 1) for x in a[2:])
fs = a[1]
if "-" in fs:
    ab, st = (fs.split("/") + ["1"])[:2]; lo_, hi_ = map(int, ab.split("-")); FR = list(range(lo_, hi_ + 1, int(st)))
else: FR = [int(x) for x in fs.split(",") if x]
W, H = map(int, opt.get("res", "1280x720").split("x")); sc = bpy.context.scene
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = W, H, 100
vs = sc.view_settings; vs.view_transform, vs.look, vs.exposure, vs.gamma = "Standard", "None", 0.0, 1.0   # the battle scene has AgX:
sc.display_settings.display_device = "sRGB"                         # it recolored the id color numbers
if opt.get("camera"): sc.camera = bpy.data.objects[opt["camera"]]
elif opt.get("shot"):                                                   # a shot from the solver's file: the camera from the description, as in the scene (with shake),
    sf_, sn_ = opt["shot"].rsplit(":", 1); sh_ = next(s_ for s_ in json.load(open(sf_)) if s_["name"] == sn_)   # the file's markers are ignored
    for m_ in sc.timeline_markers: m_.camera = None
    tc_ = bpy.data.collections.new("_shot_from_file"); sc.collection.children.link(tc_); sc.camera = CR.shot_camera(sc, tc_, sh_)
    for m_ in [m_ for m_ in sc.timeline_markers if m_.name == sn_]: sc.timeline_markers.remove(m_)
else: CR.bind_shot_markers(sc)                                         # the shot marker → its camera (not bound in the file)

COLL_KIND = {"terrain": "ground", "fortress": "wall", "palace": "tower", "city": "wall", "props": "box", "vehicles": "siege_tower", "army": "crowd"}


def root(o):
    while o.parent is not None: o = o.parent
    return o


def kind_of(o):
    r = root(o)
    k = r.get("sketch_kind") or o.get("sketch_kind")
    if k: return k
    if r.name == "terrain" or o.name == "terrain": return "ground"
    for c in o.users_collection:
        if c.name in COLL_KIND: return COLL_KIND[c.name]
    return "box"


FARP = None                                                            # the light-and-sky node (battle_frames: node.json of the keys folder; core/light/far.py):
if opt.get("far"):                                                    # far mountains and city are in the passes (classes ground, tower), the sun and sky
    from core.light import far
    from core.light import bl_far                                                 # of the passes come from the node (the moon at night), the previs is on a transparent background with the sky
    FARP = json.load(open(opt["far"])); FARP["path"] = opt["far"]   # of the node (sky.exr, far.json → previs.py)
    for o_ in bl_far.attach_terrain(sc, FARP, None): o_["sketch_kind"] = "ground"
    bl_far.attach_city(sc, FARP)                                       # the city is an instance creator with class tower
    bpy.context.view_layer.update()                                    # the layer's list (INVL below) — with the new city and without its prototypes

# --- effects and lamps are removed; parts and numbers
for o in sc.objects:
    if o.get("sketch_fx") or root(o).get("sketch_fx") or o.type == "LIGHT": o.hide_render = True
parts = {}                                                             # root → [meshes]
INVL = {o.name for o in bpy.context.view_layer.objects}               # not in excluded collections (prototypes handled separately)
REND = ("MESH", "CURVE", "SURFACE", "FONT", "META")
inst = {}                                                              # GN instance creator → prototype collection
for o in sc.objects:
    for m in o.modifiers:
        if m.type == "NODES" and m.node_group:
            c_ = next((n.inputs["Collection"].default_value for n in m.node_group.nodes
                       if n.bl_idname == "GeometryNodeCollectionInfo" and n.inputs["Collection"].default_value), None)
            if c_: inst[o.name] = c_
for o in sc.objects:                                                   # everything that will get into the render: objects hidden only in the viewport (hide_set) —
    if o.type not in REND or o.hide_render or o.name not in INVL: continue   # are rendered too, otherwise there are holes in the id
    parts.setdefault(o if o.name in inst else root(o), []).append(o)   # an instance creator is its own part: Workbench paints the instance
table, k = {}, 1                                                       # with the creator's color, not the prototype's (crowds and spires used to be white)
for r, ms in parts.items():
    kd = kind_of(r); cls = CL.class_of(kd); pr = [x for x in inst[r.name].all_objects if x.type in REND] if r.name in inst else []
    for o in ms + pr: o.color = B.id_color(k)
    if r.name in inst: h = 1.9 if kd == "crowd" else 30.0
    else:
        zs = [(o.matrix_world @ Vector(c)).z for o in ms for c in o.bound_box]; h = max(zs) - min(zs)
    table[k] = {"name": r.name, "kind": kd, "class": cls, "id": r.name, "height_m": round(h, 3), "member": False, "line": r.name, "_ms": ms + pr}
    k += 1
tab_json = {n: {kk: vv for kk, vv in v.items() if kk != "_ms"} for n, v in table.items()}
heroes = [r for r in parts if r.get("sketch_kind") == "figure"]
print(f"parts {len(parts)}, instance creators {sum(r.name in inst for r in parts)}, numbers {len(table)}, heroes {len(heroes)}", flush=True)

# --- pass materials: body parts (all are part0), tone by class
pm = B.part_materials(); tone = {}
for n, v in table.items():
    tone.setdefault(v["class"], B.tone_material(CL.ALBEDO.get(v["class"], 0.5)))
SW, SKY = {"az": 220, "el": 32}, (0.78, 0.94)
if FARP:
    S_ = far.sketch_world(FARP); SW, SKY = {"az": S_["sun"]["azimuth"], "el": S_["sun"]["elevation"]}, (S_["sky"]["top"], S_["sky"]["horizon"])


def spec_of(cam):
    ev = cam.evaluated_get(bpy.context.evaluated_depsgraph_get()); cd = ev.data
    return {"name": cam.name, "lens_mm": round(cd.lens, 4), "sensor_width_mm": cd.sensor_width, "sensor_height_mm": cd.sensor_height,
            "sensor_fit": cd.sensor_fit, "shift_x": cd.shift_x, "shift_y": cd.shift_y, "resolution": [W, H],
            "matrix_world": [list(map(float, row)) for row in ev.matrix_world], "clip": [cd.clip_start, cd.clip_end]}


def hero_joints():
    J = {}
    for r in heroes:
        arm = next((o for o in [r, *r.children_recursive] if o.type == "ARMATURE" and rig_kind(o)), None)
        if arm is None: continue
        ms = [o for o in r.children_recursive if o.type == "MESH" and not o.hide_render]
        zs = [(o.matrix_world @ Vector(c)).z for o in ms for c in o.bound_box] or [0, 1.8]
        h = max(zs) - min(zs); jj, hd, mir = joints_of(arm, rig_kind(arm), h)
        J[r.name] = {"joints": jj, "heading": hd, "height": round(h, 3), "pos": [round(sum(v[0] for v in jj.values()) / len(jj), 3),
                     round(sum(v[1] for v in jj.values()) / len(jj), 3), round(min(v[2] for v in jj.values()), 3)], "mirrored": mir}
    return J


orig = {}                                                              # the meshes' original materials — restore before the next frame's color
for v in table.values():
    for o in v["_ms"]:
        if o.data.name not in orig: orig[o.data.name] = (o.data, list(o.data.materials))


def set_mats(fn):
    for v in table.values():
        for o in v["_ms"]:
            me = o.data
            for i in range(len(me.materials)): me.materials[i] = fn(v)


T = []
w_old = sc.world; B.add_sun(sc, SW["az"], SW["el"], strength=3.0, name="pack_sun"); sky_w = B.sky_world(sc, *SKY); sc.world = w_old
FAR_OBS = []
if FARP:                                                               # the node's world (the camera sky for the previs) and the horizon table (the previs.py haze) — once
    NW = bl_far.build_world(FARP); sc.world = NW; bl_far.render_horizon(sc, FARP, os.path.splitext(FARP["path"])[0] + "_horizon.json"); sc.world = w_old
    FAR_OBS = [o for o in sc.objects if o.name.startswith(("far_terrain", "far_city")) and not o.hide_render]
for f in FR:
    tf = time.time(); sc.frame_set(f)
    d = os.path.join(OUT, f"f{f:04d}"); raw = os.path.join(d, "raw"); os.makedirs(raw, exist_ok=True)
    cam = sc.camera
    if FARP: bl_far.fit_clip(cam, FARP)                                # far terrain — up to the skirt (the map depth and the previs see the mountains)
    spec = spec_of(cam)
    json.dump(spec, open(os.path.join(raw, "camera.json"), "w")); json.dump(tab_json, open(os.path.join(raw, "objects.json"), "w"), ensure_ascii=False)
    json.dump(hero_joints(), open(os.path.join(d, "joints.json"), "w"))
    # 1. previs color — the scene's own materials, Workbench, as in bl_color.py (studio and flat)
    for me, ms in orig.values():
        for i, m_ in enumerate(ms):
            if i < len(me.materials): me.materials[i] = m_
    sc.render.engine = "BLENDER_WORKBENCH"; sh = sc.display.shading; sc.display.render_aa = "8"
    sh.light, sh.color_type, sh.show_cavity, sh.show_shadows = "STUDIO", "TEXTURE", True, True
    sh.background_type, sh.background_color = "VIEWPORT", (0.80, 0.84, 0.90)
    if FARP: sc.render.film_transparent = True                         # with a node — on a transparent background: previs.py puts the node's sky underneath
    for light, q in (("STUDIO", "color.png"), ("FLAT", "albedo.png")):
        sh.light = light; q = os.path.join(d, q)
        bl_far.two_pass(sc, B.render_to, q, FAR_OBS) if FAR_OBS else B.render_to(sc, q)   # the far terrain has no Workbench shadows (streaks)
    if FARP:
        sc.render.film_transparent = False; sc.world = NW; bl_far.per_camera(sc, spec); bl_far.render_sky(sc, os.path.join(d, "sky.exr")); sc.world = w_old
        json.dump({"plan": FARP["path"], "hash": FARP["hash"], "ev": FARP["light"]["ev"], "mode": "previs", "sky_bg": [0.80, 0.84, 0.90],
                   "camera": spec, "fx": False, "frame": f - 1}, open(os.path.join(d, "far.json"), "w"))
    sh.show_cavity = sh.show_shadows = False
    # 2. part numbers
    B.flat_workbench(sc); B.render_to(sc, os.path.join(raw, "id.png"))
    # 3. body parts — a battle scene has no part templates: everything is part0
    set_mats(lambda v: pm[0]); B.flat_workbench(sc, color_type="MATERIAL"); B.render_to(sc, os.path.join(raw, "part.png"))
    # 4. Eevee: tone by class, the sun and sky; depth and normals — one sample, tone — 16
    set_mats(lambda v: tone[v["class"]])
    sc.world = sky_w; B.look_eevee(sc)
    vl = sc.view_layers[0]; vl.use_pass_z = vl.use_pass_normal = True; im = sc.render.image_settings
    im.media_type, im.file_format, im.color_depth = "MULTI_LAYER_IMAGE", "OPEN_EXR_MULTILAYER", "32"
    sc.render.use_motion_blur = False; sc.eevee.taa_render_samples = 1; B.render_to(sc, os.path.join(raw, "passes.exr"))
    vl.use_pass_z = vl.use_pass_normal = False; sc.eevee.taa_render_samples = 16; B.render_to(sc, os.path.join(raw, "beauty.exr"))
    im.media_type, im.file_format, im.color_depth = "IMAGE", "PNG", "8"; sc.world = w_old
    T.append(round(time.time() - tf, 1)); sys.__stdout__.write(f"__FRAME__ {raw} {T[-1]} s\n"); sys.__stdout__.flush()
print("__RESULT__" + json.dumps({"frames": FR, "s_per_frame": T, "parts": len(parts), "instancers": sum(r.name in inst for r in parts), "heroes": len(heroes),
                                 "s": round(time.time() - t0, 1)}, ensure_ascii=False))

"""Shared functions for Blender scripts: a sketch25 scene (scene.obj + cameras.json) → a bpy scene, a camera, render modes.
Import: sys.path.insert(0, <tools folder>); import bl_scene as B. Works only inside blender."""
import json
import math
import os

import bpy
from mathutils import Matrix

SKY_ID = 0


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


def import_obj(path):
    """sketch25 OBJ: meters, Z up, Forward Y — no axis rotation."""
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=path, forward_axis="Y", up_axis="Z")
    return [o for o in bpy.data.objects if o not in before]


def add_far_ground(objs, reach=25000.0, name="ground_far", z=None):   # the corners (R·√2) are closer than the 40 km camera clip: otherwise Workbench shadows give a false strip
    """The ground of the old engine is infinite (far=30000 m), while the OBJ has only strips around the scene. The far ground is a ring
    from the bounds of the `ground*` objects out to `reach`, cut out along those bounds: no overlap, otherwise the depth conflicts.
    The level is z, or the far_z property of the ground (a planned site, components/site/terrain.py: the relief edge is blended to it), otherwise the most frequent height."""
    from collections import Counter
    import numpy as np
    P = []
    for ob in objs:                           # ground* comes from the pipeline OBJ, terrain — from the library .blend (core/outputs/bl_color.py)
        if ob.name.startswith(("ground", "terrain")) and ob.type == "MESH":
            co = np.empty(len(ob.data.vertices) * 3); ob.data.vertices.foreach_get("co", co); M = np.array(ob.matrix_world)
            P.append(co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3])   # in world coordinates: in the .blend the terrain has its own scale
    P = np.concatenate(P); xs, ys = P[:, 0], P[:, 1]
    x0, x1, y0, y1 = float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())
    if z is None: z = next((float(ob["far_z"]) for ob in objs if ob.type == "MESH" and ob.name.startswith(("ground", "terrain")) and "far_z" in ob), None)
    if z is None: z = Counter(np.round(P[:, 2], 1).tolist()).most_common(1)[0][0]   # the scene's ground level: scenes from the library do not sit at zero (urban_squad — 255 m)
    x0, x1, y0, y1, z = x0 + 0.5, x1 - 0.5, y0 + 0.5, y1 - 0.5, z - 0.02   # tuck under the ground edge: no gap at the seam (a dotted line of sky)
    R = reach
    verts = [(-R, -R, z), (R, -R, z), (R, R, z), (-R, R, z), (x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
    faces = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def make_camera(spec, name="cam", scene=None):
    """A camera from a cameras.json entry: matrix_world, focal length on 36 mm, sensor_fit AUTO, shift, frame size."""
    scene = scene or bpy.context.scene
    cd = bpy.data.cameras.new(name)
    cd.lens = spec["lens_mm"]
    cd.sensor_width = spec.get("sensor_width_mm", 36.0)
    cd.sensor_height = spec.get("sensor_height_mm", 24.0)
    cd.sensor_fit = spec.get("sensor_fit", "AUTO")
    cd.shift_x = spec.get("shift_x", 0.0)
    cd.shift_y = spec.get("shift_y", 0.0)
    cd.clip_start, cd.clip_end = 0.05, 40000.0
    ob = bpy.data.objects.new(name, cd)
    scene.collection.objects.link(ob)
    ob.matrix_world = Matrix(spec["matrix_world"])
    scene.camera = ob
    scene.render.resolution_x, scene.render.resolution_y = spec["resolution"]
    scene.render.resolution_percentage = 100
    return ob


def load_cameras(scene_dir):
    return json.load(open(os.path.join(scene_dir, "cameras.json")))


def srgb8_to_linear(c8):
    c = c8 / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def flat_workbench(scene, bg=(0.0, 0.0, 0.0), color_type="OBJECT"):
    """Workbench: a flat object color with no lighting and no smoothing — for masks where the color = a number."""
    scene.render.engine = "BLENDER_WORKBENCH"
    d = scene.display
    d.render_aa = "OFF"
    d.shading.light = "FLAT"
    d.shading.color_type = color_type
    d.shading.show_object_outline = False
    d.shading.show_specular_highlight = False
    d.shading.show_shadows = False
    d.shading.show_cavity = False
    d.shading.background_type = "WORLD"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.display_settings.display_device = "sRGB"
    scene.render.dither_intensity = 0.0
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    if scene.world is None:
        scene.world = bpy.data.worlds.new("w")
    scene.world.use_nodes = False
    scene.world.color = bg


def id_color(k):
    """Number → color: R = k & 255, G = (k >> 8) & 255, B = k >> 16; after sRGB encoding into a PNG the number is read back exactly."""
    return (srgb8_to_linear(k & 255), srgb8_to_linear((k >> 8) & 255), srgb8_to_linear((k >> 16) & 255), 1.0)


def paint_ids(objs, start=1):
    """Each object gets a color by number. Returns {name: number}."""
    table = {}
    k = start
    for ob in objs:
        if ob.type != "MESH":
            continue
        ob.color = id_color(k)
        table[ob.name] = k
        k += 1
    return table


def render_to(scene, path):
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


# ------------------------------------------------------------------------------------------------ light, materials, world
_MATS = {}


def tone_material(tone, name=None):
    """A gray material with tone 0..1 (sRGB), identical for Workbench (viewport) and Eevee (Principled, roughness 1)."""
    key = round(float(tone), 3)
    if key in _MATS:
        return _MATS[key]
    m = bpy.data.materials.new(name or f"tone{key}")
    lin = srgb8_to_linear(key * 255)
    m.diffuse_color = (lin, lin, lin, 1.0)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (lin, lin, lin, 1.0)
    bsdf.inputs["Roughness"].default_value = 1.0
    _MATS[key] = m
    return m


KIND_TONE = {"ground": 0.36, "craters": 0.28, "wall": 0.58, "siege_tower": 0.5, "figure": 0.72, "box": 0.5}


def assign_tones(objs, reg, default=0.55):
    """Tone by object kind (reg from world.json: id → kind). Name in the OBJ: figure__X → id _X, ground_* → ground."""
    kind = {e["id"]: e["kind"] for e in reg}
    for ob in objs:
        if ob.type != "MESH":
            continue
        n = ob.name.split(".")[0]
        if n.startswith("figure_"):
            n = n[len("figure_"):]
        if n.startswith("ground"):
            n = "ground"
        k = kind.get(n, "")
        ob.data.materials.clear()
        ob.data.materials.append(tone_material(KIND_TONE.get(k, default)))


def sun_dir(azimuth_deg, elevation_deg):
    """The direction TO the sun (as in sketch25: azimuth from +x counterclockwise, elevation above the horizon)."""
    from mathutils import Vector
    a, e = math.radians(azimuth_deg), math.radians(elevation_deg)
    return Vector((math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)))


def add_sun(scene, azimuth, elevation, strength=3.0, name="sun"):
    ld = bpy.data.lights.new(name, "SUN")
    ld.energy = strength
    ld.angle = math.radians(0.6)
    ob = bpy.data.objects.new(name, ld)
    scene.collection.objects.link(ob)
    ob.rotation_euler = sun_dir(azimuth, elevation).to_track_quat("Z", "Y").to_euler()
    return ob


def sky_world(scene, top=0.6, horizon=0.88, strength=1.0):
    """A gradient sky as in sketch25: from the horizon tone to the zenith tone by elevation above the horizon (up to 50°)."""
    w = bpy.data.worlds.new("sky")
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    n = nt.nodes
    tc = n.new("ShaderNodeTexCoord")
    sep = n.new("ShaderNodeSeparateXYZ")
    asin = n.new("ShaderNodeMath"); asin.operation = "ARCSINE"
    div = n.new("ShaderNodeMath"); div.operation = "DIVIDE"; div.inputs[1].default_value = math.radians(50)
    cl = n.new("ShaderNodeMath"); cl.operation = "MAXIMUM"; cl.inputs[1].default_value = 0.0
    cl2 = n.new("ShaderNodeMath"); cl2.operation = "MINIMUM"; cl2.inputs[1].default_value = 1.0
    pw = n.new("ShaderNodeMath"); pw.operation = "POWER"; pw.inputs[1].default_value = 0.7
    mix = n.new("ShaderNodeMix"); mix.data_type = "RGBA"
    lt, lh = srgb8_to_linear(top * 255), srgb8_to_linear(horizon * 255)
    mix.inputs[6].default_value = (lh, lh, lh, 1.0)
    mix.inputs[7].default_value = (lt, lt, lt, 1.0)
    bg = n.new("ShaderNodeBackground"); bg.inputs["Strength"].default_value = strength
    out = n.new("ShaderNodeOutputWorld")
    L = nt.links.new
    L(tc.outputs["Generated"], sep.inputs[0])
    L(sep.outputs["Z"], asin.inputs[0])
    L(asin.outputs[0], div.inputs[0])
    L(div.outputs[0], cl.inputs[0]); L(cl.outputs[0], cl2.inputs[0]); L(cl2.outputs[0], pw.inputs[0])
    L(pw.outputs[0], mix.inputs[0])
    L(mix.outputs[2], bg.inputs["Color"])
    L(bg.outputs[0], out.inputs[0])
    scene.world = w
    return w


def look_workbench(scene, bg=0.9):
    """The Workbench working view: material tone, studio light, object outline."""
    scene.render.engine = "BLENDER_WORKBENCH"
    d = scene.display
    d.render_aa = "8"
    d.shading.light = "STUDIO"
    d.shading.color_type = "MATERIAL"
    d.shading.show_object_outline = True
    d.shading.object_outline_color = (0, 0, 0)
    d.shading.show_cavity = False
    d.shading.show_shadows = False
    d.shading.background_type = "VIEWPORT"
    d.shading.background_color = (bg, bg, bg)
    scene.view_settings.view_transform = "Standard"
    scene.render.image_settings.media_type = "IMAGE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"


def look_eevee(scene):
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.use_shadows = True
    scene.view_settings.view_transform = "Standard"
    scene.render.image_settings.media_type = "IMAGE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"


# ------------------------------------------------------------------------------------------------ crowds and body parts
N_PART_SLOTS = 64
_PART_MATS = []


def part_materials():
    """Marker materials of the body parts: color = part number (0 — not a figure)."""
    if not _PART_MATS:
        for i in range(N_PART_SLOTS):
            m = bpy.data.materials.new(f"part{i}")
            m.diffuse_color = id_color(i)
            _PART_MATS.append(m)
    return _PART_MATS


def _smooth_by_angle(ob, deg=40):
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    try:
        bpy.ops.object.shade_smooth_by_angle(angle=math.radians(deg))
    except Exception:
        ob.data.polygons.foreach_set("use_smooth", [True] * len(ob.data.polygons))


def template_meshes(kit, scene):
    """figkit.json templates → meshes (one per template; the crowd members share their data). The body part is the material index."""
    out = {}
    pm = part_materials()
    for key, t in kit["templates"].items():
        m = t["mesh"]
        me = bpy.data.meshes.new("tpl_" + key.replace("|", "_")[:48])
        me.from_pydata(m["v"], [], m["f"])
        me.update()
        for poly, part in zip(me.polygons, m["part"]):
            poly.material_index = min(part, N_PART_SLOTS - 1)
        for i in range(N_PART_SLOTS):
            me.materials.append(pm[i])
        tmp = bpy.data.objects.new("tmp", me)
        scene.collection.objects.link(tmp)
        _smooth_by_angle(tmp)
        scene.collection.objects.unlink(tmp)
        bpy.data.objects.remove(tmp)
        out[key] = me
    return out


def add_crowd(kit, scene):
    """Crowd members are twin objects of the templates: scale = height / archetype height, rotation heading, translation pos."""
    meshes = template_meshes(kit, scene)
    height = {k: t["height"] for k, t in kit["templates"].items()}
    objs = []
    for mb in kit["members"]:
        ob = bpy.data.objects.new(mb["id"], meshes[mb["tpl"]])
        k = mb["height"] / height[mb["tpl"]]
        ob.location = mb["pos"]
        ob.rotation_euler = (0.0, 0.0, math.radians(mb["heading"]))
        ob.scale = (k, k, k)
        ob["member"] = True
        scene.collection.objects.link(ob)
        objs.append(ob)
    return objs


def set_slots(meshes, mats):
    """Replace the materials of all template slots (the face indices are preserved)."""
    for me in meshes:
        for i in range(len(me.materials)):
            me.materials[i] = mats[min(i, len(mats) - 1)]


def load_scene(scene_dir):
    """A pipeline scene from the folder bench/scenes/X (scene.obj, world.json, figkit.json): statics, far ground, crowds as instances.
    Returns (scene, statics, members, objs, world, kit). Shared by the frame passes (bl_pack) and the "eyes" (bl_view)."""
    import json, os
    W = json.load(open(os.path.join(scene_dir, "world.json")))
    kit_path = os.path.join(scene_dir, "figkit.json")
    kit = json.load(open(kit_path)) if os.path.exists(kit_path) else None
    sc = reset()
    statics = import_obj(os.path.join(scene_dir, "scene.obj"))
    statics.append(add_far_ground(statics, z=W.get("far_z")))
    for ob in statics:
        if ob.type == "MESH" and not ob.name.startswith("ground"):
            _smooth_by_angle(ob, 50)      # capsule faces (trees, figures) are smooth, boxes and walls are sharp
    members = add_crowd(kit, sc) if kit else []
    objs = [o for o in statics + members if o.type == "MESH"]
    return sc, statics, members, objs, W, kit


def appended_parts(scene_dir, base_ids):
    """Parts appended to the scene after the bridge (append_parts: trees, boulders, banners, ridges) — they are not among base_ids (the
    .blend assembly ids from source.json): {id: (vertices, faces)} from scene.obj, in the world.json order. One reader for the frame
    color (bl_color) and the maps from the .blend (core/outputs/bl_scene_blend.py) — a single truth about the appended parts."""
    import json, os
    wj = os.path.join(scene_dir, "world.json")
    if not os.path.exists(wj): return {}
    add = [e["id"] for e in json.load(open(wj)).get("reg", []) if e["id"] not in base_ids]
    if not add: return {}
    V, F, cur = [], {k: [] for k in add}, None
    for l in open(os.path.join(scene_dir, "scene.obj")):
        if l.startswith("v "): V.append(tuple(map(float, l.split()[1:4])))
        elif l.startswith("o "): cur = l[2:].strip()
        elif l.startswith("f ") and cur in F: F[cur].append([int(x.split("/")[0]) - 1 for x in l.split()[1:]])
    out = {}
    for pid, faces in F.items():
        used = sorted({i for f in faces for i in f}); rm = {v: j for j, v in enumerate(used)}
        out[pid] = ([V[i] for i in used], [[rm[i] for i in f] for f in faces])
    return out


def append_parts(scene_dir, parts):
    """Append parts to a pipeline scene: parts — [(id prefix, kind, vertices (N, 3), faces)] → "o <prefix>_<n>" in scene.obj,
    {"id", "kind", "gid"} in world.json. References to another scene (bench/scenes/X → another scene's files) are replaced with own
    copies. Returns the ids of the new parts. Shared by components/plants/bl_tree.py and components/site/bl_terrain.py."""
    import json, os
    obj, wj = os.path.join(scene_dir, "scene.obj"), os.path.join(scene_dir, "world.json")
    W = json.load(open(wj)); src = open(obj).read().rstrip("\n")
    gid = max((e["gid"] for e in W["reg"]), default=-1) + 1
    nv = sum(1 for l in src.splitlines() if l.startswith("v ")); out, ids = [src], []
    for prefix, kind, V, F in parts:
        n = sum(1 for e in W["reg"] if e["id"].startswith(prefix + "_")); pid = f"{prefix}_{n}"
        out.append(f"o {pid}\n" + "".join(f"v {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in V)
                   + "".join("f " + " ".join(str(nv + j + 1) for j in f) + "\n" for f in F))
        nv += len(V); W["reg"].append({"id": pid, "kind": kind, "gid": gid}); gid += 1; ids.append(pid)
    for p in (obj, wj):
        if os.path.islink(p): os.remove(p)
    open(obj, "w").write("\n".join(out) + "\n"); json.dump(W, open(wj, "w"), ensure_ascii=False)
    return ids

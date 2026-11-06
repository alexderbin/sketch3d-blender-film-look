"""A scene for maps straight from the build's .blend ("one scene truth"): the same as
tools/bl_scene.load_scene from the gray copy (scene.obj + world.json + figkit.json), but the parts come from the scene model's .blend in memory
(bl_parts.collect — the same partition as the bridge's), the plan edits edits.json are applied to objects, as in bl_color. Returns the same
tuple (scene, statics, members, objs, world, kit), so the map passes (bl_pack) do not change.
Transitional (until parts are assembled in the .blend): the parts appended to the gray copy (append_parts: trees, boulders, banners, ridges) come
from scene.obj, as in bl_color. Blender only."""
import json, math, os, sys
import core.paths

import bpy
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
from core.outputs import bl_parts  # noqa: E402
from core.model import scene_model  # noqa: E402

FRESH = {}                                                              # the last load: named, heights, reg from the .blend


def blend_of(scene_dir):
    """The scene build's .blend if it is in place; for battle scenes (crowds as instances, their own time) — None: their path is bl_pack_blend."""
    M = scene_model.load(scene_dir)
    b = M["build"]["blend"]
    if not b or not os.path.exists(b) or os.path.exists(os.path.join(scene_dir, "battle.yaml")): return None
    if os.path.exists(os.path.join(core.paths.scene_dir(os.path.splitext(os.path.basename(b))[0]), "battle.yaml")): return None   # a copy of a battle scene
    return b                                                            # (eternity_gate_4k, _fly: the assault's .blend, the gray copy is its own)


def apply_edits(scene_dir):
    """Plan edits (edits.json, core/model/scene_edit.py) applied to .blend objects — as in bl_color."""
    ep = os.path.join(scene_dir, "edits.json")
    for e in (json.load(open(ep)) if os.path.exists(ep) else []):
        ob = bpy.data.objects.get(e["id"])
        if ob is None: print("edit: no object", e["id"]); continue
        if e["op"] == "turn":
            p = Vector((e["pivot"][0], e["pivot"][1], 0.0))
            ob.matrix_world = Matrix.Translation(p) @ Matrix.Rotation(math.radians(e["deg"]), 4, "Z") @ Matrix.Translation(-p) @ ob.matrix_world
        else:
            ob.matrix_world = Matrix.Translation(Vector(e["d"])) @ ob.matrix_world
    bpy.context.view_layer.update()


def _appended(scene_dir, base_ids):
    """Parts appended to the gray copy after the bridge — the core's shared reader bl_scene.appended_parts (the frame color bl_color uses it too)."""
    from core.blender import bl_scene as B
    return B.appended_parts(scene_dir, base_ids)


SKIP_LAYERS = ("light", "cameras", "fx")                             # stepwise-build sections that are not map-frame geometry: sky, far
                                                                        # terrain and the light's clouds (the maps build their own far terrain from the node), cameras


def load_scene(scene_dir, frame=None):
    """Like bl_scene.load_scene, but the parts come from the build's .blend. → (scene, statics, members, objs, world, kit)."""
    blend = blend_of(scene_dir)
    if blend is None: raise SystemExit(f"{scene_dir}: no build .blend (or a battle scene) — maps from the gray copy")
    bpy.ops.wm.open_mainfile(filepath=blend, load_ui=False)
    return scene_for_maps(bpy.context.scene, scene_dir, frame)


def scene_for_maps(sc0, scene_dir, frame=None, edits=True):
    """A maps scene from an ALREADY open scene sc0 (built stepwise or opened by load_scene): the parts come from it (bl_parts.collect), the scene
    description (source.json, world.json, figkit.json) from scene_dir if present; a scene without a description (stepwise build, castle_rock) — the kinds
    of parts by sk_layer and names (the ground is ground, otherwise box), there are no crowd templates. The SKIP_LAYERS sections (light, cameras, effects) are
    excluded from the parts. edits — apply the edits.json edits (the "frames" output applies them before the lit frame and calls with edits=False).
    WARNING: the maps scene is built in a clean file (B.reset: read_factory_settings) — the open build scene is destroyed after the call;
    everything needed from it (the lit frame, lamps, the camera sky) must be captured BEFORE. → (scene, statics, members, objs, world, kit)."""
    from core.blender import bl_scene as B
    if frame is not None: sc0.frame_set(frame)
    if edits: apply_edits(scene_dir)
    skipped = [o for o in sc0.objects if o.get("sk_layer") in SKIP_LAYERS and "sketch_fx" not in o]
    for o in skipped: o["sketch_fx"] = 1                                # top_parts skips such objects
    try: parts, reg, named, heights = bl_parts.collect(sc0)
    finally:
        for o in skipped: del o["sketch_fx"]
    sj = os.path.join(scene_dir, "source.json")
    base_ids = set(json.load(open(sj)).get("ids", [])) if os.path.exists(sj) else {r["id"] for r in reg}
    extra = _appended(scene_dir, base_ids)
    wj, kj = os.path.join(scene_dir, "world.json"), os.path.join(scene_dir, "figkit.json")
    W = json.load(open(wj)) if os.path.exists(wj) else {"reg": [], "world": {}}       # the sketch light and far_z come from the scene description;
    wk = {e["id"]: e["kind"] for e in W.get("reg", [])}                                # parts come from the .blend; the kind from the description if the part is in it
    W["reg"] = [{**r, "kind": wk.get(r["id"], r["kind"])} for r in reg] + [e for e in W.get("reg", []) if e["id"] in extra]   # (water, ridge:
                                                                        # the kind is set by the site builder), new parts — the kind from the graph; the appended ones — from the description
    FRESH.update(named=named, heights=heights, reg=reg)                               # figure joints from the .blend go to the passes (bl_pack → joints.json)
    kit = json.load(open(kj)) if os.path.exists(kj) else {"templates": {}, "members": [], "named": named, "named_heights": heights}
    sc = B.reset()                                                      # the maps scene has only the parts (as after importing the gray copy)
    statics = []
    for p in parts:
        me = bpy.data.meshes.new(p["name"]); me.from_pydata([tuple(v) for v in p["V"]], [], p["F"]); me.update()
        ob = bpy.data.objects.new(p["name"], me); sc.collection.objects.link(ob); statics.append(ob)
    for pid, (V, F) in extra.items():
        me = bpy.data.meshes.new(pid); me.from_pydata(V, [], F); me.update()
        ob = bpy.data.objects.new(pid, me); sc.collection.objects.link(ob); statics.append(ob)
    made = set(statics)                                                 # order as after an OBJ import: bpy.data.objects by name
    statics = [o for o in bpy.data.objects if o in made]                # (the id numbers follow this order)
    statics.append(B.add_far_ground(statics, z=W.get("far_z")))
    for ob in statics:
        if ob.type == "MESH" and not ob.name.startswith("ground"):
            B._smooth_by_angle(ob, 50)
    members = B.add_crowd(kit, sc) if kit and kit.get("members") else []
    objs = [o for o in statics + members if o.type == "MESH"]
    return sc, statics, members, objs, W, kit

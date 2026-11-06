"""Scene parts from a .blend: one partition for the bridge into the gray copy (core/blender/bl_export_scene.py → scene.obj)
and for maps straight from the .blend (bl_scene_blend.py): top-level objects, their meshes together with the pose at the frame; a figure is a part with a human
skeleton (18 COCO points); the kind of a part (wall, cover, figure…) comes from the builder (sketch_kind) or from the model's role in the graph
library/graph/graph.db. Blender only. Factored out of bl_export_scene.py without changing the logic."""
import json, os, re, sqlite3, sys

import bpy
from mathutils import Vector

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.assets.rig_joints import rig_kind, joints_of  # noqa: E402   shared with the take (components/time/bl_take.py)

# model roles from the library graph: the game model name (a .model file) → role → pipeline kind
ROLE_KIND = {"wall": "wall", "structure": "wall", "building": "wall", "cover": "box", "rubble": "rocks", "rock": "rock", "vegetation": "tree",
             "vehicle": "siege_tower", "vehicle_part": "siege_tower", "pipe": "box", "prop": "box", "accessory": "box", "light": "box",
             "figure": "figure", "armour": "figure", "head": "figure", "figure_part": "figure", "weapon": "box"}
SKIP_ROLES = {"decal", "invisible", "fx"}
_ROLES = None


def roles():
    global _ROLES
    if _ROLES is None:
        _ROLES = {}
        db = os.path.join(ROOT, "library/graph/graph.db")
        if os.path.exists(db):
            for (attrs,) in sqlite3.connect(db).execute("select a from node where kind='model'"):
                try: d = json.loads(attrs)
                except Exception: continue
                raw = (d.get("raw") or "").replace("\\", "/")
                if raw: _ROLES[os.path.splitext(os.path.basename(raw))[0].lower()] = d.get("role")
    return _ROLES


def meshes_under(ob):
    return [o for o in [ob, *ob.children_recursive] if o.type == "MESH" and o.visible_get()]


def humans_under(ob):
    return [o for o in [ob, *ob.children_recursive] if o.type == "ARMATURE" and rig_kind(o)]


def top_parts(sc):
    """Scene parts: descend through groups (an empty with several descendants that have meshes); a node with a single human skeleton is
    a whole figure (a figure can have a body skeleton and a weapon skeleton); everything else is a part with all its meshes."""
    parts = []
    def walk(ob):
        if ob.get("sketch_fx") or not meshes_under(ob): return     # light, smoke, beams, shield (components/battle/bl_battle.py) — not frame geometry
        kids = [c for c in ob.children if meshes_under(c)]
        if ob.get("sketch_kind") or len(humans_under(ob)) == 1 or ob.type == "MESH" or len(kids) < 2: parts.append(ob); return
        for c in kids: walk(c)
    for ob in sc.objects:
        if ob.parent is None and ob.type not in ("CAMERA", "LIGHT"): walk(ob)
    return parts


def collect(sc):
    """Scene parts (see _collect). The plants of the Plants component (kind=plant: cards with alpha, forest and meadow as Geometry Nodes instances —
    their mesh without instances is empty) are replaced for the duration of the partition by proxies for the maps — the crown as shells, the trunk, the bark of their own trees
    (extension points maps_before / maps_after, core/hooks.py: Plants — components/plants/api.py); afterwards they are removed. A handler
    failure does not crash the maps."""
    from core import hooks
    pre = hooks.call("maps_before", sc)
    for comp, r in pre:
        if r is not None and r is not hooks.FAILED: print(f"maps — {comp}:", r, flush=True)
    try:
        return _collect(sc)
    finally:
        hooks.call("maps_after", sc, only={c for c, r in pre if r is not hooks.FAILED})


def _collect(sc):
    """Scene parts at the current frame → (parts, reg, named, heights): parts — [{name, oid, kind, V (world points), F (faces by V)}]
    in bridge order; reg — [{id, kind, gid}] for world.json; named, heights — figures for figkit.json."""
    roles_ = roles(); dg = bpy.context.evaluated_depsgraph_get()
    parts, reg, named, heights, gid = [], [], {}, {}, 0
    for part in top_parts(sc):
        ms = meshes_under(part)
        arms = [o for o in [part, *part.children_recursive] if o.type == "ARMATURE"]
        human = next(((x, rig_kind(x)) for x in arms if rig_kind(x)), (None, None))
        base = re.sub(r"[^A-Za-z0-9_]+", "_", part.name.split(".")[0]).strip("_") or "part"
        if part.get("sketch_kind") and part["sketch_kind"] != "figure":   # the kind from the builder (components/battle/bl_battle.py): vehicles with a skeleton are not fighters
            kind, oid = part["sketch_kind"], base; human = (None, None)
        elif human[0] is not None:
            kind, oid = "figure", base
        elif part.name == "terrain" or base.startswith("ground"):
            kind, oid = "ground", "ground"
        else:
            role = roles_.get(base.lower())
            if role in SKIP_ROLES: continue
            kind, oid = ROLE_KIND.get(role, "box"), base
            if kind == "figure" and not arms: kind = "box"              # the role of a figure without a skeleton (a statue, a detail) is a prop; with a skeleton
                                                                        # that is not human (a creature) is a figure for the mask and the sketch, without joints
        while any(r["id"] == oid for r in reg) and kind != "ground": oid += "_"
        V, F = [], []
        lo, hi = Vector((1e9, 1e9, 1e9)), Vector((-1e9, -1e9, -1e9))
        for o in ms:
            me = o.evaluated_get(dg).to_mesh(); off = len(V)
            for v in me.vertices:
                w = o.matrix_world @ v.co; V.append(w); lo = Vector(map(min, lo, w)); hi = Vector(map(max, hi, w))
            F += [[off + i for i in p.vertices] for p in me.polygons]
            o.evaluated_get(dg).to_mesh_clear()
        if not F: continue
        name = f"figure_{oid}" if kind == "figure" else oid
        if kind != "ground" or not any(r["id"] == "ground" for r in reg): reg.append({"id": oid, "kind": kind, "gid": gid}); gid += 1
        parts.append({"name": name, "oid": oid, "kind": kind, "V": V, "F": F})
        if kind == "figure" and human[0] is not None:
            h = hi.z - lo.z
            jt, hd, mir = joints_of(human[0], human[1], h)
            named[oid] = {"oid": reg[-1]["gid"], "height": round(h, 3), "pos": [round((lo.x + hi.x) / 2, 3), round((lo.y + hi.y) / 2, 3), round(lo.z, 3)],
                          "heading": hd, "k": 1.0, "joints": jt, "rig": human[1], "mirrored": mir}
            heights[oid] = round(h, 3)
    return parts, reg, named, heights

"""A single BVH tree over the static geometry of a large scene: rays through it take microseconds. Why not scene.ray_cast: Blender
iterates over the scene objects for every ray — in an assault scene (~15 thousand objects and instances) that is milliseconds per
ray, and an obstacle map built by rays from above took over an hour. Static means everything except the relief and the movable
things: crowds, heroes, animated vehicles (a mover tag or location keys on the root), effects, cameras, prototypes; anything hidden
from the render is skipped (the models' service spheres). Used by components/battle/bl_battle_data.py (the obstacle map) and
components/camera/bl_shot_check.py (camera paths).
  import bl_static_bvh as SB; st = SB.build(sc)   →   hit, dist, normal, idx = st.ray(origin, direction, distance); st.who(idx)"""
import time
import bpy
import numpy as np
from mathutils.bvhtree import BVHTree

DYN = ("terrain", "army", "vehicles", "fx", "anim", "protos")


def _root(o):
    while o.parent is not None: o = o.parent
    return o


def dynamic_test():
    dyn = {o.name for cn in DYN if bpy.data.collections.get(cn) for o in bpy.data.collections[cn].all_objects}
    memo = {}
    def is_dyn(o):
        r = _root(o); k = (o.name, r.name)
        if k in memo: return memo[k]
        res = o.name in dyn or r.name in dyn or any(x.get("sketch_fx") or x.get("sketch_kind") in ("crowd", "figure") or x.get("mover") for x in (o, r))
        if not res:
            ad = r.animation_data
            res = bool(ad and ad.action and r.type != "CAMERA" and any(fc.data_path == "location" for L in ad.action.layers for s in L.strips
                                                                        for cb in s.channelbags for fc in cb.fcurves))
        memo[k] = res; return res
    return is_dyn


def soup(sc, terrain=False):
    """Static triangles in the world: (vertices N×3, triangles M×3); the mesh is taken once per prototype, instances via their matrix."""
    dg = bpy.context.evaluated_depsgraph_get(); is_dyn = dynamic_test(); local = {}; V, T, nv = [], [], 0; names, starts, nt = [], [], 0
    for inst in dg.object_instances:
        ob = inst.object
        if ob.type != "MESH": continue
        host = inst.parent.original if inst.is_instance and inst.parent else ob.original
        if host.name == "terrain" and terrain: pass
        elif is_dyn(host) or ob.original.hide_render or (not inst.is_instance and not ob.original.visible_get()): continue
        me = ob.data; k = me.as_pointer()
        if k not in local:
            me.calc_loop_triangles(); co = np.empty(len(me.vertices) * 3, np.float64); me.vertices.foreach_get("co", co)
            tr = np.empty(len(me.loop_triangles) * 3, np.int64); me.loop_triangles.foreach_get("vertices", tr)
            local[k] = (co.reshape(-1, 3), tr.reshape(-1, 3))
        co, tr = local[k]
        if not len(tr): continue
        m = np.array(inst.matrix_world); w = co @ m[:3, :3].T + m[:3, 3]
        V.append(w); T.append(tr + nv); nv += len(w); names.append(_root(host).name); starts.append(nt); nt += len(tr)
    if not V: return np.zeros((0, 3)), np.zeros((0, 3), np.int64), [], np.zeros(0, np.int64)
    return np.concatenate(V), np.concatenate(T), names, np.array(starts, np.int64)


class Static:
    """The tree and the names: who(index) is the root of the part that owns the triangle (for remarks)."""
    def __init__(self, bvh, names, starts, info): self.bvh, self.names, self.starts, self.info = bvh, names, starts, info
    def ray(self, o, d, dist):
        loc, nrm, idx, dd = self.bvh.ray_cast(o, d, dist)
        return (loc is not None, dd, nrm, idx)
    def who(self, idx): return self.names[int(np.searchsorted(self.starts, idx, side="right")) - 1] if idx is not None and len(self.starts) else None


def build(sc, terrain=False):
    t0 = time.time(); V, T, names, starts = soup(sc, terrain); t1 = time.time()
    bvh = BVHTree.FromPolygons(V.tolist(), T.tolist(), all_triangles=True)
    return Static(bvh, names, starts, {"vertices": int(len(V)), "triangles": int(len(T)), "parts": len(names), "soup_s": round(t1 - t0, 1),
                                       "bvh_s": round(time.time() - t1, 1)})

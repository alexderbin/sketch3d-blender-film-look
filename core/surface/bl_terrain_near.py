"""Ground near the camera — for close-ups (small scenes): the patch under cameras with close ground is split off the site
grid (0.5 m step) into "terrain_near", refined by simple subdivision (Blender) down to 0.06–0.125 m, and the ground
shader (bl_terrain_surf) displaces its vertices by the height of the scans of the same layers (like tessellation with a
height map in engines); the displacement strength is the attribute a_near, which fades toward the edge of the patch (no
gaps with the neighboring grid). Called from bl_relief (near=1), after the ground surface."""
import os

import bpy
import numpy as np


def footprint(G, cam, reach=35.0, n=13):
    """Where the camera sees ground closer than reach m: the ray hit points (x, y); empty — the ground is far (a wide shot)."""
    M = cam.matrix_world; e = np.array(M.translation); f = -np.array(M.col[2][:3]); r = np.array(M.col[0][:3]); u = np.array(M.col[1][:3])
    hx = 36.0 / (2 * cam.data.lens); hy = hx * 9 / 16; pts = []
    for a in np.linspace(-hx, hx, n):
        for b in np.linspace(-hy, hy, n // 2 + 1):
            d = f + a * r + b * u; d /= np.linalg.norm(d)
            for t in np.arange(0.5, reach, 0.5):
                p = e + d * t
                if p[2] <= float(G.top(p[0], p[1])): pts.append(p[:2]); break
    return np.array(pts)


def build(ter, G, cams, levels=3, margin=4.0, fade=3.0, sc=None):
    """The patch near cameras: returns a summary (how many cameras with close ground, faces, vertices after subdivision)."""
    import bmesh
    sc = sc or bpy.context.scene; boxes = []
    for c in cams:
        P = footprint(G, c)
        if len(P) >= 4: lo, hi = P.min(0) - margin, P.max(0) + margin; boxes.append((lo, hi))
    if not boxes: return {"near": 0}
    me = ter.data; co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    inside = np.zeros(len(co), bool)
    for lo, hi in boxes: inside |= (co[:, 0] >= lo[0]) & (co[:, 0] <= hi[0]) & (co[:, 1] >= lo[1]) & (co[:, 1] <= hi[1]) & (co[:, 2] > co[:, 2].min() + 1)
    bm = bmesh.new(); bm.from_mesh(me); bm.verts.ensure_lookup_table()
    faces = [f for f in bm.faces if all(inside[v.index] for v in f.verts)]
    if not faces: bm.free(); return {"near": 0}
    for f in bm.faces: f.select_set(False)
    for f in faces: f.select_set(True)
    bm.to_mesh(me); bm.free()
    for o in sc.objects: o.select_set(False)
    bpy.context.view_layer.objects.active = ter; ter.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.mesh.select_mode(type="FACE"); bpy.ops.mesh.separate(type="SELECTED"); bpy.ops.object.mode_set(mode="OBJECT")
    near = next(o for o in sc.objects if o.select_get() and o is not ter); near.name = "terrain_near"; near.data.name = "terrain_near"
    nm = near.data; nc = np.empty(len(nm.vertices) * 3); nm.vertices.foreach_get("co", nc); nc = nc.reshape(-1, 3)
    bnd = set()                                                     # the patch edge — the displacement fades toward it
    bm = bmesh.new(); bm.from_mesh(nm)
    for e in bm.edges:
        if e.is_boundary: bnd.update(v.index for v in e.verts)
    bm.free()
    B = nc[sorted(bnd), :2] if bnd else np.zeros((1, 2)); d = np.full(len(nc), fade)
    for i in range(0, len(nc), 4096):                               # distance to the edge — in chunks (memory)
        d[i:i + 4096] = np.min(np.hypot(nc[i:i + 4096, None, 0] - B[None, :, 0], nc[i:i + 4096, None, 1] - B[None, :, 1]), axis=1)
    k = np.clip(d / fade, 0, 1); k = k * k * (3 - 2 * k)
    nm.attributes.new("a_near", "FLOAT", "POINT").data.foreach_set("value", k.astype(np.float32))
    sub = near.modifiers.new("near_sub", "SUBSURF"); sub.subdivision_type = "SIMPLE"; sub.levels = sub.render_levels = levels
    nm.shade_smooth()
    for e in nm.edges: e.use_edge_sharp = False                     # smooth (sharp edges produced "pyramids" up close)
    near["kind"] = "ground"; near["relief"] = G.folder
    return {"near": len(boxes), "near_faces": len(faces), "near_verts_sub": len(nm.vertices) * 4 ** levels}

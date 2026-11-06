# what is visible over a shot is decided by the frustum of all cameras;
#         it works at the object level, without cutting meshes
"""Visible in the frame: an object is hidden from the render if no camera of the frame sees its box with a 15 % margin together with its shadow
along the sun (the box is extended to the ground along the sun ray). Never hidden: prototypes (excluded collections, they are
instanced), instance creators (nodes, particles, parent points) and everything in keep (actors, frame effects).
Measured on a 500-fighter battle (4307 meshes, 8.9 M vertices): Workbench is 2.0–2.6× faster, Eevee 1.3–2.9× (the machine is shared with
neighbors), the frame is the same — 0.01 % of pixels differ at the edges; the selection takes 0.04 s per camera. A pitfall: camera markers in a .blend (timeline_markers)
switch the render camera — objects were hidden for one camera while another was rendered; unbind them beforehand (unbind_markers).
Usage in Blender: VS.prepare(scene) → hid = VS.hide_outside(scene, [camera specs], keep) → render → VS.restore(hid)."""
import math
import bpy
from mathutils import Matrix, Vector

_BOX = {}


def unbind_markers(sc):
    """Timeline markers with a camera switch the render camera — unbind them (in memory, the file is not changed)."""
    for m in sc.timeline_markers: m.camera = None


def hidable(o):
    return (o.type == "MESH" and not o.hide_render and o.visible_get() and o.instance_type == "NONE"
            and not any(m.type in ("NODES", "PARTICLE_SYSTEM") for m in o.modifiers))


def prepare(sc, keep=()):
    """World boxes — once (the world is static): a computed box (for meshes built by code the original is sometimes stale) and its shadow
    along the scene's sun down to the bottom of the box."""
    _BOX.clear(); dg = bpy.context.evaluated_depsgraph_get(); keep = {o.name for o in keep}
    sun = next((o for o in sc.objects if o.type == "LIGHT" and o.data.type == "SUN" and not o.hide_render), None)
    L = (sun.matrix_world.to_3x3() @ Vector((0, 0, -1))).normalized() if sun else None
    for o in sc.objects:
        if o.name in keep or not hidable(o): continue
        ev = o.evaluated_get(dg); P = [ev.matrix_world @ Vector(c) for c in ev.bound_box]
        if L is not None and L.z < -0.05:
            zmin = min(p.z for p in P); P += [p + L * ((p.z - zmin) / -L.z) for p in P]
        _BOX[o.name] = (o, P)
    return len(_BOX)


def _frustum(spec, margin):
    W, H = spec["resolution"]; lens = spec["lens_mm"]; fit = spec.get("sensor_fit", "AUTO")
    if fit == "VERTICAL" or (fit == "AUTO" and H > W): ty = spec.get("sensor_height_mm", 24.0) / 2 / lens; tx = ty * W / H
    else: tx = spec.get("sensor_width_mm", 36.0) / 2 / lens; ty = tx * H / W
    return Matrix(spec["matrix_world"]).inverted(), tx * (1 + margin) + abs(spec.get("shift_x", 0.0)) * 2, ty * (1 + margin) + abs(spec.get("shift_y", 0.0)) * 2


def _seen(P, inv, tx, ty):
    pts = [inv @ p for p in P]
    if all(p.z > -0.05 for p in pts): return False                  # everything is behind the camera (the view is along −Z)
    if any(p.z >= 0 for p in pts): return True                       # the box surrounds the camera
    xs = [p.x / -p.z for p in pts]; ys = [p.y / -p.z for p in pts]
    return not (max(xs) < -tx or min(xs) > tx or max(ys) < -ty or min(ys) > ty)


def hide_outside(sc, specs, keep=(), margin=0.15):
    """Hide from the render whatever no camera sees (cameras.json records); returns the hidden ones, for restore."""
    if not _BOX: prepare(sc, keep)
    F = [_frustum(s, margin) for s in specs]; keep = {o.name for o in keep}; out = []
    for name, (o, P) in _BOX.items():
        if name in keep or o.hide_render: continue
        if not any(_seen(P, *f) for f in F): o.hide_render = True; out.append(o)
    return out


def restore(objs):
    for o in objs: o.hide_render = False

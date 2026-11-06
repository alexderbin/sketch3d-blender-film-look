"""Blender building blocks shared by the components and the core outputs (moved out of Actors, whose crowd module is internal to it,
since the core, Camera, Battle, Site and the trailer format all used them): a Geometry Nodes tree in code (Graph), a point cloud
with attributes (points_object), instance prototypes (proto_collection), a collection excluded from the view layer (exclude), an
NLA clip loop (nla_loop), a driver and keys (driver, keys), a shot cinema camera and shot markers (shot_camera, bind_shot_markers)."""
import bpy
import numpy as np


class Graph:
    """A Geometry Nodes tree in code: the inputs are named point attributes, the math is Math nodes."""
    def __init__(self, name):
        self.ng = bpy.data.node_groups.new(name, "GeometryNodeTree")
        self.ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        self.ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        self.gin, self.gout = self.ng.nodes.new("NodeGroupInput"), self.ng.nodes.new("NodeGroupOutput")

    def new(self, idn, **kw):
        n = self.ng.nodes.new(idn)
        for k, v in kw.items(): setattr(n, k, v)
        return n

    def put(self, sock, v):
        if hasattr(v, "is_linked") or hasattr(v, "links"): self.ng.links.new(v, sock)
        elif v is not None: sock.default_value = v

    def attr(self, name, dtype="FLOAT"):
        n = self.new("GeometryNodeInputNamedAttribute", data_type=dtype); n.inputs["Name"].default_value = name
        return n.outputs["Attribute"]

    def m(self, op, a, b=None, c=None):
        n = self.new("ShaderNodeMath", operation=op)
        for i, v in enumerate((a, b, c)):
            if v is not None: self.put(n.inputs[i], v)
        return n.outputs[0]

    def xyz(self, x, y, z):
        n = self.new("ShaderNodeCombineXYZ")
        for i, v in enumerate((x, y, z)): self.put(n.inputs[i], v)
        return n.outputs[0]

    def sep(self, v):
        n = self.new("ShaderNodeSeparateXYZ"); self.put(n.inputs[0], v); return n.outputs


def points_object(name, coll, co, attrs):
    """A mesh of vertices only (a point = a fighter or a building) with point attributes: {name: (type, array)}; type FLOAT | INT | FLOAT_VECTOR."""
    me = bpy.data.meshes.new(name); n = len(co); me.vertices.add(n); me.vertices.foreach_set("co", np.asarray(co, np.float32).ravel())
    for k, (dt, arr) in attrs.items():
        a = me.attributes.new(k, dt, "POINT"); arr = np.asarray(arr)
        if dt == "FLOAT_VECTOR": a.data.foreach_set("vector", arr.astype(np.float32).ravel())
        elif dt == "INT": a.data.foreach_set("value", arr.astype(np.int32).ravel())
        else: a.data.foreach_set("value", arr.astype(np.float32).ravel())
    me.update(); ob = bpy.data.objects.new(name, me); coll.objects.link(ob)
    return ob


def proto_collection(name, meshes, parent):
    """Instance prototypes in order (Collection Info lays out the children by name): objects 00000, 00001, …; the collection is excluded
    from the view layer — only the instances appear in the frame (the source objects underground are not needed)."""
    c = bpy.data.collections.new(name); parent.children.link(c)
    for i, me in enumerate(meshes):
        o = bpy.data.objects.new(f"{name}_{i:05d}", me); c.objects.link(o)
    return c


def exclude(view_layer, coll):
    def walk(lc):
        if lc.collection == coll: lc.exclude = True; return True
        return any(walk(ch) for ch in lc.children)
    walk(view_layer.layer_collection)


def nla_loop(obj, action, start, scale=1.0, repeat=60):
    """A clip loop on a figure copy: an NLA strip with its own start (phase) and time scale (the clip's 30 fps → the scene's 24 fps)."""
    ad = obj.animation_data or obj.animation_data_create(); ad.action = None
    tr = ad.nla_tracks.new(); st = tr.strips.new(action.name, int(start), action)
    st.scale = scale; st.repeat = repeat; st.extrapolation = "HOLD"
    return st


def driver(idblock, path, expr, index=-1):
    """A driver of a simple expression (sin, cos, frame — no Python, a windowless render evaluates it)."""
    fc = idblock.driver_add(path, index) if index >= 0 else idblock.driver_add(path)
    fc.driver.type = "SCRIPTED"; fc.driver.expression = expr
    return fc


def keys(obj, path, pts, index=-1, interp="BEZIER"):
    """Path keys (frame, value); interp — for all keys."""
    for f, v in pts:
        if index >= 0:
            getattr(obj, path)[index] = v; obj.keyframe_insert(path, index=index, frame=f)
        else:
            setattr(obj, path, v); obj.keyframe_insert(path, frame=f)
    ad = obj.animation_data
    if ad and ad.action:
        from bpy_extras import anim_utils
        cb = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
        for fc in cb.fcurves:
            if fc.data_path == path:
                for kp in fc.keyframe_points: kp.interpolation = interp


def shot_camera(sc, coll, sh, phase=0.0):
    """A shot cinema camera: the path and the aim point are Bezier keys through control points, "Track To" onto the aim point, the lens
    is keyed, the shake is noise on the aim point; the marker at the start of the shot carries the shot name on the timeline, and the
    trailer render binds the camera to it in memory (formats/battle_trailer/bl_trailer.py): a marker with a camera in the file would
    replace the camera for other renders of this file. sh — {name, frames: [a, b], path, aim, lens | lenses, shake}."""
    from bpy_extras import anim_utils
    cd = bpy.data.cameras.new(sh["name"]); cd.lens = sh.get("lens", 24); cd.sensor_width = 36.0; cd.clip_start, cd.clip_end = 0.2, 40000
    cam = bpy.data.objects.new(sh["name"], cd); coll.objects.link(cam); aim = bpy.data.objects.new(sh["name"] + "_aim", None); coll.objects.link(aim)
    con = cam.constraints.new("TRACK_TO"); con.target = aim; con.track_axis = "TRACK_NEGATIVE_Z"; con.up_axis = "UP_Y"
    a, b = sh["frames"]; n = len(sh["path"])
    for i, (p, q) in enumerate(zip(sh["path"], sh["aim"])):
        f = a + (b - a) * i / (n - 1); cam.location = p; cam.keyframe_insert("location", frame=f); aim.location = q; aim.keyframe_insert("location", frame=f)
    for i, ln in enumerate(sh.get("lenses", [])):
        cd.lens = ln; cd.keyframe_insert("lens", frame=a + (b - a) * i / max(len(sh["lenses"]) - 1, 1))
    if sh.get("shake"):
        cb = anim_utils.action_get_channelbag_for_slot(aim.animation_data.action, aim.animation_data.action_slot)
        for k, fc in enumerate(cb.fcurves):
            md = fc.modifiers.new("NOISE"); md.strength = sh["shake"]; md.scale = 9.0; md.phase = phase + 17 * k
    sc.timeline_markers.new(sh["name"], frame=a); cam["shot"] = [a, b]
    return cam


def bind_shot_markers(sc):
    """A shot marker → its camera — in memory only (the trailer render, a copy for viewing); in the scene file the markers are not bound."""
    for m in sc.timeline_markers:
        cam = bpy.data.objects.get(m.name)
        if m.camera is None and cam is not None and cam.type == "CAMERA" and cam.get("shot"): m.camera = cam

"""A digest of a .blend without rendering: what the file contains — objects, meshes and their attributes, materials and node trees,
node groups, lamps, cameras, keys and their modifiers, drivers, NLA strips, collections, images, markers, the world and the scene
settings — as rounded numbers and short hashes. It is needed to compare two builds of one scene (the code against a hand-edited
file, a cached build against an uncached one): core/services/digest.py diff. Node positions in the editor, selection, the current
frame and block users are not part of the digest — they do not affect the scene.
blender -b --factory-startup <file.blend> -P core/blender/run.py -- core/blender/bl_digest.py -- <out.json>     → the __RESULT__ line (block counts and time)."""
import hashlib, json, re, sys, time
import bpy
import numpy as np

t0 = time.time(); OUT = sys.argv[sys.argv.index("--") + 1]; sc = bpy.context.scene
sc.frame_set(sc.frame_start)                                           # keys and drivers — at the first frame in both builds
bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)   # orphans (only the next save removes them) do not count
base = lambda s: re.sub(r"\.\d{3}$", "", s or "")


def H(*parts):
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(p.tobytes() if isinstance(p, np.ndarray) else json.dumps(p, sort_keys=True, ensure_ascii=False, default=str).encode())
    return h.hexdigest()


def q(a, k=1e4):
    """Rounding before hashing: build noise (the last float digits) does not count."""
    return np.round(np.asarray(a, np.float64) * k).astype(np.int64)


def val(v):
    if isinstance(v, bool) or v is None or isinstance(v, (int, str)): return v
    if isinstance(v, float): return round(v, 4)
    if isinstance(v, bpy.types.ID): return base(v.name)
    if isinstance(v, set): return sorted(v)
    try: return [val(x) for x in v]
    except TypeError: return str(v)


SIMPLE = {"BOOLEAN", "INT", "FLOAT", "STRING", "ENUM"}
ID_SKIP = {"name", "name_full", "rna_type", "users", "use_fake_user", "use_extra_user", "tag", "is_evaluated", "is_embedded_data",
           "is_missing", "is_runtime_data", "session_uid", "id_type", "is_library_indirect", "use_nodes", "filepath_raw", "preview"}
NODE_SKIP = ID_SKIP | {"location", "location_absolute", "width", "width_hidden", "height", "dimensions", "select", "show_options",
                       "show_preview", "show_texture", "hide", "use_custom_color", "color", "color_tag", "label", "warning_propagation"}


def props(x, skip=ID_SKIP):
    """The editable simple properties of a block (numbers, flags, enums, strings, references to blocks) — without a separate list for each type."""
    out = {}
    for p in x.bl_rna.properties:
        k = p.identifier
        if k in skip or p.is_readonly or k.startswith("bl_"): continue
        if p.type in SIMPLE or p.type == "POINTER":
            try: v = getattr(x, k)
            except Exception: continue
            if p.type == "POINTER" and not isinstance(v, bpy.types.ID) and v is not None: continue
            out[k] = val(v)
    return out


def tree(nt):
    """Node tree: for each node — a hash of its type, properties and the values of unlinked inputs; links — as a list; the group interface."""
    if nt is None: return None
    nodes = {}
    for n in nt.nodes:
        ins = {}
        for s in n.inputs:
            if not s.is_linked and hasattr(s, "default_value"):
                try: ins[s.identifier] = val(s.default_value)
                except Exception: pass
        nodes[n.name] = H(n.bl_idname, props(n, NODE_SKIP), ins, getattr(getattr(n, "node_tree", None), "name", None) and base(n.node_tree.name))
    links = sorted((l.from_node.name, l.from_socket.identifier, l.to_node.name, l.to_socket.identifier, l.is_muted) for l in nt.links)
    iface = [(i.item_type, getattr(i, "in_out", ""), getattr(i, "socket_type", ""), i.name) for i in nt.interface.items_tree] if hasattr(nt, "interface") else []
    return {"n": len(nodes), "nodes": nodes, "links": H(links), "nl": len(links), "iface": H(iface)}


ATTR = {"FLOAT": ("value", 1, np.float32), "INT": ("value", 1, np.int32), "FLOAT_VECTOR": ("vector", 3, np.float32),
        "FLOAT_COLOR": ("color", 4, np.float32), "BYTE_COLOR": ("color", 4, np.float32), "BOOLEAN": ("value", 1, bool),
        "FLOAT2": ("vector", 2, np.float32), "INT8": ("value", 1, np.int32), "INT32_2D": ("value", 2, np.int32),
        "QUATERNION": ("value", 4, np.float32), "FLOAT4X4": ("value", 16, np.float32)}


def canon(me, lv, nf):
    """The face order, the start of a face loop and the edge order in bmesh can differ from run to run (create_uvsphere gives the same
    vertices but different loops): a face starts from its smallest vertex, faces and edges are sorted. Permutations of faces, loops
    and edges — to compare the mesh, not the order of its construction."""
    ls = np.empty(nf, np.int64); me.polygons.foreach_get("loop_start", ls)
    lt = np.empty(nf, np.int64); me.polygons.foreach_get("loop_total", lt)
    fp, lp = [], []
    for k in np.unique(lt):
        sel = np.nonzero(lt == k)[0]; idx = ls[sel][:, None] + np.arange(k)[None, :]
        r = np.argmin(lv[idx], axis=1); idx = np.take_along_axis(idx, (r[:, None] + np.arange(k)[None, :]) % k, axis=1)
        o = np.lexsort(lv[idx].T[::-1]); fp.append(sel[o]); lp.append(idx[o].ravel())
    ev = np.empty(len(me.edges) * 2, np.int64); me.edges.foreach_get("vertices", ev); ev = np.sort(ev.reshape(-1, 2), axis=1)
    ep = np.lexsort(ev.T[::-1]) if len(ev) else np.zeros(0, np.int64)
    cat = lambda L: np.concatenate(L) if L else np.zeros(0, np.int64)
    return cat(fp), cat(lp), ep, ev[ep]


def mesh(me):
    nv, nf = len(me.vertices), len(me.polygons)
    co = np.empty(nv * 3, np.float32); me.vertices.foreach_get("co", co)
    lv = np.empty(len(me.loops), np.int64); me.loops.foreach_get("vertex_index", lv)
    lt = np.empty(nf, np.int32); me.polygons.foreach_get("loop_total", lt)
    mi = np.empty(nf, np.int32); me.polygons.foreach_get("material_index", mi)
    fp, lp, ep, edges = canon(me, lv, nf); perm = {"FACE": fp, "CORNER": lp, "EDGE": ep}
    attrs = {}
    for a in me.attributes:
        if a.name.startswith(".") or a.name in ("position",): continue
        spec = ATTR.get(a.data_type)
        if spec is None: attrs[a.name] = [a.domain, a.data_type]; continue
        key, w, dt = spec
        try:
            arr = np.empty(len(a.data) * w, dt); a.data.foreach_get(key, arr); arr = arr.reshape(len(a.data), w)
            if a.domain in perm and len(perm[a.domain]) == len(arr): arr = arr[perm[a.domain]]
            attrs[a.name] = [a.domain, a.data_type, H(q(arr) if dt is np.float32 else np.ascontiguousarray(arr))]
        except Exception as e_:
            attrs[a.name] = [a.domain, a.data_type, f"? {type(e_).__name__}"]
    return {"v": nv, "f": nf, "geo": H(q(co), lt[fp], lv[lp], mi[fp], edges), "mats": [base(m.name) if m else None for m in me.materials],
            "attrs": attrs}


def fcurves_of(act):
    out = []
    try:
        for L in act.layers:
            for s in L.strips:
                for cb in s.channelbags: out += list(cb.fcurves)
    except AttributeError:
        pass
    return out or list(getattr(act, "fcurves", []))


def fcurve(fc):
    kp = fc.keyframe_points; co = np.empty(len(kp) * 2, np.float32); kp.foreach_get("co", co)
    hl = np.empty(len(kp) * 2, np.float32); kp.foreach_get("handle_left", hl)
    return {"k": len(kp), "range": [round(float(co[0]), 2), round(float(co[-2]), 2)] if len(kp) else None, "h": H(q(co, 1e3), q(hl, 1e3)),
            "interp": sorted({k.interpolation for k in kp}), "mods": [[m.type, props(m)] for m in fc.modifiers]}


def drivers(ad):
    if ad is None: return []
    return [[d.data_path, d.array_index, d.driver.type, d.driver.expression,
             [[v.name, v.type, [[base(t.id.name) if t.id else None, t.data_path] for t in v.targets]] for v in d.driver.variables]]
            for d in ad.drivers]


def anim(ad):
    if ad is None: return None
    return {"action": base(ad.action.name) if ad.action else None, "slot": getattr(getattr(ad, "action_slot", None), "identifier", None),
            "nla": [[t.name, t.mute, [[base(s.action.name) if s.action else None, round(s.frame_start, 2), round(s.frame_end, 2), round(s.scale, 4),
                                       round(s.repeat, 3), s.blend_type, s.extrapolation, s.use_reverse] for s in t.strips]] for t in ad.nla_tracks],
            "drivers": drivers(ad)}


def layer_excl():
    out = {}
    def walk(lc):
        out[lc.collection.name] = [lc.exclude, lc.hide_viewport]
        for ch in lc.children: walk(ch)
    walk(bpy.context.view_layer.layer_collection)
    return out


def nodes_inputs(md):
    """Node modifier inputs: in 5.2 — md.properties (the group interface); earlier — the modifier's IDProperty."""
    try: return props(md.properties)
    except Exception: pass
    try: return {k: val(md[k]) for k in md.keys()}
    except Exception: return {}


D = {"file": bpy.data.filepath, "objects": {}, "meshes": {}, "materials": {}, "node_groups": {}, "lights": {}, "cameras": {}, "images": {},
     "actions": {}, "collections": {}, "worlds": {}}
for o in bpy.data.objects:
    m = np.array(o.matrix_world)
    D["objects"][o.name] = {
        "type": o.type, "data": base(o.data.name) if o.data else None, "parent": o.parent.name if o.parent else None,
        "colls": sorted(c.name for c in o.users_collection), "loc": [round(float(x), 3) for x in m[:3, 3]], "m": H(q(m, 1e3)),
        "hide_render": o.hide_render, "vis": o.visible_get(), "instance": [o.instance_type, base(o.instance_collection.name) if o.instance_collection else None],
        "props": {k: val(o[k]) for k in o.keys() if not k.startswith("_")},
        "mods": [[md.type, md.name, base(md.node_group.name) if getattr(md, "node_group", None) else None,
                  nodes_inputs(md) if md.type == "NODES" else props(md)] for md in o.modifiers],
        "cons": [[c.type, base(getattr(getattr(c, "target", None), "name", None)), props(c)] for c in o.constraints],
        "anim": anim(o.animation_data), "data_anim": anim(getattr(o.data, "animation_data", None)) if o.data else None}
for me in bpy.data.meshes:
    if me.users: D["meshes"][me.name] = mesh(me)
for ma in bpy.data.materials:
    if ma.users: D["materials"][ma.name] = {"props": props(ma), "tree": tree(ma.node_tree), "anim": anim(ma.animation_data)}
for ng in bpy.data.node_groups:
    if ng.users: D["node_groups"][ng.name] = {"type": ng.bl_idname, "tree": tree(ng)}
for la in bpy.data.lights:
    if la.users: D["lights"][la.name] = {"props": props(la), "anim": anim(la.animation_data), "tree": tree(la.node_tree) if la.use_nodes else None}
for ca in bpy.data.cameras:
    if ca.users: D["cameras"][ca.name] = {"props": props(ca), "anim": anim(ca.animation_data)}
for im in bpy.data.images:
    if not im.users: continue
    pf = im.packed_file
    D["images"][im.name] = {"size": list(im.size), "source": im.source, "packed": pf.size if pf else None,
                            "h": H(np.frombuffer(pf.data, np.uint8)) if pf else base(bpy.path.basename(im.filepath)),
                            "colorspace": im.colorspace_settings.name, "alpha": im.alpha_mode}
for ac in bpy.data.actions:
    if ac.users: D["actions"][ac.name] = {"curves": {f"{fc.data_path}[{fc.array_index}]": fcurve(fc) for fc in fcurves_of(ac)},
                                          "range": [round(x, 2) for x in ac.frame_range]}
for c in bpy.data.collections:
    D["collections"][c.name] = {"n": len(c.objects), "objs": H(sorted(o.name for o in c.objects)), "children": sorted(ch.name for ch in c.children),
                                "hide_render": c.hide_render}
for w in bpy.data.worlds:
    if w.users: D["worlds"][w.name] = {"props": props(w), "tree": tree(w.node_tree), "anim": anim(w.animation_data)}
r = sc.render
D["scene"] = {"frames": [sc.frame_start, sc.frame_end], "fps": [r.fps, round(r.fps_base, 4)], "engine": r.engine,
              "res": [r.resolution_x, r.resolution_y, r.resolution_percentage], "motion_blur": [r.use_motion_blur, round(r.motion_blur_shutter, 3)],
              "film_transparent": r.film_transparent, "eevee": props(sc.eevee), "view": props(sc.view_settings),
              "world": base(sc.world.name) if sc.world else None, "camera": sc.camera.name if sc.camera else None,
              "markers": sorted([m.name, m.frame, m.camera.name if m.camera else None] for m in sc.timeline_markers),
              "layers": layer_excl(), "props": {k: val(sc[k]) for k in sc.keys() if not k.startswith("_")}}
json.dump(D, open(OUT, "w"), ensure_ascii=False, indent=0, sort_keys=True)
counts = {k: len(v) for k, v in D.items() if isinstance(v, dict) and k != "scene"}
print("__RESULT__" + json.dumps({"out": OUT, "counts": counts, "s": round(time.time() - t0, 1)}, ensure_ascii=False))

"""The "frames" output of a built scene (stepwise build, core/blender/build.py): a lit frame and a map bundle for every camera in
the run scene's cameras.json — from the same .blend the steps built (light, sky and materials are already in the scene — there are no render keys).
The step function: render(ctx) — keys: cameras (all | K1,K2 | a-b), samples, light (true by default), maps (true by default),
video (true — a fly-around of the orbit_* cameras into orbit.mp4), fps, look (lit — Eevee with light; sketch — a sketch frame core/outputs/bl_sketch.py,
compositing core/outputs/sketch.py, the same lit.png; the default of the sketch stage). Next to it — objects.json: sk_id object boxes (for stage comparison).
Order (the maps scene is built in a clean file and destroys the open build scene — bl_scene_blend.scene_for_maps):
  1. a lit frame of the built scene, all cameras: a light plan exists (ctx["gives"]["light_plan"]) and light: true — as in bl_color
     (core/outputs/bl_color.py render_colour): cam<K>/lit.exr (geometry on a transparent background, depth) + sky.exr (the camera sky) + far.json;
     otherwise a direct Eevee PNG — cam<K>/lit.png;
  2. maps (maps: true): parts from the open scene (bl_scene_blend.scene_for_maps), camera passes (tools/bl_pack.Pack.camera) →
     cam<K>/raw/*, then the map build core/outputs/pack.py (the .venv Python, as for core/outputs/frames.py) → ids, depth, normal, line, pose, mask_fig …;
  3. haze by depth, AgX, the plan's exposure: core/light/far.py post → cam<K>/lit.png (depth from raw/passes.exr);
  4. the fly-around video orbit.mp4.
→ <folder>/cam<K>/…, <folder>/node.json (the node plan: the maps and pack.py take the sketch light from there), frames.json."""
import json, os, shutil, subprocess, sys, time

import bpy

SAMPLES = {"sketch": 8, "preview": 16, "final": 64}          # the profile is the vocabulary canon (the Russian aliases of sketch|preview|final are synonyms)
SKY_BG = [0.80, 0.84, 0.90]                                             # same as SKY in core/outputs/bl_color.py (far.json sky_bg)


def _select(spec, n, names=None):
    """Output cameras: all | a-b | comma-separated numbers | names and name prefixes of records (establish, s1_arc_gate_) — from cameras.json."""
    s = str(spec or "all")
    if s == "all": return list(range(1, n + 1))
    out = []
    for tok in [t.strip() for t in s.split(",") if t.strip()]:
        if tok.isdigit(): out.append(int(tok))
        elif "-" in tok and tok.replace("-", "").isdigit(): a, b = tok.split("-"); out += list(range(int(a), int(b) + 1))
        elif names: out += [i + 1 for i, nm in enumerate(names) if nm == tok or nm.startswith(tok)]
    return sorted(set(k for k in out if 1 <= k <= n))


def _on(v, default=True):
    """A flag key: true/false (also "yes"/"no", 0/1); no value — the default."""
    if v is None or v == "": return default
    return str(v).strip().lower() not in ("false", "no", "0", "no", "off")


def _camera(B, sc, spec, k):
    cam = bpy.data.objects.get(spec["name"]) if spec.get("name") else None
    if cam is None or cam.type != "CAMERA": cam = B.make_camera(spec, spec.get("name", f"cam{k}"), sc)
    sc.camera = cam
    W, H = spec["resolution"]; sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = W, H, 100
    return cam


PASS = {}                                                                 # pass times of the last frame: sky, light, effects (s)
FX_CASTERS = True                                                         # shadows of held-out objects in the effects pass: final — yes, trial — no


def _lit_far(B, bl_far, sc, P, spec, k, d):
    """A lit frame of the plan through camera k into folder d (the bl_color.render_colour recipe): the camera sky (sky.exr), geometry on a transparent background
    with depth (lit.exr), far.json for far.py post. The film settings are restored."""
    cam = _camera(B, sc, spec, k); PASS.clear(); t0 = time.time()
    bl_far.fit_clip(cam, P); bl_far.per_camera(sc, spec)
    bl_far.render_sky(sc, os.path.join(d, "sky.exr")); PASS["sky"] = round(time.time() - t0, 1)
    fxl = _fx_layer(sc)                                                  # volumes and sparks of the frame's explosions (the "explosions" step): a layer over the haze, hidden in the frame
    for o in fxl: o.hide_render = True
    far = {"plan": P["path"], "hash": P["hash"], "ev": P["light"]["ev"], "mode": "lit", "sky_bg": SKY_BG, "camera": spec,
           "fx": False, "frame": None, "scene_fx": False, "meter": True}
    json.dump(far, open(os.path.join(d, "far.json"), "w"))
    im, vl = sc.render.image_settings, sc.view_layers[0]
    sc.render.film_transparent = True; im.color_mode = "RGBA"            # the sky is the backdrop (far.py post)
    vl.use_pass_z = True
    im.media_type, im.file_format, im.color_depth = "MULTI_LAYER_IMAGE", "OPEN_EXR_MULTILAYER", "32"
    t1 = time.time(); B.render_to(sc, os.path.join(d, "lit.exr")); PASS["lit"] = round(time.time() - t1, 1)
    vl.use_pass_z = False
    im.media_type, im.file_format, im.color_depth = "IMAGE", "PNG", "8"
    sc.render.film_transparent = False; im.color_mode = "RGB"
    if fxl:                                                              # effects layer (scene_fx.exr): far.py post lays it over the frame with haze
        t2 = time.time(); far["scene_fx"] = bool(bl_far.render_scene_fx(sc, fxl, os.path.join(d, "scene_fx.exr"), casters=FX_CASTERS)); PASS["fx"] = round(time.time() - t2, 1)
        for o in fxl: o.hide_render = False
        json.dump(far, open(os.path.join(d, "far.json"), "w"))
        im.media_type, im.file_format, im.color_depth = "IMAGE", "PNG", "8"; im.color_mode = "RGB"
    elif os.path.exists(os.path.join(d, "scene_fx.exr")): os.remove(os.path.join(d, "scene_fx.exr"))   # a trace of the previous run in the same folder


def _fx_layer(sc):
    """Frame effects that are drawn as a layer (the fx_layer tag: volumes and sparks of explosions, the "explosions" step of Effects): those visible in the scene's frame — by hide_render keys."""
    return [o for o in sc.objects if o.get("fx_layer") and not o.hide_render]


def _frame_of(spec, sc):
    """The scene frame of a camera: for take cameras (the "take" step, cameras.json: the frame field) — the take's frame, for the others — the scene's first frame."""
    return int(spec["frame"]) if spec.get("frame") is not None else sc.frame_start


def _take_files(sc, SD, out, ks, cams):
    """Take by steps: the figures and joints of every take frame (cameras with a frame field) — from the keys of the open scene, in the bl_take format (the old take
    path): cam<K>/figures.obj (meshes in the frame's pose, "o figure_<id>") and cam<K>/joints.json (joints by skeleton, height from the scene's figkit.json).
    The map bundle substitutes the figures with them (bl_pack.Pack.swap_figures). → the numbers of cameras whose files were written."""
    import numpy as np
    from core.assets.rig_joints import rig_kind, joints_of
    kp = os.path.join(SD, "figkit.json")
    named = json.load(open(kp)).get("named", {}) if os.path.exists(kp) else {}
    ids = [f for f in named if bpy.data.objects.get(f) is not None]
    done = []
    for k in ks:
        if cams[k - 1].get("frame") is None or not ids: continue
        sc.frame_set(_frame_of(cams[k - 1], sc)); dg = bpy.context.evaluated_depsgraph_get()
        d = os.path.join(out, f"cam{k}"); os.makedirs(d, exist_ok=True); lines, base, J = [], 1, {}
        for fid in ids:
            root = bpy.data.objects[fid]; V, F = [], []
            for o in [root, *root.children_recursive]:
                if o.type != "MESH" or not o.visible_get(): continue
                ev = o.evaluated_get(dg); me = ev.to_mesh(); Mw = np.array(ev.matrix_world)
                co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3) @ Mw[:3, :3].T + Mw[:3, 3]
                off = len(V); V.extend(co.tolist()); F += [[off + i for i in p.vertices] for p in me.polygons]; ev.to_mesh_clear()
            lines.append(f"o figure_{fid}"); lines += [f"v {x:.4f} {y:.4f} {z:.4f}" for x, y, z in V]
            lines += ["f " + " ".join(str(base + i) for i in f) for f in F]; base += len(V)
            arm = next((o for o in root.children_recursive if o.type == "ARMATURE" and rig_kind(o)), None)
            if arm is None: continue
            jj, hd, _ = joints_of(arm, rig_kind(arm), float(named[fid]["height"]))
            J[fid] = {"joints": jj, "heading": hd, "pos": [round((jj["r_hip"][0] + jj["l_hip"][0]) / 2, 3), round((jj["r_hip"][1] + jj["l_hip"][1]) / 2, 3),
                                                            round(min(v[2] for v in jj.values()), 3)]}
        open(os.path.join(d, "figures.obj"), "w").write("\n".join(lines) + "\n"); json.dump(J, open(os.path.join(d, "joints.json"), "w"))
        done.append(k)
    return done


def _maps(ctx, sc, SD, out, ks, cams, P):
    """Map bundle for cameras ks: bl_pack passes over a maps scene made from the open build scene, then core/outputs/pack.py. → (done, failures)."""
    from core.blender import bl_scene as B
    from core.outputs import bl_scene_blend
    from core.outputs import bl_pack
    take = _take_files(sc, SD, out, ks, cams)                            # take: figures and joints of the frames — while the open scene is intact
    sc2, statics, members, objs, W, kit = bl_scene_blend.scene_for_maps(sc, SD, frame=_frame_of(cams[ks[0] - 1], sc), edits=False)   # the open build scene is destroyed here
    pk = bl_pack.Pack(sc2, statics, members, objs, W, kit, SD, out, plan=P)
    named = bl_scene_blend.FRESH.get("named")
    raws = {}
    for k in ks:
        d = os.path.join(out, f"cam{k}"); raw = os.path.join(d, "raw"); os.makedirs(raw, exist_ok=True)
        if k in take: pk.swap_figures(os.path.join(d, "figures.obj"))    # take frame: figures in the frame's pose (the frame's joints are in cam<K>/joints.json, written by _take_files)
        elif named: json.dump(named, open(os.path.join(d, "joints.json"), "w"))   # figure joints from the .blend — pack.py takes them next to raw
        pk.camera(k, raw, cams[k - 1]); raws[k] = raw
    py = os.path.join(ctx["root"], ".venv", "bin", "python"); done, fail = [], []
    for k in ks:
        r = subprocess.run([py, "-m", "core.outputs.pack", SD, str(k), raws[k], os.path.dirname(raws[k])],
                           capture_output=True, text=True, cwd=ctx["root"])
        if r.returncode:
            fail.append(k); print(f"pack.py, camera {k}:\n{(r.stdout + r.stderr)[-800:]}", flush=True)
        else: done.append(k)
    return done, fail


def _weight(sc):
    """Scene weight by collections (set by the steps): objects, instances, triangles — what is expensive in the frame; once per run."""
    import bpy
    dg = bpy.context.evaluated_depsgraph_get(); tri_of, W = {}, {}
    for inst in dg.object_instances:
        ob = inst.object
        if ob.type == "VOLUME":                                         # volumes (smoke, fire of effects): no triangles, yet the frame is expensive
            col = (ob.original.users_collection[0].name if ob.original.users_collection else "—")
            w = W.setdefault(col, {"objects": 0, "instances": 0, "tris": 0}); w["volumes"] = w.get("volumes", 0) + 1; continue
        if ob.type != "MESH": continue
        src = inst.parent.original if inst.is_instance and inst.parent else ob.original
        col = src.users_collection[0].name if src.users_collection else "—"
        key = getattr(ob.data, "name_full", ob.name)
        if key not in tri_of: tri_of[key] = sum(len(p.vertices) - 2 for p in ob.data.polygons)
        w = W.setdefault(col, {"objects": 0, "instances": 0, "tris": 0})
        w["instances" if inst.is_instance else "objects"] += 1; w["tris"] += tri_of[key]
    return dict(sorted(W.items(), key=lambda kv: -kv[1]["tris"]))


def _objects(sc, path):
    """Boxes of objects with sk_id (a root with its descendants is one box) that are visible to the render: the stage service compares the sketch stand-in
    with the trial model by sk_id (core.services.stage diff). Written for every frame view."""
    from mathutils import Vector
    out = {}
    for o in sc.objects:
        sid = o.get("sk_id")
        if sid is None or o.hide_render or (o.parent is not None and o.parent.get("sk_id") == sid): continue
        pts = [q.matrix_world @ Vector(c) for q in (o, *o.children_recursive) if q.type in ("MESH", "CURVE", "CURVES", "VOLUME")
               and not q.hide_render for c in q.bound_box]
        if not pts: continue
        out[str(sid)] = {"kind": o.get("sk_kind"), "layer": o.get("sk_layer"), "stand_in": bool(o.get("sk_stand_in")),
                         "min": [round(min(p[i] for p in pts), 2) for i in range(3)], "max": [round(max(p[i] for p in pts), 2) for i in range(3)]}
    json.dump(out, open(path, "w"), ensure_ascii=False)
    return len(out)


def _contact(ctx, out, ks):
    """Contact sheet: the lit frames in a single JPG (up to 24 thumbnails, core/outputs/sheet.py) — for the caller to view at once. → the path or None."""
    lits = [os.path.join(out, f"cam{k}", "lit.png") for k in ks if os.path.exists(os.path.join(out, f"cam{k}", "lit.png"))]
    if not lits: return None
    sel = lits if len(lits) <= 24 else [lits[round(i * (len(lits) - 1) / 23)] for i in range(24)]
    path = os.path.join(out, "contact.jpg")
    r = subprocess.run([os.path.join(ctx["root"], ".venv", "bin", "python"), "-m", "core.outputs.sheet", path, "4", "320",
                        *[f"{q}:{os.path.basename(os.path.dirname(q))}" for q in sel]], capture_output=True, text=True, cwd=ctx["root"])
    return path if r.returncode == 0 and os.path.exists(path) else None


def render(ctx):
    from core.blender import bl_scene as B
    from core.outputs import bl_scene_blend
    sc, SD, out, keys = ctx["sc"], ctx["scene"], ctx["out"], ctx["keys"]
    global FX_CASTERS; FX_CASTERS = ctx.get("profile") == "final"     # trial and sketch — held-out objects cast no shadows in the effects pass
    cams = json.load(open(os.path.join(SD, "cameras.json")))
    ks = _select(keys.get("cameras"), len(cams), [c.get("name", "") for c in cams])
    want_light, want_maps = _on(keys.get("light")), _on(keys.get("maps"))
    sketch = str(keys.get("look") or "lit") == "sketch"                  # sketch frame (core/outputs/bl_sketch.py; stage sketch)
    P = None
    fp = ctx["gives"].get("light_plan")
    if want_light and fp and os.path.exists(fp):
        from core.light import bl_far
        P = json.load(open(fp)); P["path"] = fp
    bl_scene_blend.apply_edits(SD)                                       # plan edits (edits.json) — before the frame, shared with the maps
    os.makedirs(out, exist_ok=True)
    if sketch:
        from core.outputs import bl_sketch as SK
        S = SK.prepare(sc, P)                                            # Workbench, color from an attribute, shadow from the plan's luminary; the far terrain gets its own pass
    else:
        sc.render.engine = "BLENDER_EEVEE"
        if P is not None:                                                # the light extension point also during the frames: the components' light state,
            from core import hooks; hooks.call("light", sc, P)           # which is not saved in the .blend (foliage shadow doubles — Plants)
        sc.eevee.taa_render_samples = int(keys.get("samples") or SAMPLES.get(ctx["profile"], 16))
        sc.render.film_transparent = False
        im = sc.render.image_settings; im.media_type, im.file_format, im.color_mode, im.color_depth = "IMAGE", "PNG", "RGB", "8"
    weight = _weight(sc); tm, passes = {}, {}                                         # frame cost: scene weight by steps and the frame time
    _objects(sc, os.path.join(out, "objects.json"))                       # sk_id object boxes — for stage comparison (core.services.stage diff)
    resume = _on(keys.get("resume"), False); skipped_done = 0
    for k in ks:                                                         # 1. the lit frame, while the build scene is intact
        d = os.path.join(out, f"cam{k}"); os.makedirs(d, exist_ok=True); t0 = time.time()
        if resume and (all(os.path.exists(os.path.join(d, f_)) for f_ in ("lit.exr", "sky.exr", "far.json")) if (P and not sketch)
                       else os.path.exists(os.path.join(d, "lit.png"))):   # resuming after a failure: an already rendered frame is left alone
            skipped_done += 1; continue
        sc.frame_set(_frame_of(cams[k - 1], sc))                         # a take camera has a take frame (actor and camera keys)
        if sketch:
            cam = _camera(B, sc, cams[k - 1], k); PASS.clear()
            PASS.update(SK.frame(sc, cam, d, S, P, min(1.0, 1280.0 / max(cams[k - 1]["resolution"][0], 1))))   # at most 1280 wide
        elif P: _lit_far(B, bl_far, sc, P, cams[k - 1], k, d)
        else:
            _camera(B, sc, cams[k - 1], k); B.render_to(sc, os.path.join(d, "lit.png"))
        tm[k] = round(time.time() - t0, 1)
        if PASS: passes[k] = dict(PASS)
    info = {"frames": len(ks), "cameras": ks, "light": bool(P), "look": "sketch" if sketch else "lit", "sec_per_frame": tm, "resumed_skip": skipped_done,
            "sec_mean": round(sum(tm.values()) / max(len(tm), 1), 1), "weight": weight, "passes": passes}
    if sketch:                                                           # sketch compositing: sky, far terrain, near part → lit.png (project Python)
        info["sketch"] = S["stats"]
        r = subprocess.run([os.path.join(ctx["root"], ".venv", "bin", "python"), "-m", "core.outputs.sketch", *[os.path.join(out, f"cam{k}") for k in ks]],
                           capture_output=True, text=True, cwd=ctx["root"])
        if r.returncode: raise SystemExit("sketch.py: " + (r.stdout + r.stderr)[-600:])
    if want_maps:                                                        # 2. maps — the build scene is destroyed
        from core.light import bl_far
        node = os.path.join(out, "node.json")                            # pack.py looks for the run node next to cam<K>
        if P:
            if os.path.abspath(P["path"]) != os.path.abspath(node): shutil.copy(P["path"], node)
            plan = P
        else:                                                            # without the build's light — the scene's node (as in frames.py without --light)
            if os.path.exists(node): os.remove(node)
            plan = bl_far.ensure_plan(SD, node)
        done, fail = _maps(ctx, sc, SD, out, ks, cams, plan)
        info.update(maps=done, maps_failed=fail)
    if P and not sketch:                                                 # 3. haze by depth, AgX, the plan's exposure → lit.png
        dirs = [os.path.join(out, f"cam{k}") for k in ks]
        if resume:                                                       # a composited frame (lit.png newer than lit.exr) — do not composite again
            dirs = [d_ for d_ in dirs if not (os.path.exists(os.path.join(d_, "lit.png")) and os.path.exists(os.path.join(d_, "lit.exr"))
                                             and os.path.getmtime(os.path.join(d_, "lit.png")) >= os.path.getmtime(os.path.join(d_, "lit.exr")))]
        r = subprocess.run([os.path.join(ctx["root"], ".venv", "bin", "python"), "-m", "core.light.far", "post", *dirs],
                           capture_output=True, text=True, cwd=ctx["root"])
        if r.returncode: raise SystemExit("far.py post: " + r.stderr[-600:])
    orbit = [k for k in ks if str(cams[k - 1].get("name", "")).startswith("orbit")]
    if keys.get("video") and len(orbit) > 1:                             # 4. the fly-around video
        seq = os.path.join(out, "_orbit"); shutil.rmtree(seq, ignore_errors=True); os.makedirs(seq)
        for i, k in enumerate(orbit): shutil.copy(os.path.join(out, f"cam{k}", "lit.png"), os.path.join(seq, f"{i:04d}.png"))
        fps = str(keys.get("fps") or 6); mp4 = os.path.join(out, "orbit.mp4")
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", fps, "-i", os.path.join(seq, "%04d.png"),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", mp4], capture_output=True, text=True)
        info["video"] = mp4 if r.returncode == 0 else f"ffmpeg: {r.stderr[-200:]}"
        shutil.rmtree(seq, ignore_errors=True)
    info["contact"] = _contact(ctx, out, ks)
    json.dump(info, open(os.path.join(out, "frames.json"), "w"), ensure_ascii=False, indent=1)
    if want_maps and info.get("maps_failed"): raise SystemExit(f"maps: pack.py failed for cameras {info['maps_failed']}")
    return info

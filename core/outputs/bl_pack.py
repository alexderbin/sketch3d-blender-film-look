"""Frame passes for the map bundle: object ids and body parts (Workbench, flat color), depth, normals and light (Eevee, EXR).
Run: blender -b -P core/blender/run.py -- core/outputs/bl_pack.py -- <scene folder> <camera 1..> <raw folder>
     blender -b -P core/blender/run.py -- core/outputs/bl_pack.py -- <scene folder> <K1,K2,…> <output folder>   — the scene is loaded once for all cameras,
     the passes of camera K go to <output folder>/cam<K>/raw, after each one a line "__FRAME__ <raw>" (the map build is driven by core/outputs/frames.py;
     the Blender process stays warm).
Result in raw: id.png, part.png, passes.exr, beauty.exr, objects.json, camera.json.
Take (components/time/bl_take.py): if <output folder>/cam<K>/figures.obj exists, the figures of frame K are replaced by the poses of that frame.
Source of parts ("one scene truth"): by default the build's .blend (core/outputs/bl_scene_blend.py), if it exists
and the scene is not a battle; otherwise the gray copy scene.obj. source=gray|blend (or SK_PACK_SOURCE) — forced.
Module (stepwise build): preparation and passes are functions, importing starts nothing. `Pack(sc, statics, members, objs, W, kit,
scene_dir, dst[, plan])` — the object table, tones, the light node (one for all cameras); `Pack.camera(k, raw, spec)` — the passes of one camera
into the raw folder. Script launch — `main()` (as before: arguments, "__FRAME__ <raw>", DONE). The "frames" output of a built scene takes the same
functions over a maps scene built from the open build scene (core/outputs/bl_scene_blend.scene_for_maps)."""
import core.paths        # noqa: E402  core parts on sys.path
import json
import os
import sys

import bpy

from core.blender import bl_scene as B
from core.contract import classes as C

ROOT = core.paths.ROOT


def plan_for(scene_dir, dst):
    """The run's light-and-sky node (frames.py → node.json next to the output folder) or the scene's (far.yaml / world.json → out/far/<scene>/pack_plan.json)."""
    from core.light import bl_far
    from core.light import far
    return far.node_near(dst, os.path.dirname(dst), os.path.dirname(os.path.dirname(dst))) or bl_far.ensure_plan(scene_dir, os.path.join(
        ROOT, "out", "far", os.path.basename(os.path.abspath(scene_dir)), "pack_plan.json"))


class Pack:
    """Maps scene: objects (statics, members, objs), the node's far terrain and city, the object table, tones, the sketch light.
    One for all cameras; the camera passes are camera()."""

    def __init__(self, sc, statics, members, objs, W, kit, scene_dir, dst, plan=None):
        from core.light import bl_far
        from core.light import far
        self.sc, self.statics, self.members, self.objs, self.W, self.kit = sc, statics, members, objs, W, kit
        self.scene_dir, self.dst = scene_dir, dst
        self.FARP = FARP = plan or plan_for(scene_dir, dst)
        _fo = bl_far.attach_terrain(sc, FARP, bpy.data.objects.get("ground_far")); self.statics += _fo; self.objs += _fo   # far terrain — if the node has it
        _city, _protos = bl_far.attach_city(sc, FARP)                         # far city: one object (class tower), the number's color goes to the prototypes too
        if _city is not None: self.statics.append(_city); self.objs.append(_city)
        reg_kind = {e["id"]: e["kind"] for e in W["reg"]}
        # --- object table: number → class, kind, height (one for all cameras)
        named_h = dict(kit.get("named_heights", {})) if kit else {}
        member_info = {mb["id"]: mb for mb in kit["members"]} if kit else {}
        ids = B.paint_ids(self.objs)
        if _city is not None:
            for _p in _protos: _p.color = _city.color
        table = {}
        for ob in self.objs:
            num = ids[ob.name]
            if ob.get("member"):
                mb = member_info[ob.name]
                table[num] = {"name": ob.name, "kind": "member", "class": "member", "crowd": mb["crowd"], "height_m": mb["height"],
                              "member": True, "line": ob.name}
                continue
            cid = C.canon_name(ob.name)
            kind = "ground" if ob.name == "ground_far" else "ridge" if ob.name.startswith("far_terrain") else "tower" if ob.name == "far_city" else reg_kind.get(cid, "box")
            table[num] = {"name": ob.name, "kind": kind, "class": C.class_of(kind), "id": cid,
                          "height_m": round(named_h.get(cid, ob.dimensions.z), 3), "member": False,
                          "line": "ground" if ob.name == "ground_far" else ob.name}
        self.table, self.ids = table, ids
        self.tpl_meshes = list({o.data.name: o.data for o in members}.values())
        self.pm = B.part_materials()
        self.tone = {ob.name: B.tone_material(C.ALBEDO.get(table[ids[ob.name]]["class"], 0.5)) for ob in self.statics if ob.type == "MESH"}
        self.member_tone = B.tone_material(C.ALBEDO["member"])
        SW = far.sketch_world(FARP)                                           # sketch light comes from the node's key luminary (the moon at night), the sky takes the node's tones
        self.sun, self.sky = SW["sun"], SW["sky"]
        self.lit = False

    def swap_figures(self, path):
        """Take (components/time/bl_take.py): the frame's figures come from cam<K>/figures.obj (poses of this frame); the id number, id color and tone are the same by name."""
        old = [o for o in self.statics if o.name.startswith("figure_")]; keep = {o.name: tuple(o.color) for o in old}
        for o in old: bpy.data.objects.remove(o, do_unlink=True)
        before = set(bpy.data.objects); bpy.ops.wm.obj_import(filepath=path, forward_axis="Y", up_axis="Z")
        new = [o for o in bpy.data.objects if o not in before]
        for o in new:
            o.color = keep.get(o.name, (0, 0, 0, 1)); B._smooth_by_angle(o, 50)
        self.statics = [o for o in self.statics if o not in old] + new
        self.objs = [o for o in self.objs if o not in old] + [o for o in new if o.type == "MESH"]

    def camera(self, k, raw, cam_spec):
        """Passes of camera K into the raw folder: id.png, part.png, passes.exr (depth, normals, one sample), beauty.exr, objects.json, camera.json."""
        from core.light import bl_far
        sc, statics = self.sc, self.statics
        os.makedirs(raw, exist_ok=True)
        _cam = B.make_camera(cam_spec, f"cam{k}", sc)
        bl_far.fit_clip(_cam, self.FARP)
        json.dump(self.table, open(os.path.join(raw, "objects.json"), "w"), ensure_ascii=False)
        json.dump(cam_spec, open(os.path.join(raw, "camera.json"), "w"), ensure_ascii=False)
        # 1. object ids (object color, materials do not take part)
        B.flat_workbench(sc)
        B.render_to(sc, os.path.join(raw, "id.png"))
        # 2. body parts: a template has a tag — the material by part index; everything else gets part0
        for ob in statics:
            if ob.type == "MESH":
                ob.data.materials.clear(); ob.data.materials.append(self.pm[0])
        B.set_slots(self.tpl_meshes, self.pm)
        B.flat_workbench(sc, color_type="MATERIAL")
        B.render_to(sc, os.path.join(raw, "part.png"))
        # 3. Eevee: tone by class, the scene world's sun and sky; depth and normals in EXR
        for ob in statics:
            if ob.type == "MESH":
                ob.data.materials.clear(); ob.data.materials.append(self.tone[ob.name])
        B.set_slots(self.tpl_meshes, [self.member_tone])
        if not self.lit:
            B.add_sun(sc, self.sun.get("azimuth", 220), self.sun.get("elevation", 30), strength=3.0)
            B.sky_world(sc, self.sky.get("top", 0.78), self.sky.get("horizon", 0.94)); self.lit = True
        B.look_eevee(sc)
        vl = sc.view_layers[0]
        vl.use_pass_z = True
        vl.use_pass_normal = True
        sc.render.image_settings.media_type = "MULTI_LAYER_IMAGE"
        sc.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
        sc.render.image_settings.color_depth = "32"
        # depth and normals — a single sample without blur (otherwise at the edge by the sky the depth is mixed with 1e10), light — separately, with antialiasing
        sc.eevee.taa_render_samples = 1
        sc.render.filepath = os.path.join(raw, "passes.exr")
        bpy.ops.render.render(write_still=True)
        sc.eevee.taa_render_samples = 32
        vl.use_pass_z = False
        vl.use_pass_normal = False
        sc.render.filepath = os.path.join(raw, "beauty.exr")
        bpy.ops.render.render(write_still=True)
        sc.render.image_settings.media_type = "IMAGE"


def main():
    scene_dir, ks, dst = sys.argv[sys.argv.index("--") + 1:][:3]
    OPT = dict(x.split("=", 1) for x in sys.argv[sys.argv.index("--") + 1:][3:] if "=" in x)
    cams = [int(x) for x in ks.split(",") if x]          # "2," is one camera in list mode (frames.py)
    multi = "," in ks
    SRC = OPT.get("source", os.environ.get("SK_PACK_SOURCE", "auto"))   # auto: the build's .blend if it exists and the scene is not a battle; otherwise the gray copy
    if SRC in ("blend", "auto"):                                          # maps straight from the build's .blend — one scene truth
        from core.outputs import bl_scene_blend
        if SRC == "auto" and not bl_scene_blend.blend_of(scene_dir): SRC = "gray"
    if SRC in ("blend", "auto"):
        sc, statics, members, objs, W, kit = bl_scene_blend.load_scene(scene_dir)
    else:
        sc, statics, members, objs, W, kit = B.load_scene(scene_dir)
    P = Pack(sc, statics, members, objs, W, kit, scene_dir, dst)        # the light-and-sky node: the run's (frames.py → node.json) or the scene's (far.yaml / world.json)
    specs = B.load_cameras(scene_dir)
    for k in cams:
        raw = os.path.join(dst, f"cam{k}", "raw") if multi else dst
        os.makedirs(raw, exist_ok=True)
        if multi and os.path.exists(os.path.join(dst, f"cam{k}", "figures.obj")): P.swap_figures(os.path.join(dst, f"cam{k}", "figures.obj"))
        elif multi and "bl_scene_blend" in sys.modules and bl_scene_blend.FRESH.get("named"):   # from the .blend: figure joints come from there too (pack.py takes
            json.dump(bl_scene_blend.FRESH["named"], open(os.path.join(os.path.dirname(raw), "joints.json"), "w"))   # joints.json next to raw)
        P.camera(k, raw, specs[k - 1])
        sys.__stdout__.write(f"__FRAME__ {raw}\n"); sys.__stdout__.flush()
    print("DONE", dst, len(P.objs), "n_objects,", len(cams), "frame(s)")


if __name__ == "__main__":
    main()

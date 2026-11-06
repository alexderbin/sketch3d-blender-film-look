"""A sketch frame — the Blender part of the "frames" output (core/outputs/bl_frames_blend.py, the look: sketch knob; stage sketch).
Workbench over the built scene: the Col attribute color (building masses and stand-ins have their own; for other meshes the material color
is written as an attribute here — the output does not save the scene), cavities, object outline, studio light in world axes with the key from the plan's
luminary (the sun; at night the moon): the lit side of forms faces the luminary; there is no cast shadow (Workbench shadows are volumes,
and open relief shades the whole frame with them). Two passes: the near part clipped by its own depth and the plan's far terrain (far_*) with its own
depth scale. Compositing with the sky in the node's tones and the levels — core/outputs/sketch.py (project Python) → cam<K>/lit.png.
    S = prepare(sc, P) — once per output; frame(sc, cam, d, S, P, scale) — per frame: _near.png, _far.png, sketch.json"""
import json, math, os, time

import bpy
import numpy as np
from mathutils import Vector

from core.blender import stand_in as SI

FAR_KEEP = ("far_terrain", "far_city")                  # in the sketch the plan's far terrain is mountains and city as silhouettes; clouds, fires, flashes are omitted
SKY_DAY = ((0.52, 0.64, 0.84), (0.80, 0.84, 0.90))      # sky top and horizon without a light plan (display)
HAZE = 0.35                                             # far-terrain haze toward the horizon tone
STUDIO_KEY_AZ = 244.4                                   # azimuth of the key light of the "Default" studio in world axes at rotation 0 (trial:
                                                        # rotation 0/90/180° → 244/155/65°, azimuth counterclockwise from +x, as in far.sun_vec)


def _lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


_IMG, _MAT = {}, {}


def _image_rgb(img):
    """Mean color of an image (linear) from a downscaled 16×16 copy; no data — None."""
    if img.name in _IMG: return _IMG[img.name]
    rgb = None
    try:
        cp = img.copy(); cp.scale(16, 16); px = np.array(cp.pixels[:], float).reshape(-1, 4); bpy.data.images.remove(cp)
        px = px[px[:, 3] > 0.5] if (px[:, 3] > 0.5).any() else px
        rgb = px[:, :3].mean(0)
        if not img.is_float and img.colorspace_settings.name == "sRGB": rgb = _lin(rgb)   # 8-bit — as in the file
        rgb = tuple(float(x) for x in rgb)
    except Exception:                                    # noqa: BLE001 — no file, or it does not load: the color comes from the object's class
        rgb = None
    _IMG[img.name] = rgb
    return rgb


def _upstream_image(sock, depth=5):
    """The first image upstream of an input (an Image Texture through mixes and color correction); none — None."""
    if depth == 0 or not sock.is_linked: return None
    n = sock.links[0].from_node
    if n.type == "TEX_IMAGE" and n.image: return n.image
    for s in n.inputs:
        im = _upstream_image(s, depth - 1) if s.is_linked else None
        if im: return im
    return None


def mat_rgb(m):
    """Material color for the sketch (linear): an unlinked Base Color, the mean color of an image upstream of Base Color, emission, a set
    viewport color; otherwise None — the color comes from the object's class."""
    if m is None: return None
    if m.name in _MAT: return _MAT[m.name]
    rgb = None
    if m.use_nodes and m.node_tree:
        bs = next((n for n in m.node_tree.nodes if n.type in ("BSDF_PRINCIPLED", "BSDF_DIFFUSE")), None)
        em = next((n for n in m.node_tree.nodes if n.type == "EMISSION"), None)
        if bs:
            s = bs.inputs["Base Color" if bs.type == "BSDF_PRINCIPLED" else "Color"]
            if not s.is_linked: rgb = tuple(s.default_value[:3])
            else:
                im = _upstream_image(s)
                rgb = _image_rgb(im) if im else None
        elif em: rgb = tuple(em.inputs["Color"].default_value[:3])
    if rgb is None and tuple(round(x, 3) for x in m.diffuse_color[:3]) != (0.8, 0.8, 0.8): rgb = tuple(m.diffuse_color[:3])
    _MAT[m.name] = rgb
    return rgb


def _paint(me, kind_rgb):
    """Per-face color as a Col attribute, if there is none: the face's material color, otherwise the object's class color. → 1 if written."""
    if "Col" in me.color_attributes:
        me.color_attributes.active_color = me.color_attributes["Col"]; return 0
    n = len(me.polygons)
    if not n: return 0
    pal = np.array([(mat_rgb(m) or kind_rgb) for m in me.materials] + [kind_rgb], float)   # the last one is for an out-of-range index
    mi = np.empty(n, np.int32); me.polygons.foreach_get("material_index", mi)
    lt = np.empty(n, np.int32); me.polygons.foreach_get("loop_total", lt)
    rgba = np.c_[pal[np.clip(mi, 0, len(pal) - 1)], np.ones(n)]
    ca = me.color_attributes.new("Col", "BYTE_COLOR", "CORNER"); ca.data.foreach_set("color", np.repeat(rgba, lt, axis=0).ravel())
    me.color_attributes.active_color = ca
    return 1


def _sky(P):
    """Sky top and horizon (display) and the key luminary's color: the node's sketch tones (far.sketch_world) with the tint of the plan's sky."""
    if not P: return SKY_DAY
    from core.light import far as F
    W = F.sketch_world(P)
    e = np.asarray(P["light"]["sky_e_rgb"], float); tint = e / max(float(e.max()), 1e-6)
    top = np.clip(W["sky"]["top"] * (0.55 + 0.45 * tint), 0, 1); hor = np.clip(W["sky"]["horizon"] * (0.8 + 0.2 * tint), 0, 1)
    return [round(float(x), 4) for x in top], [round(float(x), 4) for x in hor]


def prepare(sc, P):
    """Prepares the scene for a sketch: removes volumes, layered effects (fx_layer, sketch_fx) and the plan's far terrain except mountains and the city; writes color as an attribute;
    Workbench; studio light from the plan's key luminary. → S: near and far objects (including ones hidden at this frame: visibility by keys
    changes from frame to frame — explosion stand-ins), the corners of the near boxes, the sky, counters."""
    t0 = time.time(); hidden = 0
    for o in list(sc.objects):                                          # removed from the scene, not just hide_render: the frame would restore keyed visibility
        nm = o.name
        if o.type == "VOLUME" or o.get("sketch_fx") or o.get("fx_layer") or (nm.startswith("far_") and not nm.startswith(FAR_KEEP)):
            for c in list(o.users_collection): c.objects.unlink(o)       # (the output does not save the scene)
            hidden += 1
    kinds = {}
    for o in sc.objects:
        if o.type == "MESH" and o.data is not None: kinds.setdefault(o.data.name, o.get("sk_kind") or o.get("kind"))
    painted = sum(_paint(me, SI.KIND_RGB.get(kinds.get(me.name), SI.GREY)) for me in bpy.data.meshes if me.users)
    sc.render.engine = "BLENDER_WORKBENCH"
    d = sc.display; sh = d.shading
    d.render_aa = "8"
    sh.light, sh.color_type = "STUDIO", "VERTEX"
    sh.show_cavity, sh.cavity_type = True, "WORLD"
    sh.show_shadows = False                                              # the Workbench cast shadow is made of volumes: for an open mesh (relief)
    sh.show_object_outline, sh.object_outline_color = True, (0.0, 0.0, 0.0)   # the shadow falls on everything (trial: the frame went 0.57 → 0.195 for any
    sh.show_specular_highlight = False                                   # direction); the light direction comes from the studio in world axes
    sh.use_world_space_lighting = True
    if P:                                                                # key luminary: the sun; at night the moon (as in far.sketch_world)
        key = P["moon"] if P["sun"]["el"] <= 0 and P["moon"].get("on") else P["sun"]
        sh.studiolight_rotate_z = math.radians(STUDIO_KEY_AZ - float(key["az"]))
    vs = sc.view_settings; vs.view_transform, vs.look, vs.exposure, vs.gamma = "Standard", "None", 0.0, 1.0
    sc.render.film_transparent = True
    im = sc.render.image_settings; im.media_type, im.file_format, im.color_mode, im.color_depth = "IMAGE", "PNG", "RGBA", "8"
    vis = [o for o in sc.objects if o.type in ("MESH", "CURVE", "CURVES", "POINTCLOUD", "FONT")]   # and those hidden now (keyed visibility)
    far = [o for o in vis if o.name.startswith(FAR_KEEP)]
    near = [o for o in vis if not o.name.startswith(FAR_KEEP)]
    bpy.context.view_layer.update()
    corners = np.array([[tuple(o.matrix_world @ Vector(c)) for c in o.bound_box] for o in near], float).reshape(-1, 3)
    top, hor = _sky(P)
    return {"near": near, "far": far, "corners": corners, "sky_top": top, "sky_horizon": hor,
            "stats": {"hidden": hidden, "painted": painted, "near": len(near), "far": len(far), "sec": round(time.time() - t0, 1)}}


def _horizon(cam, sc):
    """The horizon row (a fraction of the frame height from the top) from the camera tilt, field of view and shift; roll is ignored."""
    W, H = sc.render.resolution_x, sc.render.resolution_y; cd = cam.data
    f = cam.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0)); pitch = math.asin(max(-1.0, min(1.0, f.z)))
    if cd.sensor_fit == "VERTICAL": tv = cd.sensor_height / (2 * cd.lens)
    elif cd.sensor_fit == "AUTO" and H > W: tv = cd.sensor_width / (2 * cd.lens)
    else: tv = cd.sensor_width / (2 * cd.lens) * H / W
    y = H / 2 + (H / 2) / tv * math.tan(pitch) + cd.shift_y * max(W, H)
    return round(y / H, 4)


def frame(sc, cam, d, S, P, scale=1.0):
    """The frame of camera cam into folder d: the near part (_near.png), the plan's far terrain (_far.png), sketch.json for compositing. → pass times."""
    from core.blender import bl_scene as B
    sc.render.resolution_percentage = max(10, min(100, int(round(scale * 100))))
    cd, sh = cam.data, sc.display.shading; clip0 = (cd.clip_start, cd.clip_end); t0 = time.time()
    pos = np.array(cam.matrix_world.translation)
    reach = float(np.linalg.norm(S["corners"] - pos, axis=1).max()) * 1.2 + 50.0 if len(S["corners"]) else clip0[1]
    cd.clip_end = max(50.0, reach); cd.clip_start = max(min(clip0[0], 0.1), cd.clip_end / 4.0e5)   # depth of the outline and cavities: near ≥ far / 400,000
    pf = [(o, o.hide_render) for o in S["far"]]                          # restore the frame's visibility (including keyed) as it was
    for o, _ in pf: o.hide_render = True
    B.render_to(sc, os.path.join(d, "_near.png")); out = {"near": round(time.time() - t0, 1)}
    for o, h in pf: o.hide_render = h
    has_far = False
    if S["far"]:
        t1 = time.time(); pn = [(o, o.hide_render) for o in S["near"]]
        for o, _ in pn: o.hide_render = True
        cd.clip_start, cd.clip_end = clip0
        if P:
            from core.light import bl_far
            bl_far.fit_clip(cam, P)
        B.render_to(sc, os.path.join(d, "_far.png")); has_far = True
        for o, h in pn: o.hide_render = h
        out["far"] = round(time.time() - t1, 1)
    elif os.path.exists(os.path.join(d, "_far.png")): os.remove(os.path.join(d, "_far.png"))
    cd.clip_start, cd.clip_end = clip0
    json.dump({"horizon": _horizon(cam, sc), "sky_top": S["sky_top"], "sky_horizon": S["sky_horizon"], "haze": HAZE, "far": has_far},
              open(os.path.join(d, "sketch.json"), "w"))
    return out

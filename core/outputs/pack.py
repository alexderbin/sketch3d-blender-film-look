"""A frame map bundle from Blender passes (bl_pack.py): control maps for local diffusion and a marker sketch for sb.

ControlNet maps: depth, normal, line (black on white), canny (white on black), seg, ids, pose (OpenPose), mask_fig.
For viewing: tone, notan, toneline. For sb: marker. For checking without images: frame.json.

Usage: .venv/bin/python -m core.outputs.pack <scene folder> <camera 1..> <raw folder> <out folder> [--seed N] [--exposure X]"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image

from core.contract import classes as C
from core.outputs import maps as M
from core.outputs import marker as MK
from core.outputs import openpose as OP

ap = argparse.ArgumentParser()
ap.add_argument("scene_dir")
ap.add_argument("cam", type=int)
ap.add_argument("raw")
ap.add_argument("out")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--exposure", type=float, default=1.0)
ap.add_argument("--pose-min-px", type=float, default=40.0)
ap.add_argument("--depth-window", help="1/z window per shot: lo,hi (from depth_m.inv of the first frame) so the tone does not jump")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def save(name, arr):
    Image.fromarray(arr).save(os.path.join(a.out, name))
    files.append(name)


def u8(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


files = []
cam_spec = json.load(open(f"{a.raw}/camera.json"))
cam = M.Cam(cam_spec)
H, W = cam.H, cam.W
_wj = f"{a.scene_dir}/world.json"                                     # a stepwise-build scene without a description (castle_rock) has no world.json:
world = json.load(open(_wj)) if os.path.exists(_wj) else {"reg": [], "world": {}, "sprites": []}   # the sky and haze come from the light node, there are no sprites
wd = world["world"]
from core.light import far                                                          # the sketch's light and sky come from the run's node (frames.py → node.json) or the scene's
wd = {**wd, **far.sketch_world(far.node_near(a.out, os.path.dirname(a.out)) or far.plan_of(a.scene_dir))}   # (far.yaml / world.json): luminary, sky, haze
table = {int(n): v for n, v in json.load(open(f"{a.raw}/objects.json")).items()}
kit_path = f"{a.scene_dir}/figkit.json"
kit = json.load(open(kit_path)) if os.path.exists(kit_path) else None
fj = os.path.join(os.path.dirname(a.raw), "joints.json")               # take: the joints of this frame (components/time/bl_take.py)
if kit and os.path.exists(fj):
    for n_, v_ in json.load(open(fj)).items(): kit["named"].setdefault(n_, {}).update(v_)

idm = M.read_id(f"{a.raw}/id.png")
part = M.read_id(f"{a.raw}/part.png")
ex = M.read_exr(f"{a.raw}/passes.exr")
if os.path.exists(f"{a.raw}/beauty.exr"):
    ex["combined"] = M.read_exr(f"{a.raw}/beauty.exr")["combined"]
sky = idm == 0
z = np.where(sky, np.inf, ex["z"]).astype(np.float32)
t = cam.ray_len(z)
nrm = ex["normal"]
zw = cam.world(np.where(sky, 1.0, z))[..., 2]

# --- lookup tables by object number
NUM = max(table) + 1
cls_names = C.CLASS_ORDER
cls_of = np.zeros(NUM, np.int32)
role = np.full(NUM, 3, np.int32)
alb = np.full(NUM, 0.5, np.float32)
height = np.zeros(NUM, np.float32)
is_fig = np.zeros(NUM, bool)
grp_names = {"": 0}
grp_of = np.zeros(NUM, np.int32)
for n, v in table.items():
    cls_of[n] = cls_names.index(v["class"])
    role[n] = C.ROLE.get(v["class"], 3)
    alb[n] = C.ALBEDO.get(v["class"], 0.5)
    height[n] = v["height_m"]
    is_fig[n] = v["class"] in ("figure", "member")
    grp_of[n] = grp_names.setdefault(v["line"], len(grp_names))
role[0] = 9
cls = cls_of[idm]
fig = is_fig[idm]

# --- figure heights in the frame
cnt = np.bincount(idm.ravel(), minlength=NUM).astype(np.float64)
zsum = np.bincount(idm.ravel(), weights=np.where(sky, 0, z).ravel().astype(np.float64), minlength=NUM)
zmean = np.where(cnt > 0, zsum / np.maximum(cnt, 1), np.inf)
hpx = np.where(is_fig & (cnt > 0), cam.f * height / np.maximum(zmean, 1e-3), 0).astype(np.float32)
eng = is_fig & (height > 3.2)
big_num = is_fig & (hpx >= MK.FIG_BIG)
small_num = is_fig & ~big_num & (hpx < MK.CROWD_PX) & ~eng
big_grp = np.zeros(len(grp_names), bool)
big_grp[grp_of[big_num]] = True
fig_grp = np.zeros(len(grp_names), bool)
fig_grp[grp_of[is_fig]] = True

# --- depth, normals, lines, segmentation
depth8, (inv_lo, inv_hi) = M.encode_depth(z, sky, tuple(map(float, a.depth_window.split(","))) if a.depth_window else None)
save("depth.png", depth8)
save("normal.png", M.encode_normal(nrm, cam, sky))
gdot = np.abs((nrm * cam.ray_dirs()).sum(-1))
E, T = M.edges(grp_of[idm], part, t, nrm, gdot, 1.0 / cam.f, big_grp, fig_grp)
cov = M.line_cover(E, T)
save("line.png", u8(1 - cov))
save("canny.png", u8(cov > 0.35))
save("seg.png", M.seg_map(cls, cls_names))
pal = M.id_palette(NUM)
save("ids.png", pal[idm])
save("mask_fig.png", u8(fig))

# --- tone: Eevee light with shadows, air, sky, pit, smoke and fire
lum = ex["combined"] @ np.array([0.2126, 0.7152, 0.0722], np.float32)
base = np.clip(M.srgb(lum * a.exposure), 0, 1).astype(np.float32)
base *= M.figure_shade(idm, part, big_num, hpx, big_num[idm])
cav = M.cavity(zw, sky, fig)
base *= 1 - 0.55 * cav
haze = wd.get("haze", {}) or {}
hz = (float(haze.get("distance", 1500)), float(haze.get("tone", 0.86)))
e = np.exp(-np.where(np.isfinite(t), t, 1e9) / hz[0]).astype(np.float32)
img = base * e + hz[1] * (1 - e)
img = np.where(sky, M.sky_tone(cam, wd), img)
fx_c, fx_a, fx_p = M.fx_layer(cam, world["sprites"], t, hz)
tone = np.clip(fx_c + img * (1 - fx_a), 0, 1)
tl = MK.toneline(tone, cov)
save("tone.png", u8(tone))
save("toneline.png", u8(tl))

# --- notan by roles (sketch25.views.notan_design)
sun_az, sun_el = float(wd["sun"]["azimuth"]), float(wd["sun"]["elevation"])
sa, se = math.radians(sun_az), math.radians(sun_el)
sun = np.array([math.cos(sa) * math.cos(se), math.sin(sa) * math.cos(se), math.sin(se)], np.float32)
D, Mid, L_ = 0.1, 0.55, 0.95
r = role[idm]
notan = np.full((H, W), Mid, np.float32)
notan[sky] = L_
notan[r == 0] = D
st = r == 3
lit = (nrm @ sun) > 0.25
notan[st & lit & (alb[idm] >= 0.45)] = L_
notan[st & (alb[idm] < 0.45)] = D
notan[(cav > 0.24) & (r != 0)] = D
ratio = fx_c / np.maximum(fx_a, 1e-3)
notan[(fx_p > 0.3) & (ratio < 0.7) & (notan == L_)] = Mid
notan[(fx_a > 0.45) & (ratio >= 0.9)] = L_
save("notan.png", u8(notan))

# --- marker sketch for sb
bigm = fig & big_num[idm]
smallm = fig & ~bigm & small_num[idm]
midm = fig & ~bigm & ~smallm
mk = MK.make(tone, cov, fig, notan, bigm, smallm, midm, seed=a.seed)
save("marker.png", u8(mk))

# --- OpenPose pose map
poses = []
by_name = {v["name"]: n for n, v in table.items()}
by_id = {v["id"]: n for n, v in table.items() if not v["member"]}
if kit:
    for n, v in kit["named"].items():
        if v.get("pose") is False: continue                          # not a human (the bridge sets this by race) — not drawn in the pose map
        poses.append({"joints": {c: np.array(p) for c, p in v["joints"].items()}, "height_m": v["height"], "num": by_id.get(n, -1)})
    tpl_h = {k_: t_["height"] for k_, t_ in kit["templates"].items()}
    for mb in kit["members"]:
        tj = kit["templates"][mb["tpl"]]["joints"]
        if not tj:
            continue
        d = np.linalg.norm(np.array(mb["pos"]) - cam.pos)
        if cam.f * mb["height"] / max(d, 1e-3) < a.pose_min_px * 0.8:
            continue
        j = OP.mirrored(tj) if mb["tpl"].split("|")[3] == "True" else tj
        poses.append({"joints": OP.to_world(j, mb["pos"], mb["heading"], mb["height"] / tpl_h[mb["tpl"]]), "height_m": mb["height"], "num": by_name.get(mb["id"], -1)})
pose_img, n_pose = OP.render(cam, poses, np.where(sky, 1e9, z), idm, min_px=a.pose_min_px)
save("pose.png", pose_img)

# --- frame.json: what is in the frame, in numbers (to check the composition without viewing images)
order = np.argsort(idm.ravel(), kind="stable")
starts = np.concatenate([[0], np.cumsum(np.bincount(idm.ravel(), minlength=NUM))])
yy, xx = np.divmod(order, W)
objects, crowds = [], {}
for n, v in table.items():
    c_ = int(cnt[n])
    if c_ == 0:
        continue
    sl = slice(starts[n], starts[n + 1])
    bb = [int(xx[sl].min()), int(yy[sl].min()), int(xx[sl].max()), int(yy[sl].max())]
    rec = {"area_px": c_, "bbox": bb, "dist_m": round(float(zmean[n]), 1)}
    if v["member"]:
        c = crowds.setdefault(v["crowd"], {"visible": 0, "area_px": 0, "bbox": [W, H, 0, 0], "dist_m": []})
        c["visible"] += 1
        c["area_px"] += c_
        c["bbox"] = [min(c["bbox"][0], bb[0]), min(c["bbox"][1], bb[1]), max(c["bbox"][2], bb[2]), max(c["bbox"][3], bb[3])]
        c["dist_m"].append(float(zmean[n]))
    else:
        rec.update({"id": v.get("id", v["name"]), "class": v["class"]})
        if v["class"] == "figure":
            rec["height_px"] = round(float(hpx[n]), 1)
        objects.append(rec)
for c in crowds.values():
    d = c.pop("dist_m")
    c["dist_m"] = [round(min(d), 1), round(float(np.median(d)), 1), round(max(d), 1)]
objects.sort(key=lambda o: -o["area_px"])
share = {n: round(100 * float((cls == i).mean()), 2) for i, n in enumerate(cls_names) if (cls == i).any()}
fig_sizes = {"big": int(((big_num) & (cnt > 0)).sum()), "mid": int((is_fig & ~big_num & ~small_num & (cnt > 0)).sum()),
             "small": int((small_num & (cnt > 0)).sum())}
frame = {
    "camera": {"name": cam_spec["name"], "lens_mm": cam_spec["lens_mm"], "pos": [round(float(v), 2) for v in cam.pos],
               "size": [W, H], "horizon_y": None if cam.horizon_y() is None else round(cam.horizon_y(), 3)},
    "sun": {"azimuth": sun_az, "elevation": sun_el},
    "class_share_pct": share,
    "figures_visible": fig_sizes,
    "pose_figures_drawn": n_pose,
    "depth_m": {"near_p2": round(1 / inv_hi, 1), "far_p98": round(1 / max(inv_lo, 1e-6), 1), "inv": [round(inv_lo, 8), round(inv_hi, 8)]},
    "objects": objects[:30],
    "crowds": crowds,
    "files": files,
}
# compact: indentation took 43 % of the size of frame.json
json.dump(frame, open(os.path.join(a.out, "frame.json"), "w"), ensure_ascii=False, separators=(",", ":"))
print("batch:", a.out, "| files", len(files), "| poses", n_pose, "| figures in frame", fig_sizes)

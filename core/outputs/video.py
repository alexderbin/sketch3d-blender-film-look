"""A video from the camera map bundles (push-in, fly-around, crane — components/camera/cam_move.py; any cameras in order): frame K → video.
The maps for the model are lossless (libx264rgb -qp 0: x264 yuv420 changes 95 % of the pose-stick pixels and 87 % of the normals),
the sketch is for the eyes (x264), quad — four maps in one frame.
.venv/bin/python -m core.outputs.video <out/X> <K1,K2,…|a-b> [--fps 12] [--hold 1] [--name shot]
Output: <out/X>/<name>/: marker.mp4, pose.mp4, line.mp4, depth.mp4, quad.mp4."""
import argparse, os, shutil, subprocess

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("cams"); ap.add_argument("--fps", type=int, default=12)
ap.add_argument("--hold", type=int, default=1); ap.add_argument("--name", default="shot"); A = ap.parse_args()
ks = [int(x) for x in A.cams.split(",")] if "-" not in A.cams else list(range(int(A.cams.split("-")[0]), int(A.cams.split("-")[1]) + 1))
D = os.path.join(A.out, A.name); MAPS = ("marker", "pose", "line", "depth")
for m in MAPS:
    shutil.rmtree(os.path.join(D, "_" + m), ignore_errors=True); os.makedirs(os.path.join(D, "_" + m))
i = 0
for k in ks:
    for _ in range(A.hold):
        i += 1
        for m in MAPS: os.symlink(os.path.abspath(os.path.join(A.out, f"cam{k}", f"{m}.png")), os.path.join(D, "_" + m, f"{i:04d}.png"))
ff = lambda *a: subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *a], check=True)
src = lambda m: ["-framerate", str(A.fps), "-i", os.path.join(D, "_" + m, "%04d.png")]
for m in ("pose", "line", "depth"):
    ff(*src(m), "-c:v", "libx264rgb", "-qp", "0", "-preset", "veryfast", "-pix_fmt", "rgb24", os.path.join(D, f"{m}.mp4"))
ff(*src("marker"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", os.path.join(D, "marker.mp4"))
for nm in ("previs", "previs_lit"):                                  # previs (color + outline) and the lit one — for library scenes
    if not all(os.path.exists(os.path.join(A.out, f"cam{k}", f"{nm}.png")) for k in ks): continue
    shutil.rmtree(os.path.join(D, f"_{nm}"), ignore_errors=True); os.makedirs(os.path.join(D, f"_{nm}")); i_ = 0
    for k in ks:
        for _ in range(A.hold):
            i_ += 1; os.symlink(os.path.abspath(os.path.join(A.out, f"cam{k}", f"{nm}.png")), os.path.join(D, f"_{nm}", f"{i_:04d}.png"))
    ff(*src(nm), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", os.path.join(D, f"{nm}.mp4"))
ff(*src("marker"), *src("pose"), *src("line"), *src("depth"), "-filter_complex",   # rgb24: otherwise the concatenation takes the sketch's gray format
   "[0]format=rgb24[a];[1]format=rgb24[p];[2]format=rgb24[l];[3]format=rgb24[d];[a][p]hstack[t];[l][d]hstack[b];[t][b]vstack,scale=1536:-2",
   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", os.path.join(D, "quad.mp4"))
import json
inv = [json.load(open(os.path.join(A.out, f"cam{k}", "frame.json")))["depth_m"].get("inv") for k in ks]
lo, hi = min(v[0] for v in inv if v), max(v[1] for v in inv if v)
same = all(v and abs(v[0] - inv[0][0]) <= 0.01 * abs(inv[0][0]) + 1e-9 and abs(v[1] - inv[0][1]) <= 0.01 * abs(inv[0][1]) + 1e-9 for v in inv)
res = {"frames": i, "fps": A.fps, "seconds": round(i / A.fps, 2), "out": D, "depth_window_same": same}
if not same: res["hint"] = f"depth tone jumps: core/outputs/frames.py … --depth-window={lo:.6g},{hi:.6g} and video.py again"   # one window per shot
print("__RESULT__ " + json.dumps(res, ensure_ascii=False))

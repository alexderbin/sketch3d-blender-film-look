"""Editing — a core output (ARCHITECTURE.md: the mechanism belongs to its owner): shots (frames back to back) → a clip per shot → the video back to back and a title card at the end.
Stepwise build: the "cut" output after "frames" — shots by the run's camera names (orbit_* — fly-around, take_* — take), frames — the lit.png
of the "frames" output. The editing of the "Battle Trailer" format (formats/battle_trailer/trailer_cut.py) is a special case: shots by frame numbers.
    cut(shots, out, fps=12, title="", grade="") → {"shots": [...], "trailer": path}; shots — [(name, [.png frames in order])]"""
import glob
import json
import os
import subprocess

FONT = "/System/Library/Fonts/Supplemental/Georgia Bold.ttf"


def _ff(*a):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *a], check=True)


def _title(out, w, h, text, fps):
    """Title card: 2.5 s of black with a caption, faded (this ffmpeg build has no drawtext — PIL draws the caption)."""
    from PIL import Image, ImageDraw, ImageFont
    png, card = os.path.join(out, "_title.png"), os.path.join(out, "_title.mp4")
    im = Image.new("RGB", (w, h), (0, 0, 0)); d = ImageDraw.Draw(im); fnt = ImageFont.truetype(FONT, max(h // 9, 12))
    x0, y0, x1, y1 = d.textbbox((0, 0), text, font=fnt); d.text(((w - x1 + x0) / 2 - x0, (h - y1 + y0) / 2 - y0), text, font=fnt, fill=(232, 200, 120)); im.save(png)
    _ff("-loop", "1", "-framerate", str(fps), "-i", png, "-t", "2.5", "-vf", "fade=t=in:st=0:d=0.6,fade=t=out:st=1.9:d=0.6",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17", card)
    return card


def cut(shots, out, fps=12, title="", grade="", suffix=""):
    """A clip per shot (frames back to back, color correction grade — an ffmpeg filter), a video of all shots back to back, a title card at the end (title).
    A shot without frames is skipped (so you can see what is still being computed). → {"shots": [names], "trailer": path | None}."""
    os.makedirs(out, exist_ok=True); done = []; vf = "pad=ceil(iw/2)*2:ceil(ih/2)*2" + (f",{grade}" if grade else "")
    for name, frames in shots:
        frames = [f for f in frames if os.path.exists(f)]
        if not frames: continue
        seq = os.path.join(out, f"_{name}"); os.makedirs(seq, exist_ok=True)
        for f in glob.glob(os.path.join(seq, "*.png")): os.remove(f)
        for i, f in enumerate(frames): os.symlink(os.path.abspath(f), os.path.join(seq, f"{i:04d}.png"))
        _ff("-framerate", str(fps), "-i", os.path.join(seq, "%04d.png"), "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17",
            os.path.join(out, f"{name}{suffix}.mp4"))
        done.append(name)
    res = {"shots": done, "trailer": None}
    if not done: return res
    lst = os.path.join(out, "_list.txt"); open(lst, "w").write("".join(f"file '{n}{suffix}.mp4'\n" for n in done))
    body = os.path.join(out, "_body.mp4"); _ff("-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", body)
    final = os.path.join(out, f"trailer{suffix}.mp4")
    if title:
        w, h = map(int, subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", body],
                                       capture_output=True, text=True).stdout.strip().split(","))
        _title(out, w, h, title, fps)
        open(lst, "w").write("file '_body.mp4'\nfile '_title.mp4'\n"); _ff("-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", final)
    else: os.replace(body, final)
    res["trailer"] = final
    return res


def shots_of(run, scene, spec):
    """Shots of a stepwise run: spec "name:camera_name_prefix,…" (orbit:orbit_,take:take_) → [(name, [lit.png in camera order])]
    from the last folder of the "frames" output (<run>/<NN>_frames/cam<K>/lit.png) and the camera names of the run scene (cameras.json)."""
    fr = sorted(glob.glob(os.path.join(run, "*_frames")))
    if not fr: return []
    po = sorted(glob.glob(os.path.join(run, "*_post")))                 # post-processing after the frames: editing takes its cam<K>.png frames
    post = po[-1] if po and os.path.basename(po[-1]) > os.path.basename(fr[-1]) else None
    cams = json.load(open(os.path.join(scene, "cameras.json"))) if os.path.exists(os.path.join(scene, "cameras.json")) else []
    out = []
    def frame_of(i):
        p = os.path.join(post, f"cam{i + 1}.png") if post else None
        return p if p and os.path.exists(p) else os.path.join(fr[-1], f"cam{i + 1}", "lit.png")
    for part in [p for p in str(spec).split(",") if ":" in p]:
        name, prefix = (x.strip() for x in part.split(":", 1))
        out.append((name, [frame_of(i) for i, c in enumerate(cams) if str(c.get("name", "")).startswith(prefix)]))
    return out


def plan_cut(M, keys, ctx):
    """The "cut" output of a stepwise build (project Python): the frames of this run's "frames" output → shots by camera names → the video and the title card."""
    run = os.path.dirname(ctx["out"])
    sh = shots_of(run, ctx["scene"], keys.get("shots") or "orbit:orbit_,take:take_")
    r = cut(sh, ctx["out"], int(keys.get("fps") or 12), str(keys.get("title") or ""), str(keys.get("grade") or ""))
    if not r["shots"]: r["_skipped"] = "no shot frames: this run's 'frames' output did not render cameras with such names"
    return {"trailer": r["trailer"], "shots": ", ".join(r["shots"]), **({"_skipped": r["_skipped"]} if r.get("_skipped") else {})}

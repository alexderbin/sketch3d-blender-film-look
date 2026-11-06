"""Entry of the "Post" component for the step-by-step build: the post output — after the "frames" output.
plan (project Python, no Blender): the frames of this run's last "frames" output (<run>/NN_frames/cam<K>/lit.png, depth,
far.json) → a grading preset (presets.yaml) with knob adjustments → <step folder>/cam<K>.png; the palette by a reference — one transform
for the whole run (the frames do not flicker); a clip (video: true) — by the core's montage (core.outputs.cut) from the graded frames in camera order.
It does not touch the scene; the pack's maps (depth, id, outline) stay as they were.
The preset is chosen by format, at the final stage (the main one is trailer). To change it on a finished run without the pipeline —
into a separate folder next to the frames, for comparison and replacement:
  .venv/bin/python -m components.post.steps <run folder> [preset=anime] [palette=… mix=… letterbox=0 video=true …]
      → <run>/post_<preset>/cam<K>.png, contact.jpg, post.json"""
import glob
import json
import os
import re
import time


def _frames_dirs(run):
    fr = sorted(glob.glob(os.path.join(run, "*_frames")))
    if not fr: return []
    cams = [d for d in glob.glob(os.path.join(fr[-1], "cam*")) if os.path.exists(os.path.join(d, "lit.png"))]
    return sorted(cams, key=lambda d: int(re.sub(r"\D", "", os.path.basename(d)) or 0))


def _overrides(keys):
    """Step knobs → preset adjustments (look.resolve)."""
    o = {}
    if keys.get("palette"): o.setdefault("palette", {})["ref"] = keys["palette"]
    if keys.get("mix") is not None: o.setdefault("palette", {})["mix"] = float(keys["mix"])
    if keys.get("haze_km"): o["haze"] = {"visibility_km": float(keys["haze_km"])}
    if keys.get("fstop"): o.setdefault("dof", {})["fstop"] = float(keys["fstop"])
    if keys.get("focus_m"): o.setdefault("dof", {})["focus_m"] = float(keys["focus_m"])
    if keys.get("grain"): o["grain"] = {"strength": float(keys["grain"])}
    if keys.get("letterbox") is not None: o["letterbox"] = {"aspect": float(keys["letterbox"])} if keys["letterbox"] else False
    return o


def plan_post(M, keys, ctx):
    from components.post import look as L
    t0 = time.time()
    run = os.path.dirname(ctx["out"])
    cams = _frames_dirs(run)
    if not cams:
        return {"frames": 0, "_skipped": 'no frames: the "frames" output of this run did not render lit.png (post goes after frames)'}
    preset = keys.get("preset") or L.default_preset()
    ops = L.resolve(preset, _overrides(keys))
    frames = [L.frame_from_dir(d, i) for i, d in enumerate(cams)]
    if "palette" in ops:                                             # one transform for the whole run
        pal = L.palettes().get(ops["palette"].get("ref"))
        if pal:
            mu, cov = L.stats_of([f.img[::4, ::4] for f in frames])
            ops["palette"]["_transform"] = L.mkl(mu, cov, pal["mean"], pal["cov"])
    notes, outs = {}, []
    for d, fr in zip(cams, frames):
        st = {}
        img = L.apply(fr, ops, st)
        p = os.path.join(ctx["out"], os.path.basename(d) + ".png"); L.write_png(p, img); outs.append(p)
        for k in ("skipped", "notes"):
            for v in st.get(k, []): notes[v] = notes.get(v, 0) + 1
    _contact(cams, outs, os.path.join(ctx["out"], "contact.jpg"))
    res = {"frames": len(outs), "preset": preset, "sec_per_frame": round((time.time() - t0) / len(outs), 2)}
    if "palette" in ops: res["palette"] = ops["palette"].get("ref")
    if notes: res["notes"] = "; ".join(f"{k} ×{n}" for k, n in notes.items())
    if keys.get("video"):
        from core.outputs import cut
        r = cut.cut([("post", outs)], ctx["out"], int(keys.get("fps") or 12))
        res["video"] = r["trailer"]
    json.dump({"preset": preset, "ops": {k: {kk: vv for kk, vv in (v.items() if isinstance(v, dict) else []) if not kk.startswith("_")}
                                         for k, v in ops.items()}}, open(os.path.join(ctx["out"], "post.json"), "w"), ensure_ascii=False, indent=1)
    return res


def _contact(cams, outs, path, n=8, w=480):
    """A "before / after" sheet: a pair per camera (up to 8 cameras)."""
    import cv2
    import numpy as np
    rows = []
    for d, o in list(zip(cams, outs))[:n]:
        a, b = cv2.imread(os.path.join(d, "lit.png")), cv2.imread(o)
        h = round(a.shape[0] * w / a.shape[1])
        rows.append(np.hstack([cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA), cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)]))
    if rows:
        cv2.imwrite(path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main(argv):
    """A preset on a finished run: <run> [key=value …] → <run>/post_<preset>/ (the pipeline's post output is not touched)."""
    import sys
    import yaml
    if not argv: print(__doc__); sys.exit(1)
    run = os.path.abspath(argv[0])
    keys = {k: yaml.safe_load(v) for k, v in (a.split("=", 1) for a in argv[1:] if "=" in a)}
    from components.post import look as L
    out = os.path.join(run, "post_" + str(keys.get("preset") or L.default_preset()))
    os.makedirs(out, exist_ok=True)
    print("__RESULT__" + json.dumps(dict(plan_post({}, keys, {"out": out, "scene": ""}), out=out), ensure_ascii=False))


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])

"""Editing: take videos back to back as a single video with cuts (shots after shots: components/time/take.py, the "after" key continues an action)
and angle cutaways: one action filmed by different cameras (the same take with another camera) is cut by time — "@a-b", s.
.venv/bin/python -m core.outputs.montage <output.mp4> <out/takes/shot1>[@a-b] <out/takes/shot2>[@a-b] …   — the previs with light if it exists
for all, otherwise the previs. The reply is a __RESULT__ line: the duration, the shots, a strip of the first frames of the pieces (<output>.jpg)."""
import json, math, os, subprocess, sys
from PIL import Image
OUTV = os.path.abspath(sys.argv[1]); TK = []
for a in sys.argv[2:]:                                              # a piece: a take and, if given, a time window (s)
    p, _, w = a.partition("@"); job = json.load(open(os.path.join(os.path.abspath(p), "job.json"))); full = job["frames"] / job["fps"]
    t0, t1 = (float(x) for x in w.split("-")) if w else (0.0, full)
    TK.append((os.path.abspath(p), max(0.0, t0), min(full, t1), job["fps"]))
has = lambda n: all(os.path.exists(os.path.join(t, "take", n + ".mp4")) for t, *_ in TK)   # 3D preview (take.py --fps) — preview.mp4
lay = "previs_lit" if has("previs_lit") else "previs" if has("previs") else "preview"
lst = OUTV + ".txt"; open(lst, "w").write("".join(f"file '{os.path.join(t, 'take', lay + '.mp4')}'\ninpoint {a:.3f}\noutpoint {b:.3f}\n" for t, a, b, _ in TK))
os.makedirs(os.path.dirname(OUTV), exist_ok=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", OUTV], check=True)
os.remove(lst); dur = sum(b - a for _, a, b, _ in TK)
pic = lambda d: os.path.join(d, f"{lay}.png") if lay != "preview" else next(p for p in (os.path.join(d, "lit.png"), os.path.join(d, "color.png")) if os.path.exists(p))
ims = [Image.open(pic(os.path.join(t, f"cam{int(math.floor(a * f + 1e-6)) + 1}"))).convert("RGB") for t, a, _, f in TK]
w = 400; ims = [im.resize((w, round(im.height * w / im.width))) for im in ims]
sh = Image.new("RGB", (w * len(ims), ims[0].height), "white")
for i, im in enumerate(ims): sh.paste(im, (i * w, 0))
sh.save(os.path.splitext(OUTV)[0] + ".jpg", quality=86)
print("__RESULT__" + json.dumps({"video": OUTV, "shots": len(TK), "seconds": round(dur, 1), "layer": lay}, ensure_ascii=False))

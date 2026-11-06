"""Test bench of the "Post" component (film look): the post step on frames of a real run (no Blender, seconds) — numeric checks of the operations and a sheet of presets.
Data — out/dev/post_bench/run (4 showcase frames of the film look, with a README there); another run — by the path to its folder with NN_frames.
  .venv/bin/python -m components.post.bench.bench [run folder]   → OK/FAIL lines, the sheet sheet_presets.jpg, exit code = number of failures
Checks (7): clean does not spoil the frame; the palette moves the frame toward the reference; DoF blurs out-of-focus areas and leaves the focus alone; Kuwahara — the distance
only; the letterbox — exact height of the bars, a vertical frame — no bars; the pipeline step — all frames, cost per frame."""
import glob
import os
import shutil
import sys

import cv2
import numpy as np

from components.post import look as L, steps as S

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
RUN = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(ROOT, "out", "dev", "post_bench", "run")
OUT = os.path.join(os.path.dirname(RUN), "bench_out")
fails = []


def check(name, ok, value):
    print(("OK   " if ok else "FAIL ") + name, value, flush=True)
    if not ok: fails.append(name)


def psnr(a, b):
    m = float(((a - b) ** 2).mean())
    return 99.0 if m == 0 else 10 * np.log10(1.0 / m)


def sat(img):
    hsv = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
    return float(hsv[..., 1][hsv[..., 2] > 0.1].mean())


def hf(img, mask):
    y = L.luma(img); d = y - cv2.GaussianBlur(y, (0, 0), 1.5)
    return float(np.abs(d[mask]).mean()) if mask.any() else 0.0


def main():
    cams = S._frames_dirs(RUN)
    if not cams:
        print(f"no frames in {RUN}: a run with the frames output is needed (README in out/dev/post_bench)"); sys.exit(1)
    os.makedirs(OUT, exist_ok=True)
    frames = [L.frame_from_dir(d, i) for i, d in enumerate(cams)]
    f0 = frames[0]

    # 1. clean — dithering only: the frame stays the same
    a = L.apply(f0, L.resolve("clean"))
    check("clean does not spoil the frame (PSNR ≥ 45 dB)", psnr(a, f0.img) >= 45, round(psnr(a, f0.img), 1))

    # 2. palette: the set's saturation moves toward the reference; with mix 1 and no accent protection — the set's mean = the reference's mean
    pal = L.palettes()["muted_warm"]
    mu, cov = L.stats_of([f.img for f in frames]); tr = L.mkl(mu, cov, pal["mean"], pal["cov"])
    full = [L.op_palette(f.img, f, {"ref": "muted_warm", "mix": 1.0, "protect": 0.0, "_transform": tr}, {}) for f in frames]
    m_out = np.concatenate([x.reshape(-1, 3) for x in full]).mean(0)
    check("MKL palette: the set's mean = the reference (|Δ| ≤ 0.02)", float(np.abs(m_out - pal["mean"]).max()) <= 0.02,
          [round(float(v), 3) for v in m_out - np.asarray(pal["mean"])])
    s_in = np.mean([sat(f.img) for f in frames]); s_tr = np.mean([sat(x) for x in full])
    check("palette: saturation twice as close to the reference", abs(s_tr - pal["saturation"]) <= 0.5 * abs(s_in - pal["saturation"]),
          f"{s_in:.2f} → {s_tr:.2f} (reference {pal['saturation']})")

    # 3. DoF on a frame with a foreground: focus on the distant — the foreground is softer, the focal area untouched
    fn = min(frames, key=lambda f: float(np.nanmin(np.where(np.isfinite(f.z), f.z, np.nan))) if f.z is not None else 1e9)
    zf = np.where(np.isfinite(fn.z), fn.z, np.nan)
    s = float(np.nanpercentile(zf, 80)); near = zf < s * 0.4; focus = np.abs(zf - s) < s * 0.05
    st = {}
    d = L.op_dof(fn.img, fn, {"fstop": 1.4, "focus_m": s, "max_px": 24}, st)
    r_near, r_focus = hf(d, near) / max(hf(fn.img, near), 1e-6), hf(d, focus) / max(hf(fn.img, focus), 1e-6)
    check("DoF: the foreground is blurred (≤ 0.7), the focus is intact (≥ 0.85)", r_near <= 0.7 and r_focus >= 0.85,
          f"{fn.name}: foreground {r_near:.2f}, focus {r_focus:.2f}, circle up to {st.get('dof_max_px')} px")

    # 4. Kuwahara — the distance only
    fk = frames[0]
    k = L.op_kuwahara(fk.img, fk, {"radius": 3, "from_m": 250, "soft_m": 150}, {})
    z = np.where(np.isfinite(fk.z), fk.z, 1e9)
    dn, dfar = float(np.abs(k - fk.img)[z < 200].mean()) if (z < 200).any() else 0.0, float(np.abs(k - fk.img)[z > 400].mean())
    check("Kuwahara: untouched closer than 200 m, touched beyond 400 m", dn < 1e-4 and dfar > 1e-3, f"near {dn:.5f}, far {dfar:.4f}")

    # 5. letterbox
    lb = L.op_letterbox(f0.img, f0, {"aspect": 2.39}, {})
    W, H = f0.size; want = round((H - W / 2.39) / 2); got = int((lb.max(axis=(1, 2)) == 0).sum() // 2)
    vert = L.Frame(np.ones((640, 360, 3), np.float32) * 0.5)
    check("letterbox: bars exact for 2.39, a vertical frame — no bars", got == want and L.op_letterbox(vert.img, vert, {"aspect": 2.39}, {}).min() > 0,
          f"{got} of {want} rows")

    # 6. the pipeline step on a run: all frames, cost per frame
    step = os.path.join(RUN, "99_post"); shutil.rmtree(step, ignore_errors=True); os.makedirs(step)
    r = S.plan_post({}, {"preset": "trailer"}, {"out": step, "scene": ""})
    check("post step: all frames graded, ≤ 3 s per frame", r.get("frames") == len(cams) and r.get("sec_per_frame", 99) <= 3,
          {k: r[k] for k in ("frames", "sec_per_frame", "preset") if k in r})

    # preset sheet: rows — cameras, columns — presets
    names = list(L.presets())
    font = cv2.FONT_HERSHEY_SIMPLEX; w = 360; rows = []
    for f in frames:
        cells = []
        for n in names:
            img = L.apply(f, L.resolve(n))
            c = cv2.resize((np.clip(img, 0, 1) * 255).astype(np.uint8)[..., ::-1], (w, round(f.size[1] * w / f.size[0])), interpolation=cv2.INTER_AREA)
            cv2.putText(c, n, (6, 18), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA); cells.append(c)
        rows.append(np.hstack(cells))
    sheet = os.path.join(OUT, "sheet_presets.jpg"); cv2.imwrite(sheet, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("sheet:", os.path.relpath(sheet, ROOT))
    print(f"__RESULT__{{\"n_fails\": {len(fails)}, \"n_checks\": 7}}")
    sys.exit(len(fails))


if __name__ == "__main__":
    main()

"""Frame versions (shot-plan previs register): a snapshot of the input and output in rNNN, a repeat of the same input —
the same version. The output hash is by pixels (Blender writes the render date and time into PNG/EXR), files are stored
by content in blobs/ (a 6 MB scene is not copied into every version).

  .venv/bin/python -m core.outputs.rev register <store, e.g. out/X/revs> --scene bench/scenes/lab26 --cam 1 --frame out/X/cam1 [--flags "..."] [--desc description.md]
  .venv/bin/python -m core.outputs.rev show <store>

The version number is issued by the script, not by a model (a pitfall: numbers counted by an agent diverged from the core).
The reply is a single __RESULT__ line + JSON.
"""
import core.paths        # noqa: E402  core parts on sys.path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(core.paths.ROOT)
TOOLS = ["make_frame.sh", "frames.py", "bl_pack.py", "bl_scene.py", "pack.py", "maps.py", "marker.py", "openpose.py", "classes.py"]
IMAGES = {".png", ".jpg", ".jpeg"}


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def content_id(path):
    """Content hash: images and EXR by pixels, everything else by bytes."""
    suf = path.suffix.lower()
    if suf in IMAGES:
        from PIL import Image
        with Image.open(path) as im:
            a = np.asarray(im)
        return "px:" + sha_bytes(str(a.shape).encode() + a.tobytes())
    if suf == ".exr":
        from core.outputs import maps
        d = maps.read_exr(str(path))
        h = hashlib.sha256()
        for k in sorted(d):
            h.update(k.encode() + str(d[k].shape).encode() + np.ascontiguousarray(d[k]).tobytes())
        return "px:" + h.hexdigest()
    return "b:" + sha_bytes(path.read_bytes())


def tree(base, files=None):
    """Folder files by relative paths. Links (pipeline scenes link to the files of the source one) — by the target's content;
    logs (blender.log with timestamps) are not output: otherwise an identical run would always give a new version."""
    base = Path(base)
    paths = [base / f for f in files] if files else sorted(p for p in base.rglob("*") if p.is_file() and p.suffix != ".log")
    return {str(p.relative_to(base)): (p.resolve() if p.is_symlink() else p) for p in paths}


def put_blob(store, path):
    b = path.read_bytes()
    h = sha_bytes(b)
    dst = store / "blobs" / h[:2] / h
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".part")
        tmp.write_bytes(b)
        tmp.rename(dst)
    return h, len(b)


def env_versions():
    try:
        bl = subprocess.run(["blender", "--version"], capture_output=True, text=True, timeout=60).stdout.splitlines()[0]
    except Exception as e:                       # noqa: BLE001 — a version for the passport, not for work
        bl = f"unknown ({e})"
    import cv2, PIL
    return {"blender": bl, "numpy": np.__version__, "opencv": cv2.__version__, "pillow": PIL.__version__}


def register(a):
    t0 = time.perf_counter()
    store = Path(a.store)
    store.mkdir(parents=True, exist_ok=True)
    inputs = {f"scene/{k}": v for k, v in tree(a.scene).items()}
    inputs.update({f"tools/{k}": Path(core.paths.script(k)) for k in TOOLS})   # the key stays the old one ("tools/…"): the frame versions match
    params = {"cam": a.cam, "flags": a.flags, "env": env_versions()}
    in_files = {k: sha_bytes(p.read_bytes()) for k, p in inputs.items()}
    input_hash = sha_bytes(json.dumps([sorted(in_files.items()), params], sort_keys=True, ensure_ascii=False).encode())
    outputs = tree(a.frame)
    out_ids = {k: content_id(p) for k, p in outputs.items()}
    output_hash = sha_bytes(json.dumps(sorted(out_ids.items())).encode())
    revs = sorted(store.glob("r[0-9][0-9][0-9]"))
    nondet = None
    for r in revs:
        m = json.loads((r / "manifest.json").read_text())
        if m["input_hash"] == input_hash:
            changed = sorted(k for k in out_ids if m["outputs"].get(k, {}).get("id") != out_ids[k])
            if m["output_hash"] == output_hash:
                return {"action": "reuse", "revision": r.name, "outputs_changed": [], "seconds": round(time.perf_counter() - t0, 2)}
            nondet = {"same_input_as": r.name, "outputs_changed": changed}   # the same input, a different output — a new version, with a note
    num = (int(revs[-1].name[1:]) + 1) if revs else 1
    rdir = store / f"r{num:03d}"
    rdir.mkdir()
    new_bytes = 0
    before = sum(p.stat().st_size for p in (store / "blobs").rglob("*") if p.is_file()) if (store / "blobs").exists() else 0
    man_in = {}
    for k, p in inputs.items():
        h, n = put_blob(store, p)
        man_in[k] = {"sha256": h, "size": n}
    man_out = {}
    for k, p in outputs.items():
        h, n = put_blob(store, p)
        man_out[k] = {"sha256": h, "size": n, "id": out_ids[k]}
    after = sum(p.stat().st_size for p in (store / "blobs").rglob("*") if p.is_file())
    new_bytes = after - before
    prev = json.loads((revs[-1] / "manifest.json").read_text()) if revs else None
    diff_in = sorted(k for k in man_in if not prev or prev["inputs"].get(k, {}).get("sha256") != man_in[k]["sha256"])
    diff_out = sorted(k for k in man_out if not prev or prev["outputs"].get(k, {}).get("id") != man_out[k]["id"])
    manifest = {"revision": rdir.name, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "input_hash": input_hash,
                "output_hash": output_hash, "params": params, "inputs": man_in, "outputs": man_out,
                "changed_vs_previous": {"inputs": diff_in, "outputs": diff_out}}
    (rdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    if a.desc:
        shutil.copyfile(a.desc, rdir / "description.md")     # the description is the full current direction, not a log
    return {"action": "new" if not nondet else "new — NONDETERMINISM: the same input, a different output", **({"nondeterministic": nondet} if nondet else {}),
            "revision": rdir.name, "inputs": len(man_in), "outputs": len(man_out),
            "changed_inputs": diff_in, "changed_outputs": len(diff_out), "new_blob_bytes": new_bytes,
            "full_copy_bytes": sum(v["size"] for v in man_in.values()) + sum(v["size"] for v in man_out.values()),
            "seconds": round(time.perf_counter() - t0, 2)}


def show(a):
    store = Path(a.store)
    rows = []
    for r in sorted(store.glob("r[0-9][0-9][0-9]")):
        m = json.loads((r / "manifest.json").read_text())
        rows.append({"rev": r.name, "in": m["input_hash"][:10], "out": m["output_hash"][:10],
                     "changed_inputs": m["changed_vs_previous"]["inputs"][:6],
                     "changed_outputs": len(m["changed_vs_previous"]["outputs"])})
    blobs = sum(p.stat().st_size for p in (store / "blobs").rglob("*") if p.is_file()) if (store / "blobs").exists() else 0
    return {"revisions": rows, "blob_bytes": blobs}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    r.add_argument("store")
    r.add_argument("--scene", required=True)
    r.add_argument("--cam", type=int, required=True)
    r.add_argument("--frame", required=True)
    r.add_argument("--flags", default="")
    r.add_argument("--desc")
    s = sub.add_parser("show")
    s.add_argument("store")
    a = ap.parse_args()
    print("__RESULT__", json.dumps(register(a) if a.cmd == "register" else show(a), ensure_ascii=False))

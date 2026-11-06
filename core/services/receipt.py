"""Edit receipt: a snapshot of the scene files before and after any command — only the fields that differ (~100 tokens
instead of a retelling and re-reading the files). The truth of a pipeline scene is its files: figures (figkit), cameras,
parts (world.json), plan edits.
.venv/bin/python -m core.services.receipt scenes/<name> -- <command …>      — for example: -- .venv/bin/python -m core.model.scene_edit bench/scenes/X turn m1 30
The reply is a __RESULT__ line: {"edit": {…what changed…}, "command": <its __RESULT__, shortened>, "code": exit code}."""
import json, os, subprocess, sys

SD = os.path.abspath(sys.argv[1]); CMD = sys.argv[sys.argv.index("--") + 1:]
R = lambda v, n=2: [round(x, n) for x in v] if isinstance(v, (list, tuple)) else round(v, n) if isinstance(v, float) else v


def snap():
    j = lambda f: json.load(open(os.path.join(SD, f))) if os.path.exists(os.path.join(SD, f)) else None
    kit, cams, W, ed = j("figkit.json") or {}, j("cameras.json") or [], j("world.json") or {"reg": []}, j("edits.json") or []
    return {"figures": {k: {"where": R(f["pos"]), "heading": R(f.get("heading", 0.0), 1), "growth": R(f.get("height", 0.0))} for k, f in kit.get("named", {}).items()},
            "cameras": {str(i + 1): {"name": c.get("name", "")[:40], "where": R(c.get("location", [0, 0, 0])), "mm": R(c.get("lens_mm", 0), 1),
                                    "frame": c.get("resolution")} for i, c in enumerate(cams)},
            "parts": {e["id"]: e["kind"] for e in W["reg"]}, "plan_edits": len(ed)}


def delta(a, b, path=""):
    out = {}
    for k in sorted(set(a) | set(b), key=str):
        if k not in a: out[f"{path}{k}"] = {"+": b[k]}
        elif k not in b: out[f"{path}{k}"] = "−"
        elif isinstance(a[k], dict) and isinstance(b[k], dict): out.update(delta(a[k], b[k], f"{path}{k}."))
        elif a[k] != b[k]: out[f"{path}{k}"] = [a[k], b[k]]
    return out


s0 = snap(); r = subprocess.run(CMD, capture_output=True, text=True); s1 = snap()
res = next((l.split("__RESULT__", 1)[1].strip() for l in r.stdout.splitlines() if "__RESULT__" in l), None)
print("__RESULT__" + json.dumps({"edit": delta(s0, s1) or "no changes", "command": (res[:300] if res else (r.stderr or r.stdout)[-300:]),
                                 "code": r.returncode}, ensure_ascii=False, separators=(",", ":")))

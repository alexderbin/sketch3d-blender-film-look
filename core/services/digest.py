"""A .blend digest and a comparison of two: does the scene build match the file — code against a file edited by script, a build with a cache against a build without one.
The digest itself is taken by core/blender/bl_digest.py (windowless Blender, no rendering); the comparison is by section, only the fields that differ (like the receipt
core/services/receipt.py), ~100–600 tokens instead of frame sheets.
.venv/bin/python -m core.services.digest <file.blend> [<out.json>]          — a digest (by default next to it: <file>.digest.json)
.venv/bin/python -m core.services.digest diff <a.json> <b.json> [--max 40]  — what differs; the reply is a __RESULT__ line (same, a count per section)
Block names with a number (.001) are merged: a block that is "X" in one build and "X.001" with the same content in the other is renamed, not changed."""
import core.paths        # noqa: E402  core parts on sys.path
import json, os, re, subprocess, sys

base = lambda s: re.sub(r"\.\d{3}$", "", s)


def short(v, n=90):
    s = json.dumps(v, ensure_ascii=False, sort_keys=True) if not isinstance(v, str) else v
    return s if len(s) <= n else s[:n - 1] + "…"


def walk(a, b, path, out):
    """Fields that differ between two values: a path and a pair of values."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a: out.append((f"{path}.{k}", "—", b[k]))
            elif k not in b: out.append((f"{path}.{k}", a[k], "—"))
            elif a[k] != b[k]: walk(a[k], b[k], f"{path}.{k}", out)
    elif a != b:
        out.append((path, a, b))


def diff(A, B, mx=40):
    lines, counts = [], {}
    for sec in sorted(set(A) | set(B)):
        if sec == "file": continue
        a, b = A.get(sec, {}), B.get(sec, {})
        if sec == "scene":
            ch = []; walk(a, b, "scene", ch)
            counts[sec] = {"changed": len(ch)}; lines += [("~", p, x, y) for p, x, y in ch]; continue
        oa, ob = sorted(set(a) - set(b)), sorted(set(b) - set(a))
        ren = []
        for k in list(oa):                                             # "X" ↔ "X.001" with the same content is a rename
            m = next((j for j in ob if base(j) == base(k) and a[k] == b[j]), None)
            if m: ren.append((k, m)); oa.remove(k); ob.remove(m)
        chg = []
        for k in sorted(set(a) & set(b)):
            if a[k] != b[k]: d_ = []; walk(a[k], b[k], f"{sec}:{k}", d_); chg += d_
        counts[sec] = {"only_a": len(oa), "only_b": len(ob), "changed": len({p.split(".")[0] for p, _, _ in chg}), "renamed": len(ren)}
        lines += [("−", f"{sec}:{k}", "", "") for k in oa] + [("+", f"{sec}:{k}", "", "") for k in ob]
        lines += [("↔", f"{sec}:{x} → {y}", "", "") for x, y in ren] + [("~", p, x, y) for p, x, y in chg]
    same = not any(v.get("only_a") or v.get("only_b") or v.get("changed") for v in counts.values())
    print(f"A: {A.get('file')}\nB: {B.get('file')}")
    for sec, c in counts.items():
        if any(c.values()): print(f"{sec}: " + ", ".join(f"{k} {v}" for k, v in c.items() if v))
    for sign, p, x, y in lines[:mx]:
        print(f"  {sign} {p}" + (f": {short(x)} → {short(y)}" if sign == "~" else ""))
    if len(lines) > mx: print(f"  … {len(lines) - mx} more lines (--max N)")
    print("__RESULT__" + json.dumps({"same": same, "counts": {k: v for k, v in counts.items() if any(v.values())}}, ensure_ascii=False))
    return same


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "diff":
        mx = int(a[a.index("--max") + 1]) if "--max" in a else 40
        sys.exit(0 if diff(json.load(open(a[1])), json.load(open(a[2])), mx) else 1)
    if not a:
        print(__doc__); sys.exit(2)
    blend = os.path.abspath(a[0]); out = os.path.abspath(a[1]) if len(a) > 1 else blend[:-6] + ".digest.json"
    r = subprocess.run(["blender", "-b", "--factory-startup", blend, "--python-exit-code", "1", "-P", core.paths.RUN, "--", core.paths.script("bl_digest.py"), "--", out],
                       capture_output=True, text=True)
    res = next((l for l in r.stdout.splitlines() if l.startswith("__RESULT__")), None)
    if r.returncode or not res:
        print("__RESULT__" + json.dumps({"error": "bl_digest", "rc": r.returncode, "tail": (r.stdout + r.stderr)[-1500:]}, ensure_ascii=False)); sys.exit(1)
    print(res)

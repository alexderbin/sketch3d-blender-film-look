"""Archive and cleanup as part of the project.

Zones and rules (what may be deleted, by age in days; never touches git-tracked files):
  out/dev/**        working dumps of tool building (runs, benches, probes): heavy files older than DEV_DAYS, whole run folders older than DEV_RUN_DAYS
  out/projects/**   productions for publication: only intermediates (raw passes, checkpoints) older than PROJECT_DAYS; finals never
  out/agent/**      agent transcripts/logs (promo agent, stopped 29.10): everything older than AGENT_DAYS
  out/<other>/**    legacy dumps before the zones (runs, build, takes, relief, far …): treated as out/dev
Heavy classes: image, video, exr, mesh (.obj/.ply/.fbx/.glb), blend (.blend/.blend1), array (.npz/.npy), archive (.zip/.7z/.tar*).
Finals (never deleted automatically): out/projects/**/final/**, *.mp4 in out/projects, sheet_*.jpg / REPORT_*.jpg next to a REPORT.md.

    .venv/bin/python -m core.services.archive report [--top 15]        what weighs how much, by zone and class
    .venv/bin/python -m core.services.archive sweep [--apply] [--days N]   dry run by default; --apply deletes and journals to out/archive.jsonl
Stdlib only.
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "out")
JOURNAL = os.path.join(OUT, "archive.jsonl")

CLASSES = {
    "image": {".png", ".jpg", ".jpeg", ".webp", ".gif", ".tif", ".tiff", ".bmp"},
    "video": {".mp4", ".mov", ".webm", ".mkv", ".gif"},
    "exr": {".exr", ".hdr"},
    "mesh": {".obj", ".ply", ".fbx", ".glb", ".gltf", ".stl"},
    "blend": {".blend", ".blend1", ".blend2"},
    "array": {".npz", ".npy", ".bin"},
    "archive": {".zip", ".7z", ".tar", ".gz", ".tgz"},
    "log": {".log", ".jsonl", ".txt"},
}
HEAVY = {"image", "video", "exr", "mesh", "blend", "array", "archive"}
DEV_DAYS, DEV_RUN_DAYS, PROJECT_DAYS, AGENT_DAYS = 7, 14, 30, 1
PROTECTED = ("out/dev/night/backup/",)                    # copies of the data from before the migration (sites, relief windows, the Effects knowledge) —
                                                          # do not clean up until they are accepted; lift the protection only on explicit instruction
INTERMEDIATE_DIRS = {"raw", "passes", "cache", "tmp", "work", "scene"}      # inside projects: may go; finals stay


def classify(path):
    ext = os.path.splitext(path)[1].lower()
    for cls, exts in CLASSES.items():
        if ext in exts:
            return cls
    return "other"


def zone(rel):
    top = rel.split("/")[0]
    if top != "out":
        return None
    parts = rel.split("/")
    sub = parts[1] if len(parts) > 1 else ""
    if sub == "projects":
        return "projects"
    if sub == "agent":
        return "agent"
    return "dev"                                              # out/dev and legacy out/<runs|build|takes|…>


def is_final(rel):
    low = rel.lower()
    if zone(rel) != "projects":
        return False
    if "/final/" in low or low.endswith(".mp4"):
        return True
    base = os.path.basename(low)
    return base.startswith(("sheet_", "report_")) and base.endswith((".jpg", ".png"))


def tracked():
    """Git-tracked paths (never deleted)."""
    try:
        import subprocess
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True).stdout
        return set(out.split("\0"))
    except Exception:
        return set()


def walk(root):
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in (".git",)]
        for f in fn:
            p = os.path.join(dp, f)
            try:
                st = os.stat(p)
            except OSError:
                continue
            yield os.path.relpath(p, ROOT), st.st_size, st.st_mtime


def report(top=15):
    by_zone, by_class, by_dir = {}, {}, {}
    total = 0
    for rel, size, _ in walk(OUT):
        z, c = zone(rel) or "?", classify(rel)
        by_zone[z] = by_zone.get(z, 0) + size
        by_class[c] = by_class.get(c, 0) + size
        d = "/".join(rel.split("/")[:3])
        by_dir[d] = by_dir.get(d, 0) + size
        total += size
    gb = lambda b: f"{b / 2**30:.1f} GB"
    print(f"out/: {gb(total)}")
    print("by zone:  " + ", ".join(f"{k} {gb(v)}" for k, v in sorted(by_zone.items(), key=lambda x: -x[1])))
    print("by class: " + ", ".join(f"{k} {gb(v)}" for k, v in sorted(by_class.items(), key=lambda x: -x[1])))
    print(f"heaviest folders (top {top}):")
    for d, v in sorted(by_dir.items(), key=lambda x: -x[1])[:top]:
        print(f"  {gb(v):>9}  {d}")
    plan = sweep(apply=False, quiet=True)
    print(f"the rule-based cleanup would free: {gb(plan['bytes'])} ({plan['files']} files) — `sweep --apply`")


def sweep(apply=False, days=None, quiet=False):
    """Delete (or list) what the rules allow. Returns {files, bytes}."""
    now = time.time()
    keep = tracked()
    victims = []
    for rel, size, mtime in walk(OUT):
        if rel in keep or rel == os.path.relpath(JOURNAL, ROOT) or rel.startswith(PROTECTED):
            continue
        age = (now - mtime) / 86400
        z, c = zone(rel), classify(rel)
        limit = None
        if z == "agent":
            limit = days or AGENT_DAYS
        elif z == "projects":
            parts = set(rel.lower().split("/"))
            if is_final(rel):
                continue
            if c in HEAVY and (parts & INTERMEDIATE_DIRS or c in ("exr", "blend", "array")):
                limit = days or PROJECT_DAYS
        else:                                                 # dev and legacy dumps
            if c in HEAVY:
                limit = days or DEV_DAYS
            elif c == "log":
                limit = days or DEV_RUN_DAYS
        if limit is not None and age > limit:
            victims.append((rel, size, c, z))
    total = sum(v[1] for v in victims)
    if not quiet:
        by = {}
        for rel, size, c, z in victims:
            by[(z, c)] = by.get((z, c), 0) + size
        print(("removed" if apply else "to delete") + f": {len(victims)} files, {total / 2**30:.1f} GB")
        for (z, c), v in sorted(by.items(), key=lambda x: -x[1])[:12]:
            print(f"  {v / 2**30:6.1f} GB  {z}/{c}")
    if apply and victims:
        os.makedirs(OUT, exist_ok=True)
        with open(JOURNAL, "a", encoding="utf-8") as j:
            for rel, size, c, z in victims:
                try:
                    os.remove(os.path.join(ROOT, rel))
                    j.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "path": rel, "bytes": size, "class": c, "zone": z}, ensure_ascii=False) + "\n")
                except OSError as e:
                    j.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "path": rel, "error": str(e)}, ensure_ascii=False) + "\n")
        # empty directories left behind
        for dp, dn, fn in os.walk(OUT, topdown=False):
            if not dn and not fn and dp != OUT:
                try:
                    os.rmdir(dp)
                except OSError:
                    pass
    return {"files": len(victims), "bytes": total}


def main(argv):
    if not argv or argv[0] not in ("report", "sweep"):
        print(__doc__); return 2
    if argv[0] == "report":
        top = int(argv[argv.index("--top") + 1]) if "--top" in argv else 15
        report(top)
    else:
        days = int(argv[argv.index("--days") + 1]) if "--days" in argv else None
        sweep(apply="--apply" in argv, days=days)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""Resource guards for the pipeline and for an AI agent: disk space, free memory, one heavy run per machine,
a time limit per command, leftover Blender processes, cleanup of a shot's raw files with a recipe, a ban on reading huge files.
Pure Python — works both in the .venv and in Blender. The thresholds are LIMITS below.
From the command line (.venv/bin/python guard.py …):
  check                          — one line: disk, memory, Blender in the background, whether a heavy run is busy
  mem                            — memory for load calculation (total, available, pressure, swap) and cores; on Linux — from cgroup
  tokens <file> …                — how many tokens reading a file costs; above the threshold — "do not read"
  recipe <shot folder> <scene folder> [key=value …] — write a shot's recipe (how to rebuild the raw files)
  purge <shot folder> <scene folder> — delete a shot's raw EXRs; without a recipe it writes the recipe first
  sweep [--apply]                — cleanup by age (RETENTION): without --apply only the list and how much would be freed
  gate [GB]                      — can a heavy run start: space, memory, not busy; if not — exit code 1
  heavy [--wait S] [--need GB] [--max-gb GB] [--what TEXT] [--where server|local] -- <command …>  — a heavy command (the
                                   "server" mode — remote where server: on a rented machine, core/services/remote.py) under a shared lock: space, memory
                                   and pressure; if busy — wait (up to 1800 s by default); the exit code is the command's (a long render, frames
                                   of a large scene); --max-gb — a ceiling on the memory of the command's processes for the memory guard (memguard.py)
In code: import guard as G; G.require_disk(need_gb); with G.heavy_lock(): …; G.run(cmd, timeout)."""
import core.paths        # noqa: E402  core parts on sys.path
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time

ROOT = os.environ.get("SKETCH3D_ROOT") or core.paths.ROOT   # project root
LIMITS = {
    "disk_warn_gb": 15,          # below this — a warning
    "disk_stop_gb": 10,          # below this — long runs do not start
    "mem_free_min_pct": 30,      # below this — a heavy run does not start
    "swap_warn_gb": 10,          # above this — a warning (frames took 5–9 min while swapping 10.7–13 GB; the swap volume itself
                                 # stays high even after the pressure has subsided — only memory pressure stops a run)
    "read_max_tokens": 10000,    # a file costing more than this is not read whole (a frame's objects.json is 14k)
    "cmd_timeout_s": 600,        # a command running longer is killed
    "pack_workers": 4,           # map builds running in the background at the same time
}
LOCK_DIR = os.environ.get("GUARD_LOCK_DIR") or os.path.join(ROOT, ".locks")


class Stop(RuntimeError):
    """A safeguard tripped: the reason is in the message."""


# ---------------------------------------------------------------------------------------------------- disk and memory
def free_gb(path=ROOT):
    return shutil.disk_usage(path).free / 1e9


def require_disk(need_gb=0.0, path=ROOT):
    """Is there enough disk space for the run: at least disk_stop_gb must remain after it."""
    free = free_gb(path)
    if free - need_gb < LIMITS["disk_stop_gb"]:
        raise Stop(f"low disk space: {free:.1f} GB free, the run needs ~{need_gb:.1f} GB, threshold {LIMITS['disk_stop_gb']} GB")
    return free


LINUX = sys.platform.startswith("linux")      # rented machine (a Linux container): memory and cores come from the container's cgroup, pressure from PSI


def _text(p):
    try:
        with open(p) as f:
            return f.read()
    except OSError:
        return None


def linux_mem(proc="/proc", cg="/sys/fs/cgroup"):
    """Linux memory adjusted for the container: /proc/meminfo sees the whole host, the allotment is memory.max (cgroup v2) or
    limit_in_bytes (v1); the file cache (inactive_file) counts as free. Pressure is PSI (the share of time spent waiting for
    memory over 10 s): full ≥ 10 % or available < 5 % — 4, some ≥ 10 % or < 15 % — 2, otherwise 1 (the macOS scale). Paths
    are parameters (the selftest probe)."""
    mi = {}
    for line in (_text(f"{proc}/meminfo") or "").splitlines():
        k, _, v = line.partition(":")
        if v.split():
            mi[k] = int(v.split()[0]) * 1024
    if "MemTotal" not in mi:
        return None
    total, avail = mi["MemTotal"], mi.get("MemAvailable", mi.get("MemFree", 0))
    swap = mi.get("SwapTotal", 0) - mi.get("SwapFree", 0)
    lim, use, stat = _text(f"{cg}/memory.max"), _text(f"{cg}/memory.current"), _text(f"{cg}/memory.stat")
    key = "inactive_file"
    if lim is None:                                                   # cgroup v1
        lim, use, stat = (_text(f"{cg}/memory/memory.{n}") for n in ("limit_in_bytes", "usage_in_bytes", "stat"))
        key = "total_inactive_file"
    if lim and use and lim.strip().isdigit() and int(lim) < total:
        inact = next((int(l.split()[1]) for l in (stat or "").splitlines() if l.startswith(key + " ")), 0)
        total = int(lim); avail = min(avail, total - (int(use) - inact))
    psi = _text(f"{cg}/memory.pressure") or _text(f"{proc}/pressure/memory") or ""
    some, full = (float(m.group(1)) if m else 0.0 for m in (re.search(rf"^{w} avg10=([\d.]+)", psi, re.M) for w in ("some", "full")))
    frac = max(avail, 0) / total
    level = 4 if frac < 0.05 or (full >= 10 and frac < 0.25) else 2 if some >= 10 or full >= 10 or frac < 0.15 else 1   # PSI full with 3/4 of memory free is
    # a stall of the FUSE volume (a .blend write), not a shortage: critical only when little is left
    return {"total_gb": round(total / 2 ** 30, 2), "avail_gb": round(max(avail, 0) / 2 ** 30, 2), "free_pct": int(100 * frac),
            "pressure": level, "swap_gb": round(swap / 2 ** 30, 1)}


_MEMSIZE = []


def mem():
    """Memory for the load calculation: total and available (GB), free (%), pressure (1/2/4), swap (GB). macOS — memory_pressure
    and sysctl; Linux — linux_mem (container)."""
    if LINUX:
        return linux_mem()
    if not _MEMSIZE:
        try:
            _MEMSIZE.append(int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout) / 2 ** 30)
        except Exception:
            _MEMSIZE.append(None)
    total, pct = _MEMSIZE[0], mem_free_pct()
    return {"total_gb": total, "avail_gb": round(total * pct / 100, 2) if total and pct is not None else None, "free_pct": pct,
            "pressure": pressure_level(), "swap_gb": swap_used_gb()}


def cpus():
    """CPU cores for the calculation: on Linux — those allowed to the process and the cgroup quota (cpu.max, v1 — cfs_quota_us),
    otherwise all of them."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    if LINUX:
        q = (_text("/sys/fs/cgroup/cpu.max") or "").split()
        if len(q) != 2:
            q = [(_text(f"/sys/fs/cgroup/cpu/cpu.cfs_{n_}_us") or "").strip() for n_ in ("quota", "period")]
        if len(q) == 2 and q[0].isdigit() and q[1].isdigit() and int(q[0]) > 0:
            n = min(n, max(1, -(-int(q[0]) // int(q[1]))))
    return n


def mem_free_pct():
    if LINUX:
        return (linux_mem() or {}).get("free_pct")
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=10).stdout
        m = re.search(r"free percentage:\s*(\d+)%", out)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def pressure_level():
    """Memory pressure: 1 — normal, 2 — elevated, 4 — critical (macOS — kern.memorystatus_vm_pressure_level, Linux — PSI)."""
    if LINUX:
        return (linux_mem() or {}).get("pressure")
    try:
        return int(subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], capture_output=True, text=True, timeout=5).stdout)
    except Exception:
        return None


def swap_used_gb():
    if LINUX:
        return (linux_mem() or {}).get("swap_gb")
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=5).stdout
        m = re.search(r"used = ([\d.]+)M", out)
        return round(float(m.group(1)) / 1024, 1) if m else None
    except Exception:
        return None


def require_mem():
    pct = mem_free_pct()
    if pct is not None and pct < LIMITS["mem_free_min_pct"]:
        raise Stop(f"low memory: {pct} % free, threshold {LIMITS['mem_free_min_pct']} %")
    if (pressure_level() or 1) >= 4:
        raise Stop(f"memory pressure is critical (swap {swap_used_gb()} GB) — the heavy run does not start")
    return pct


# ------------------------------------------------------------------------------------------- one heavy run
def _alive(pid):
    try: os.kill(pid, 0); return True
    except ProcessLookupError: return False
    except PermissionError: return True


@contextlib.contextmanager
def heavy_lock(name="pipeline", what="", wait=0):
    """One heavy run per machine (pipeline, benchmark, long render). If busy — wait up to wait seconds in a queue in order of
    arrival (ticket <time>-<pid> in <name>.queue/; the lock goes to the oldest live ticket, polled once per second; the
    "WAITING" line says who holds it and what your place in the queue is), then refuse, naming the holder. Polling used to
    happen once every 15 s with no queue — back-to-back runs kept grabbing the lock and the waiter stood for 18 min. A nested
    guard (a heavy step inside a pipeline that is already under guard) passes at once: the parent holds the lock and marks
    its children with the variable SK_GUARD_HELD."""
    if os.environ.get("SK_GUARD_HELD") == name:          # nested call (a pipeline step under guard inside a pipeline under guard): the lock is already held by the parent
        yield; return
    os.makedirs(LOCK_DIR, exist_ok=True)
    path = os.path.join(LOCK_DIR, name + ".lock"); qdir = os.path.join(LOCK_DIR, name + ".queue"); os.makedirs(qdir, exist_ok=True)
    f = open(path, "a+"); t_end = time.time() + wait; said = None
    ticket = os.path.join(qdir, f"{time.time_ns()}-{os.getpid()}")
    with open(ticket, "w") as t_: t_.write(what)
    try:
        while True:
            live = []
            for tk in sorted(os.listdir(qdir)):                     # oldest first; tickets of dead processes are dropped
                try: pid = int(tk.split("-")[1])
                except (IndexError, ValueError): continue
                if _alive(pid): live.append(tk)
                else:
                    try: os.remove(os.path.join(qdir, tk))
                    except OSError: pass
            if live and live[0] == os.path.basename(ticket):
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); break
                except BlockingIOError: pass
            f.seek(0); who = f.read().strip() or "?"; left = t_end - time.time()
            if left <= 0:
                f.close(); raise Stop(f"a heavy run is already in progress: {who}")
            pos = live.index(os.path.basename(ticket)) + 1 if os.path.basename(ticket) in live else "?"
            if (who, pos) != said: print(f"WAITING: the heavy run is busy ({who}); you are #{pos} in the queue", file=sys.stderr, flush=True); said = (who, pos)
            time.sleep(min(1.0, max(left, 0.01)))
    finally:
        try: os.remove(ticket)
        except OSError: pass
    f.seek(0); f.truncate()
    f.write(f"pid {os.getpid()}, since {time.strftime('%H:%M:%S')}, {what}"); f.flush()
    os.environ["SK_GUARD_HELD"] = name                     # for the children (subprocess): their guard of the same name does not wait for itself
    try:
        yield
    finally:
        os.environ.pop("SK_GUARD_HELD", None)
        f.seek(0); f.truncate(); fcntl.flock(f, fcntl.LOCK_UN); f.close()


def lock_busy(name="pipeline"):
    path = os.path.join(LOCK_DIR, name + ".lock")
    if not os.path.exists(path):
        return None
    with open(path, "a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(f, fcntl.LOCK_UN)
            return None
        except BlockingIOError:
            f.seek(0)
            return f.read().strip() or "occupied"


# ------------------------------------------------------------------------------------------ time and processes
def run(cmd, timeout=None, **kw):
    """A command with a time limit: if it runs longer, it is killed together with its child processes (Blender, builds)."""
    timeout = timeout or LIMITS["cmd_timeout_s"]
    p = subprocess.Popen(cmd, start_new_session=True, **kw)
    try:
        return p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, 9)
        raise Stop(f"the command ran longer than {timeout} s and was killed: {' '.join(map(str, cmd))[:120]}")


def stray_blenders():
    """Background Blender processes (none should be left after the work is done)."""
    out = subprocess.run(["pgrep", "-af" if LINUX else "-fl", "Blender.app|/blender "], capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if "pgrep" not in l]


# ------------------------------------------------------------------------------------------------------ tokens
def tokens(path):
    """Token cost of reading a file: an image — w×h/750 (after downscaling to 1568 on the long side), text —
    characters/4."""
    with open(path, "rb") as f:
        head = f.read(32)
    w = h = None
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack(">II", head[16:24])
    elif head[:2] == b"\xff\xd8":
        b = open(path, "rb").read(); i = 2
        while i + 9 < len(b) and b[i] == 0xFF:
            m, L = b[i + 1], struct.unpack(">H", b[i + 2:i + 4])[0]
            if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
                h, w = struct.unpack(">HH", b[i + 5:i + 9]); break
            i += 2 + L
    if w:
        k = min(1.0, 1568 / max(w, h))
        return round(int(w * k) * int(h * k) / 750)
    return os.path.getsize(path) // 4


def safe_to_read(path):
    t = tokens(path)
    return t <= LIMITS["read_max_tokens"], t


# ------------------------------------------------------------------------------------------- recipe and cleanup
def _hash(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def write_recipe(shot_dir, scene_dir, **params):
    """Shot recipe: what to rebuild the raw passes from if they were deleted. The scene (file fingerprints), cameras, Blender
    version, fingerprints of the pipeline scripts, parameters. A copy of cameras.json sits alongside (the cameras may have
    been edited afterwards)."""
    files = {f: _hash(os.path.join(scene_dir, f)) for f in sorted(os.listdir(scene_dir))
             if os.path.isfile(os.path.join(scene_dir, f))}
    tools = {f: _hash(core.paths.script(f)) for f in ("bl_scene.py", "bl_pack.py", "pack.py", "maps.py", "marker.py",
                                                  "openpose.py", "classes.py") if os.path.exists(core.paths.script(f))}
    try:
        bver = subprocess.run(["blender", "--version"], capture_output=True, text=True, timeout=60).stdout.splitlines()[0]
    except Exception:
        bver = None
    rec = {"date": time.strftime("%Y-%m-%d %H:%M"), "scene_dir": os.path.relpath(scene_dir, ROOT), "scene_files": files,
           "tools": tools, "blender": bver, "params": params}
    os.makedirs(shot_dir, exist_ok=True)
    json.dump(rec, open(os.path.join(shot_dir, "recipe.json"), "w"), ensure_ascii=False, indent=1)
    cams = os.path.join(scene_dir, "cameras.json")
    if os.path.exists(cams):
        shutil.copy(cams, os.path.join(shot_dir, "recipe_cameras.json"))
    return rec


def purge_raw(shot_dir, scene_dir=None):
    """Delete the shot's raw EXR files (the PNG maps, ids, part, objects.json and the recipe stay). With no recipe — write the
    recipe first."""
    if not os.path.exists(os.path.join(shot_dir, "recipe.json")):
        if not scene_dir:
            raise Stop(f"shot {shot_dir} has no recipe.json — give the scene folder so that a recipe is written before cleanup")
        write_recipe(shot_dir, scene_dir, note="recipe written during cleanup")
    freed, n = 0, 0
    for d, _, fs in os.walk(shot_dir):
        for f in fs:
            if f.endswith(".exr"):
                p = os.path.join(d, f); freed += os.path.getsize(p); n += 1; os.remove(p)
    return {"files": n, "freed_mb": round(freed / 1e6, 1)}


# ------------------------------------------------------------------------------------------- cleanup by age
RETENTION = {
    "raw_days": 7,               # raw EXR of a shot: delete after 7 days without edits (a recipe exists)
    "raw_days_no_recipe": 30,    # with no recipe — only after 30 days (there will be nothing to rebuild from, so be more careful)
    "bench_keep": 3,             # reference runs: the last 3 + the reference
    "info_only": ["library/_raw"],   # other people's and shared items — only list them; the user decides
}


def _age_days(path):
    """How many days the folder has been untouched: judged by the newest file inside."""
    newest = os.path.getmtime(path)
    for d, _, fs in os.walk(path):
        for f in fs:
            try:
                newest = max(newest, os.path.getmtime(os.path.join(d, f)))
            except OSError:
                pass
    return (time.time() - newest) / 86400


def _size(path, pred=lambda f: True):
    n = 0
    for d, _, fs in os.walk(path):
        for f in fs:
            if pred(f):
                try:
                    n += os.path.getsize(os.path.join(d, f))
                except OSError:
                    pass
    return n


def _shot_of(raw_dir):
    """Shot folder for a raw folder: the nearest ancestor containing recipe.json, otherwise the parent of the camera folder."""
    d = os.path.dirname(raw_dir)
    for _ in range(4):
        if os.path.exists(os.path.join(d, "recipe.json")):
            return d, True
        d = os.path.dirname(d)
    return os.path.dirname(os.path.dirname(raw_dir)), False


def sweep(apply=False, roots=("out", "research/analysis/runs"), now_days=None):
    """What can be cleaned up under the RETENTION rules. apply=False — list only. Returns the plan rows and the total."""
    plan, seen = [], set()
    for r in roots:
        base = os.path.join(ROOT, r)
        for d, dirs, fs in os.walk(base):
            if os.path.basename(d) == "raw" and any(f.endswith(".exr") for f in fs):
                shot, has_rec = _shot_of(d)
                if shot in seen:
                    continue
                seen.add(shot)
                age = _age_days(shot) if now_days is None else now_days
                limit = RETENTION["raw_days"] if has_rec else RETENTION["raw_days_no_recipe"]
                mb = _size(shot, lambda f: f.endswith(".exr")) / 1e6
                plan.append({"what": "raw EXR", "path": os.path.relpath(shot, ROOT), "age_days": round(age, 1), "mb": round(mb, 1),
                             "rule": f"{limit} days, recipe: {'present' if has_rec else 'no'}", "due": age >= limit})
            if os.path.basename(d) == "bench" and os.path.exists(os.path.join(os.path.dirname(d), "bench_baseline.json")):
                runs = sorted((x for x in dirs), key=lambda x: os.path.getmtime(os.path.join(d, x)), reverse=True)
                base_json = json.load(open(os.path.join(os.path.dirname(d), "bench_baseline.json")))
                for i, x in enumerate(runs):
                    p = os.path.join(d, x)
                    is_base = os.path.exists(os.path.join(p, "summary.json")) and json.load(open(os.path.join(p, "summary.json"))) == base_json
                    plan.append({"what": "reference run", "path": os.path.relpath(p, ROOT), "age_days": round(_age_days(p), 1),
                                 "mb": round(_size(p) / 1e6, 1), "rule": "reference — keep" if is_base else f"keep the last {RETENTION['bench_keep']}",
                                 "due": (not is_base) and i >= RETENTION["bench_keep"]})
                dirs[:] = []
    for p in RETENTION["info_only"]:
        full = p if os.path.isabs(p) else os.path.join(ROOT, p)
        if os.path.isdir(full):
            plan.append({"what": "list only", "path": p, "age_days": None, "mb": round(_size(full) / 1e6), "rule": "the user decides",
                         "due": False})
    freed = 0.0
    if apply:
        for x in plan:
            if not x["due"]:
                continue
            full = os.path.join(ROOT, x["path"])
            if x["what"] == "raw EXR":
                freed += purge_raw(full, None)["freed_mb"] if os.path.exists(os.path.join(full, "recipe.json")) else _purge_exr(full)
            elif x["what"] == "reference run":
                freed += x["mb"]; shutil.rmtree(full)
    return plan, round(freed, 1)


def _purge_exr(shot_dir):
    """Raw data with no recipe older than 30 days: before deletion, a note that there is nothing to rebuild it from."""
    json.dump({"date": time.strftime("%Y-%m-%d %H:%M"), "note": "raw data deleted by cleanup without a recipe (past its retention age)"},
              open(os.path.join(shot_dir, "recipe.json"), "w"), ensure_ascii=False)
    freed = 0
    for d, _, fs in os.walk(shot_dir):
        for f in fs:
            if f.endswith(".exr"):
                p = os.path.join(d, f); freed += os.path.getsize(p); os.remove(p)
    return freed / 1e6


def gate(need_gb=0.5):
    """Before a heavy run: disk space, memory, and whether another heavy run is in progress. If the machine is not fit — an
    exception with the reason."""
    require_disk(need_gb)
    require_mem()
    busy = lock_busy()
    if busy:
        raise Stop(f"a heavy run is already in progress: {busy}")
    return status()


def status():
    free = free_gb()
    mem = mem_free_pct()
    flags = []
    if free < LIMITS["disk_stop_gb"]:
        flags.append("DISK: stop")
    elif free < LIMITS["disk_warn_gb"]:
        flags.append("disk: low")
    if mem is not None and mem < LIMITS["mem_free_min_pct"]:
        flags.append("MEMORY: low")
    pr, sw = pressure_level(), swap_used_gb()
    if (pr or 1) >= 4:
        flags.append("MEMORY PRESSURE: critical")
    elif (pr or 1) >= 2:
        flags.append("memory pressure: elevated")
    warn = [f"swap {sw} GB"] if sw is not None and sw >= LIMITS["swap_warn_gb"] else []
    guard_on, paused = None, 0
    if not LINUX:                                                         # the memory guard is macOS only (launchd, libproc); on Linux
        from core.services import memguard as M                          # the load is reduced by the executor (executor.py)
        guard_on, paused = M.alive(), len(M._read().get("paused", {}))
        warn += [] if guard_on else ["memory guard is not running (memguard install)"]
    stray = stray_blenders()
    busy = lock_busy()
    return {"disk_free_gb": round(free, 1), "mem_free_pct": mem, "pressure": pr, "swap_used_gb": sw, "blender_running": len(stray),
            "heavy_run": busy, "memguard": guard_on, "paused": paused, "ok": not flags, "flags": flags + warn}


if __name__ == "__main__":
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else "check"), sys.argv[2:]
    if cmd == "check":
        print(json.dumps(status(), ensure_ascii=False))
    elif cmd == "mem":
        print(json.dumps({**(mem() or {}), "cpus": cpus()}, ensure_ascii=False))
    elif cmd == "tokens":
        for p in args:
            ok, t = safe_to_read(p)
            print(f"{t:>9} tokens  {'ok' if ok else 'DO NOT READ'}  {p}")
    elif cmd == "recipe":
        kv = dict(x.split("=", 1) for x in args[2:])
        print(json.dumps(write_recipe(args[0], args[1], **kv), ensure_ascii=False)[:300])
    elif cmd == "purge":
        print(json.dumps(purge_raw(args[0], args[1] if len(args) > 1 else None), ensure_ascii=False))
    elif cmd == "sweep":
        plan, freed = sweep(apply="--apply" in args)
        for x in plan:
            age = "" if x["age_days"] is None else f"{x['age_days']:5.1f} days"
            print(f"{'REMOVE' if x['due'] else '      '} {x['mb']:>9} MB {age:>9}  {x['what']:16s} {x['path']}  ({x['rule']})")
        due = sum(x["mb"] for x in plan if x["due"])
        print(f"to clean up: {due:.0f} MB" + (f"; freed {freed:.0f} MB" if "--apply" in args else " — to clean up: sweep --apply"))
    elif cmd == "gate":
        try:
            print(json.dumps(gate(float(args[0]) if args else 0.5), ensure_ascii=False))
        except Stop as e:
            print(f"STOP: {e}"); sys.exit(1)
    elif cmd == "heavy":                                                # a heavy command under the shared lock: disk space, memory, wait if busy
        if "--" not in args:
            print("heavy [--wait S] [--need GB] [--max-gb GB] [--what TEXT] [--where server|local] -- <command …>"); sys.exit(2)
        head, run_ = args[:args.index("--")], args[args.index("--") + 1:]
        kv = {head[i]: head[i + 1] for i in range(0, len(head) - 1, 2)}
        if "--max-gb" in kv:
            os.environ["SK_MEM_MAX_GB"] = kv["--max-gb"]                # memory guard ceiling: inherited by all processes of the command
        if not LINUX and os.environ.get("SK_WHERE") != "local" and kv.get("--where") != "local":   # on the rented machine — always here
            try:
                from core.services import remote as R                  # "run on the server" (remote where server): the command —
            except ImportError:                                        # without the rental module (public release) — always here
                R = None
            if R and (kv.get("--where") == "server" or R.where() == "server"):  # goes to the rented machine; if unavailable — carry on here, as usual
                try:
                    rc = R.exec_routed(run_, kv.get("--what", " ".join(run_)[:80]), kv.get("--max-gb"))
                except Stop as e:
                    print(f"server: {e} — running here instead", flush=True); rc = None
                except Exception as e:                                 # routing failure (the network dropped in the middle of the launch) — do not crash
                    print(f"server: failure {type(e).__name__}: {str(e)[:160]} — running here instead", flush=True); rc = None
                if rc is not None:
                    sys.exit(rc)
        try:
            require_disk(float(kv.get("--need", 0.5)))
            with heavy_lock(what=kv.get("--what", " ".join(run_)[:80]), wait=float(kv.get("--wait", 1800))):
                require_mem()
                t0 = time.time(); rc = subprocess.run(run_).returncode
                if rc and sys.platform == "darwin":      # memguard uses macOS calls (libSystem); the rented machine (Linux) does not have them
                    from core.services import memguard as M
                    why = M.why(os.getpid(), t0)
                    if why:
                        print(f"STOP: {why}", file=sys.stderr)
                sys.exit(rc)
        except Stop as e:
            print(f"STOP: {e}"); sys.exit(1)
    else:
        print(__doc__)

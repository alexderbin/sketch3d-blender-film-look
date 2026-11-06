"""Memory guard: a per-process ceiling and a pause instead of letting the machine fall over — for all background Blender and
project Python processes, whoever started them (~70 files launch Blender themselves, bypassing guard). macOS does not
set a memory limit on a process: it rejects ulimit -v and -d, and the kernel limit (jetsam, memorystatus_control) works
only as root and also kills rather than slows down. Hence our own guard, sampling every 0.5 s (a run once grew to 16 GB
in under 3 s):
  1. ceiling: a process's footprint (phys_footprint — including compressed memory and swap, like "Memory" in Activity
     Monitor and MEM in top) > the ceiling → SIGTERM, SIGKILL after 5 s. The default ceiling is CONF max_gb; for a single
     run it is the variable SK_MEM_MAX_GB in its environment (guard heavy --max-gb sets it itself; it never goes above
     hard_max_gb). A Blender with a window (no -b) is not touched — only a notification is shown;
  2. memory pressure critical (4) and ≥ 2 heavy processes (≥ 1 GB) running — pause (SIGSTOP) the youngest; the tree of the
     guard lock holder goes last. When it subsides (normal for 10 s in a row, or elevated and the pause longer than
     5 min, or no heavy processes left running) — resume (SIGCONT) the oldest of the paused ones, one at a time;
  3. critical for longer than 3 min with nothing left to pause — kill the largest one, if it is ≥ 4 GB (smaller — the
     pressure comes from a process that is not ours).
A killed process returns exit code -15 (-9) to its parent; the reason is in the log ~/Library/Logs/sketch3d-memguard.log
(one JSON line per event), guard heavy prints it itself. A command's time limit (guard.run) is not stopped by a pause.
From the command line (.venv/bin/python -m core.services.memguard …):
  status               — pressure, whether the guard is alive, watched processes: footprint, ceiling, pause
  daemon [--only RE]   — the guard itself (started by launchd); --only — only processes whose command line matches (probes)
  install | uninstall  — launchd agent: start at login, restart on crash; uninstall resumes the paused ones
  resume [pid …]       — resume the paused processes manually (no pid — all of them)
  log [N]              — the last N events
  selftest             — the rules on made-up cases (no processes or signals)
In code: from core.services import memguard as M; M.why(pid, since) — why the guard killed pid or its descendant; M.alive()."""
import ctypes
import json
import os
import plistlib
import re
import signal
import struct
import subprocess
import sys
import time

from core.services import guard as G  # noqa: E402

CONF = {
    "tick_s": 0.5,               # measured every … s
    "max_gb": 8.0,               # default ceiling (a 16 GB machine; the frame pipeline ~4.5, trailer — peak 5.8, a face with hair 6–8)
    "hard_max_gb": 12.0,         # an SK_MEM_MAX_GB above this is clamped to it
    "heavy_gb": 1.0,             # a process heavier than this is a candidate for a pause
    "pause_gap_s": 10,           # between pauses — give the system time to compress the paused one
    "resume_hold_s": 10,         # pressure normal this long in a row — resume
    "resume_gap_s": 20,          # between resumes (and after a pause)
    "pause_max_s": 300,          # paused longer and the pressure is elevated (2) — resume anyway, otherwise it would stay paused forever
    "crit_kill_s": 180,          # critical for longer than this and nobody left to pause — kill the largest one …
    "kill_min_gb": 4.0,          # … if it is not smaller than this
    "term_grace_s": 5,           # SIGTERM did not help within … s — SIGKILL
    "heartbeat_s": 30,           # the "alive" mark in the state
}
LOG = os.environ.get("SK_MEMGUARD_LOG") or os.path.expanduser("~/Library/Logs/sketch3d-memguard.log")
STATE = os.path.join(G.LOCK_DIR, "memguard.json")
LABEL = "com.sketch3d.memguard"
AGENTS = os.path.expanduser("~/Library/LaunchAgents")
ROOT = str(G.ROOT)
BLENDER_RE = re.compile(r"/Blender\.app/Contents/MacOS/Blender$|(^|/)blender$")
PY_RE = re.compile(r"(^|/)python[\d.]*$")
SSTOP = 4                        # pbi_status: process stopped (SIGSTOP)


# ------------------------------------------------------------------------------------- kernel calls: processes without top and ps
_lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
_lib.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
_lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
_lib.proc_pidpath.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
_lib.proc_listallpids.argtypes = [ctypes.c_void_p, ctypes.c_int]


def footprint_gb(pid):
    """Process footprint, GB; no such process — None. RUSAGE_INFO_V2, field [9] — ri_phys_footprint (for a run at 13 GB, ps
    shows an RSS of 0.3)."""
    b = (ctypes.c_uint64 * 64)()
    return b[9] / 2 ** 30 if _lib.proc_pid_rusage(pid, 2, b) == 0 else None


def _bsd(pid):
    """(state, parent, owner, start in epoch seconds) from proc_bsdinfo; no such process — None."""
    b = ctypes.create_string_buffer(136)
    if _lib.proc_pidinfo(pid, 3, 0, b, 136) != 136:          # PROC_PIDTBSDINFO
        return None
    (st,), (pp, uid), (start,) = struct.unpack_from("<I", b.raw, 4), struct.unpack_from("<II", b.raw, 16), struct.unpack_from("<Q", b.raw, 120)
    return st, pp, uid, start


def _path(pid):
    b = ctypes.create_string_buffer(4096)
    return b.value.decode(errors="replace") if _lib.proc_pidpath(pid, b, 4096) > 0 else None


def _cwd(pid):
    """The process's working directory (PROC_PIDVNODEPATHINFO: the pvi_cdir path — at offset 152)."""
    b = ctypes.create_string_buffer(2352)
    return b.raw[152:1176].split(b"\0")[0].decode(errors="replace") if _lib.proc_pidinfo(pid, 9, 0, b, 2352) == 2352 else None


def _args(pid):
    """A process's command line and environment (sysctl KERN_PROCARGS2: argc, path, argv, env); someone else's or dead —
    (None, {})."""
    mib = (ctypes.c_int * 3)(1, 49, pid); size = ctypes.c_size_t(1 << 20); b = ctypes.create_string_buffer(size.value)
    if _lib.sysctl(mib, 3, b, ctypes.byref(size), None, 0) != 0:
        return None, {}
    raw = b.raw[:size.value]; argc, = struct.unpack_from("<i", raw)
    parts = raw[4:].split(b"\0", 1)[1].lstrip(b"\0").split(b"\0")
    env = dict(p.decode(errors="replace").split("=", 1) for p in parts[argc:] if b"=" in p)
    return [p.decode(errors="replace") for p in parts[:argc]], env


def _pids():
    n = _lib.proc_listallpids(None, 0); a = (ctypes.c_int * (n + 256))()
    return a[:_lib.proc_listallpids(a, ctypes.sizeof(a))]


def pressure():
    """Memory pressure: 1 — normal, 2 — elevated, 4 — critical (sysctl called directly, no subprocess — once every 0.5 s)."""
    v, n = ctypes.c_int(1), ctypes.c_size_t(4)
    return v.value if _lib.sysctlbyname(b"kern.memorystatus_vm_pressure_level", ctypes.byref(v), ctypes.byref(n), None, 0) == 0 else 1


# ------------------------------------------------------------------------------------------- what the guard sees
_known = {}                      # (pid, start) → description, or None (not ours): path, arguments, environment — once per process


def _describe(pid, only):
    path = _path(pid)
    if not path:
        return None
    if BLENDER_RE.search(path):
        kind = None
    elif PY_RE.search(path) and ((_cwd(pid) or "") + "/").startswith(ROOT + "/"):
        kind = "py"
    else:
        return None
    argv, env = _args(pid)
    if argv is None:
        return None
    kind = kind or ("bg" if {"-b", "--background"} & set(argv) else "gui")
    cmd = " ".join(argv)
    if only and not re.search(only, cmd):
        return None
    try:
        lim = min(float(env["SK_MEM_MAX_GB"]), CONF["hard_max_gb"]) if env.get("SK_MEM_MAX_GB") else CONF["max_gb"]
    except ValueError:
        lim = CONF["max_gb"]
    return {"kind": kind, "cmd": cmd[:240], "limit": lim}


def scan(only=None):
    """Watched processes of the current user: background Blender (bg), Blender with a window (gui), Python from the project
    folder (py). Each has pid, footprint gb, ceiling limit, whether it is paused (stopped), and ancestors chain (for the guard
    lock and for "why a descendant was killed")."""
    me, uid, alive, parent, found = os.getpid(), os.getuid(), set(), {}, []
    for pid in _pids():
        b = _bsd(pid)
        if not b or b[2] != uid:
            continue
        parent[pid] = b[1]; key = (pid, b[3]); alive.add(key)
        if pid == me:
            continue
        if key not in _known:
            _known[key] = _describe(pid, only)
        if _known[key]:
            found.append((pid, b))
    for k in set(_known) - alive:
        del _known[k]
    out = []
    for pid, b in found:
        gb = footprint_gb(pid)
        if gb is None:
            continue
        chain, q = [], parent.get(pid)
        while q and q > 1 and len(chain) < 40:
            chain.append(q); q = parent.get(q)
        out.append({**_known[(pid, b[3])], "pid": pid, "start": b[3], "gb": gb, "stopped": b[0] == SSTOP, "chain": chain})
    return out


def _lock_holder():
    """pid of the holder of the guard heavy-run lock: the lock file is non-empty (it is erased on release) and the process is
    alive. Read without flock — the memory guard does not interfere with the guard queue."""
    try:
        m = re.match(r"pid (\d+)", open(os.path.join(G.LOCK_DIR, "pipeline.lock")).read())
    except OSError:
        return None
    return int(m.group(1)) if m and G._alive(int(m.group(1))) else None


# ------------------------------------------------------------------------------------------------------ rules
def new_state():
    return {"paused": {}, "killing": {}, "warned": [], "crit_since": None, "calm_since": None, "last_pause": 0.0,
            "last_resume": 0.0, "beat": 0.0, "pid": os.getpid(), "level": 1}


def _kill(st, p, now, why, acts):
    st["killing"][p["pid"]] = {"start": p["start"], "at": now, "why": why}
    st["paused"].pop(p["pid"], None)
    acts.append(("kill", p["pid"], why))


def decide(procs, level, now, st, protected=frozenset()):
    """The rules for one sample → actions (kill | kill9 | pause | resume | notify, pid, reason). st is the state between samples,
    mutated here: paused processes, those being killed, since when the pressure has been critical and normal, the time of the
    last pause and resume."""
    acts, by = [], {p["pid"]: p for p in procs}
    for pid in list(st["paused"]):                                     # died or was resumed by someone else — forget it
        p = by.get(pid)
        if not p or p["start"] != st["paused"][pid]["start"] or not p["stopped"]:
            del st["paused"][pid]
    for pid, k in list(st["killing"].items()):
        p = by.get(pid)
        if not p or p["start"] != k["start"]:
            del st["killing"][pid]
        elif now - k["at"] >= CONF["term_grace_s"] and not k.get("hard"):
            k["hard"] = True; acts.append(("kill9", pid, k["why"] + "; SIGTERM did not help"))
    for p in procs:                                                    # 1. ceiling
        if p["pid"] in st["killing"] or p["gb"] <= p["limit"]:
            continue
        if p["kind"] == "gui":
            key = f"{p['pid']}:{p['start']}"
            if key not in st["warned"]:
                st["warned"] = st["warned"][-50:] + [key]
                acts.append(("notify", p["pid"], f"Blender with a window: {p['gb']:.1f} GB > {p['limit']:g} GB — save your work"))
            continue
        _kill(st, p, now, f"ceiling: {p['gb']:.2f} GB > {p['limit']:g} GB", acts)
    if level >= 4:
        st["crit_since"] = st["crit_since"] or now; st["calm_since"] = None
    elif level <= 1:
        st["calm_since"] = st["calm_since"] or now; st["crit_since"] = None
    else:
        st["crit_since"] = st["calm_since"] = None
    live = [p for p in procs if p["kind"] != "gui" and p["pid"] not in st["killing"]]
    running = [p for p in live if p["pid"] not in st["paused"] and p["gb"] >= CONF["heavy_gb"]]
    if level >= 4:                                                     # 2. pause the youngest; 3. prolonged — kill the biggest
        if len(running) >= 2:
            if now - st["last_pause"] >= CONF["pause_gap_s"]:
                v = max(running, key=lambda p: (p["pid"] not in protected, p["start"], p["pid"]))
                st["paused"][v["pid"]] = {"start": v["start"], "at": now}; st["last_pause"] = now
                acts.append(("pause", v["pid"], f"pressure critical, {len(running)} heavy processes running — the youngest waits"))
        elif now - st["crit_since"] >= CONF["crit_kill_s"]:
            big = max(live, key=lambda p: p["gb"], default=None)
            if big and big["gb"] >= CONF["kill_min_gb"]:
                _kill(st, big, now, f"pressure critical for {now - st['crit_since']:.0f} s, nothing to pause;"
                                    f" the largest is {big['gb']:.1f} GB", acts)
                st["crit_since"] = now
    elif st["paused"] and now - max(st["last_resume"], st["last_pause"]) >= CONF["resume_gap_s"]:
        pid = min(st["paused"], key=lambda q: (st["paused"][q]["start"], q))
        waited = now - st["paused"][pid]["at"]
        why = ("pressure back to normal" if st["calm_since"] and now - st["calm_since"] >= CONF["resume_hold_s"] else
               "no heavy processes left running" if not running else
               f"paused for {waited:.0f} s, pressure elevated — do not hold any longer" if waited >= CONF["pause_max_s"] else None)
        if why:
            del st["paused"][pid]; st["last_resume"] = now
            acts.append(("resume", pid, why))
    return acts


# ------------------------------------------------------------------------------------------- actions and log
def _log(ev):
    ev = {"ts": round(time.time(), 1), "t": time.strftime("%Y-%m-%d %H:%M:%S"), **ev}
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")


_shown = []                      # notifications in flight: osascript takes up to a second, the sampling does not wait for it


def _notify(text):
    _shown[:] = [q for q in _shown if q.poll() is None]
    try:
        _shown.append(subprocess.Popen(["osascript", "-e", f"display notification {json.dumps(text, ensure_ascii=False)}"
                                        " with title \"sketch3d: memory\""], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    except OSError:
        pass


SIG = {"kill": signal.SIGTERM, "kill9": signal.SIGKILL, "pause": signal.SIGSTOP, "resume": signal.SIGCONT}


def execute(a, by, dry=False):
    what, pid, why = a; p = by.get(pid, {})
    ev = {"act": what, "pid": pid, "why": why, "kind": p.get("kind"), "gb": round(p.get("gb", 0), 2), "limit": p.get("limit"),
          "cmd": p.get("cmd", "")[:160], "chain": p.get("chain", [])}
    if dry:
        ev["dry"] = True
    elif what in SIG:
        try:
            os.kill(pid, SIG[what])
            if what == "kill":
                os.kill(pid, signal.SIGCONT)                           # a paused process will not receive SIGTERM while it is paused
        except ProcessLookupError:
            ev["gone"] = True
    _log(ev)
    if what in ("kill", "notify") and not dry:
        _notify(f"{why}: {ev['cmd'][:60]}" if what == "kill" else why)


def tick(st, level=None, only=None, dry=False):
    """One sample: processes, pressure (level — can be injected for probes), rules, signals, log. Returns the actions."""
    procs = scan(only)
    st["level"] = level = pressure() if level is None else level
    holder = _lock_holder()
    protected = {p["pid"] for p in procs if holder and holder in p["chain"]}
    acts = decide(procs, level, time.time(), st, protected)
    by = {p["pid"]: p for p in procs}
    for a in acts:
        execute(a, by, dry)
    return acts


def _save(st):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    keep = {k: st[k] for k in ("paused", "beat", "pid", "level")}
    with open(STATE + ".tmp", "w") as f:
        json.dump(keep, f, ensure_ascii=False)
    os.replace(STATE + ".tmp", STATE)


def _read():
    try:
        return json.load(open(STATE))
    except (OSError, ValueError):
        return {}


def alive():
    """The guard is running: the "alive" mark is fresh and its process exists."""
    st = _read()
    return bool(st.get("pid")) and time.time() - st.get("beat", 0) < 2 * CONF["heartbeat_s"] + 5 and G._alive(st["pid"])


def daemon(only=None, dry=False):
    """The guard loop. Picks up the processes left paused from a previous life (crashed, restarted by launchd); on stop — resumes
    all of them."""
    st = new_state()
    st["paused"] = {int(k): v for k, v in _read().get("paused", {}).items()}
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    _log({"act": "start", "pid": os.getpid(), "why": f"ceiling {CONF['max_gb']:g} GB, sampling every {CONF['tick_s']} s"
                                                     + (f", only {only}" if only else "") + (", dry run" if dry else "")})
    try:
        while True:
            t0 = time.time()
            if tick(st, only=only, dry=dry) or t0 - st["beat"] >= CONF["heartbeat_s"]:
                st["beat"] = t0; _save(st)
            time.sleep(max(0.05, CONF["tick_s"] - (time.time() - t0)))
    finally:
        for pid in list(st["paused"]):
            execute(("resume", pid, "guard stopped"), {}, dry)
        st["paused"] = {}; st["beat"] = 0.0; _save(st)
        _log({"act": "stop", "pid": os.getpid(), "why": "guard stopped"})


def resume(pids=()):
    st = _read(); todo = [int(p) for p in pids] or [int(p) for p in st.get("paused", {})]
    for pid in todo:
        execute(("resume", pid, "manual"), {}, False)
    return todo


def why(pid, since=0.0):
    """Why the guard killed process pid or its descendant after since (epoch seconds) — text; if it did not kill — None."""
    if not os.path.exists(LOG):
        return None
    for line in reversed(open(LOG).readlines()[-1000:]):
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("ts", 0) < since:
            break
        if ev.get("act") in ("kill", "kill9") and (ev.get("pid") == pid or pid in ev.get("chain", [])):
            return f"the memory guard killed {ev.get('kind')} pid {ev['pid']}: {ev['why']} ({ev['t']}; {ev.get('cmd', '')[:80]})"
    return None


def status():
    st = _read(); level = pressure()
    print(f"pressure {level} (1 normal, 2 elevated, 4 critical), swap {G.swap_used_gb()} GB; guard: "
          + (f"running (pid {st['pid']}), default ceiling {CONF['max_gb']:g} GB" if alive() else "NOT RUNNING — memguard install"))
    paused = {int(k) for k in st.get("paused", {})}
    for p in sorted(scan(), key=lambda p: -p["gb"]):
        mark = "PAUSE" if p["pid"] in paused else ("stands" if p["stopped"] else "")
        print(f"  {p['pid']:>6} {p['kind']:3s} {p['gb']:6.2f} / {p['limit']:g} GB  {mark:5s} {p['cmd'][:100]}")


# -------------------------------------------------------------------------------------------------- self-test
def selftest():
    """The rules on made-up cases: ceiling, window, custom ceiling, pausing the youngest, lock, resuming, prolonged pressure."""
    def P(pid, gb, start, kind="bg", limit=8.0, stopped=False):
        return {"pid": pid, "gb": gb, "start": start, "kind": kind, "limit": limit, "stopped": stopped, "chain": [], "cmd": f"p{pid}"}

    def run(procs, level, now, st, prot=frozenset()):
        return [(a, pid) for a, pid, _ in decide(procs, level, now, st, prot)]

    ok, st = [], new_state()
    ok.append(("ceiling → SIGTERM", run([P(1, 9, 10)], 1, 100, st) == [("kill", 1)]))
    ok.append(("alive after 5 s → SIGKILL", run([P(1, 9, 10)], 1, 105, st) == [("kill9", 1)]))
    ok.append(("died — forgotten", run([], 1, 106, st) == [] and not st["killing"]))
    st = new_state()
    ok.append(("window — one notification", run([P(2, 9, 10, "gui")], 1, 100, st) == [("notify", 2)]
               and run([P(2, 9, 10, "gui")], 1, 101, st) == []))
    ok.append(("custom ceiling 12 — 9 GB is fine", run([P(3, 9, 10, limit=12)], 1, 100, new_state()) == []))
    st = new_state(); a, b = P(1, 5, 10), P(2, 5, 20)
    ok.append(("critical, two processes — pause the youngest", run([a, b], 4, 100, st) == [("pause", 2)]))
    b["stopped"] = True
    ok.append(("no second pause — only one is running", run([a, b], 4, 105, st) == []))
    ok.append(("normal for 10 s — too early (20 s after the pause)", run([a, b], 1, 112, st) == [] and run([a, b], 1, 118, st) == []))
    ok.append(("normal, 20 s since the pause — resume", run([a, b], 1, 122, st) == [("resume", 2)]))
    b["stopped"] = False
    st = new_state()
    ok.append(("the guard lock is held by the youngest — the older one waits", run([P(1, 5, 10), P(2, 5, 20)], 4, 100, st, {2}) == [("pause", 1)]))
    st = new_state(); a, b = P(1, 5, 10), P(2, 5, 20, stopped=True); st["paused"] = {2: {"start": 20, "at": 0}}
    ok.append(("elevated, paused for > 5 min — resume", run([a, b], 2, 320, st) == [("resume", 2)]))
    st = new_state(); st["paused"] = {2: {"start": 20, "at": 100}}
    ok.append(("elevated, no heavy processes running — resume", run([P(1, 0.3, 10), P(2, 5, 20, stopped=True)], 2, 130, st) == [("resume", 2)]))
    st = new_state()
    ok.append(("one heavy process, critical for 3 min, 3 GB — leave it alone",
               run([P(1, 3, 10)], 4, 0.1, st) == [] and run([P(1, 3, 10)], 4, 200, st) == []))
    st = new_state()
    ok.append(("one heavy process, critical for 3 min, 6 GB — kill it",
               run([P(1, 6, 10)], 4, 0.1, st) == [] and run([P(1, 6, 10)], 4, 181, st) == [("kill", 1)]))
    st = new_state(); st["paused"] = {2: {"start": 20, "at": 100}}
    ok.append(("resumed by someone else — forgotten", run([P(2, 5, 20)], 4, 101, st) == [] and not st["paused"]))
    for name, good in ok:
        print(f"  {'yes ' if good else 'NO'} {name}")
    return all(g for _, g in ok)


# ---------------------------------------------------------------------------------------------- launchd agent
def install():
    py = os.path.join(ROOT, ".venv", "bin", "python"); path = os.path.join(AGENTS, LABEL + ".plist")
    os.makedirs(AGENTS, exist_ok=True)
    plist = {"Label": LABEL, "ProgramArguments": [py, "-m", "core.services.memguard", "daemon"], "WorkingDirectory": ROOT,
             "EnvironmentVariables": {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"},
             "RunAtLoad": True, "KeepAlive": True, "ProcessType": "Interactive",     # do not throttle the guard under load
             "StandardOutPath": LOG.replace(".log", ".launchd.log"), "StandardErrorPath": LOG.replace(".log", ".launchd.log")}
    subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}/{LABEL}"], capture_output=True)
    with open(path, "wb") as f:
        plistlib.dump(plist, f)
    r = subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", path], capture_output=True, text=True)
    print(f"{LABEL}: {'installed' if r.returncode == 0 else 'error ' + r.stderr.strip()} ({path})")


def uninstall():
    path = os.path.join(AGENTS, LABEL + ".plist")
    subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}/{LABEL}"], capture_output=True)
    if os.path.exists(path):
        os.remove(path)
    print(f"{LABEL}: removed")


if __name__ == "__main__":
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else "status"), sys.argv[2:]
    if cmd == "status":
        status()
    elif cmd == "daemon":
        daemon(only=args[args.index("--only") + 1] if "--only" in args else None, dry="--dry" in args)
    elif cmd == "install":
        install()
    elif cmd == "uninstall":
        uninstall()
    elif cmd == "resume":
        print("resumed:", resume(args) or "nobody")
    elif cmd == "log":
        n = int(args[0]) if args else 20
        print("".join(open(LOG).readlines()[-n:]) if os.path.exists(LOG) else "no log yet")
    elif cmd == "selftest":
        sys.exit(0 if selftest() else 1)
    else:
        print(__doc__)

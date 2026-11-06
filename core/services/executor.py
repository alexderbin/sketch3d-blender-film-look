"""Executor of heavy computations — a core service next to the guard (guard.py), working on its memory thresholds.
A component supplies independent tasks and a memory estimate for each; the executor decides how many to run at once: cores (minus one — left for the user
and other work on the machine), available memory minus a reserve (guard.LIMITS mem_free_min_pct — as for a heavy run), memory pressure, other
heavy runs (the guard lock, optional). A task is its own process (`python -m core.services.executor _one`): the footprint is measured, the failure
of one does not bring down the others, memory goes back to the system at once. When pressure is elevated it starts no new ones; when critical it pauses the youngest
(SIGSTOP), when it subsides it resumes them one by one; a task killed by a signal (the memory guard, shortage) is resubmitted with an estimate from the measurement.
With no estimate it learns on the fly: the footprint per cost unit (or the peak) of finished tasks × learn_k. The report is a line per task and a total (wall,
sum of tasks, speedup, parallelism, peak memory), JSON optionally. The core does not import components: the task function is a string
"module:function" (like the hooks points), imported by the task process; the call is fn(*args).
From the command line (.venv/bin/python -m core.services.executor …):
  plan [GB]                              — how many tasks of the given GB can run at once now and why (cores, memory, pressure, lock)
  dir <folder> <module:function> [--workers N] [--lock] [--what TEXT] — the tasks of a folder job_*.pkl (fields mem_gb, cost — optional):
                                           the function receives a tuple (folder, name); the report is <folder>/executor_report.json
  selftest [n=6] [mb=150] [s=4]          — trials: Linux memory measurement against invented cgroup files, a pool on invented tasks
                                           (memory and numpy counting) against a single thread, resubmission of a killed task, a pause under pressure
In code: from core.services import executor as X; X.run([X.Task("id", "pkg.mod:fn", (a, b), mem_gb=1.5, cost=…)]) → a report."""
import core.paths        # noqa: F401  the core parts in sys.path
import contextlib
import dataclasses
import importlib
import json
import os
import pickle
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from core.services import guard as G

CONF = {
    "tick_s": 0.5,               # measured every … s
    "reserve_pct": G.LIMITS["mem_free_min_pct"],   # free after the computation — no less than this, % of all memory (as for a heavy run)
    "spare_cpus": 1,             # cores to leave for the user and other work on the machine (a rented machine — SK_EXEC_SPARE=0)
    "default_gb": 1.0,           # task estimate if the component gave none and there are no measurements yet
    "learn_k": 1.2,              # margin over the measured footprint
    "retries": 2,                # a task killed by a signal is resubmitted this many times
    "pause_gap_s": 10,           # between pauses — let the system compress the paused one (as in the memory guard)
    "resume_hold_s": 10,         # pressure is normal this long in a row — resume one
    "crit_wait_s": 300,          # critical and nothing is computing — after this many seconds start one task anyway
    "nice": 10,                  # background computations get a lower priority
}
THREADS_1 = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


@dataclasses.dataclass
class Task:
    id: str
    fn: str                      # "module:function"
    args: tuple = ()
    mem_gb: float = None         # the component's estimate (peak footprint); if absent, the executor learns
    cost: float = None           # relative weight (the order: large ones first; the memory estimate from measurements)


def python():
    """The Python of task processes: the project environment (inside Blender sys.executable is Blender itself)."""
    p = os.environ.get("SK_PYTHON") or os.path.join(G.ROOT, ".venv", "bin", "python")
    return p if os.path.exists(p) else sys.executable


def footprint_gb(pid):
    """The process footprint now: macOS — phys_footprint (the memory guard; RSS lies under compression), Linux — VmRSS."""
    if G.LINUX:
        for line in (G._text(f"/proc/{pid}/status") or "").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 2 ** 20
        return None
    from core.services import memguard as M
    return M.footprint_gb(pid)


def plan(mem_gb=1.0, mem=None):
    """How many tasks of mem_gb can run at once now — the numbers and the reason (the same rule as run applies at start)."""
    m = mem or G.mem(); cpu = max(1, G.cpus() - _spare())
    reserve = m["total_gb"] * CONF["reserve_pct"] / 100; room = (m["avail_gb"] or 0) - reserve
    by_mem = max(0, int(room // max(mem_gb, 0.05)))
    n = min(cpu, by_mem) if m["pressure"] in (None, 1) else 0
    why = "memory pressure" if m["pressure"] not in (None, 1) else "memory" if by_mem < cpu else "cores"
    return {"workers": n, "cpu": cpu, "by_mem": by_mem, "limit": why, "avail_gb": m["avail_gb"], "reserve_gb": round(reserve, 1),
            "pressure": m["pressure"], "swap_gb": m["swap_gb"], "heavy_run": G.lock_busy()}


def _spare():
    return int(os.environ.get("SK_EXEC_SPARE", CONF["spare_cpus"]))


class _Run:
    """A task in progress: process, time, footprint, pause."""
    def __init__(self, task, est, tries, pid, t0, log):
        self.task, self.est, self.tries, self.pid, self.t0, self.log = task, est, tries, pid, t0, log
        self.peak = self.cur = 0.0; self.paused_at = None; self.paused_s = 0.0


def run(tasks, where="local", workers=None, lock=False, what="", report=None, log=print, mem=None):
    """Run the tasks; the report is a dict (and JSON in report). where: local — here; remote/auto — a rented machine (not implemented yet).
    lock — take guard's shared lock for heavy runs (a run nested under guard passes at once). mem — a memory measurement (trials)."""
    if where != "local":
        raise NotImplementedError("the executor's remote mode is not implemented yet")
    ctx = G.heavy_lock(what=what or f"executor: {len(tasks)} tasks", wait=1800) if lock else contextlib.nullcontext()
    with ctx:
        return _local(list(tasks), workers, report, log, mem or G.mem)


def _local(tasks, workers, report, log, mem):
    cpu = workers or max(1, G.cpus() - _spare())
    m = mem(); reserve = m["total_gb"] * CONF["reserve_pct"] / 100
    work = tempfile.mkdtemp(prefix="sk_exec_")
    env = {**os.environ, **{k: "1" for k in THREADS_1}}
    pending = [(t, 0, None) for t in sorted(tasks, key=lambda t: -(t.cost or 0))]   # (task, attempt, lower estimate after being killed)
    running, rows, results = {}, [], {}
    learn = {"per_cost": 0.0, "peak": 0.0, "corr": 1.0}
    stat = {"pauses": 0, "retries": 0, "max_parallel": 0, "mem_min_gb": m["avail_gb"]}
    ok_since, last_pause, last_resume, crit_since = None, 0.0, 0.0, None
    t_start = time.time()

    def est(t, floor):
        e = (t.mem_gb * learn["corr"]) if t.mem_gb else (learn["per_cost"] * t.cost if t.cost and learn["per_cost"] else learn["peak"] or CONF["default_gb"])
        return max(e, floor or 0.0)

    def start(t, tries, floor):
        spec = os.path.join(work, f"{t.id}.{tries}.spec"); lg = os.path.join(work, f"{t.id}.{tries}.log")
        pickle.dump({"fn": t.fn, "args": t.args, "ret": os.path.join(work, f"{t.id}.ret")}, open(spec, "wb"))
        with open(lg, "wb") as f_:
            p = subprocess.Popen([python(), "-m", "core.services.executor", "_one", spec], cwd=G.ROOT, env=env, stdout=f_,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        running[p.pid] = _Run(t, est(t, floor), tries, p.pid, time.time(), lg); p.returncode = 0   # reaped by wait4, not by Popen
        stat["max_parallel"] = max(stat["max_parallel"], len(running))

    def signal_group(r, sig):
        try: os.killpg(r.pid, sig)
        except (ProcessLookupError, PermissionError): pass

    try:
        while pending or running:
            now = time.time()
            for pid, r in list(running.items()):                                 # measurement and reaping
                fp = footprint_gb(pid)
                if fp is not None: r.cur = fp; r.peak = max(r.peak, fp)
                try: wp, st, ru = os.wait4(pid, os.WNOHANG)
                except ChildProcessError: wp, st, ru = pid, 0, None
                if not wp: continue
                del running[pid]
                code = os.waitstatus_to_exitcode(st)
                if ru: r.peak = max(r.peak, ru.ru_maxrss / (2 ** 20 if G.LINUX else 2 ** 30))
                if r.paused_at: r.paused_s += now - r.paused_at
                dur = round(now - r.t0 - r.paused_s, 1); t = r.task
                row = {"id": t.id, "s": dur, "peak_gb": round(r.peak, 2), "est_gb": round(r.est, 2), "code": code, "try": r.tries + 1,
                       "parallel": len(running) + 1}
                if code == 0:
                    if t.cost: learn["per_cost"] = max(learn["per_cost"], r.peak / t.cost * CONF["learn_k"])
                    if t.mem_gb: learn["corr"] = max(learn["corr"], r.peak / t.mem_gb)
                    learn["peak"] = max(learn["peak"], r.peak * CONF["learn_k"])
                    try: results[t.id] = pickle.load(open(os.path.join(work, f"{t.id}.ret"), "rb"))
                    except (OSError, EOFError, pickle.UnpicklingError): results[t.id] = None
                    log(f"{t.id}: {dur} s, peak {r.peak:.2f} GB (estimate {r.est:.2f}), in parallel {row['parallel']}")
                elif code < 0 and r.tries < CONF["retries"]:                    # killed by a signal — once more, with an estimate no lower than the measurement
                    floor = max(r.est, r.peak * CONF["learn_k"]); pending.insert(0, (t, r.tries + 1, floor)); stat["retries"] += 1
                    row["requeued"] = True
                    log(f"{t.id}: killed by signal {-code} at {r.peak:.2f} GB — resubmitted (estimate {floor:.2f} GB)")
                else:
                    tail = open(r.log, errors="replace").read()[-800:]
                    row["error"] = tail[-300:]; log(f"{t.id}: ERROR code {code}\n{tail}")
                rows.append(row)
            m = mem(); lvl = m["pressure"] or 1
            stat["mem_min_gb"] = round(min(stat["mem_min_gb"] or 1e9, m["avail_gb"] or 1e9), 1)
            active = [r for r in running.values() if not r.paused_at]; paused = [r for r in running.values() if r.paused_at]
            if lvl >= 4:                                                         # critical: the youngest is paused (one stays running)
                ok_since = None; crit_since = crit_since or now
                if len(active) >= 2 and now - last_pause >= CONF["pause_gap_s"]:
                    r = max(active, key=lambda r: r.t0); signal_group(r, signal.SIGSTOP); r.paused_at = now; last_pause = now
                    stat["pauses"] += 1; log(f"pressure is critical — {r.task.id} is paused")
            else:
                crit_since = None
                ok_since = (ok_since or now) if lvl <= 1 else None
                if paused and now - last_resume >= CONF["pause_gap_s"] and (not active or (ok_since and now - ok_since >= CONF["resume_hold_s"])):
                    r = min(paused, key=lambda r: r.t0); signal_group(r, signal.SIGCONT); r.paused_s += now - r.paused_at; r.paused_at = None
                    last_resume = now; log(f"pressure has subsided — {r.task.id} resumes")
            active = [r for r in running.values() if not r.paused_at]; paused = [r for r in running.values() if r.paused_at]
            room = (m["avail_gb"] or 0) - reserve - sum(max(0.0, r.est - r.cur) for r in active)
            while pending and not paused and len(active) < cpu:
                if active and lvl > 1: break                                     # elevated — start no new ones
                if not active and lvl >= 4 and now - (crit_since or now) < CONF["crit_wait_s"]: break
                k = next((i for i, (t, _, fl) in enumerate(pending) if est(t, fl) <= room), None)
                if k is None and active: break                                   # does not fit — wait until room frees up
                t, tries, fl = pending.pop(k or 0); e = est(t, fl)
                start(t, tries, fl); room -= e; active = [r for r in running.values() if not r.paused_at]
            time.sleep(CONF["tick_s"])
    finally:
        for r in running.values():                                               # interrupted — no children are left behind
            signal_group(r, signal.SIGCONT); signal_group(r, signal.SIGTERM)
        shutil.rmtree(work, ignore_errors=True)
    wall = round(time.time() - t_start, 1); done = [x for x in rows if x["code"] == 0]
    s_sum = round(sum(x["s"] for x in done), 1)
    rep = {"where": "local", "cpu": cpu, "total_gb": m["total_gb"], "reserve_gb": round(reserve, 1), "tasks": len(tasks), "ok": len(done),
           "failed": [x["id"] for x in rows if x["code"] != 0 and not x.get("requeued")], "wall_s": wall, "sum_s": s_sum,
           "speedup": round(s_sum / wall, 2) if wall else None, "peak_gb": round(max((x["peak_gb"] for x in done), default=0), 2), **stat,
           "rows": rows}
    log(f"executor: {len(done)}/{len(tasks)} tasks in {wall} s (sum {s_sum} s, ×{rep['speedup']}), up to {stat['max_parallel']} at once "
        f"of {cpu} cores, pauses {stat['pauses']}, resubmissions {stat['retries']}, lowest availability {stat['mem_min_gb']} GB")
    if report:
        json.dump(rep, open(report, "w"), ensure_ascii=False, indent=1)
    rep["results"] = results
    return rep


def run_dir(folder, fn, **kw):
    """The tasks of a folder: job_*.pkl (a dict; mem_gb and cost optional) → the function fn((folder, name)); the report is next to them."""
    folder = os.path.abspath(folder); tasks = []
    for n in sorted(x for x in os.listdir(folder) if x.startswith("job_") and x.endswith(".pkl")):
        j = pickle.load(open(os.path.join(folder, n), "rb"))
        tasks.append(Task(n[:-4], fn, ((folder, n),), mem_gb=j.get("mem_gb"), cost=j.get("cost")))
    return run(tasks, report=os.path.join(folder, "executor_report.json"), **kw)


def _one(spec):
    """The task process: import the function by its string, call it, the reply is a pickle next to it."""
    os.nice(CONF["nice"])
    s = pickle.load(open(spec, "rb")); mod, _, name = s["fn"].partition(":")
    ret = getattr(importlib.import_module(mod), name)(*s["args"])
    with open(s["ret"] + ".part", "wb") as f:
        pickle.dump(ret, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(s["ret"] + ".part", s["ret"])


# ------------------------------------------------------------------------------------------------------------------- trials
def _probe(mb, iters, die_once=None):
    """A synthetic task: holds mb MB and runs iters numpy passes. die_once — a marker file: if absent, create it and die by SIGKILL
    (like a task killed by the memory guard), if present — compute."""
    import numpy as np
    if die_once and not os.path.exists(die_once):
        open(die_once, "w").close(); os.kill(os.getpid(), signal.SIGKILL)
    a = np.ones(int(mb * 2 ** 20 / 8)); s = 0.0
    for i in range(iters):
        a = np.sin(a) * 0.5 + 0.5; s += float(a[::4096].sum())
    return round(s, 3)


def _calibrate(mb, seconds):
    import numpy as np
    a = np.ones(int(mb * 2 ** 20 / 8)); t = time.time(); n = 0
    while time.time() - t < 0.5:
        a = np.sin(a) * 0.5 + 0.5; n += 1
    return max(1, int(seconds * n / (time.time() - t)))


def selftest(n=6, mb=150, s=4):
    import numpy as np  # noqa: F401  the trial computes with numpy — in the project environment
    tmp = tempfile.mkdtemp(prefix="sk_exec_test_"); ok = []
    try:
        p, c = os.path.join(tmp, "proc"), os.path.join(tmp, "cg"); os.makedirs(os.path.join(p, "pressure")); os.makedirs(c)
        open(f"{p}/meminfo", "w").write("MemTotal: 263852904 kB\nMemFree: 100000000 kB\nMemAvailable: 200000000 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n")
        open(f"{c}/memory.max", "w").write("66571993088\n"); open(f"{c}/memory.current", "w").write("40000000000\n")
        open(f"{c}/memory.stat", "w").write("anon 30000000000\ninactive_file 5000000000\n")
        open(f"{c}/memory.pressure", "w").write("some avg10=12.50 avg60=3.00 avg300=1.00 total=1\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n")
        lm = G.linux_mem(p, c); good = lm["total_gb"] == 62.0 and abs(lm["avail_gb"] - 29.4) < 0.05 and lm["pressure"] == 2
        ok.append(good); print(f"{'yes ' if good else 'NO'} Linux: container 62 GB, used 35 − cache 5 → available {lm['avail_gb']} GB, PSI 12.5 % → pressure {lm['pressure']}")

        real = G.mem(); pl = plan(mb / 1024 * 2, real)
        print(f"    here now: available {real['avail_gb']} GB, reserve {pl['reserve_gb']}, pressure {real['pressure']}, swap {real['swap_gb']} GB, "
              f"another heavy run: {pl['heavy_run'] or 'no'} → at once {pl['workers']} (limit: {pl['limit']})")
        room = lambda k_gb: (lambda: {**real, "avail_gb": real["total_gb"] * CONF["reserve_pct"] / 100 + k_gb, "pressure": 1})   # memory
        iters = _calibrate(mb, s)                                             # the trial is synthetic: the decision does not depend on other people's runs
        one = run([Task("serial", "core.services.executor:_probe", (mb, iters))], workers=1, log=lambda *_: None, mem=room(2.0))
        t1 = one["rows"][0]["s"]
        rep = run([Task(f"p{i}", "core.services.executor:_probe", (mb, iters), cost=1.0) for i in range(n)], log=lambda x: print("   ", x),
                  mem=room(n * mb / 1024 * 2.5))
        par = n * t1 / rep["wall_s"]
        good = rep["ok"] == n and par > 1.5; ok.append(good)
        print(f"{'yes ' if good else 'NO'} pool: {n} tasks of {mb:.0f} MB, one takes {t1} s; at once {rep['max_parallel']} of {rep['cpu']} cores, "
              f"wall {rep['wall_s']} s against ~{round(n * t1, 1)} single-threaded — ×{round(par, 1)}; task peak {rep['peak_gb']} GB")
        rep = run([Task(f"m{i}", "core.services.executor:_probe", (mb, max(1, iters // 4))) for i in range(4)], log=lambda *_: None,
                  mem=room(2 * mb / 1024 * 1.1), workers=4)                  # room for two tasks by the estimate: it takes no more than two
        good = rep["ok"] == 4 and rep["max_parallel"] <= 2; ok.append(good)
        print(f"{'yes ' if good else 'NO'} memory: room for ~2 tasks — at once {rep['max_parallel']} of 4 (estimate from the measurement: {rep['rows'][-1]['est_gb']} GB)")

        mark = os.path.join(tmp, "die")
        rep = run([Task("die", "core.services.executor:_probe", (50, 3, mark))], log=lambda *_: None)
        good = rep["ok"] == 1 and rep["retries"] == 1; ok.append(good)
        print(f"{'yes ' if good else 'NO'} resubmission: the task killed by SIGKILL was completed on the second attempt ({rep['rows'][0].get('requeued')}, {rep['rows'][-1]['code']})")

        t0 = []; it7 = _calibrate(50, 7)                                     # the pressure is synthetic: normal for 1.5 s, critical for 4 s, normal
        def fake():
            t0.append(t0[0] if t0 else time.time())
            return {**room(2.0)(), "pressure": 4 if 1.5 < time.time() - t0[0] < 5.5 else 1}
        CONF_ = dict(CONF); CONF.update(pause_gap_s=0.5, resume_hold_s=1.0)
        try:
            rep = run([Task(f"q{i}", "core.services.executor:_probe", (50, it7), mem_gb=0.2) for i in range(3)], workers=3, mem=fake,
                      log=lambda x: print("   ", x))
        finally:
            CONF.clear(); CONF.update(CONF_)
        good = rep["ok"] == 3 and rep["pauses"] >= 1; ok.append(good)
        print(f"{'yes ' if good else 'NO'} pressure: pauses {rep['pauses']}, all {rep['ok']} of 3 finished after resuming")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"selftest: {sum(ok)}/{len(ok)}")
    return all(ok)


if __name__ == "__main__":
    cmd, a = (sys.argv[1] if len(sys.argv) > 1 else ""), sys.argv[2:]
    if cmd == "_one":
        _one(a[0])
    elif cmd == "plan":
        print(json.dumps(plan(float(a[0]) if a else 1.0), ensure_ascii=False))
    elif cmd == "dir":
        kv = {a[i]: a[i + 1] for i in range(2, len(a) - 1) if a[i].startswith("--") and a[i] != "--lock"}
        rep = run_dir(a[0], a[1], workers=int(kv["--workers"]) if "--workers" in kv else None, lock="--lock" in a, what=kv.get("--what", ""))
        sys.exit(1 if rep["failed"] or rep["ok"] < rep["tasks"] else 0)
    elif cmd == "selftest":
        kv = dict(x.split("=", 1) for x in a if "=" in x)
        sys.exit(0 if selftest(int(kv.get("n", 6)), float(kv.get("mb", 150)), float(kv.get("s", 4))) else 1)
    else:
        print(__doc__)

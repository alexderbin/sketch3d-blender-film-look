"""Step-by-step scene assembly in one Blender (ARCHITECTURE.md).
One step = a component function over the open scene: the manager opens the previous step's checkpoint .blend (or an empty scene),
calls `function(ctx)` from the passport module ("build: file.py:function"), marks the new objects with the section (`sk_layer`),
puts them into the section collection and saves the step's checkpoint .blend. An output (kind: output) is the same function
without saving: it reads the scene and writes files.
    blender -b --factory-startup --python-exit-code 1 -P core/blender/build.py -- <step.json>
step.json (written by the manager): step, section, module (path to the .py), function, plan (the step's plan.json or null), input
(a .blend or null), output (a .blend or null), scene (a copy of the run's scene), out (the step folder), profile, keys, gives
(data from earlier steps), component.
ctx of the function: sc (the bpy scene), section, collection (the section collection), plan (dict), keys, scene, out, profile, gives, root.
The function returns a dict — data for the next steps (the "gives" context) and for the step's build.json. Prints __RESULT__."""
import importlib, importlib.util, json, os, sys, time

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path: sys.path.insert(0, ROOT)                 # bootstrap (with run.py): packages core/, components/, formats/
from core.paths import module_name, blender_site  # noqa: E402
blender_site()                                                    # venv site-packages (yaml for the dictionary) — a single Python 3.13


def collection(sc, name):
    """The section collection in the scene (one per section)."""
    c = bpy.data.collections.get(name)
    if c is None: c = bpy.data.collections.new(name)
    if c.name not in sc.collection.children: sc.collection.children.link(c)
    return c


def load_module(path):
    """The step module by the path from step.json: inside packages — an import by the dotted name (components.light.steps), otherwise —
    loading from the file (sys.path is left untouched)."""
    name = module_name(path)
    if name: return importlib.import_module(name)
    spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0] + "_step", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def main():
    if os.environ.get("SK_IN_BLENDER"):                                # fork-bomb guard, as in run.py
        sys.stderr.write("build.py: nested Blender launch refused (SK_IN_BLENDER)\n"); sys.exit(3)
    os.environ["SK_IN_BLENDER"] = "1"
    S = json.load(open(sys.argv[sys.argv.index("--") + 1]))
    t0 = time.time()
    if S.get("input"):
        bpy.ops.wm.open_mainfile(filepath=S["input"], load_ui=False)
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sec = S.get("section") or S["step"]
    coll = collection(sc, sec) if not S.get("is_output") else None
    before = {o.name for o in bpy.data.objects}
    mod = load_module(S["module"]); fn = getattr(mod, S["function"])
    plan = json.load(open(S["plan"])) if S.get("plan") and os.path.exists(S["plan"]) else {}
    ctx = dict(sc=sc, section=sec, collection=coll, plan=plan, keys=S.get("keys") or {}, scene=S["scene"], out=S["out"],
               profile=S.get("profile", "preview"), gives=S.get("gives") or {}, root=ROOT, step=S["step"])   # the ctx keys are English: the steps' build functions read them
    os.makedirs(S["out"], exist_ok=True)
    ret = fn(ctx) or {}
    sc = bpy.context.scene                                       # the frames output builds the maps scene in a clean file — the previous sc is dead
    bpy.context.view_layer.update()
    new = [o for o in bpy.data.objects if o.name not in before]
    if coll is not None:
        for o in new:
            if "sk_layer" not in o: o["sk_layer"] = sec           # the function may have marked it itself (the site's ground while building)
            if o.name in sc.collection.objects:                  # directly in the scene — move to the section collection; in its own collection — leave it
                sc.collection.objects.unlink(o); coll.objects.link(o)
            elif not o.users_collection:
                coll.objects.link(o)
        bpy.context.view_layer.update()
    contract = {}                                                # the object contract: for now a warning in build.json, not a failure
    if coll is not None and new:
        from core.contract import contract as C
        under = {coll.name} | {c.name for c in coll.children_recursive}
        contract = {"no_sk_id": sum(1 for o in new if C.SK_ID not in o), "no_sk_kind": sum(1 for o in new if C.SK_KIND not in o),
                    "outside_section_collection": sum(1 for o in new if not any(c.name in under for c in o.users_collection))}
        contract = {k: v for k, v in contract.items() if v}
        if contract: print(f"contract: step '{S['step']}' — {contract} of {len(new)} new objects", flush=True)
    if S.get("output"):
        os.makedirs(os.path.dirname(S["output"]), exist_ok=True)
        bpy.context.preferences.filepaths.save_version = 0       # no .blend1 backup copy next to it (on a sample it doubled the size of the overwritten step)
        bpy.ops.wm.save_as_mainfile(filepath=S["output"], compress=True, copy=False)   # zstd explicitly, not per the settings (measured: 237 → 110 MB on the "textures" step)
    try:                                                         # step cost: the peak memory of the Blender process and the size of the checkpoint .blend
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        mem_gb = round(rss / (1 << 30 if sys.platform == "darwin" else 1 << 20), 2)   # macOS gives bytes, Linux gives KB
    except Exception:                                            # noqa: BLE001
        mem_gb = None
    blend_mb = round(os.path.getsize(S["output"]) / (1 << 20), 1) if S.get("output") and os.path.exists(S["output"]) else None
    info = {"step": S["step"], "section": sec, "new_objects": len(new), "sec": round(time.time() - t0, 1), "memory_gb": mem_gb, "blend_mb": blend_mb,
            "collections": {c.name: len(c.all_objects) for c in bpy.data.collections if c.name in sc.collection.children},
            "contract": contract, "blend": S.get("output"), "gives": ret,
            "layers": dict(getattr(sys.modules.get("core.layers"), "SEEN", None) or {})}   # the layer read accounting goes to the manager
    json.dump(info, open(os.path.join(S["out"], "build.json"), "w"), ensure_ascii=False, indent=1)
    print("__RESULT__" + json.dumps({k: v for k, v in info.items() if k != "gives"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

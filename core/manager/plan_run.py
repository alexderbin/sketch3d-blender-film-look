"""The "plan" phase of a build step ("plan: file.py:function"): project Python (with yaml, PIL, cv2) — the scene model
and the step keys → a JSON plan for the "build" phase in Blender (which has no yaml).
    .venv/bin/python -m core.manager.plan_run <step.json>   → <step folder>/plan.json
Function: `plan(M, keys, ctx) -> dict`; M is the run's scene model (scene_model.load), ctx: scene, folder, profile, gives, root."""
import importlib, importlib.util, json, os, sys
import core.paths

HERE = os.path.dirname(os.path.abspath(__file__))
from core.model import scene_model  # noqa: E402


def load_module(path):
    """The plan module by the path from step.json: inside packages — by dotted name, otherwise from the file."""
    name = core.paths.module_name(path)
    if name: return importlib.import_module(name)
    spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0] + "_plan", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def main(path):
    S = json.load(open(path))
    mod = load_module(S["plan_module"]); fn = getattr(mod, S["plan_function"])
    M = scene_model.load(S["scene"])
    ctx = dict(scene=S["scene"], out=S["out"], profile=S.get("profile", "preview"), gives=S.get("gives") or {},
                    root=core.paths.ROOT, step=S["step"])
    os.makedirs(S["out"], exist_ok=True)
    plan = fn(M, S.get("keys") or {}, ctx) or {}
    LY = sys.modules.get("core.layers")                                  # accounting of scene-layer reads — for the manager ("no layer …")
    if LY is not None and LY.SEEN and isinstance(plan, dict): plan["_layers"] = dict(LY.SEEN)
    json.dump(plan, open(S["plan"], "w"), ensure_ascii=False, indent=1, default=str)
    print("__PLAN__" + json.dumps({k: v for k, v in plan.items() if isinstance(v, (int, float, str, bool))}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main(sys.argv[1])

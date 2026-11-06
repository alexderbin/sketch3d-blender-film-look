"""The component's entry for step-by-step assembly. A template: the example step.
plan (project Python): scene model M + knobs keys + ctx (scene, out, profile, gives, root, step) → plan (dict;
written to plan.json).
build (Blender): ctx (sc, section, collection, plan, keys, scene, out, profile, gives, root, step) → objects in
ctx["collection"] with contract labels (sk_id, sk_kind; core/contract/contract.py) → return: the names from the
passport's gives go to the following steps, other scalars are log statistics (stats).
The manager checks the step boundary before launch: the knobs read are declared in the passport, the gives read
come from the source step, the returned names are English snake_case."""
import json
import os

# if the component's internal knowledge uses other value names — an explicit table of knob values at the step
# boundary (not needed here):
# MODE = {"fast": "quick", "full": "thorough"}


def plan_example(M, keys, ctx):
    seed = int(keys.get("seed") or M.get("seed") or 1)
    plan = {"seed": seed, "density": float(keys.get("density") or 0.5), "mode": keys.get("mode") or "fast",
            "site_files": ((M.get("sections") or {}).get("site") or {}).get("files", [])}
    return plan


def build_example(ctx):
    import bpy
    plan, coll = ctx["plan"], ctx["collection"]
    me = bpy.data.meshes.new("example")
    ob = bpy.data.objects.new("example", me)
    coll.objects.link(ob)
    ob["sk_id"], ob["sk_kind"] = "example_0", "example"      # object contract labels
    out = os.path.join(ctx["out"], "example_plan.json")
    json.dump(plan, open(out, "w"), ensure_ascii=False, indent=1)
    return {"example_plan": out, "objects": 1}              # example_plan is a gives entry (in the passport), objects is a log statistic

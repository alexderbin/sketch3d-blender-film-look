"""A format step from a description: a format command that takes a YAML description with the keys scene and out
(page, shot, take) runs in the step folder and on the run's scene: the manager itself decides where the scene is and where to put the output.
.venv/bin/python -m core.manager.spec_run <tool.py> <description.yaml> <step folder> <run scene> [key=value …] [--flag …]
  key=value — a top-level key of the description (light=night, weather=rain, res=2k): the value is read as YAML (numbers, lists);
  --flag … — the tool command's tail as is (for a take, --fps=24).
→ <folder>/spec.yaml (the description with scene, out and keys replaced), the tool's output — <folder>/out (a take cleans its output
folder entirely — the step journal step.log stays next to it), run from the project root; the exit code is the tool's.
Paths inside the description (lib:, after:, panel scenes) are relative to the project root, as for the tool itself."""
import os, subprocess, sys

import yaml

import core.paths

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))


def main(a):
    tool, src, out, scene = (os.path.abspath(os.path.join(ROOT, x)) for x in a[:4])
    keys = [x for x in a[4:] if "=" in x and not x.startswith("--")]; tail = [x for x in a[4:] if x.startswith("--")]
    spec = yaml.safe_load(open(src, encoding="utf-8")) or {}
    for kv in keys:
        k, v = kv.split("=", 1); spec[k] = yaml.safe_load(v)
    spec["scene"], spec["out"] = os.path.relpath(scene, ROOT), os.path.relpath(os.path.join(out, "out"), ROOT)
    os.makedirs(out, exist_ok=True); job = os.path.join(out, "spec.yaml")
    yaml.safe_dump(spec, open(job, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
    print("step description →", os.path.relpath(job, ROOT), flush=True)
    return subprocess.run([os.path.join(ROOT, ".venv", "bin", "python"), *core.paths.python_target(tool), job, *tail], cwd=ROOT).returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

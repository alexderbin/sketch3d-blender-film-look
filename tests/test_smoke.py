"""Smoke tests without Blender: the tree compiles, imports resolve, the scene model and the pipeline manager pass their
stands, and every recipe (core samples and component benches) passes the dry run: step order, layer contract, keys."""
import compileall
import glob
import os
import subprocess
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(*args):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, (r.stdout[-3000:] + "\n" + r.stderr[-3000:])
    return r.stdout


def recipes():
    out = sorted(glob.glob(os.path.join(ROOT, "core", "manager", "pipelines", "*.yaml")))
    for p in sorted(glob.glob(os.path.join(ROOT, "components", "**", "bench", "**", "*.yaml"), recursive=True)):
        doc = yaml.safe_load(open(p, encoding="utf-8"))
        if isinstance(doc, dict) and "pipeline" in doc and isinstance(doc.get("steps"), list):
            out.append(p)
    return [os.path.relpath(p, ROOT) for p in out]


def test_compiles():
    for part in ("core", "components", "bench"):
        if os.path.isdir(os.path.join(ROOT, part)):
            assert compileall.compile_dir(os.path.join(ROOT, part), quiet=1, force=False), part


def test_static_imports():
    run("-m", "core.services.import_check")


def test_scene_model():
    run("-m", "core.model.scene_model", "check", "bench/scenes/castle_rock")


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "core", "manager", "pipelines", "castle_rock.yaml")),
                    reason="the manager stand runs the castle_rock sample, not part of this repository")
def test_manager_stand():
    run("bench/manager_stand.py", "check")


@pytest.mark.parametrize("recipe", recipes())
def test_recipe_dry_run(recipe):
    run("-m", "core.manager.pipeline", recipe, "--dry")

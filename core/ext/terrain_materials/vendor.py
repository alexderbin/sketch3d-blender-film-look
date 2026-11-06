"""Vendoring of the Infinigen terrain materials as is (princeton-vl/infinigen 224598d, 2026-10-14, BSD-3) — only the needed modules
(the list was measured by importing in Blender 5.2) and gin-config 0.5.0 (Apache-2.0). The files are not edited.
.venv/bin/python core/ext/infinigen_terrain/vendor.py"""
import os, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SRC = os.path.join(ROOT, "projects/infinigen/src/infinigen/src/infinigen")
GIN = os.path.join(ROOT, "projects/analysis/runs/deep-infinigen/pylib/gin")
FILES = ["__init__.py", "core/__init__.py", "core/surface.py", "core/tags.py", "core/nodes/__init__.py", "core/nodes/compatibility.py",
         "core/nodes/node_info.py", "core/nodes/node_utils.py", "core/nodes/node_wrangler.py", "core/nodes/utils.py",
         "core/util/__init__.py", "core/util/blender.py", "core/util/color.py", "core/util/logging.py", "core/util/math.py",
         "core/util/organization.py", "core/util/random.py", "assets/__init__.py", "assets/colors.py", "assets/materials/__init__.py",
         "assets/materials/utils/__init__.py", "assets/materials/utils/surface_utils.py"] + \
        [f"assets/materials/terrain/{n}.py" for n in ("__init__", "chunkyrock", "cobble_stone", "cracked_ground", "dirt", "ice", "mountain",
                                                       "mud", "sand", "sandstone", "soil", "stone")] + \
        [f"assets/materials/fluid/{n}.py" for n in ("lava", "water", "river_water")]
for f in FILES:
    d = os.path.join(HERE, "infinigen", f); os.makedirs(os.path.dirname(d), exist_ok=True); shutil.copy2(os.path.join(SRC, f), d)
shutil.copytree(GIN, os.path.join(HERE, "pylib", "gin"), dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
PAL = "infinigen_examples/configs_nature/palette"                  # color palettes (sand, sandstone, mountain soil, water) — the path is relative to the package parent
shutil.copytree(os.path.join(SRC, "..", PAL), os.path.join(HERE, PAL), dirs_exist_ok=True)
shutil.copy2(os.path.join(SRC, "../../LICENSE"), os.path.join(HERE, "LICENSE"))
print("__RESULT__", len(FILES), "files + gin")

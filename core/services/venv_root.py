"""Make the project root importable for every script run by the project venv: writes <venv site-packages>/sketch3d_root.pth
with the project root (what an editable install does). Scripts outside the packages (bench/, research/, projects/) then
`import core…` without their own sys.path lines. Blender is unaffected (its entry core/blender/run.py puts the root itself).
Run once after (re)creating .venv:   .venv/bin/python core/services/venv_root.py
Stdlib only, no project imports (it runs before the root is importable)."""
import os
import site
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if __name__ == "__main__":
    sp = next((p for p in site.getsitepackages() if p.endswith("site-packages")), None)
    if not sp or not sys.prefix.endswith(".venv"):
        raise SystemExit(f"venv_root: run with the project venv (.venv/bin/python), not {sys.executable}")
    path = os.path.join(sp, "sketch3d_root.pth")
    open(path, "w", encoding="utf-8").write(ROOT + "\n")
    print(f"venv_root: {path} → {ROOT}")

"""Substance surfaces (a material texture) from Poly Haven — for buildings and masonry: library/materials/surfaces.yaml.
.venv/bin/python -m core.surface.surface_tex fetch  → library/textures/surface/<id>/<id>_<map>_<res>.jpg, meta.json; index.json
The Poly Haven license is CC0. The size of the square comes from their description (mm → m): for scans it is the
real one (not a 1 m default). The shader removes the repetition (core/surface/bl_surface.py):
each masonry block gets its own piece of surface (a shift and rotation by the stone number)."""
import core.paths        # noqa: E402  core parts on sys.path
import json, sys, urllib.request
from pathlib import Path

import yaml

ROOT = Path(core.paths.ROOT)
OUT = ROOT / "library/textures/surface"
API = "https://api.polyhaven.com"
UA = {"User-Agent": "sketch3d/1.0 (+local pipeline)"}


def _get(url, raw=False):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
        b = r.read()
    return b if raw else json.loads(b)


def fetch(path=ROOT / "library/materials/surfaces.yaml"):
    cfg = yaml.safe_load(Path(path).read_text())
    res, maps = cfg.get("res", "2k"), cfg.get("maps", ["Diffuse", "nor_gl", "Rough", "Displacement"])
    ip = OUT / "index.json"
    idx = json.loads(ip.read_text()) if ip.exists() else {}
    mb = 0.0
    for sid, spec in cfg["surfaces"].items():
        d = OUT / sid; d.mkdir(parents=True, exist_ok=True)
        info, files = _get(f"{API}/info/{sid}"), None
        got = {}
        for m in maps:
            p = d / f"{sid}_{m}_{res}.jpg"
            if not p.exists():
                files = files or _get(f"{API}/files/{sid}")
                b = _get(files[m][res]["jpg"]["url"], raw=True); mb += len(b) / 1e6
                p.write_bytes(b)
            got[m] = str(p.relative_to(ROOT))
        meta = {"id": sid, "name": info.get("name"), "size_m": [round(x / 1000.0, 3) for x in info.get("dimensions", [1000, 1000])],
                "tags": info.get("tags", []), "authors": list((info.get("authors") or {}).keys()), "license": "CC0",
                "res": res, "maps": got, **spec}
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        idx[sid] = meta
        print(sid, meta["size_m"], flush=True)
    ip.write_text(json.dumps(idx, ensure_ascii=False, indent=1))
    print(f"downloaded {mb:.1f} MB")
    return idx


if __name__ == "__main__":
    if sys.argv[1:2] == ["fetch"]:
        fetch(*(sys.argv[2:3]))
    else:
        sys.exit(__doc__)

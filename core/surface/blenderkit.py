"""BlenderKit — the base library of ready-made materials. A material comes whole: a .blend with nodes and settings (a
resolution variant 0.5K–4K, 2K by default ~2 MB; the source ~20 MB), a description, a thumbnail. The free ones need no
key: the server hands out a signed link. The "royalty free" license: usable in works, distributing the assets as they are
is not allowed (in git — only the descriptions).
  .venv/bin/python -m core.surface.blenderkit search "<words>" [--style npr|realistic] [--procedural 0|1] [--n 24] [--all] [--json output]
  .venv/bin/python -m core.surface.blenderkit get <assetBaseId | name> [--res 2K|1K|4K|0_5K|blend]      → library/materials/blenderkit/<id>/
  .venv/bin/python -m core.surface.blenderkit sheet <output.jpg> <search.json | id …>                     — a thumbnail sheet for selection
  .venv/bin/python -m core.surface.blenderkit set [library/materials/blenderkit.yaml]                    — download everything listed by roles
Search order — by BlenderKit's rating (score: quality, bookmarks, the author's effort); --all — paid ones too (by default
only the free ones).
In Blender the material is applied by core/surface/bl_matlib.py (a "box" UV in meters, scale by the tile size, bl_mat
states)."""
import core.paths        # noqa: E402  core parts on sys.path
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import uuid

ROOT = core.paths.ROOT
LIB = os.path.join(ROOT, "library", "materials", "blenderkit")
API = "https://www.blenderkit.com/api/v1"
UA = {"User-Agent": "sketch3d/1.0 (+local pipeline)"}
PARAMS = ("materialStyle", "procedural", "textureSizeMeters", "textureCount", "textureResolutionMax", "nodeCount", "uv",
          "engine", "shaders", "purePbr", "pbrType")


def _get(url, raw=False, tries=3):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                b = r.read()
            return b if raw else json.loads(b)
        except Exception as e:                                          # network: two retries with a pause
            if k == tries - 1:
                raise SystemExit(f"BlenderKit: {url[:90]} — {e}")
            time.sleep(2 + 3 * k)


def brief(x):
    """What we keep about a material: name, author, license, tags, score, parameters, files and thumbnail."""
    p = x.get("dictParameters") or {}
    q, c = x.get("ratingsAverage") or {}, x.get("ratingsCount") or {}
    th = next((f for f in x.get("files", []) if f.get("fileType") == "thumbnail"), {})
    return {"id": x["assetBaseId"], "version": x["id"], "name": x["name"], "author": (x.get("author") or {}).get("fullName"),
            "license": x.get("license"), "access": x.get("access"), "tags": x.get("tags", []), "score": x.get("score"),
            "quality": q.get("quality"), "quality_votes": c.get("quality"), "bookmarks": c.get("bookmarks"),
            **{k: p.get(k) for k in PARAMS},
            "files": {f["fileType"]: {"url": f["downloadUrl"], "bytes": f.get("fileUploadSize")} for f in x.get("files", [])
                      if f.get("fileType") not in ("thumbnail",)},
            "thumb": th.get("thumbnailMiddleUrlNonsquared") or th.get("thumbnailMiddleUrl")}


def search(words, style=None, procedural=None, n=24, free=True):
    q = ["asset_type:material"] + ([f"materialStyle:{style}"] if style else []) \
        + ([f"procedural:{'true' if procedural else 'false'}"] if procedural is not None else []) + words.split() + ["order:-score"]
    url, out = f"{API}/search/?" + urllib.parse.urlencode({"query": " ".join(q), "page_size": min(max(n, 1) * 2, 100)}), []
    while url and len(out) < n:
        d = _get(url)
        for x in d.get("results", []):
            if not free or x.get("access") == "free" or x.get("isFree"):
                out.append(brief(x))
        url = d.get("next")
    return out[:n]


def by_id(asset):
    """Description by assetBaseId (or by name — the first match by score)."""
    for q in (f"asset_base_id:{asset}", f'"{asset}"'):
        d = _get(f"{API}/search/?" + urllib.parse.urlencode({"query": f"asset_type:material {q}", "page_size": 5}))
        for x in d.get("results", []):
            if x["assetBaseId"] == asset or x["name"].lower() == asset.lower():
                return brief(x)
    raise SystemExit(f"BlenderKit: material '{asset}' not found")


def get(item, res="2K"):
    """Download a material into library/materials/blenderkit/<id>/: <res>.blend, thumb.jpg, meta.json; update index.json."""
    m = item if isinstance(item, dict) else by_id(item)
    d = os.path.join(LIB, m["id"])
    os.makedirs(d, exist_ok=True)
    key = "blend" if res == "blend" else f"resolution_{res}"
    if key not in m["files"]:                                           # no such resolution (procedural ones — blend only)
        key = "blend"
    dst = os.path.join(d, ("source" if key == "blend" else res) + ".blend")
    if not os.path.exists(dst):
        link = _get(m["files"][key]["url"] + "?" + urllib.parse.urlencode({"scene_uuid": str(uuid.uuid4())}))["filePath"]
        b = _get(link, raw=True)
        open(dst + ".part", "wb").write(b); os.replace(dst + ".part", dst)
    if m.get("thumb") and not os.path.exists(os.path.join(d, "thumb.jpg")):
        open(os.path.join(d, "thumb.jpg"), "wb").write(_get(m["thumb"], raw=True))
    meta = {k: v for k, v in m.items() if k != "files"}
    meta["blend"] = os.path.relpath(dst, ROOT)
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), ensure_ascii=False, indent=1)
    ip = os.path.join(LIB, "index.json")
    idx = json.load(open(ip)) if os.path.exists(ip) else {}
    idx[m["id"]] = {k: meta.get(k) for k in ("name", "author", "license", "materialStyle", "procedural", "textureSizeMeters",
                                             "textureCount", "score", "quality", "tags", "blend")}
    json.dump(idx, open(ip, "w"), ensure_ascii=False, indent=1)
    return meta


def sheet(items, out, cols=6, w=256):
    """Thumbnail sheet: name, score, quality, style, tile size, procedural."""
    from io import BytesIO
    from PIL import Image, ImageDraw, ImageFont
    F = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13)
    rows = (len(items) + cols - 1) // cols
    S = Image.new("RGB", (cols * w, rows * (w + 34)), (18, 18, 18)); dr = ImageDraw.Draw(S)
    for i, m in enumerate(items):
        x0, y0 = (i % cols) * w, (i // cols) * (w + 34)
        try:
            im = Image.open(BytesIO(_get(m["thumb"], raw=True))).convert("RGB"); im.thumbnail((w, w)); S.paste(im, (x0, y0))
        except SystemExit:
            pass
        dr.text((x0 + 4, y0 + w + 2), m["name"][:34], fill=(255, 230, 0), font=F)
        dr.text((x0 + 4, y0 + w + 17), f"{m.get('score') or 0:.0f} · {m.get('quality') or 0:.1f} · {m.get('materialStyle') or '?'}"
                f" · {m.get('textureSizeMeters') or '?'} m{' · proc.' if m.get('procedural') else ''}", fill=(200, 200, 200), font=F)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    S.save(out, quality=88)
    return out


def from_set(path):
    """library/materials/blenderkit.yaml: roles {role: [{id, res?, tiling?}]}, elements {role: {axis, m} | {tile}} → download
    everything; the set for Blender (there is no yaml there) — library/materials/blenderkit/set.json {roles: {role: [{id,
    name, tile_m, style, tiling}]}, elements: {…}}."""
    import yaml
    spec = yaml.safe_load(open(path)) or {}
    got, out = {}, {}
    for role, lst in (spec.get("roles") or {}).items():
        for e in lst or []:
            e = e if isinstance(e, dict) else {"id": e}
            m = get(e["id"], e.get("res", spec.get("resolution", "2K")))
            got.setdefault(role, []).append(m["name"])
            out.setdefault(role, []).append({"id": m["id"], "name": m["name"], "tile_m": m.get("textureSizeMeters"),
                                             "style": m.get("materialStyle"), "blend": m["blend"],
                                             **({"tiling": e["tiling"]} if e.get("tiling") else {})})
    json.dump({"roles": out, "elements": spec.get("elements") or {}}, open(os.path.join(LIB, "set.json"), "w"),
              ensure_ascii=False, indent=1)
    return got


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("args", nargs="*")
    ap.add_argument("--style"); ap.add_argument("--procedural", type=int); ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--all", action="store_true"); ap.add_argument("--json"); ap.add_argument("--res", default="2K")
    a = ap.parse_args()
    if a.cmd == "search":
        r = search(" ".join(a.args), a.style, None if a.procedural is None else bool(a.procedural), a.n, not a.all)
        for m in r:
            print(f"{m['id']}  {m['score'] or 0:6.1f}  q{m['quality'] or 0:4.1f}  {m['materialStyle'] or '?':9s}  "
                  f"{m['textureSizeMeters'] or '?':>4} m  {'proc' if m['procedural'] else 'maps'}  {m['name']}")
        if a.json:
            os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
            json.dump(r, open(a.json, "w"), ensure_ascii=False, indent=1)
    elif a.cmd == "get":
        for x in a.args:
            m = get(x, a.res); print(m["name"], "→", m["blend"])
    elif a.cmd == "sheet":
        out, src = a.args[0], a.args[1:]
        items = json.load(open(src[0])) if len(src) == 1 and src[0].endswith(".json") else [by_id(x) for x in src]
        print(sheet(items, out))
    elif a.cmd == "set":
        print(json.dumps(from_set(a.args[0] if a.args else os.path.join(ROOT, "library", "materials", "blenderkit.yaml")), ensure_ascii=False))
    else:
        sys.exit(__doc__)

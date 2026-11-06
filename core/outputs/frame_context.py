"""Frame context by plain code (no model call) on our frame passport.

Input: a frame folder out/<scene>/cam<K> (frame.json + maps), a world library (JSON: id → who it is, sheet, what to preserve).
Output: context.json (references with non-overlapping roles, a layout of boxes in the FLUX 3 format, checks)
and prompt.txt (a prompt draft "reference → role → what to preserve → what not to take").

  .venv/bin/python -m core.outputs.frame_context <frame folder out/X/cam<K>> <library.json> <output folder> [--level K1|K3] [--max-refs 4]

Rules:
  - the model sees only what is attached: the prompt has no ids, file names, or the words "selected", "approved";
  - each reference has one role, roles do not overlap; the style is the only source of the look;
  - gaps (no hero sheet, the file is missing on disk) are warnings, not prohibitions;
  - FLUX 3 boxes: [top, left, bottom, right] on a 0–1000 grid; new elements in very small boxes often vanish.
The reply is a single __RESULT__ line + JSON.
"""
import core.paths        # noqa: E402  core parts on sys.path
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(core.paths.ROOT)
FORBIDDEN = ["selected", "approved", "current", "studio", "card", "tab", "database", "asset id", "file name",
             "project path", ".png", ".jpg", "cam1", "revision", "frame.json"]
SMALL_BOX = 40            # out of 1000: a box shorter than this on its smaller side risks the element not appearing
CONTROL = {
    "K1": ("marker.png", "the composition sketch",
           "where each figure and mass stands, its scale, the camera angle and the horizon line",
           "its marker strokes, flat grey shapes and simplified mannequin bodies"),
    "K3": ("line.png", "the line drawing",
           "every contour, the perspective of the wall and the exact silhouettes of the figures",
           "its thin uniform line weight and empty white fills"),
}


def resolve(p):
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def where(b, W, H):
    """Words about the position from a box: the third horizontally, the tier vertically."""
    cx, cy = (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H
    hx = "on the left" if cx < 1 / 3 else ("on the right" if cx > 2 / 3 else "in the centre")
    vy = "high in the frame" if cy < 0.33 else ("low in the frame" if cy > 0.67 else "")
    return " ".join(x for x in (hx, vy) if x)


def box1000(b, W, H):
    return [round(b[1] / H * 1000), round(b[0] / W * 1000), round((b[3] + 1) / H * 1000), round((b[2] + 1) / W * 1000)]


def shot_facts(fd, fr):
    """Frame facts for the prompt: the lens, the shot size from the height fraction of the largest hero, the camera
    level from the tilt of its axis. Frame facts outrank style words."""
    import math
    W, H = fr["camera"]["size"]; lens = fr["camera"].get("lens_mm")
    figs = [o for o in fr["objects"] if o["class"] in ("figure", "member")]
    hero = max(figs, key=lambda o: o.get("height_px", 0), default=None)
    share = hero.get("height_px", 0) / H if hero else 0
    whole = hero is not None and hero["bbox"][1] > 0 and hero["bbox"][3] < H - 1        # the hero is fully in the frame — not "waist-up"
    size = ("full shot" if whole and share > 0.25 else "close-up" if share > 0.8 else "medium shot" if share > 0.5 else
            "full shot" if share > 0.25 else "wide shot" if share > 0.08 else "extreme wide shot")
    level = "eye level"
    cj = fd / "raw" / "camera.json"
    if cj.exists():
        cam = json.loads(cj.read_text())
        it = (cam.get("intent") or {}).get("intent")                  # an intent-driven camera (components/camera/cam_intent.py) outranks the estimate
        size = {"wide": "wide shot", "medium": "medium shot", "closeup": "close-up", "heroic": "full shot", "overhead": "wide shot"}.get(it, size)
        M = cam["matrix_world"]; fz = -M[2][2]
        pitch = math.degrees(math.asin(max(-1, min(1, fz))))
        level = "high angle looking down" if pitch < -15 else ("low angle looking up" if pitch > 8 else "eye level")
    view = ((json.loads(cj.read_text()).get("intent") or {}).get("view_en") if cj.exists() else None)   # the hero's angle (cam_intent)
    return {"lens_mm": lens, "shot": size, "level": level, "hero_height_share": round(share, 2), "view": view}


def main(frame_dir, lib_path, out_dir, level="K1", max_refs=4):
    t0 = time.perf_counter()
    fd, out = Path(frame_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fr = json.loads((fd / "frame.json").read_text())
    lib = json.loads(Path(lib_path).read_text())
    W, H = fr["camera"]["size"]
    warnings, refs = [], []

    def add_ref(role, path, use_for, do_not_copy, subject=None):
        p = resolve(path)
        ok = p.exists()
        if not ok:
            warnings.append(f"reference unavailable: {path}")
            return None
        refs.append({"n": len(refs) + 1, "role": role, "file": str(p), "use_for": use_for,
                     "do_not_copy": do_not_copy, **({"subject": subject} if subject else {})})
        return refs[-1]

    fname, cname, cuse, cnot = CONTROL[level]
    layout_ref = add_ref("composition", fd / fname, cuse, cnot)
    style = lib.get("style")
    style_ref = add_ref("appearance", style["file"], style["use_for"], style["do_not_copy"]) if style else None
    if not style:
        warnings.append("no style reference: the look is set by the text only")

    subs = lib.get("subjects", {})
    objs = sorted(fr["objects"], key=lambda o: -o["area_px"])
    rows, people, seen = [], [], set()
    for o in objs:
        s = subs.get(o["id"])
        if o["id"] in seen or o["class"] in ("sky",):
            continue
        seen.add(o["id"])
        if not s:
            if o["class"] in ("figure", "siege"):
                warnings.append(f"no description in the library: {o['id']} ({o['class']})")
            continue
        b = box1000(o["bbox"], W, H)
        share = o["area_px"] / (W * H)
        if s["kind"] in ("character", "figure", "prop") or share > 0.02:
            rows.append({"id": f"{s['kind'][:4]}_{len(rows) + 1}", "bbox": b, "desc": s["desc"],
                         "_src": o["id"], "_area_pct": round(share * 100, 2)})
        if s["kind"] == "character":
            people.append((o, s))
            if s.get("sheet") and len(refs) < max_refs:
                add_ref("identity", s["sheet"], f"identity, {s['keep']}",
                        "its lighting, finish, background, pose and sheet layout", subject=o["id"])
            elif not s.get("sheet"):
                warnings.append(f"the hero has no sheet — the look is given in words only: {o['id']}")
    for cid, c in sorted(fr.get("crowds", {}).items(), key=lambda kv: -kv[1]["area_px"]):
        d = lib.get("crowds", {}).get(cid)
        if not d:
            warnings.append(f"no crowd description: {cid}")
            continue
        rows.append({"id": f"crowd_{len(rows) + 1}", "bbox": box1000(c["bbox"], W, H), "desc": d["desc"],
                     "_src": cid, "_count": c["visible"]})

    # Prompt draft: artifact → reference roles → who is where → look → exclusions (the order comes from scene-storyboard-sheet.md).
    from fractions import Fraction
    fr_ = Fraction(W, H).limit_denominator(20); ratio = f"{fr_.numerator}:{fr_.denominator}" if abs(fr_ - Fraction(W, H)) < 0.02 else f"{W}x{H}"
    facts = shot_facts(fd, fr)
    lines = [f"Create one finished comic panel in a {ratio} {'square' if W == H else 'landscape' if W > H else 'portrait'} frame: {facts['shot']}, "
             f"{facts['level']}, {round(facts['lens_mm'] or 35)} mm lens" + (f", the main figure {facts['view']}" if facts.get("view") else "") + ".", "", "References:"]
    for r in refs:
        lines.append(f"- Reference {r['n']} is {cname if r['role'] == 'composition' else ('the style reference' if r['role'] == 'appearance' else 'a character sheet')}."
                     f" Use it only for {r['use_for']}. Do not copy {r['do_not_copy']}.")
    lines += ["", "In the panel:"]
    for o, s in people:
        sheet = next((r for r in refs if r.get("subject") == o["id"]), None)
        tail = f", matching Reference {sheet['n']}" if sheet else ""
        lines.append(f"- {s['desc'][0].upper() + s['desc'][1:]}, {where(o['bbox'], W, H)}{tail}.")
    told = {}
    for r in rows:
        if r["_src"] in {o["id"] for o, _ in people}:
            continue
        b = r["bbox"]
        key = (r["desc"], where([b[1] * W / 1000, b[0] * H / 1000, b[3] * W / 1000, b[2] * H / 1000], W, H))
        told[key] = told.get(key, 0) + 1
    words = {2: "Two", 3: "Three", 4: "Four", 5: "Five"}
    for (desc, place), n in told.items():
        text = desc[0].upper() + desc[1:]
        if n > 1:        # "a space marine in …" ×3 → "Three space marines in …": the noun part before the preposition is made plural
            body = re.sub(r"^(a|an) ", "", desc)
            m = re.search(r" (in|with|on|at|of|from|behind|near|aiming|firing|holding|carrying|wearing|running|standing)\b", body)
            head, tail = (body[:m.start()], body[m.start():]) if m else (body, "")
            head = re.sub(r"(s|x|ch|sh)$", r"\1es", head) if re.search(r"(s|x|ch|sh)$", head) else (head[:-1] + "ies" if re.search(r"[^aeiou]y$", head) else head + "s")
            text = f"{words.get(n, str(n))} {head}{tail}"
        lines.append(f"- {text}, {place}.")
    if style_ref:
        lines += ["", f"Render everything only in the visual language of Reference {style_ref['n']}; "
                      "character sheets give identity, not lighting or finish."]
    lines += ["", "Keep the placement, scale and horizon exactly as in Reference 1. Use Reference 1 only as a camera and staging "
                  "guide: draw the described characters and places, not its grey mannequins, sketch lines or flat lighting. "
                  "No text, captions, panel borders or speech balloons."]
    prompt = "\n".join(lines)

    # Visibility checks (prompt-input-visibility.md): every role is named in the text, there are no extras, there are no internal words.
    mentioned = {int(n) for n in re.findall(r"Reference (\d+)", prompt)}
    attached = {r["n"] for r in refs}
    ids = {o["id"] for o in fr["objects"]} | set(fr.get("crowds", {}))
    low = prompt.lower()
    lint = {
        "refs_without_mention": sorted(attached - mentioned),
        "mentions_without_ref": sorted(mentioned - attached),
        "forbidden_words": [w for w in FORBIDDEN if re.search(r"(?<![a-z\u0430-\u044f\u0451])" + re.escape(w) + r"(?![a-z\u0430-\u044f\u0451])", low)],   # whole word only
        # Only "code-like" ids (XB, we_b1, def_line) count as internal; ordinary words (wall, tower) are not a leak.
        "internal_ids_in_prompt": sorted(i for i in ids if re.search(r"[A-Z]{2}|_|\d", i)
                                         and re.search(rf"\b{re.escape(i)}\b", prompt)),
        "small_boxes": [r["_src"] for r in rows if min(r["bbox"][2] - r["bbox"][0], r["bbox"][3] - r["bbox"][1]) < SMALL_BOX],
    }
    flux = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    ctx = {"purpose": "shot.image", "level": level, "facts": facts, "frame": {"camera": fr["camera"]["name"], "size": [W, H],
           "class_share_pct": fr.get("class_share_pct")}, "references": refs, "layout_rows": rows,
           "flux3_layout": flux, "prompt": prompt, "lint": lint, "warnings": warnings}
    (out / "context.json").write_text(json.dumps(ctx, ensure_ascii=False, indent=1))
    (out / "prompt.txt").write_text(prompt + "\n")
    (out / "prompt_flux3.txt").write_text(prompt + " " + json.dumps(flux, ensure_ascii=False) + "\n")
    pb = len(prompt.encode())
    return {"frame": str(fd), "refs": len(refs), "rows": len(rows), "people": len(people), "warnings": len(warnings),
            "lint_problems": sum(len(v) for v in lint.values()), "small_boxes": len(lint["small_boxes"]),
            "prompt_bytes": pb, "prompt_tokens_est": round(pb / 4),
            "flux_bytes": len((out / "prompt_flux3.txt").read_bytes()),
            "context_bytes": len((out / "context.json").read_bytes()), "ms": round((time.perf_counter() - t0) * 1000, 1)}


if __name__ == "__main__":
    a = sys.argv
    lvl = a[a.index("--level") + 1] if "--level" in a else "K1"
    mr = int(a[a.index("--max-refs") + 1]) if "--max-refs" in a else 4
    print("__RESULT__", json.dumps(main(a[1], a[2], a[3], lvl, mr), ensure_ascii=False))

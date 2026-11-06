"""Editing the scene plan: rotate or shift a hero — the mesh in scene.obj and the joints,
the position and heading in figkit.json. Needed when a squad stands with its back to the open side and the intent-driven camera sees only backs.
.venv/bin/python -m core.model.scene_edit scenes/<name> turn <id|all> <degrees | face=x,y | face=cam:K>
.venv/bin/python -m core.model.scene_edit scenes/<name> move <id> <dx,dy[,dz]>
  turn — about the vertical through the midpoint of the feet (all — each about its own); face — toward a point or toward scene camera K.
Links to another scene are replaced with its own copies; before the first edit scene.obj.orig and figkit.json.orig are saved; the edits
are written to edits.json — core/outputs/bl_color.py repeats them in the library's .blend."""
import json, math, os, shutil, sys
import numpy as np

SD, OP, WHO, ARG = os.path.abspath(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
obj_p, kit_p = os.path.join(SD, "scene.obj"), os.path.join(SD, "figkit.json")
for p in (obj_p, kit_p):                                            # its own copy and a backup before the first edit
    if os.path.islink(p): real = os.path.realpath(p); os.remove(p); shutil.copy(real, p)
    if not os.path.exists(p + ".orig"): shutil.copy(p, p + ".orig")
kit = json.load(open(kit_p)); named = kit["named"]
ids = list(named) if WHO == "all" else [WHO]
lines = open(obj_p).read().split("\n")
vline = [i for i, l in enumerate(lines) if l.startswith("v ")]          # vertex index → line index
used, cur = {}, None
for l in lines:
    if l.startswith("o "): cur = l[2:].strip()
    elif l.startswith("f ") and cur and cur.startswith("figure_"):
        used.setdefault(cur[7:], set()).update(int(t.split("/")[0]) for t in l.split()[1:])
cams = json.load(open(os.path.join(SD, "cameras.json"))) if os.path.exists(os.path.join(SD, "cameras.json")) else []
done = []
for fid in ids:
    f = named[fid]; P = np.array(f["pos"], float)
    if "r_ankle" in f["joints"] and "l_ankle" in f["joints"]:          # the rotation axis is the midpoint of the feet: the bridge's pos is the center of the box
        P[:2] = (np.array(f["joints"]["r_ankle"][:2]) + np.array(f["joints"]["l_ankle"][:2])) / 2   # (with weapon and backpack), up to 0.3 m off
    if OP == "turn":
        if ARG.startswith("face="):
            t = ARG[5:]; tgt = np.array(cams[int(t[4:]) - 1]["matrix_world"])[:2, 3] if t.startswith("cam:") else np.array([float(x) for x in t.split(",")[:2]])
            want = math.degrees(math.atan2(tgt[1] - P[1], tgt[0] - P[0])); deg = (want - (f.get("heading") or 0) + 180) % 360 - 180
        else:
            deg = float(ARG)
        c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
        fx = lambda q: [P[0] + (q[0] - P[0]) * c - (q[1] - P[1]) * s, P[1] + (q[0] - P[0]) * s + (q[1] - P[1]) * c, q[2]]
        f["heading"] = round(((f.get("heading") or 0) + deg + 180) % 360 - 180, 1); f["pos"] = [round(v, 3) for v in fx(f["pos"])]
    else:
        d = [float(x) for x in ARG.split(",")] + [0.0] * (3 - len(ARG.split(","))); fx = lambda q: [q[0] + d[0], q[1] + d[1], q[2] + d[2]]
        f["pos"] = [round(v, 3) for v in fx(f["pos"])]; deg = None
    for k, q in f["joints"].items(): f["joints"][k] = [round(v, 4) for v in fx(q)]
    for vi in used.get(fid, ()):                                         # vertices of the figure's mesh (via its faces)
        li = vline[vi - 1]; q = [float(x) for x in lines[li].split()[1:4]]; x, y, z = fx(q); lines[li] = f"v {x:.5f} {y:.5f} {z:.5f}"
    done.append({"id": fid, "verts": len(used.get(fid, ())), **({"turn_deg": round(deg, 1), "heading": f["heading"]} if deg is not None else {"pos": f["pos"]})})
open(obj_p, "w").write("\n".join(lines)); json.dump(kit, open(kit_p, "w"), ensure_ascii=False)
ep = os.path.join(SD, "edits.json"); edits = json.load(open(ep)) if os.path.exists(ep) else []   # for color from the .blend (core/outputs/bl_color.py)
for d_ in done:
    f = named[d_["id"]]
    if OP == "turn":
        a_ = np.array(f["joints"]["r_ankle"][:2]) + np.array(f["joints"]["l_ankle"][:2]) if "r_ankle" in f["joints"] else None
        edits.append({"op": "turn", "id": d_["id"], "deg": d_["turn_deg"], "pivot": (a_ / 2).round(4).tolist() if a_ is not None else f["pos"][:2]})
    else:
        edits.append({"op": "move", "id": d_["id"], "d": [float(x) for x in ARG.split(",")] + [0.0] * (3 - len(ARG.split(",")))})
json.dump(edits, open(ep, "w"), ensure_ascii=False, indent=1)
print("__RESULT__" + json.dumps({"op": OP, "figures": done}, ensure_ascii=False))

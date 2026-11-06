"""Joints of a figure with a skeleton — 18 COCO points in world meters and the heading: shared by the bridge
(core/blender/bl_export_scene.py) and the take (components/time/bl_take.py — joints per frame). The bone tables for a humanoid
rig and a biped rig; the face — by the face bones."""
import math
from mathutils import Vector

# --- bone tables: COCO joint → bone (the bone head); the skeleton kind — by names
BONES = {
    "humanoid": {"r_shoulder": "RightArm", "r_elbow": "RightForeArm", "r_wrist": "RightHand", "l_shoulder": "LeftArm", "l_elbow": "LeftForeArm",
               "l_wrist": "LeftHand", "r_hip": "RightUpLeg", "r_knee": "RightLeg", "r_ankle": "RightFoot", "l_hip": "LeftUpLeg", "l_knee": "LeftLeg",
               "l_ankle": "LeftFoot", "head": "Head"},
    "biped": {"r_shoulder": "bip01 r upperarm", "r_elbow": "bip01 r forearm", "r_wrist": "bip01 r hand", "l_shoulder": "bip01 l upperarm",
              "l_elbow": "bip01 l forearm", "l_wrist": "bip01 l hand", "r_hip": "bip01 r thigh", "r_knee": "bip01 r calf", "r_ankle": "bip01 r foot",
              "l_hip": "bip01 l thigh", "l_knee": "bip01 l calf", "l_ankle": "bip01 l foot", "head": "bip01 head"},
}
FACE = {"biped": ("bip01 jaw", "bip01 cheekbones", "bip01 forehead")}      # the face bones sit in front of the head center
COCO = ["nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist", "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee",
        "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear"]


def rig_kind(arm):
    names = {b.name.split(":")[-1] for b in arm.data.bones}
    for kind, tab in BONES.items():
        if all(v in names for v in tab.values()): return kind
    return None


def bone_head(arm, name):
    for pb in arm.pose.bones:
        if pb.name.split(":")[-1] == name: return arm.matrix_world @ pb.head
    raise KeyError(name)


def joints_of(arm, kind, height):
    """18 COCO points in world meters and the figure's heading. Joints are the bone heads; the neck is the middle of the shoulders; the
    face is placed from the head along the gaze direction, in fractions of the height. "Forward" is taken from the face bones (a biped rig:
    jaw, cheekbones, forehead in front of the head center), without them — from the hip line. The feet cannot be used: in biped-rig
    models the toe bone often sits behind the ankle (checked: in 4 of 5 figures the direction by the feet is turned by 150–180°). If
    the "left" bones turn out to be to the right of the face, mirror: left and right swap places, otherwise the pose skeleton for
    ControlNet will mix up the sides."""
    t = BONES[kind]; J = {k: bone_head(arm, v) for k, v in t.items() if k != "head"}
    up = Vector((0, 0, 1)); hd = bone_head(arm, t["head"])
    fwd = Vector((0, 0, 0))
    names = {pb.name.split(":")[-1] for pb in arm.pose.bones}
    face = [b for b in FACE.get(kind, ()) if b in names]             # orc models have no cheekbones — we take what exists
    if face:
        fwd = sum((bone_head(arm, b) for b in face), Vector((0, 0, 0))) / len(face) - hd
    fwd.z = 0
    if fwd.length < 1e-4:
        fwd = (J["l_hip"] - J["r_hip"]).cross(up); fwd.z = 0
        # without the face bones the hip-based mirroring cannot be seen (the hips would be checked against themselves)
        # — an independent cue: the knees bend forward
        kf = sum((J[f"{s_}_knee"] - (J[f"{s_}_hip"] + J[f"{s_}_ankle"]) / 2 for s_ in ("r", "l")), Vector((0, 0, 0))); kf.z = 0
        if kf.length > 0.03 * height / 1.8 and kf.dot(fwd) < 0: fwd = -fwd
    fwd.normalize()
    left = up.cross(fwd)
    mirrored = (J["l_hip"] - J["r_hip"]).dot(left) < 0
    if mirrored:
        for k in [k for k in J if k.startswith("l_")]:
            J[k], J["r_" + k[2:]] = J["r_" + k[2:]], J[k]
    J["neck"] = (J["r_shoulder"] + J["l_shoulder"]) / 2
    s = height / 1.8
    hc = bone_head(arm, t["head"]) + up * 0.10 * s                 # the head center above the base of the skull
    J["nose"] = hc + fwd * 0.10 * s - up * 0.01 * s
    J["r_eye"], J["l_eye"] = hc + fwd * 0.08 * s - left * 0.032 * s + up * 0.02 * s, hc + fwd * 0.08 * s + left * 0.032 * s + up * 0.02 * s
    J["r_ear"], J["l_ear"] = hc - left * 0.075 * s, hc + left * 0.075 * s
    return {k: [round(c, 4) for c in J[k]] for k in COCO}, round(math.degrees(math.atan2(fwd.y, fwd.x)), 1), mirrored

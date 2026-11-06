"""A pose map in the OpenPose format (18 COCO points) for ControlNet: sticks and dots on black — as controlnet_aux draws them.
The joints come from figkit.json (world coordinates), are projected by the frame's camera and checked against depth (occluded ones are not drawn)."""
import math

import cv2
import numpy as np

COCO = ["nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist", "r_hip", "r_knee",
        "r_ankle", "l_hip", "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear"]
# joint pairs (1-indexed, as in OpenPose) and stick colors
LIMBS = [(2, 3), (2, 6), (3, 4), (4, 5), (6, 7), (7, 8), (2, 9), (9, 10), (10, 11), (2, 12), (12, 13), (13, 14), (2, 1),
         (1, 15), (15, 17), (1, 16), (16, 18)]
COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
          (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
          (255, 0, 255), (255, 0, 170), (255, 0, 85)]
SWAP = {"r": "l", "l": "r"}


def mirrored(j):
    """Joint mirroring: y → −y and swapping left with right (a template with mirror)."""
    out = {}
    for n, p in j.items():
        m = n
        if n[:2] in ("r_", "l_"):
            m = SWAP[n[0]] + n[1:]
        out[m] = [p[0], -p[1], p[2]]
    return out


def to_world(j_local, pos, heading_deg, k):
    a = math.radians(heading_deg)
    R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
    T = np.asarray(pos, float)
    return {n: T + R @ (np.asarray(p, float) * k) for n, p in j_local.items()}


def draw_figure(canvas, uv, ok, h_px):
    """One figure: uv — 18×2 pixels, ok — 18 "visible" flags. The stick and dot thickness follows the height in the frame but is no thicker than
    0.6 % of the frame width: in a close-up the height in the frame is thousands of pixels and the sticks turned into blobs (controlnet_aux
    sets the thickness from the image size, not from the person)."""
    cap = canvas.shape[1] * 0.006
    sw = min(max(1.5, h_px * 0.022), cap)
    rad = max(2, int(round(min(h_px * 0.02, cap * 1.1))))
    for i, (a, b) in enumerate(LIMBS):
        a, b = a - 1, b - 1
        if not (ok[a] and ok[b]):
            continue
        (x0, y0), (x1, y1) = uv[a], uv[b]
        L = math.hypot(x1 - x0, y1 - y0)
        ang = math.degrees(math.atan2(y0 - y1, x0 - x1))
        poly = cv2.ellipse2Poly((int((x0 + x1) / 2), int((y0 + y1) / 2)), (max(1, int(L / 2)), max(1, int(sw))), int(ang), 0, 360, 1)
        cv2.fillConvexPoly(canvas, poly, tuple(int(c * 0.6) for c in COLORS[i]))      # sticks are dimmer than dots, as in controlnet_aux
    for i in range(18):
        if ok[i]:
            cv2.circle(canvas, (int(uv[i][0]), int(uv[i][1])), rad, COLORS[i], -1)


def render(cam, figures, zbuf, idm, min_px=40.0, tol=0.35):
    """figures: [{joints: {name: world xyz}, height_m, num}]; num — the figure's object number in the id map: a joint is not occluded by its own body,
    it is occluded by other objects (a wall, a neighboring figure). Returns the pose map (H, W, 3) and the number of figures drawn."""
    H, W = zbuf.shape
    canvas = np.zeros((H, W, 3), np.uint8)
    layer = np.zeros_like(canvas)
    n = 0
    for fg in sorted(figures, key=lambda f: -float(np.linalg.norm(f["joints"]["neck"] - cam.pos))):     # far ones first
        J = np.array([fg["joints"][c] for c in COCO], float)
        u, v, z = cam.project(J)
        h_px = cam.f * fg["height_m"] / max(float(np.nanmedian(z)), 1e-3)
        if not np.isfinite(u).all() or h_px < min_px:
            continue
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        inside = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        ok = inside.copy()
        idx = np.where(inside)[0]
        zb = zbuf[vi[idx], ui[idx]]
        own = idm[vi[idx], ui[idx]] == fg.get("num", -1)
        ok[idx] = own | (zb + tol >= z[idx])        # occluded if another surface is closer than the joint
        # the head occludes itself: from behind the nose and eyes are not visible, in profile — the far eye and the far ear (otherwise
        # ControlNet draws a face on a hero standing with his back to the camera)
        hc = (J[16] + J[17]) / 2; to_cam = cam.pos - hc; to_cam /= max(np.linalg.norm(to_cam), 1e-9)
        for i, lim in ((0, 0.0), (14, 0.0), (15, 0.0), (16, -0.35), (17, -0.35)):     # nose, eyes, ears
            dv = J[i] - hc; n_ = np.linalg.norm(dv)
            if n_ > 1e-6 and (dv / n_) @ to_cam < lim: ok[i] = False
        if ok.sum() < 3:
            continue
        draw_figure(layer, np.stack([u, v], 1), ok, h_px)
        m = layer.any(-1)
        canvas[m] = layer[m]
        layer[m] = 0
        n += 1
    return canvas, n

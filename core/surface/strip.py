"""Unrolled strip of a wall side — the shared geometry of the masonry surfaces layer (core.layers: surfaces.json): the
writer (Buildings) builds strips for its walls and towers and writes their passports (Strip.spec), the reader
(Constructions, masonry) assembles the same strip with strip_from(spec) and lays stones along it. Pure numpy: works both
in the project's Python and in Blender."""
import math

import numpy as np

BEND = math.radians(20.0)                          # a path bend steeper than this — a forced joint
CORNER = math.radians(45.0)                        # a bend steeper than this is a corner: quoins in a bond, wrapping around the corner
FOLD = math.radians(3.0)                           # a bend steeper than this — the stone face is cut into flat parts across it


class Strip:
    """Unrolled strip of one wall side along a path. pts — stations of the centerline (x, y); base, top — bottom and top
    heights per station; t — thickness; side = +1 — outer (along the normal N, with a batter), −1 — inner. A stair-step
    ledge is two stations at the same point: along u they are separated by 1 mm, and the top profile there is a vertical
    step."""

    def __init__(self, pts, base, top, t, side=1, closed=False, batter=0.0, name="side", walk=None, bat=None, knee=None):
        jl = lambda x: None if x is None else (float(x) if np.ndim(x) == 0 else np.asarray(x, float).tolist())
        self._spec = dict(pts=np.asarray(pts, float).tolist(), base=jl(base), top=jl(top), t=jl(t), side=int(side), closed=bool(closed),
                          batter=float(batter), name=name, walk=jl(walk), bat=jl(bat), knee=jl(knee))
        P = np.asarray(pts, float); n = len(P)
        if closed: P = np.vstack([P, P[:1]]); base = list(base) + [base[0]]; top = list(top) + [top[0]]
        self.P, self.closed, self.side, self.name = P, closed, side, name
        self.base = np.asarray(base, float); self.top = np.asarray(top, float)
        self.t = np.full(len(P), float(t)) if np.isscalar(t) else np.concatenate([np.asarray(t, float), np.asarray(t, float)[:1]] if closed else [np.asarray(t, float)])
        self.batter = batter
        wk = np.asarray(walk if walk is not None else top, float)
        if closed and len(wk) == n: wk = np.concatenate([wk, wk[:1]])
        h = np.maximum(wk - self.base, 0.0)
        self.bat = batter * np.minimum(h, 12.0) * 0.5                            # batter projection at the bottom (bl_castle)
        self.knee = self.base + np.minimum(h * 0.4, 1.0 + h * 0.25)
        def per_station(x):                                                      # per station; for a closed one — same as the first
            x = np.atleast_1d(np.asarray(x, float))
            if closed and len(x) == n: x = np.concatenate([x, x[:1]])
            return np.broadcast_to(x, self.base.shape).copy()
        if bat is not None: self.bat = per_station(bat)
        if knee is not None: self.knee = per_station(knee)
        T = np.diff(P, axis=0); Tn = np.linalg.norm(T, axis=1, keepdims=True)
        T = np.where(Tn > 1e-6, T / np.maximum(Tn, 1e-9), np.nan)
        for i in range(len(T)):                                                  # a zero-length segment (a ledge) takes its neighbor's direction
            if np.isnan(T[i, 0]):
                j = i - 1 if i > 0 else i + 1
                while 0 <= j < len(T) and np.isnan(T[j, 0]): j += 1 if j > i else -1
                T[i] = T[j] if 0 <= j < len(T) else (1.0, 0.0)
        self.T = T                                                               # direction of each segment
        nseg = np.column_stack([T[:, 1], -T[:, 0]])                              # segment normal (outward in bl_castle)
        n_next = np.vstack([nseg, nseg[:1] if closed else nseg[-1:]])           # at a station: the segment forward and backward
        n_prev = np.vstack([nseg[-1:] if closed else nseg[:1], nseg])
        Nst = n_prev + n_next; Nst /= np.maximum(np.linalg.norm(Nst, axis=1, keepdims=True), 1e-9)
        Nst = Nst / np.clip((Nst * n_next).sum(1), 0.35, 1.0)[:, None]          # mitered joint, as in bl_castle.wall
        self.N, self.nseg = Nst, nseg
        A = P + Nst * (side * self.t / 2)[:, None]                               # face at the stations (without the batter)
        Lf = np.maximum(np.einsum("ij,ij->i", np.diff(A, axis=0), T), 0.001)      # length of the segment's face: u runs along the face, not the axis
        U = np.concatenate([[0.0], np.cumsum(Lf)]); self.U = U                   # (otherwise the face is stretched at a corner and the brick is longer)
        ang = np.zeros(len(P))                                                   # path turn at the station
        for i in range(1, len(P) - 1):
            ang[i] = math.acos(float(np.clip(np.dot(T[i - 1], T[i]), -1, 1)))
        if closed: ang[0] = ang[-1] = math.acos(float(np.clip(np.dot(T[-1], T[0]), -1, 1)))
        self.turn = U[ang > 1e-4]                                                # stations with a turn: the depth there follows the mitered joint
        self.bends = [float(U[i]) for i in range(len(P)) if BEND < ang[i] <= CORNER]
        self.folds = [float(U[i]) for i in range(len(P)) if ang[i] > FOLD]          # where the stone face folds along the path
        self.corners = [float(U[i]) for i in range(len(P) - (1 if closed else 0)) if ang[i] > CORNER]   # corners — quoins

    @property
    def L(self): return float(self.U[-1])

    def spec(self):
        """A side's passport (JSON): centerline stations, bottom and top per station, thickness, side, ring, batter —
        strip_from(spec) rebuilds the same strip (the same u) in the project's Python and in Blender."""
        return dict(self._spec)

    def base_at(self, u): return np.interp(u, self.U, self.base)
    def top_at(self, u): return np.interp(u, self.U, self.top)

    def world(self, u, v, d=0.0):
        """(u, v, d) → world: u — length along the side's face (without the batter), the face — the centerline shifted along the
        normal by half the thickness (+ the batter on the outside), depth d — into the wall along the segment normal (the sides
        of stones and of opening reveals are at a right angle to the face); at a station with a turn (u ± 0.2 mm) — along the
        mitered joint. Returns (x, y, z)."""
        u = np.atleast_1d(np.asarray(u, float)); v = np.atleast_1d(v); d = np.broadcast_to(np.atleast_1d(d), u.shape)
        x = np.interp(u, self.U, self.P[:, 0]); y = np.interp(u, self.U, self.P[:, 1])
        nx = np.interp(u, self.U, self.N[:, 0]); ny = np.interp(u, self.U, self.N[:, 1])   # with a mitered joint — not a unit vector
        t = np.interp(u, self.U, self.t); b = self.base_at(u)
        if self.side > 0:
            bat = np.interp(u, self.U, self.bat); kn = np.interp(u, self.U, self.knee)
            off = t / 2 + bat * np.clip((kn - v) / np.maximum(kn - b, 1e-3), 0, 1)
        else:
            off = t / 2
        i = np.clip(np.searchsorted(self.U, u, side="right") - 1, 0, len(self.nseg) - 1)
        at = np.zeros(u.shape, bool)
        if len(self.turn): at = np.min(np.abs(u[:, None] - self.turn[None, :]), axis=1) < 2e-4
        dx = np.where(at, nx, self.nseg[i, 0]); dy = np.where(at, ny, self.nseg[i, 1])
        o = self.side * off; k = self.side * d
        return np.stack([x + nx * o - dx * k, y + ny * o - dy * k, v], 1)


def strip_from(spec):
    """A side from a passport (Strip.spec) or from a request description: pts, base, top, t, side, closed, batter, walk, bat,
    knee — per station (a list of length pts) or one number for all. A ring with the key start begins at the middle of the
    edge start: the closing joint does not land on a corner (the station values for the new point are the average of the
    neighbors)."""
    P = np.asarray(spec["pts"], float); closed = spec.get("closed", True); n0 = len(P)
    def st(x):
        if x is None: return None
        a = np.atleast_1d(np.asarray(x, float)); return a if len(a) == n0 else np.full(n0, float(a.mean()))
    vals = {k: st(spec.get(k)) for k in ("base", "top", "walk", "bat", "knee")}
    t = spec.get("t", 0.0); t = st(t) if np.ndim(t) else float(t)
    if closed and spec.get("start") is not None:
        k = int(spec["start"]) % n0; m = (P[k] + P[(k + 1) % n0]) / 2
        P = np.vstack([m[None], P[k + 1:], P[:k + 1]])
        rot = lambda a: None if a is None else np.concatenate([[(a[k] + a[(k + 1) % n0]) / 2], a[k + 1:], a[:k + 1]])
        vals = {key: rot(a) for key, a in vals.items()}
        if np.ndim(t): t = rot(t)
    return Strip(P, vals["base"], vals["top"], t, side=spec.get("side", 1), closed=closed, batter=float(spec.get("batter", 0.0) or 0.0),
                 name=spec.get("name", "ring"), walk=vals["walk"], bat=vals["bat"], knee=vals["knee"])

"""Queries to the site (components/site/terrain.py → relief.npz + location.json): height, normal, slope, water, road,
platform, ground color, anchor, path point, free places, a path over the relief (A* with a slope and water cost), line of
sight. Only numpy — read by both .venv and Blender: layout, cameras, motion, simulations. Ground(<folder with relief.npz>
| scenes/<name>); coordinates are meters, z up.
All functions take numbers or arrays: g.height(x, y), g.slope(xs, ys)."""
import heapq, json, math, os
import numpy as np

# ground color (sRGB 0..1): base, slope, height, bank — by the plan's ground type; road and platform have their own
PAL = {"grass": (0.45, 0.50, 0.31), "soil": (0.53, 0.45, 0.35), "rock": (0.53, 0.51, 0.48), "cliff": (0.45, 0.43, 0.41),
       "sand": (0.74, 0.67, 0.52), "snow": (0.82, 0.84, 0.88), "ice": (0.66, 0.74, 0.82), "mud": (0.41, 0.36, 0.29),
       "ash": (0.37, 0.36, 0.36), "road": (0.60, 0.53, 0.43), "pad": (0.58, 0.53, 0.46), "bank": (0.64, 0.58, 0.46), "bed": (0.36, 0.33, 0.28)}
ENV = {"grass": ("grass", "rock", "snow", "bank"), "soil": ("soil", "rock", "snow", "mud"), "sand": ("sand", "rock", "rock", "mud"),
       "snow": ("snow", "rock", "snow", "ice"), "ice": ("ice", "rock", "snow", "ice"), "ash": ("ash", "cliff", "ash", "mud"),
       "mud": ("mud", "rock", "snow", "mud"), "rock": ("rock", "cliff", "snow", "sand")}


def axis(c, near_half, res, core_half, far_half, edge_res=2.5, far_res=25.0, growth=1.12):
    """The site grid lines along one axis: step res in the core |x − c| ≤ core_half, then growing (×growth) up to edge_res
    toward the edge of the near patch, beyond the edge — up to far_res toward the edge of the belt; a line falls exactly on
    the edge of the patch. Shared by the Blender mesh (bl_relief) and the texture (terrain_tex)."""
    xs, step = [0.0], res
    while xs[-1] < far_half - 1e-9:
        x = xs[-1]
        if x >= core_half - 1e-9: step = min(step * growth, edge_res if x < near_half - 1e-9 else far_res)
        nxt = x + step
        if x < near_half - 1e-9 < nxt: nxt = near_half
        xs.append(min(nxt, far_half))
    xs = np.array(xs)
    return np.concatenate([c - xs[:0:-1], c + xs])


def axes(G, core=None):
    """Site grid lines (xs, ys): a 0.5 m core — 120 m in the middle of the patch (or core — half of the core)."""
    nb, fb = G.near_bounds(), G.far_bounds(); c = ((nb[0] + nb[2]) / 2, (nb[1] + nb[3]) / 2); nh = (nb[2] - nb[0]) / 2; fh = (fb[2] - fb[0]) / 2
    core = core or min(60.0, nh)
    return axis(c[0], nh, G.nres, core, fh), axis(c[1], nh, G.nres, core, fh)


def _smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t)


def snowline_of(L, biome):
    """The site's snow line, m (a relief layer: location.json): explicit (loc.snowline) or, for mountains, the default (0.62 of
    the height of the far mountains: real ones — far_top, the belt's p99; noise-based ones — far.h); otherwise none."""
    s = L.get("snowline", 0.62 * float(L.get("far_top") or L.get("far", {}).get("h", 140)) if biome == "alpine" else None)
    return None if s in (None, "none") else float(s)


class Ground:
    def __init__(self, folder):
        z = np.load(os.path.join(folder, "relief.npz"))
        self.Hn, (self.nx0, self.ny0, self.nres) = z["near"].astype(np.float64), z["near_o"]
        self.Hf, (self.fx0, self.fy0, self.fres) = z["far"].astype(np.float64), z["far_o"]
        self.level = z["level"].astype(np.float64)
        self.masks = {k: z[k].astype(np.float32) / 255 for k in ("road", "pad", "cliff", "pave") if k in z.files}
        self.masks.setdefault("pave", np.zeros_like(self.masks["pad"]))
        self.loc = json.load(open(os.path.join(folder, "location.json")))
        self.far_z = float(self.loc.get("far_z", 0.0)); self.folder = folder
        self.decks = [(np.asarray(b["path"], float), float(b.get("w", 5.2)) / 2) for b in self.loc.get("bridges") or []]   # bridge decks
        self.amaps = {k[3:]: z[k] for k in z.files if k.startswith("an_") and k != "an_step"}   # analysis data maps (terrain.data_maps)
        self.astep = float(z["an_step"][0]) if "an_step" in z.files else None
        b = self.near_bounds(); self.size = b[2] - b[0]

    # --- grids
    def near_bounds(self):
        return (self.nx0, self.ny0, self.nx0 + (self.Hn.shape[1] - 1) * self.nres, self.ny0 + (self.Hn.shape[0] - 1) * self.nres)

    def far_bounds(self):
        return (self.fx0, self.fy0, self.fx0 + (self.Hf.shape[1] - 1) * self.fres, self.fy0 + (self.Hf.shape[0] - 1) * self.fres)

    @staticmethod
    def _bil(A, x0, y0, res, x, y):
        gx = np.clip((x - x0) / res, 0, A.shape[1] - 1.001); gy = np.clip((y - y0) / res, 0, A.shape[0] - 1.001)
        ix, iy = gx.astype(int), gy.astype(int); fx, fy = gx - ix, gy - iy
        return A[iy, ix] * (1 - fx) * (1 - fy) + A[iy, ix + 1] * fx * (1 - fy) + A[iy + 1, ix] * (1 - fx) * fy + A[iy + 1, ix + 1] * fx * fy

    def _in(self, b, x, y):
        return (x >= b[0]) & (x <= b[2]) & (y >= b[1]) & (y <= b[3])

    # --- relief
    BAND = 12.0                                                     # m: at the edge of the patch the height blends into the belt (a seam with no step)

    def height(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        z = np.where(self._in(self.far_bounds(), x, y), self._bil(self.Hf, self.fx0, self.fy0, self.fres, x, y), self.far_z)
        b = self.near_bounds(); m = self._in(b, x, y)
        if not np.any(m): return z
        d = np.minimum(np.minimum(x - b[0], b[2] - x), np.minimum(y - b[1], b[3] - y)); w = _smooth(0, self.BAND, d)
        return np.where(m, self._bil(self.Hn, self.nx0, self.ny0, self.nres, x, y) * w + z * (1 - w), z)

    def grad(self, x, y, e=None):
        x, y = np.asarray(x, float), np.asarray(y, float)
        e = e or np.where(self._in(self.near_bounds(), x, y), self.nres, self.fres)
        return (self.height(x + e, y) - self.height(x - e, y)) / (2 * e), (self.height(x, y + e) - self.height(x, y - e)) / (2 * e)

    def normal(self, x, y):
        gx, gy = self.grad(x, y); n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    def slope(self, x, y):
        """Slope, degrees."""
        gx, gy = self.grad(x, y); return np.degrees(np.arctan(np.hypot(gx, gy)))

    # --- body support: height/slope/water — the ground (mesh, texture, dressing); top/steep/wet — where the body stands (a bridge deck on top)
    def deck(self, x, y):
        """Top of a bridge deck above the point, m; NaN — not on a bridge."""
        x, y = np.asarray(x, float), np.asarray(y, float); z = np.full(np.broadcast(x, y).shape, np.nan)
        for P, hw in self.decks:
            for a, b in zip(P[:-1], P[1:]):
                d = b[:2] - a[:2]; t = ((x - a[0]) * d[0] + (y - a[1]) * d[1]) / max(float(d @ d), 1e-9)
                on = (t >= 0) & (t <= 1) & (np.hypot(x - a[0] - t * d[0], y - a[1] - t * d[1]) <= hw)
                z = np.where(on, a[2] + t * (b[2] - a[2]) + 0.05, z)
        return z

    def top(self, x, y):
        """Height of the body's support: a bridge deck or the ground."""
        h = self.height(x, y)
        if not self.decks: return h
        dz = self.deck(x, y); return np.where(np.isnan(dz), h, dz)

    def wet(self, x, y):
        """Water depth under the body: on a bridge — dry."""
        w = self.water(x, y)
        return w if not self.decks else np.where(np.isnan(self.deck(x, y)), w, 0.0)

    def steep(self, x, y):
        """Slope of the body's support, degrees (the edge of a deck is a drop)."""
        if not self.decks: return self.slope(x, y)
        x, y = np.asarray(x, float), np.asarray(y, float); e = np.where(self._in(self.near_bounds(), x, y), self.nres, self.fres)
        gx = (self.top(x + e, y) - self.top(x - e, y)) / (2 * e); gy = (self.top(x, y + e) - self.top(x, y - e)) / (2 * e)
        return np.degrees(np.arctan(np.hypot(gx, gy)))

    def water(self, x, y):
        """Water depth, m (0 — dry). Water exists only in the near patch."""
        x, y = np.asarray(x, float), np.asarray(y, float); m = self._in(self.near_bounds(), x, y)
        if getattr(self, "_lv_src", None) is not self.level:            # the level grid without NaN — once per site (not on every
            self._lv_src, self._lv = self.level, np.nan_to_num(self.level, nan=-1e9)   # call: 5540 calls cost 7.3 s in Buildings)
        lv = self._bil(self._lv, self.nx0, self.ny0, self.nres, x, y)
        return np.where(m & (lv > -1e8), np.maximum(lv - self.height(x, y), 0.0), 0.0)

    def amap(self, name, x, y):
        """Analysis data map at a point (the nearest node of the 1 m grid): area — catchment, m²; brow/foot — the brow/foot of a drop;
        flat — a flat-place label (0 — not flat); forms — the form (terrain_analysis.FORMS); dwater/droad — distance to
        water/road, m."""
        x, y = np.asarray(x, float), np.asarray(y, float); A = self.amaps[name]
        i = np.clip(np.round((y - self.ny0) / self.astep).astype(int), 0, A.shape[0] - 1); j = np.clip(np.round((x - self.nx0) / self.astep).astype(int), 0, A.shape[1] - 1)
        return A[i, j]

    def mask(self, name, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return np.where(self._in(self.near_bounds(), x, y), self._bil(self.masks[name], self.nx0, self.ny0, self.nres, x, y), 0.0)

    # --- cover (components/grass → cover.npz in the site folder): fields, group shares, density, ground — for the ground shader, the meadow, the materials
    def _cov(self):
        if not hasattr(self, "_C"):
            p = os.path.join(self.folder, "cover.npz"); self._C = None
            if os.path.exists(p):
                z = np.load(p); self._C = {k: z[k] for k in z.files if k != "meta"}; self._C["meta"] = json.loads(str(z["meta"]))
        return self._C

    COVER_BAND = 20.0                                               # m: at the edge of the cover grid — blend to the belt raster (a seam with no step)

    def cover(self, name, x, y):
        """Cover at a point (bilinear; outside the grid — the edge; None — the site has no cover): fields W (water), L (light), D
        (disturbance), E (edge); density, height, dryness, lushness; a group share — its name (cover_meta()["groups"]); ground —
        litter, needles, moss, dirt, pebbles, soil, bare; tint — (…, 3)."""
        C = self._cov()
        if C is None: return None
        x, y = np.asarray(x, float), np.asarray(y, float); M = C["meta"]; x0, y0, st = float(C["x0"]), float(C["y0"]), float(C["step"])
        if name == "tint": v = np.stack([self._bil(C["tint"][..., k].astype(np.float32), x0, y0, st, x, y) for k in range(3)], -1)
        else:
            A = C["shares"][M["groups"].index(name)] if name in M["groups"] else C["ground"][M["ground"].index(name)] if name in M["ground"] else C[name]
            v = self._bil(A.astype(np.float32) / 255.0, x0, y0, st, x, y)
        if "f_density" in C and (name in M["groups"] or name in ("density", "height", "dryness", "tint")):   # far belt (Grass, model far):
            fx0, fy0, fst = float(C["f_x0"]), float(C["f_y0"]), float(C["f_step"])          # beyond the edge of the cover grid — its raster,
            if name == "tint": vf = np.stack([self._bil(C["f_tint"][..., k].astype(np.float32), fx0, fy0, fst, x, y) for k in range(3)], -1)
            else:                                                       # within the last COVER_BAND m — smoothly toward it
                A = C["f_shares"][M["groups"].index(name)] if name in M["groups"] else C["f_" + name]
                vf = self._bil(A.astype(np.float32) / 255.0, fx0, fy0, fst, x, y)
            ny_, nx_ = C["density"].shape; e = np.minimum.reduce([x - x0, x0 + (nx_ - 1) * st - x, y - y0, y0 + (ny_ - 1) * st - y])
            w = _smooth(0, self.COVER_BAND, e); w = w[..., None] if name == "tint" else w
            v = v * w + vf * (1 - w)
        return v * float(M.get("height_m", 1.6)) if name == "height" else v

    def cover_meta(self):
        """Cover meta (biome, groups, ground, plant species per group, knobs, interpretation) or None."""
        C = self._cov(); return None if C is None else C["meta"]

    def walkable(self, x, y, max_slope=30.0, max_water=0.5):
        return (self.steep(x, y) <= max_slope) & (self.wet(x, y) <= max_water)

    # --- anchors and paths
    def anchor(self, name):
        """"<id>.<anchor>" → (x, y, z); "<path>@fraction" → a point on the path."""
        if "@" in name:
            p, _ = self.along(*name.split("@")); return p
        x, y, _ = self.loc["anchors"][name]; return np.array([x, y, float(self.top(x, y))])

    def path(self, name):
        return np.asarray(self.loc["paths"][name]["pts"], float)

    def along(self, name, t):
        """A path point at the fraction t of the length, and the tangent (unit, in plan)."""
        P = self.path(name); s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P[:, :2], axis=0).T))]); u = float(t) * s[-1]
        i = int(np.clip(np.searchsorted(s, u) - 1, 0, len(P) - 2)); f = (u - s[i]) / max(s[i + 1] - s[i], 1e-9)
        p = P[i] * (1 - f) + P[i + 1] * f; d = P[i + 1, :2] - P[i, :2]
        if self.loc["paths"][name]["kind"] != "river": p[2] = float(self.top(p[0], p[1]))
        return p, d / max(np.linalg.norm(d), 1e-9)

    # --- places and paths for staging, motion, simulations
    def spots(self, n, center=None, r=None, max_slope=15.0, gap=2.0, dry=True, avoid=("road", "pad"), seed=0, tries=6000, box=None):
        """n free places: slope no steeper than max_slope, dry, not on the avoid masks, no closer than gap to each other."""
        rng = np.random.default_rng(seed); b = self.near_bounds(); out = []
        c = np.asarray(center if center is not None else [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], float)
        r = r or (b[2] - b[0]) / 2
        for _ in range(tries):
            if len(out) >= n: break
            a, rr = rng.uniform(0, 2 * math.pi), r * math.sqrt(rng.uniform(0, 1)); x, y = c[0] + rr * math.cos(a), c[1] + rr * math.sin(a)
            if box and not (box[0] <= x <= box[2] and box[1] <= y <= box[3]): continue
            if self.steep(x, y) > max_slope or (dry and self.wet(x, y) > 0.05) or any(self.mask(m, x, y) > 0.3 for m in avoid): continue
            if all(math.hypot(x - p[0], y - p[1]) >= gap for p in out): out.append((x, y, float(self.top(x, y))))
        return np.array(out)

    def route(self, a, b, max_slope=32.0, step=1.0, water=40.0, margin=60.0, k_slope=8.0):
        """A walking path from a to b (points or anchors): A* over a grid of step m, cost — length × (1 + k·slope²), water is
        expensive, steeper than max_slope is impossible. Returns (N, 3) or None. For squads, motion over time, roads "on their
        own"."""
        a = self.anchor(a)[:2] if isinstance(a, str) else np.asarray(a, float)[:2]; b = self.anchor(b)[:2] if isinstance(b, str) else np.asarray(b, float)[:2]
        nb = self.near_bounds(); x0, y0 = max(min(a[0], b[0]) - margin, nb[0]), max(min(a[1], b[1]) - margin, nb[1])
        x1, y1 = min(max(a[0], b[0]) + margin, nb[2]), min(max(a[1], b[1]) + margin, nb[3])
        nx, ny = int((x1 - x0) / step) + 1, int((y1 - y0) / step) + 1
        X, Y = np.meshgrid(x0 + np.arange(nx) * step, y0 + np.arange(ny) * step)
        Z = self.top(X, Y); W = self.wet(X, Y); S = self.steep(X, Y); bad = S > max_slope
        idx = lambda p: (int(round((p[1] - y0) / step)), int(round((p[0] - x0) / step)))
        s0, s1 = idx(a), idx(b); g = {s0: 0.0}; prev = {}; pq = [(0.0, s0)]; tg = math.tan(math.radians(max_slope))
        nbrs = [(dj, di, math.hypot(dj, di) * step) for dj in (-1, 0, 1) for di in (-1, 0, 1) if dj or di]
        while pq:
            f_, u = heapq.heappop(pq)
            if u == s1: break
            for dj, di, L in nbrs:
                v = (u[0] + dj, u[1] + di)
                if not (0 <= v[0] < ny and 0 <= v[1] < nx) or bad[v]: continue
                gr = abs(Z[v] - Z[u]) / L
                if gr > tg: continue
                c = g[u] + L * (1 + k_slope * gr * gr) + water * L * min(W[v], 1.5)
                if c < g.get(v, 1e18):
                    g[v] = c; prev[v] = u; h = math.hypot(v[0] - s1[0], v[1] - s1[1]) * step
                    heapq.heappush(pq, (c + h, v))
        if s1 not in prev and s1 != s0: return None
        cells, u = [s1], s1
        while u != s0: u = prev[u]; cells.append(u)
        P = np.array([(x0 + c[1] * step, y0 + c[0] * step) for c in cells[::-1]])
        for _ in range(2):                                            # smooth the grid steps
            if len(P) < 3: break
            Q = np.empty((2 * len(P) - 2, 2)); Q[0::2] = 0.75 * P[:-1] + 0.25 * P[1:]; Q[1::2] = 0.25 * P[:-1] + 0.75 * P[1:]
            P = np.concatenate([P[:1], Q[1:-1], P[-1:]])
        return np.column_stack([P, self.top(P[:, 0], P[:, 1])])

    def visible(self, p, q, clear=0.0, step=None):
        """Line of sight p → q over the relief (clearance clear m)."""
        p, q = np.asarray(p, float), np.asarray(q, float); L = np.linalg.norm(q - p); n = max(2, int(L / (step or self.nres * 2)))
        t = np.linspace(0, 1, n)[1:-1, None]; P = p + (q - p) * t
        return bool(np.all(P[:, 2] - self.height(P[:, 0], P[:, 1]) > clear))

    # --- placing on the relief: props, figures, formation
    def seat(self, poly, policy="sink", sink=0.0, max_tilt=15.0):
        """Base height of a prop by the footprint poly (N × 2, m). pad — at the center (the core leveled the platform); sink — at
        the lowest point of the footprint (walls, rocks, ruins sink into the ground); tilt — at the center and tilted along the
        mean normal of the footprint (up to max_tilt°). Returns (z, a normal or None, the height drop under the footprint, m)."""
        P = np.asarray(poly, float); c = P.mean(0); Q = np.vstack([P, (P + np.roll(P, -1, 0)) / 2, c[None]])
        z = self.top(Q[:, 0], Q[:, 1]); spread = float(np.ptp(z))
        if policy == "sink": return float(z.min()) - sink, None, spread
        if policy == "tilt":
            n = self.normal(Q[:, 0], Q[:, 1]).mean(0); n /= np.linalg.norm(n); a = math.degrees(math.acos(min(1.0, n[2])))
            if a > max_tilt:                                         # tilt no steeper than max_tilt
                h = np.hypot(n[0], n[1]); t = math.tan(math.radians(max_tilt)); n = np.array([n[0] / h * t, n[1] / h * t, 1.0]); n /= np.linalg.norm(n)
            return float(self.top(c[0], c[1])), n, spread
        return float(self.top(c[0], c[1])), None, spread

    def stand(self, feet_xy, feet_z0):
        """Height of a figure by its feet: feet_xy — the feet points in plan (K × 2, m), feet_z0 — their heights at a figure shift
        of 0. Returns (dz, gaps): the shift at which the largest |gap| is smallest (one foot slightly in the ground, the other
        slightly above it), and the feet gaps after the shift (> 0 — hanging, < 0 — in the ground), m."""
        F = np.asarray(feet_xy, float); g0 = np.asarray(feet_z0, float) - self.top(F[:, 0], F[:, 1])
        dz = -(g0.max() + g0.min()) / 2
        return float(dz), g0 + dz

    def formation(self, at, face, offsets, max_slope=20.0, reach=2.0, contour=False, dry=True, search=0.0):
        """A formation on the relief: offsets — (sideways, forward) in m from the point at, the front is face (a vector or degrees
        from east). A spot with a slope steeper than max_slope or in water is shifted to the nearest suitable one within reach;
        contour — the front runs along the contour line (on a slope the formation faces up or down the slope — whichever is
        closer to face). Returns (points N × 3, the front, how many were shifted, how many are unfit)."""
        at = self.anchor(at)[:2] if isinstance(at, str) else np.asarray(at, float)[:2]
        f = np.array([math.cos(math.radians(face)), math.sin(math.radians(face))]) if np.isscalar(face) else np.asarray(face, float)[:2]
        f = f / max(np.linalg.norm(f), 1e-9)
        if contour:
            gx, gy = self.grad(at[0], at[1]); g = np.array([float(gx), float(gy)])
            if np.linalg.norm(g) > math.tan(math.radians(4)): u = g / np.linalg.norm(g); f = u if u @ f >= 0 else -u
        side = np.array([f[1], -f[0]]); ok = lambda x, y: self.steep(x, y) <= max_slope and (not dry or self.wet(x, y) <= 0.05)
        if search:                                                  # the plan point is bad — the formation looks nearby for a place where almost all are fit
            score = lambda c: sum(ok(*(c + side * a + f * b)) for a, b in offsets)
            best, sc_ = at, score(at)
            for r in np.arange(3.0, search + 1e-9, 3.0):
                if sc_ >= len(offsets) * 0.6: break                 # 60 % fit — the formation stays at the plan point (the intent matters more)
                for k in range(12):
                    c = at + r * np.array([math.cos(k * math.pi / 6), math.sin(k * math.pi / 6)]); v = score(c)
                    if v > sc_: best, sc_ = c, v
            at = best
        P = np.array([at + side * a + f * b for a, b in offsets])
        moved = bad = 0; ring = [(r, k) for r in np.arange(0.5, reach + 1e-9, 0.5) for k in range(8)]
        for i, (x, y) in enumerate(P):
            if ok(x, y): continue
            for r, k in ring:
                q = np.array([x, y]) + r * np.array([math.cos(k * math.pi / 4), math.sin(k * math.pi / 4)])
                if ok(q[0], q[1]) and all(np.hypot(*(q - P[j])) > 0.9 for j in range(len(P)) if j != i): P[i] = q; moved += 1; break
            else: bad += 1
        return np.column_stack([P, self.top(P[:, 0], P[:, 1])]), f, moved, bad

    # --- ground color (previs, sheet, texture)
    def colors(self, x, y):
        """sRGB 0..1 by the plan's ground type: a base with spots, slope — stone, height — snow, bank, road, platform."""
        x, y = np.asarray(x, float), np.asarray(y, float); g = self.loc.get("ground", "grass")
        base, steep, high, bank = ENV.get(g, ENV["grass"]); P = {k: np.asarray(v) for k, v in PAL.items()}
        z = self.height(x, y); s = self.slope(x, y); dep = self.water(x, y)
        from core.surface.noise import fbm                                     # spots by gradient noise (sines gave stripes)
        c = P[base] * (1 + 0.07 * fbm(x / 23.0, y / 23.0, 3, 11))[..., None]
        patch = _smooth(0.25, 0.75, fbm(x / 41.0, y / 41.0, 3, 12) * 0.5 + 0.5)[..., None] * 0.35
        c = c * (1 - patch) + P["soil" if base != "soil" else "mud"] * patch
        rock = np.maximum(_smooth(30, 44, s), self.mask("cliff", x, y))[..., None]
        c = c * (1 - rock) + P[steep] * rock
        sl = self.loc.get("snowline", 0.62 * float(self.loc.get("far", {}).get("h", 140)) if base not in ("sand", "ash") else None)
        if sl is not None and sl != "none":
            snow = (_smooth(float(sl) - 12, float(sl) + 12, z) * (1 - 0.6 * rock[..., 0]))[..., None]; c = c * (1 - snow) + P[high] * snow
        lv = self._bil(np.nan_to_num(self.level, nan=-1e9), self.nx0, self.ny0, self.nres, x, y)
        shore = ((lv > -1e8) & self._in(self.near_bounds(), x, y) & (z - lv < 1.0))[..., None] * 1.0
        c = c * (1 - shore) + P[bank] * shore
        wet = np.clip(dep / 1.0, 0, 1)[..., None]; c = c * (1 - wet) + P["bed"] * wet
        rd = self.mask("road", x, y)[..., None]; c = c * (1 - rd) + P["road"] * rd
        pd = (self.mask("pad", x, y) * 0.65)[..., None]; c = c * (1 - pd) + P["pad"] * pd
        return np.clip(c * float(self.loc.get("bright", 1.0)), 0, 1)


def load(path):
    """A site from a scene folder (scenes/<name>/relief.npz) or from terrain.py's output folder."""
    return Ground(path)

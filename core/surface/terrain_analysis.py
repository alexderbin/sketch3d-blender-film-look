"""Relief analysis: on any height map (our site, a real window, a game map) finds what the plan, staging, assets and
cameras rely on — the drainage network and thalwegs, saddles, summits (with a flat top), bowls, brows and feet of drops,
flat places; the shares of landforms (geomorphons). Pure numpy (works in Blender too). Grid: H[iy, ix], x = x0 + ix·res,
y = y0 + iy·res.
Sources: depression filling — Barnes et al. 2014 (Priority-Flood+ε); drainage — D8; saddles — the sign of curvature
(Hessian) under smoothing; geomorphons — Jasiewicz, Stepinski 2013, the form table — GRASS r.geomorphon (geom.c).
.venv/bin/python -m core.surface.terrain_analysis <site folder | relief.npz> → analysis summary (__RESULT__)."""
import heapq, json, math, os, sys

import numpy as np

N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
FORMS = ["flat", "peak", "ridge", "shoulder", "spur", "slope", "hollow", "footslope", "valley", "pit"]
_T = [[0, 0, 0, 7, 7, 8, 8, 8, 9], [0, 0, 7, 7, 7, 8, 8, 8, -1], [0, 3, 5, 5, 6, 6, 8, -1, -1], [3, 3, 5, 5, 5, 6, -1, -1, -1],
      [3, 3, 4, 5, 5, -1, -1, -1, -1], [2, 2, 4, 4, -1, -1, -1, -1, -1], [2, 2, 2, -1, -1, -1, -1, -1, -1], [2, 2, -1, -1, -1, -1, -1, -1, -1],
      [1, -1, -1, -1, -1, -1, -1, -1, -1]]                          # [number of "lower"][number of "higher"] → form (GRASS forms[9][9])


# ---------------------------------------------------------------- base: smoothing, slope, dilation
def blur(A, sigma):
    """Gaussian along the axes (numpy), sigma in cells."""
    if sigma < 0.3: return A.copy()
    r = int(3 * sigma); k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    P = np.pad(A, r, mode="edge")
    P = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, P)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 1, P)


def slope(H, res):
    gy, gx = np.gradient(H, res); return np.degrees(np.arctan(np.hypot(gx, gy)))


def dilate(M, k, op=np.maximum):
    """Maximum (or minimum) over a (2k+1)² square — k passes of 3×3."""
    A = M.copy()
    for _ in range(k):
        P = np.pad(A, 1, mode="edge"); B = A.copy()
        for dy, dx in N8: B = op(B, P[1 + dy:1 + dy + A.shape[0], 1 + dx:1 + dx + A.shape[1]])
        A = B
    return A


# ---------------------------------------------------------------- drainage
def fill(H, eps=1e-4):
    """Filling closed depressions with a tiny slope toward the exit (Priority-Flood+ε): every cell has a drainage path to the edge."""
    ny, nx = H.shape; n = ny * nx; F = H.ravel().astype(float).tolist(); done = bytearray(n); pq = []
    for i in list(range(nx)) + list(range(n - nx, n)) + list(range(nx, n - nx, nx)) + list(range(2 * nx - 1, n - nx, nx)):
        if not done[i]: done[i] = 1; pq.append((F[i], i))
    heapq.heapify(pq); push, pop = heapq.heappush, heapq.heappop
    while pq:
        z, i = pop(pq); r, c = divmod(i, nx)
        for dy, dx in N8:
            rr, cc = r + dy, c + dx
            if 0 <= rr < ny and 0 <= cc < nx:
                j = rr * nx + cc
                if not done[j]:
                    done[j] = 1
                    if F[j] <= z: F[j] = z + eps
                    push(pq, (F[j], j))
    return np.array(F).reshape(ny, nx)


def receivers(F, res):
    """D8: where a cell drains to (a flat index; the edge drains into itself)."""
    ny, nx = F.shape; idx = np.arange(ny * nx).reshape(ny, nx); best = np.zeros(F.shape); rec = idx.copy()
    P = np.pad(F, 1, constant_values=np.inf); I = np.pad(idx, 1, constant_values=-1)
    for dy, dx in N8:
        drop = (F - P[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx]) / (res * math.hypot(dy, dx))
        m = drop > best; best = np.where(m, drop, best); rec = np.where(m, I[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx], rec)
    return rec.ravel()


def accumulate(F, rec, res):
    """Catchment area, m²: top to bottom over the filled relief."""
    order = np.argsort(F.ravel())[::-1].tolist(); R = rec.tolist(); A = [res * res] * len(R)
    for i in order:
        j = R[i]
        if j != i: A[j] += A[i]
    return np.array(A).reshape(F.shape)


def trunk(A, rec, start, a_stop):
    """A thalweg through the cell start: upstream — along the tributary with the largest catchment (while ≥ a_stop), downstream —
    along the drainage to the edge. Returns flat indices from the source to the mouth."""
    ny, nx = A.shape; a = A.ravel(); R = rec
    up = [start]; seen = {start}
    while True:
        r, c = divmod(up[-1], nx); best = None
        for dy, dx in N8:
            rr, cc = r + dy, c + dx
            if 0 <= rr < ny and 0 <= cc < nx:
                j = rr * nx + cc
                if j not in seen and R[j] == up[-1] and a[j] >= a_stop and (best is None or a[j] > a[best]): best = j
        if best is None: break
        up.append(best); seen.add(best)
    down = [start]
    while R[down[-1]] != down[-1] and len(down) < ny * nx: down.append(int(R[down[-1]]))
    return up[::-1] + down[1:]


# ---------------------------------------------------------------- points and places
def saddles(H, res, scale=30.0, n=8, min_drop=3.0):
    """Saddles: curvature of opposite signs (a minimum along the ridge, a maximum across) under smoothing of scale m and a small
    slope; a saddle qualifies if within 3·scale the ground rises along the ridge and falls across it (the road heading) — by
    at least min_drop m on both sides. Returns [{ij, strength, drop (the smaller of four), road_deg, ridge_deg}] in
    descending order of drop."""
    Hs = blur(H, scale / res); gy, gx = np.gradient(Hs, res); hyy, hyx = np.gradient(gy, res); hxy, hxx = np.gradient(gx, res)
    tr, det = hxx + hyy, hxx * hyy - hxy * hyx; disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0)); l1, l2 = tr / 2 + disc, tr / 2 - disc
    sc = np.where((det < 0) & (np.hypot(gx, gy) < 0.1), np.minimum(l1, -l2), 0.0)
    m = int(scale / res); sc[:m] = sc[-m:] = 0; sc[:, :m] = sc[:, -m:] = 0
    G = np.where(sc > 0, np.hypot(gx, gy), np.inf)                  # a saddle is a point with no slope: a local minimum of slope
    peak = (sc > 0) & (G <= dilate(G, max(1, m // 3), np.minimum)); out = []; ny, nx = H.shape; R = 3 * scale / res

    def at(i, j, a, t):                                             # height t cells away from (i, j) along heading a
        y, x = int(round(i + t * math.sin(a))), int(round(j + t * math.cos(a)))
        return Hs[min(max(y, 0), ny - 1), min(max(x, 0), nx - 1)]
    for i, j in sorted(zip(*np.nonzero(peak)), key=lambda p: -sc[p])[:600]:   # all local candidates: the drop check is cheap
        lam, vec = np.linalg.eigh(np.array([[hxx[i, j], hxy[i, j]], [hyx[i, j], hyy[i, j]]]))
        road, crest = math.atan2(vec[1, 0], vec[0, 0]), math.atan2(vec[1, 1], vec[0, 1]); z = Hs[i, j]
        drops = [z - at(i, j, road, R), z - at(i, j, road, -R), at(i, j, crest, R) - z, at(i, j, crest, -R) - z]
        if min(drops) < min_drop: continue
        out.append({"ij": [int(i), int(j)], "strength": float(sc[i, j]), "drop": float(min(drops)), "road_deg": math.degrees(road) % 180,
                    "ridge_deg": math.degrees(crest) % 180})
    return sorted(out, key=lambda d: -d["drop"])[:n]


def peaks(H, res, radius=25.0, rel_r=60.0, min_rel=6.0, sign=1, n=12):
    """Summits (sign=1) or bowls (sign=−1): an extremum of the smoothed height within a circle of radius, the drop relative to
    the surroundings rel_r ≥ min_rel; for a summit — the area of the "flat top" (within 2 m of the summit and a slope < 12°)."""
    Hs = blur(H * sign, 2.0 / res); k = max(1, int(radius / res)); top = (Hs >= dilate(Hs, k) - 1e-9)
    kr = max(1, int(rel_r / res)); low = dilate(Hs, kr, np.minimum); rel = Hs - low; S = slope(H, res); out = []
    m = k; top[:m] = top[-m:] = False; top[:, :m] = top[:, -m:] = False
    for i, j in sorted(zip(*np.nonzero(top & (rel >= min_rel))), key=lambda p: -rel[p])[:n]:
        d = {"ij": [int(i), int(j)], "drop": float(rel[i, j])}
        if sign > 0:
            w = slice(max(0, i - k), i + k + 1), slice(max(0, j - k), j + k + 1)
            d["flat_top_m2"] = float(((H[w] > H[i, j] - 2.0) & (S[w] < 12)).sum() * res * res)
        out.append(d)
    return out


def flats(H, res, max_slope=6.0, min_area=150.0, smooth=2.0):
    """Flat places for staging: connected regions with a slope < max_slope (smoothing smooth m), area ≥ min_area m²."""
    S = slope(blur(H, smooth / res), res); M = S < max_slope; ny, nx = M.shape; lab = np.zeros(M.shape, int); out = []; L = 0
    for i0, j0 in zip(*np.nonzero(M)):
        if lab[i0, j0]: continue
        L += 1; st = [(i0, j0)]; lab[i0, j0] = L; pts = []
        while st:
            i, j = st.pop(); pts.append((i, j))
            for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                a, b = i + dy, j + dx
                if 0 <= a < ny and 0 <= b < nx and M[a, b] and not lab[a, b]: lab[a, b] = L; st.append((a, b))
        area = len(pts) * res * res
        if area >= min_area:
            P = np.array(pts); out.append({"ij": [float(P[:, 0].mean()), float(P[:, 1].mean())], "area_m2": area, "label": L,
                                           "height": float(H[P[:, 0], P[:, 1]].mean())})
    return sorted(out, key=lambda d: -d["area_m2"]), lab


def cliffs(H, res, steep=40.0):
    """Drops: cells steeper than steep; the brow — neighboring non-steep cells above the drop, the foot — below it. Masks."""
    S = slope(H, res); M = S > steep; near = dilate(M.astype(float), 1) > 0
    Hn = dilate(np.where(M, H, -np.inf), 1); Hm = dilate(np.where(M, H, np.inf), 1, np.minimum)
    brow = near & ~M & (H >= Hn - 0.5); foot = near & ~M & (H <= Hm + 0.5)
    return M, brow, foot


def knife_crests(H, res, r_m=9.0, tpi=(1.0, 3.0), walls=(45.0, 55.0)):
    """Knife-edge ridges: the grid (step res) rounds off a ridge between steep walls — a cell-wide top comes out gentle (a
    slope-based paint put forest and meadow on it: "a stream along the summits"), while in nature it is rock there. A ridge
    is a point higher than the mean within r_m by tpi m (TPI), with slopes steeper than walls within r_m of it. → the slope
    of the walls, ° (on the ridge; a share by thresholds, the edge — a one-cell Gaussian), 0 — elsewhere."""
    S = slope(H, res); k = max(1, int(round(r_m / res))); n = 2 * k + 1
    c = np.pad(np.pad(H, k, mode="edge").cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    tp = H - (c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]) / (n * n)   # height above the mean in a (2k+1)² square
    ramp = lambda x, a, b: np.clip((x - a) / (b - a), 0.0, 1.0); smax = dilate(S, k)
    return blur(ramp(tp, *tpi) * ramp(smax, *walls), 1.0) * smax


def dist(M, res, max_m=60.0):
    """Distance to mask M, m (chamfer over 8 neighbors, iteratively; beyond max_m — max_m)."""
    D = np.where(M, 0.0, max_m); s2 = math.sqrt(2) * res
    for _ in range(int(max_m / res) + 1):
        P = np.pad(D, 1, constant_values=max_m); B = D.copy()
        for dy, dx in N8: B = np.minimum(B, P[1 + dy:1 + dy + D.shape[0], 1 + dx:1 + dx + D.shape[1]] + (s2 if dy and dx else res))
        if np.array_equal(B, D): break
        D = B
    return np.minimum(D, max_m)


def geomorphons(H, res, L=20.0, t_deg=1.5):
    """Form class of each cell (edges of width L — −1): 8 directions, angles to the horizon within radius L; "higher" if
    φmax+φmin > t."""
    t = math.radians(t_deg); h, w = H.shape; npos = np.zeros(H.shape, np.int8); nneg = np.zeros(H.shape, np.int8)
    for dx, dy in ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)):
        step = math.hypot(dx, dy) * res; K = max(1, int(L / step)); hi = np.full(H.shape, -np.pi / 2); lo = np.full(H.shape, np.pi / 2)
        for k in range(1, K + 1):
            sh = np.full(H.shape, np.nan)
            ys, ye, xs, xe = max(0, -k * dy), h - max(0, k * dy), max(0, -k * dx), w - max(0, k * dx)
            sh[ys:ye, xs:xe] = H[ys + k * dy:ye + k * dy, xs + k * dx:xe + k * dx]
            a = np.arctan((sh - H) / (k * step)); hi = np.fmax(hi, a); lo = np.fmin(lo, a)
        D = hi + lo; npos += (D > t); nneg += (D < -t)
    C = np.array(_T)[nneg, npos]; m = int(math.ceil(L / res)); C[:m] = -1; C[-m:] = -1; C[:, :m] = -1; C[:, -m:] = -1
    return C


# ---------------------------------------------------------------- all together
def analyze(H, res, x0=0.0, y0=0.0, step=1.0, stream_a=2000.0):
    """Analysis of a height map (analysis step step m — coarser is not needed). Points are in site meters. Returns (summary,
    maps)."""
    k = max(1, int(round(step / res))); Hc = np.asarray(H, float)[::k, ::k]; rc = res * k
    xy = lambda ij: [round(x0 + ij[1] * rc, 2), round(y0 + ij[0] * rc, 2)]
    F = fill(Hc); rec = receivers(F, rc); A = accumulate(F, rec, rc); S = slope(Hc, rc)
    st = A >= stream_a
    a_idx = int(np.argmax(np.where(st, A, -1))) if st.any() else int(np.argmax(A))
    T = trunk(A, rec, a_idx, stream_a)                                            # the site's main stream — from the mouth upward
    sad = []                                                        # saddles of different sizes: three scales, close ones count as one
    for sc_ in (20.0, 30.0, 50.0):
        for d in saddles(Hc, rc, sc_):
            if all(math.dist(d["ij"], e["ij"]) * rc > 30 for e in sad): sad.append(dict(d, scale=sc_))
    sad = sorted(sad, key=lambda d: -d["drop"])[:8]; top = peaks(Hc, rc); bowl = peaks(Hc, rc, sign=-1, min_rel=3.0); fl, lab = flats(Hc, rc)
    M, brow, foot = cliffs(Hc, rc); C = geomorphons(Hc, rc); v = C[C >= 0]; sh = np.bincount(v, minlength=10) / max(len(v), 1)
    out = {"step_m": rc, "flow": {"main": [xy(divmod(i, Hc.shape[1])) for i in T[::max(1, int(4 / rc))]],
                                "catchment_m2": float(A.ravel()[a_idx]), "channel_share": float(st.mean())},
           "saddles": [dict(d, at=xy(d.pop("ij"))) for d in sad], "peaks": [dict(d, at=xy(d.pop("ij"))) for d in top],
           "bowls": [dict(d, at=xy(d.pop("ij"))) for d in bowl], "flats": [dict({k_: v_ for k_, v_ in d.items() if k_ != "label"}, at=xy(d.pop("ij")))
                                                                        for d in fl[:12]],
           "cliff_share": float(M.mean()), "brow_share": float(brow.mean()), "forms": dict(zip(FORMS, [round(float(x), 3) for x in sh])),
           "slope_p50_p90": [round(float(np.percentile(S, 50)), 1), round(float(np.percentile(S, 90)), 1)]}
    maps = {"area": A.astype(np.float32), "filled": (F - Hc).astype(np.float32), "flat": lab.astype(np.int32), "brow": brow, "foot": foot,
            "cliff": M, "forms": C.astype(np.int8), "step": rc}
    return out, maps


def main():
    p = sys.argv[1]; f = p if p.endswith(".npz") else os.path.join(p, "relief.npz"); z = np.load(f)
    H = z["near"].astype(float); x0, y0, res = map(float, z["near_o"])
    out, _ = analyze(H, res, x0, y0)
    brief = {"flow_points": len(out["flow"]["main"]), "catchment_m2": round(out["flow"]["catchment_m2"]), "n_saddles": len(out["saddles"]),
             "n_peaks": len(out["peaks"]), "n_bowls": len(out["bowls"]), "n_flats": len(out["flats"]), "cliff_share": round(out["cliff_share"], 3),
             "slope_p50_p90": out["slope_p50_p90"], "flat": out["forms"]["flat"], "slope": out["forms"]["slope"]}
    print("__RESULT__" + json.dumps(brief, ensure_ascii=False))


if __name__ == "__main__":
    main()

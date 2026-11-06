"""Far terrain and sky: the place's `far:` description → the far-terrain plan (the light node: sun, moon, sky, stars, exposure;
then hazes, clouds, relief, city, fires); post-processing after render — the sky under the geometry and AgX tone mapping
as in Blender (its own matrices and the AgX_Base_sRGB.cube table), in linear color.
.venv/bin/python -m core.light.far resolve scenes/<name> [description.yaml] [time= weather= style= sun_az= moon_az= moon_el=] [out=<plan.json>]
    → out/far/<name>/plan.json (without a description — scenes/<name>/far.yaml or the far: key in battle.yaml; line keys override)
.venv/bin/python -m core.light.far post out/<name>/cam<K> [more…] — a lit frame: lit.exr (geometry, transparent background) + sky.exr
    (the sky by camera) → lit.png; exposure and style come from the frame's far.json (written by core/outputs/bl_color.py light=far)
Azimuth is as everywhere in the pipeline: counterclockwise from +x (east 0, north 90), elevation above the horizon."""
import hashlib, json, math, os, sys
import core.paths as _core_paths      # the code is here, the core — core/<part> through kernel
import numpy as np
from core.light import wind as wind_model

ROOT = _core_paths.ROOT
def _ocio_dir():
    """Blender's color-management folder (the AgX tables): SK_OCIO_DIR, otherwise next to the blender found on PATH — a macOS bundle or a Linux folder
    (on a rented machine the macOS path does not work)."""
    if os.environ.get("SK_OCIO_DIR"):
        return os.environ["SK_OCIO_DIR"]
    import glob, shutil
    exe = os.path.realpath(shutil.which("blender") or "/Applications/Blender.app/Contents/MacOS/Blender")
    for root in (os.path.join(os.path.dirname(exe), "..", "Resources"), os.path.dirname(exe)):
        found = sorted(glob.glob(os.path.join(root, "[0-9]*.[0-9]*", "datafiles", "colormanagement")))
        if found:
            return os.path.normpath(found[-1])
    return "/Applications/Blender.app/Contents/Resources/5.2/datafiles/colormanagement"


OCIO = _ocio_dir()
TABLE = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "far_sky_table.json")))      # runs/backdrop/bl_sun_table.py: sky-model light by sun elevation
LUM = np.array([0.2126, 0.7152, 0.0722])

# time of day: sun elevation and side; the side's azimuth is the default (the description sun: {az} overrides)
TIME = {"dawn": (-3.0, "east"), "sunrise": (2.0, "east"), "morning": (18.0, "east"), "noon": (58.0, "south"), "day": (40.0, "south"),
        "afternoon": (32.0, "west"), "evening": (12.0, "west"), "sunset": (1.5, "west"), "afterglow": (-0.8, "west"), "dusk": (-5.0, "west"), "night": (-28.0, "west")}
SIDE = {"east": 0.0, "north": 90.0, "west": 180.0, "south": 270.0}
# weather: visibility (km; σ = 3.912 / V), the sky model's aerosol, the sun and sky fractions, sky saturation, cloud coverage
WEATHER = {"clear": dict(vis_km=40, aerosol=1.0, sun_k=1.0, sky_k=1.0, sat=1.0, cover=0.05),
           "cumulus": dict(vis_km=30, aerosol=1.3, sun_k=1.0, sky_k=1.0, sat=1.0, cover=0.35),
           "haze": dict(vis_km=8, aerosol=4.0, sun_k=0.8, sky_k=1.1, sat=0.75, cover=0.1),
           "overcast": dict(vis_km=15, aerosol=2.0, sun_k=0.15, sky_k=1.3, sat=0.35, cover=0.95, over_k=0.25, ev=-0.6),
           "storm": dict(vis_km=6, aerosol=3.0, sun_k=0.05, sky_k=0.6, sat=0.3, cover=1.0, over_k=0.10, ev=-1.4,
                         ground=dict(vis_km=2.5, height=300.0, albedo=(0.62, 0.64, 0.68), g=0.4)),
           "rain": dict(vis_km=4, aerosol=2.5, sun_k=0.1, sky_k=1.1, sat=0.3, cover=1.0, over_k=0.2, ev=-0.9,   # it is raining (materials:
                        ground=dict(vis_km=1.8, height=90.0, albedo=(0.72, 0.74, 0.78), g=0.5)),               # weather: rain) — a veil up to 4 km,
                                                                                                                # drizzle at the ground
           "war_smoke": dict(vis_km=4, aerosol=6.0, sun_k=0.45, sky_k=0.8, sat=0.6, cover=0.3, tint=(1.0, 0.86, 0.72), ev=-0.3,
                             ground=dict(vis_km=2.5, height=140.0, albedo=(0.50, 0.44, 0.38), g=0.55)),
           "mist": dict(vis_km=12, aerosol=2.5, sun_k=0.85, sky_k=1.0, sat=0.8, cover=0.15,
                        ground=dict(vis_km=0.35, height=22.0, albedo=(0.95, 0.96, 0.98), g=0.6))}
H_AIR = 1500.0                                                       # scale height of the air haze decay, m (aerosol — 1–2 km)
D_SKY = 30000.0                                                      # the sky ray's path in the ground layer, m (beyond that — Earth curvature and the far-terrain edge)
# style: sky saturation, exposure shift, night tint; anime — the physical base is brighter and cleaner, real — as is
STYLE = {"anime": dict(sky_sat=1.18, ev=0.15, night_tint=(0.55, 0.68, 1.0), stars=1.0, moon_size=2.2, night_shift=(0.3, 0.5),
                       look="medium_high_contrast", bloom=0.35, shafts=0.9, night_ev=-1.5, night_fill=-3.5),
         "real": dict(sky_sat=1.0, ev=0.0, night_tint=(0.75, 0.82, 1.0), stars=0.7, moon_size=1.0, night_shift=(0.0, 1.0), look=None, bloom=0.15,
                      shafts=0.7, night_ev=-2.5, night_fill=-4.5)}
# look — the Blender AgX view (an OCIO table, core/blender/bl_agx_bake.py): anime — contrast 1.2 (livelier day, night with depth of black);
# bloom — glow strength of bright areas (far.bloom)
# night_shift — (decades, strength) of the blue night shift: real — the eye (Jensen 2000 as is); anime — like film: a camera without the Purkinje effect,
#   cool shadows and sky, warm glow cores (the threshold is 0.3 decades lower; by a full decade the assault turns brown again); strength 0.5 under the moon — the color
#   of objects stays (fortress: the hue spread of colored ones is 15° against 14° for the albedo; at 1.0 — 4°, everything blue), warm tones under a blue moon
#   go gray (canyon sand); with fire — toward 1.0 by the glow fraction (resolve): the glow on the haze turns blue, otherwise a ginger veil
# night_ev — night exposure relative to the key (moon and sky): film shoots night from day 2–2.5 stops darker, anime — a readable night (−1.5;
#   was −3: moonlit things fell 3–4 stops below middle gray); night_fill — moonlit fill: an object in shadow sits that many
#   stops below middle gray (bl_far.moon_fill; games — "dark but visible", the shadow does not go to black)
ALT_H = np.linspace(0.0, 16000.0, 64)                                # heights of the "sun at altitude" table (clouds, smoke), m
GROUND_ALB = np.array([0.20, 0.18, 0.15])                            # ground albedo for the bounce onto the underside of clouds and smoke
KAPPA = {"cloud": 6.0, "smoke": 2.5}                                 # puff optical thickness per radius (through the center — double; cumulus are dense:
                                                                     # against the sun the body is dark, the edge glows)
NIGHT_BOOST = 2e-3 / 2.5e-6                                          # night is compressed: the moon and the moon's sky are 800 times brighter than in life (K_NIGHT); fires at night —
                                                                     # likewise (their ratio to the moon is as in life), by day — no boost
K_NIGHT = 2e-3                                                       # the moon's sky relative to the sun's sky (physically ~2.5e-6; in the frame the fire must be brighter than the moon)


def lum(c): return float(np.dot(LUM, c))


def sun_vec(az, el):
    a, e = math.radians(az), math.radians(el)
    return [math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)]


def table_at(el):
    """Sky model light at sun elevation el: (the sun perpendicular to the ray, the sky onto a horizontal surface), RGB."""
    rows = TABLE["rows"]; els = np.array([r["el"] for r in rows], float)
    e = float(np.clip(el, els[0], els[-1]))
    sun = np.array([np.interp(e, els, [r["sun_rgb"][i] for r in rows]) for i in range(3)])
    sky = np.array([np.interp(e, els, [r["sky_rgb"][i] for r in rows]) for i in range(3)])
    if el < els[0]: sky = sky * 10 ** ((el - els[0]) / 4)            # below −6°: the sky fades about tenfold per 4° (measured)
    return sun, sky


def dip_deg(h):
    """Horizon dip from height h, degrees (with refraction — R_EARTH)."""
    return math.degrees(math.sqrt(2.0 * max(h, 0.0) / R_EARTH))


def puff_depths(pos, rad, cid, kappa, dirs):
    """Optical thickness through the neighboring puffs of the same cloud or smoke column, from the puff's point facing the luminary (Harris —
    Lastra 2001: the light of a cloud's particles is a pass from the luminary; it cannot be taken from the center — for overlapped puffs even the sunlit side ended up
    in shadow); the shader computes the puff's own thickness itself. κ is the puff's thickness per radius. dirs: {name: vector} → {name: thickness per puff}."""
    out = {k: np.zeros(len(pos), np.float32) for k in dirs}
    for c in np.unique(cid):
        idx = np.nonzero(cid == c)[0]; Pc = pos[idx].astype(float); Rc = rad[idx].astype(float)
        for k, L in dirs.items():
            L = np.asarray(L, float); L = L / np.linalg.norm(L)
            dP = Pc[None, :, :] - (Pc + Rc[:, None] * L)[:, None, :]; dd = np.einsum("ijk,ijk->ij", dP, dP); t = dP @ L
            half = np.sqrt(np.maximum(Rc[None, :] ** 2 - (dd - t * t), 0.0))
            seg = np.maximum(t + half, 0.0) - np.maximum(t - half, 0.0); np.fill_diagonal(seg, 0.0)
            out[k][idx] = (kappa / Rc[None, :] * seg).sum(1)
    return out


def puff_light(npz, plan, kappa, cache_dir):
    """Puff thicknesses toward the sun, the moon, up and down for this plan (cached by puffs, luminaries and κ)."""
    dirs = {"psun": plan["sun"]["dir"], "pmoon": plan["moon"]["dir"], "pup": [0.0, 0.0, 1.0], "pdown": [0.0, 0.0, -1.0]}
    key = hashlib.sha1(json.dumps([npz, dirs, kappa, 3], sort_keys=True).encode()).hexdigest()[:10]
    path = os.path.join(cache_dir, f"light_{key}.npz")
    if not os.path.exists(path):
        D = np.load(npz); pos, rad, cid = D["pos"].astype(float), D["rad"].astype(float), D["cid"]
        out = puff_depths(pos, rad, cid, kappa, dirs)
        C = np.zeros((len(pos), 4), np.float32)                              # cloud (column) dome: center at one third of the height, radius
        for c in np.unique(cid):
            i = np.nonzero(cid == c)[0]; lo, hi = (pos[i, 2] - rad[i]).min(), (pos[i, 2] + rad[i]).max()
            ctr = np.array([pos[i, 0].mean(), pos[i, 1].mean(), lo + 0.35 * (hi - lo)])
            C[i, :3] = ctr; C[i, 3] = (np.linalg.norm(pos[i] - ctr, axis=1) + rad[i]).max()
        np.savez_compressed(path, **out, pcx=C[:, 0], pcy=C[:, 1], pcz=C[:, 2], pcr=C[:, 3])
    return path


def smooth(a, b, x):
    t = float(np.clip((x - a) / (b - a), 0, 1)); return t * t * (3 - 2 * t)


def load_spec(scene_dir, path=None):
    """The scene's light-and-sky node is part of the scene model, not just of a frame: far.yaml (the far: key) → far: in battle.yaml → the default
    from world.json (node_default). Every scene has a node; the lit frame, the previs, the map passes and the sketch all come from it."""
    if path:
        import yaml; return yaml.safe_load(open(path)) or {}
    p = os.path.join(scene_dir, "far.yaml")
    if os.path.exists(p) or os.path.exists(os.path.join(scene_dir, "battle.yaml")): import yaml   # a node from world.json does not need yaml (Blender's Python)
    if os.path.exists(p): return (yaml.safe_load(open(p)) or {}).get("far", {})
    b = os.path.join(scene_dir, "battle.yaml")
    if os.path.exists(b) and (yaml.safe_load(open(b)) or {}).get("far"): return yaml.safe_load(open(b))["far"]
    return node_default(scene_dir)


NODE_OF = {"overcast": ("day", "overcast"), "blaze": ("night", "war_smoke")}   # frame light presets on top of the node's TIME
                                              # (bl_color light=, frames.py --light): daytime with overcast weather, a fire night; for the node's times the weather is the scene's own (for a site — by biome)


def node_light(light):
    """Frame light of the old path (bl_color light=, frames.py --light) → (node time, weather | None): the node time TIME directly
    (dawn … night — protocol names) or a NODE_OF preset (overcast, blaze); far, scene, empty — None (the node as is)."""
    if light in TIME: return (light, None)
    return NODE_OF.get(light)


def light_over(light):
    """Node override for a run by frame light: {time[, weather]}; far, scene, empty — the node as is."""
    nl = node_light(light)
    return {k: v for k, v in zip(("time", "weather"), nl) if v} if nl else {}


def node_near(*dirs):
    """The run's node (frames.py writes <run folder>/node.json — one for the sketch, maps, previs and light): the first one found."""
    for d in dirs:
        p = os.path.join(d, "node.json")
        if os.path.exists(p): P = json.load(open(p)); P["path"] = p; return P
    return None


NODE_BIOME = {   # the site's node by biome (location.json: biome, otherwise by the ground): weather and the far terrain beyond the site's mountain belt
    "temperate": dict(weather="cumulus", terrain=dict(kind="ridged", height=1600)),
    "alpine": dict(weather="clear", terrain=dict(kind="ridged", height=2600)),
    "ice": dict(weather="clear", terrain=dict(kind="ridged", height=2400)),
    "desert": dict(weather="haze", terrain=dict(kind="mesa", height=700)),
    "jungle": dict(weather="haze", terrain=dict(kind="hills", height=900)),
    "volcanic": dict(weather="haze", terrain=dict(kind="ridged", height=1800)),
    "wasteland": dict(weather="haze", terrain=dict(kind="hills", height=500))}


def node_default(scene_dir):
    """The default node comes from the scene: the sun from world.json (azimuth counterclockwise from +x, elevation); for a site (location.json) — weather
    and the far terrain beyond its mountain belt by biome (NODE_BIOME; a coast gets no relief: there is no sea); for the rest — clear, with no clouds or relief.
    The sketch haze (haze in world.json) is a sketch style, not air visibility."""
    wp = os.path.join(scene_dir, "world.json"); W_ = (json.load(open(wp)).get("world") or {}) if os.path.exists(wp) else {}
    s = W_.get("sun", {}) or {}
    sp = {"sun": {"az": float(s.get("azimuth", 220.0)), "el": float(s.get("elevation", 30.0))}, "weather": "clear", "clouds": [], "node": "world.json"}
    lp = os.path.join(scene_dir, "location.json")
    if os.path.exists(lp):
        L = json.load(open(lp)); bio = L.get("biome") or GROUND_BIOME.get(L.get("ground", "grass"), "temperate")
        nb = NODE_BIOME.get(bio)
        if nb and "coast" not in str((L.get("preset") or {}).get("name", "") if isinstance(L.get("preset"), dict) else L.get("preset") or ""):
            sp.update(weather=nb["weather"], terrain=dict(nb["terrain"], biome=bio), node="location.json"); sp.pop("clouds")   # clouds — by weather
    return sp


def plan_of(scene_dir, over=None, out=None):
    """The scene node's plan (a single entry point for all consumers: bl_color, bl_pack, pack.py): node (load_spec) → resolve;
    out — where to write the JSON (default out/far/<name>/node.json)."""
    plan = resolve(load_spec(scene_dir), scene_dir, over or {})
    out = out or os.path.join(ROOT, "out", "far", os.path.basename(os.path.abspath(scene_dir)), "node.json")
    os.makedirs(os.path.dirname(out), exist_ok=True); tmp = f"{out}.{os.getpid()}"
    json.dump(plan, open(tmp, "w"), ensure_ascii=False, indent=1); os.replace(tmp, out); plan["path"] = out   # map builds run in parallel
    return plan


def _sky_disp(sky_E, over, ev):
    """Sky brightness in the frame (screen, AgX) at the zenith and the horizon: mean sky L = E/π (the illuminance of a horizontal surface from the sky); clear —
    zenith 0.75 L, horizon 1.35 L; overcast (CIE) — 1.3 L and 0.55 L; mixed by the overcast fraction."""
    L = lum(np.asarray(sky_E, float)) / math.pi
    top, hor = L * (0.75 * (1 - over) + 1.3 * over), L * (1.35 * (1 - over) + 0.55 * over)
    return [float(agx(np.array([[v, v, v]], np.float32), ev)[0] @ LUM) for v in (top, hor)]


def sketch_world(P):
    """Light and sky of the sketch (pack.py, bl_pack) from the node: the key luminary (the sun; at night — the moon); the sky and haze tones are the sketch style
    (day: zenith 0.7, horizon 0.9, haze 0.8 at 700 m) × the frame's sky brightness relative to a clear noon (night and twilight get darker);
    a dense plan haze (fog, smoke, storm) shortens the sketch haze. Returns the world.json fields: sun, sky, haze."""
    key = P["moon"] if P["sun"]["el"] <= 0 and P["moon"].get("on") else P["sun"]
    sky_E = np.asarray(P["light"]["sky_e_rgb"]) + np.asarray(P["light"].get("night_sky_e_rgb", [0, 0, 0]))
    t, h = _sky_disp(sky_E, P["sky"].get("over", 0.0), P["light"]["ev"])
    s_ref, k_ref = table_at(TIME["noon"][0]); e_ref = lum(s_ref) * math.cos(math.radians(max(0.0, 45.0 - TIME["noon"][0]))) + 0.85 * lum(k_ref)
    t0, h0 = _sky_disp(k_ref, 0.0, math.log2(math.pi / e_ref) + STYLE["anime"]["ev"])
    sig = sum(Ly["a"] for Ly in P["fog"]["layers"])                      # haze density at the ground, 1/m
    rt, rh = min(t / max(t0, 1e-4), 1.15), min(h / max(h0, 1e-4), 1.1)
    rt, rh = max(rt, 0.22), max(rh, 0.25)                                # the sketch night is no darker than the site's moonlit night (0.15/0.22): in the assault
    # war smoke and the exposure correction gave sky tones of 0.003 — the sketch and the key frames went black and forms in shadow vanished
    return {"sun": {"azimuth": round(float(key["az"]), 2), "elevation": round(float(key["el"]), 2)},
            "sky": {"top": round(min(0.7 * rt, 0.95), 3), "horizon": round(min(0.9 * rh, 0.97), 3)},
            "haze": {"distance": round(float(min(700.0, 1.0 / max(sig, 1e-6))), 1), "tone": round(min(0.8 * rh, 0.95), 3)}}


SKY_IMG_N = 1024                                                     # sky image analysis — a 1024 × 512 grid


def sky_dirs(W, H):
    """Pixel directions of an equirectangular image — as in Blender (measured, Eevee and Cycles): the center is +x, the azimuth grows counterclockwise
    (+y is at a quarter of the width), the top is the zenith: az = (0.5 − u)·360°, el = (0.5 − v)·180°."""
    return np.meshgrid((0.5 - (np.arange(W) + 0.5) / W) * 360.0, (0.5 - (np.arange(H) + 0.5) / H) * 180.0)


def read_sky_image(path, N=SKY_IMG_N):
    """A sky image → linear RGB N × N/2 (by averaging): JPG and PNG from sRGB, HDR via OpenCV, EXR via OpenEXR."""
    import cv2
    ext = os.path.splitext(path)[1].lower()
    if ext == ".exr": im = read_rgba(path)[0]
    elif ext == ".hdr": im = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32)
    else:                                                                 # an 8K JPG is decoded straight to a fraction of its size (DCT scaling), not 400 MB in memory
        from PIL import Image
        im = Image.open(path); im.draft("RGB", (N, N // 2)); im = np.asarray(im.convert("RGB"), np.float32) / 255.0
        im = np.where(im <= 0.04045, im / 12.92, ((im + 0.055) / 1.055) ** 2.4).astype(np.float32)
    return cv2.resize(np.ascontiguousarray(im), (N, N // 2), interpolation=cv2.INTER_AREA)


def sky_image_info(path):
    """Analysis of a sky image (cached per file): the illuminance of a horizontal surface from the upper hemisphere E (linear, in image
    units; without a 5° disk around the sun — the node's lamp supplies it), the sun — a compact brightest spot (brightness centroid within 2° of the peak;
    the peak is brighter than the 10–20° ring around it by a factor SUN_RATIO) or None, and whether it is HDR (brightness above 1.5)."""
    import cv2
    st_ = os.stat(path); key = hashlib.sha1(json.dumps([os.path.abspath(path), st_.st_size, int(st_.st_mtime), 1]).encode()).hexdigest()[:10]
    cp = os.path.join(ROOT, "out", "far", "_sky_images", key + ".json")
    if os.path.exists(cp): return json.load(open(cp))
    L = read_sky_image(path); Y = L @ LUM.astype(np.float32); H, W = Y.shape
    AZ, EL = sky_dirs(W, H); a_, e_ = np.radians(AZ), np.radians(EL)
    D = np.stack([np.cos(e_) * np.cos(a_), np.cos(e_) * np.sin(a_), np.sin(e_)], -1)
    p = 16; B = cv2.GaussianBlur(np.concatenate([Y[:, -p:], Y, Y[:, :p]], 1), (0, 0), 2.0)[:, p:-p]   # wrapping around in longitude
    B[(EL < -3.0) | (EL > 88.0)] = 0.0
    iy, ix = np.unravel_index(int(np.argmax(B)), B.shape); c = D @ D[iy, ix]
    core, ring = c > math.cos(math.radians(2.0)), (c < math.cos(math.radians(10.0))) & (c > math.cos(math.radians(20.0))) & (EL > -3.0)
    ratio = float(Y[core].mean() / max(Y[ring].mean(), 1e-9)) if ring.any() else 1.0
    hdr, peak = bool(Y.max() > 1.5), float(Y[core].max())
    sun = None
    if ratio >= SUN_RATIO and (hdr or peak >= 0.85):                    # in a tone-mapped (LDR) image the sun is clipped to white — the peak is nearly white
        v = (D[core] * Y[core][:, None]).sum(0); v /= np.linalg.norm(v)
        sun = [round(float(math.degrees(math.atan2(v[1], v[0]))) % 360.0, 3), round(float(math.degrees(math.asin(np.clip(v[2], -1, 1)))), 3)]
    up = EL > 0
    if sun: up &= c <= math.cos(math.radians(5.0))
    dO = (2 * math.pi / W) * (math.pi / H) * np.cos(e_)
    E = float((Y * np.sin(np.clip(e_, 0, None)) * dO)[up].sum())
    info = {"E": round(E, 6), "sun": sun, "ratio": round(ratio, 3), "peak": round(peak, 4), "hdr": hdr, "path": os.path.abspath(path)}
    os.makedirs(os.path.dirname(cp), exist_ok=True); json.dump(info, open(cp, "w")); return info


SUN_RATIO = 1.35                                                      # the sun in an image is a peak brighter than the 10–20° ring around it: a sunset HDR —
                                                                      # 3.2, a tone-mapped sunset JPG — 1.4, noon — 2.1 (measured)


def sky_image_spec(v, scene_dir):
    """Sky as an image (the `sky_image:` description or a run key): a string is a path, a dict is {path, yaw, ev, sun, drift}; sun is
    image (the luminary is taken from the image, the default), node (the node's own time) or [az, el] on the image (if it was not found, set it by hand);
    a "path:rotation" string — with a rotation. The path is relative to the project root or the scene folder. None — no image sky."""
    if not v: return None
    if isinstance(v, str):
        v = {"path": v.rsplit(":", 1)[0], "yaw": float(v.rsplit(":", 1)[1])} if v.rsplit(":", 1)[-1].lstrip("-").replace(".", "", 1).isdigit() else {"path": v}
    v = dict(v)
    p = next((q for q in (v["path"], os.path.join(ROOT, v["path"]), os.path.join(scene_dir, v["path"])) if os.path.exists(q)), None)
    if not p: raise SystemExit(f"sky_image: no file {v['path']}")
    info = dict(sky_image_info(p)); s_ = v.get("sun", "image")
    if isinstance(s_, (list, tuple)): info["sun"], s_ = [float(s_[0]), float(s_[1])], "image"   # the luminary is set by hand (on the image)
    return {"path": os.path.abspath(p), "yaw": float(v.get("yaw", 0.0)), "ev": float(v.get("ev", 0.0)), "sun": s_,
            "drift": float(v.get("drift", 0.0)), "info": info}


def resolve(spec, scene_dir, over=None):
    """Far-terrain description → plan: everything the light node knows is computed here once; Blender and numpy take the ready numbers."""
    sp = dict(spec or {}); over = over or {}
    SI = sky_image_spec(over.get("sky_image", sp.get("sky_image")), scene_dir)   # an image sky is an alternative to the sky model
    for k in ("time", "weather", "style"):
        if k in over: sp[k] = over[k]
    frame = float(over.get("frame", sp.get("frame", 0.0)))                  # frame (flashes); next, over is the overcast fraction
    W_ = json.load(open(os.path.join(scene_dir, "world.json")))["world"] if os.path.exists(os.path.join(scene_dir, "world.json")) else {}
    t = sp.get("time", {}); t = {"preset": t} if isinstance(t, str) else dict(t)
    sun = dict(sp.get("sun", {}))
    if "preset" in t:                                                    # the time on the command line (over) is stronger than the node's sun elevation; the azimuth —
        el0, side = TIME[t["preset"]]                                    # the scene's, if set (for a node from world.json — always)
        if "time" in over or "el" not in sun: sun["el"] = el0
        sun.setdefault("az", SIDE[side])
    sun.setdefault("az", W_.get("sun", {}).get("azimuth", 220.0)); sun.setdefault("el", W_.get("sun", {}).get("elevation", 30.0))
    if "sun_az" in over: sun["az"] = float(over["sun_az"])
    if "sun_el" in over: sun["el"] = float(over["sun_el"])
    wname = sp.get("weather", "clear"); wt = dict(WEATHER[wname]); wt.update(sp.get("air", {}) if isinstance(sp.get("air"), dict) else {})
    if "visibility_km" in wt: wt["vis_km"] = wt.pop("visibility_km")
    sname = (sp.get("sky", {}) or {}).get("style", sp.get("style", "anime")); st = STYLE[sname]
    moon = dict(sp.get("moon", {}))
    moon.setdefault("az", (sun["az"] + 180.0) % 360.0 if "az" not in moon else moon["az"]); moon.setdefault("el", 35.0)
    for k in ("moon_az", "moon_el"):
        if k in over: moon[k[5:]] = float(over[k])
    if SI and SI["sun"] != "node" and SI["info"]["sun"]:                # light follows the image: the sun lamp (at night — the moon's) is where the luminary is
        az_, el_ = (SI["info"]["sun"][0] + SI["yaw"]) % 360.0, SI["info"]["sun"][1]   # on the image (with its rotation); the time of day — by
        if sun["el"] < -6.0: moon["az"], moon["el"] = az_, el_                # the same, the node's night (−6° and below) stays night
        else: sun["az"], sun["el"] = az_, el_
    S, M = np.array(sun_vec(sun["az"], sun["el"])), np.array(sun_vec(moon["az"], moon["el"]))
    psi = math.degrees(math.acos(float(np.clip(S @ M, -1, 1))))         # the sun–moon angle: 180° is a full moon
    frac = (1 - math.cos(math.radians(psi))) / 2                       # the lit fraction of the disk
    w_night = smooth(-4.0, -12.0, sun["el"])
    s_rgb, k_rgb = table_at(sun["el"])
    over = float(np.clip((wt.get("cover", 0.0) - 0.6) / 0.35, 0, 1))  # overcast: the sky is gray by the CIE rule (the zenith is three times brighter than the horizon)
    m_rgb, mk_rgb = table_at(max(moon["el"], -6.0))
    m_on = moon["el"] > -1.0 and w_night > 0
    nk = K_NIGHT * w_night * (1.0 if moon["el"] > -6 else 0.3)
    e_over = (lum(k_rgb) + lum(s_rgb) * max(math.sin(math.radians(sun["el"])), 0.0) + lum(mk_rgb) * nk   # at night behind the layer — the moon and
              + (lum(m_rgb) * K_NIGHT * frac * w_night * max(math.sin(math.radians(moon["el"])), 0.0) if m_on else 0.0)) * wt.get("over_k", 0.25)
    # its sky, as the sun by day: without them an overcast night is a black dome, shadows drop to zero (the fortress in rain — 0.12, overcast — 0.085)
    grey = np.array([0.92, 0.95, 1.0]); grey = grey / lum(grey)
    s_rgb = s_rgb * wt["sun_k"] * (1.0 if sun["el"] > -1 else 0.0); k_rgb = k_rgb * wt["sky_k"] * (1 - over) + grey * e_over * over
    tint = np.array(st["night_tint"]); tint = tint / lum(tint)          # moonlight takes the night tint, not the warm sun color
    moon_lamp = (lum(m_rgb) * tint * K_NIGHT * frac * w_night * wt["sun_k"]) if m_on else np.zeros(3)
    night_sky = mk_rgb * nk * wt["sky_k"] * (1 - over)                  # the moon's clear sky — there is none behind the layer (the world mixes it with gray by over)
    key = lambda el: math.cos(math.radians(max(0.0, 45.0 - el))) if el > 0 else 0.0   # a surface turned 45° toward the luminary
    e_key = lum(s_rgb) * key(sun["el"]) + 0.85 * lum(k_rgb) + lum(moon_lamp) * key(moon["el"]) + 0.85 * lum(night_sky)
    e_key = max(e_key, K_NIGHT * 0.02)                                 # a moonless night: the exposure does not shoot up to noise
    MOOD_EL = [-12.0, -9.0, -6.0, -3.0, -1.0, 0.0, 6.0, 12.0]
    mood_fire = float(np.interp(sun["el"], MOOD_EL, [-3.0, -3.8, -4.2, -3.8, -2.4, -1.0, -0.4, 0.0]))   # night with fire stays as before (−3)
    mood = mood_moon = float(np.interp(sun["el"], MOOD_EL, [st.get("night_ev", -3.0), -3.8, -4.2, -3.8, -2.4, -1.0, -0.4, 0.0]))   # moon night — by style
    # sunset is darker than noon, twilight and night darker still; in twilight the sky fades faster than the moon comes on (K_NIGHT) — it is
    # deepest there (without the glow in the exposure, twilight came out brighter than sunset)
    ev_c = float(sp.get("ev", 0.0))                                   # scene exposure correction (stops, like the camera's correction dial): a scene
    ev = float(np.clip(math.log2(math.pi / e_key) + mood + st["ev"] + wt.get("ev", 0.0) + ev_c, -9.0, 11.0))   # with its own light
    stars = st["stars"] * smooth(-6.0, -15.0, sun["el"]) * (1 - wt.get("cover", 0.0))
    # the sun at height h — the horizon is lower by √(2h/R), the light comes from the table at the sun's elevation above it (after sunset
    # clouds still glow); the ground bounce from below — albedo × illuminance of a horizontal surface / π
    sun_alt = [(table_at(sun["el"] + dip_deg(h))[0] * wt["sun_k"] if sun["el"] + dip_deg(h) > 0 else np.zeros(3)) for h in ALT_H]
    e_gnd = (s_rgb * max(math.sin(math.radians(sun["el"])), 0.0) + k_rgb + np.asarray(moon_lamp) * max(math.sin(math.radians(moon["el"])), 0.0)
             + night_sky)
    plan = {"version": 1, "scene": os.path.basename(os.path.abspath(scene_dir)), "spec": sp,
            "sun": {"az": sun["az"], "el": sun["el"], "dir": S.round(6).tolist(), "lamp_rgb": s_rgb.round(5).tolist(), "angle_deg": 0.545},
            "moon": {"az": moon["az"], "el": moon["el"], "dir": M.round(6).tolist(), "phase_deg": round(psi, 2), "lit_frac": round(frac, 4),
                     "size_deg": 0.52 * float(moon.get("size", st["moon_size"])), "lamp_rgb": np.asarray(moon_lamp).round(7).tolist(),
                     "on": bool(m_on)},
            "sky": {"model": "MULTIPLE_SCATTERING", "air": 1.0, "aerosol": wt["aerosol"], "ozone": 1.0, "sat": st["sky_sat"] * wt["sat"], "k": wt["sky_k"],
                    "over": round(over, 4), "over_L": round(9 * e_over / (7 * math.pi), 6), "over_rgb": grey.round(5).tolist(),
                    "tint": list(wt.get("tint", (1.0, 1.0, 1.0))), "w_night": round(w_night, 4), "k_night": K_NIGHT,
                    "night_tint": list(st["night_tint"]), "stars": round(stars, 4), "night_shift": list(st["night_shift"]), "night_fill": st.get("night_fill", -3.5),
                    "look": st["look"], "bloom": st["bloom"], "shafts": st["shafts"]},
            "light": {"e_key": round(e_key, 6), "ev": round(ev, 3), "view": "AgX", "style": sname, "weather": wname,
                      "sky_e_rgb": k_rgb.round(5).tolist(), "night_sky_e_rgb": night_sky.round(7).tolist(),
                      "sun_alt": {"h_max": float(ALT_H[-1]), "rgb": np.round(sun_alt, 5).tolist()},
                      "ground_up_rgb": (GROUND_ALB * e_gnd / math.pi).round(7).tolist()},
            "air": {"vis_km": wt["vis_km"]}}
    if SI:                                                               # image sky: the camera gets the image × gain (the illuminance from its sky =
        e_sky = lum(k_rgb) + lum(night_sky)                              # the node's sky: haze, night and exposure are shared), the lighting uses the node's model
        plan["sky"]["image"] = {"path": SI["path"], "yaw": SI["yaw"], "drift": SI["drift"], "hdr": SI["info"]["hdr"],
                                "gain": round(e_sky / max(SI["info"]["E"], 1e-12) * 2.0 ** SI["ev"], 9), "sun": SI["info"]["sun"],
                                "sun_used": SI["sun"] != "node" and bool(SI["info"]["sun"])}
    # --- hazes: air — density from visibility (Koschmieder: σ = 3.912 / V), the color is the sky at the horizon by azimuth, not
    # added to the sky (this haze is already in the sky model); ground fog or smoke — its own color, scattering toward the sun and moon, toward the sky at the horizon — yes
    z0 = ground_level(scene_dir)
    layers = [{"name": "air", "a": 3.912 / (wt["vis_km"] * 1000.0), "H": H_AIR, "z0": z0, "on_sky": False, "src": "horizon"}]
    gf = dict(wt.get("ground", {})); gf.update(sp.get("ground_fog", {}) or {})
    if gf:
        vis = gf.get("visibility_km", gf.get("vis_km", 1.0))
        layers.append({"name": "ground", "a": 3.912 / (vis * 1000.0), "H": float(gf.get("height", 60.0)), "z0": float(gf.get("base", z0)),
                       "on_sky": True, "src": "light", "albedo": list(gf.get("albedo", gf.get("color", (0.9, 0.9, 0.9)))), "g": float(gf.get("g", 0.6))})
    plan["fog"] = {"layers": layers}
    plan["envelope"] = env = envelope(scene_dir)                          # camera envelope and far terrain (cached by description)
    decks, cu = clouds_spec(sp, wname) if not SI or "clouds" in sp else ([], None)   # clouds — layers and cumulus (with an image sky
    # the clouds are on it; the node's clouds only if explicitly described)
    cdir = os.path.join(ROOT, "out", "far", plan["scene"])
    plan["clouds"] = {"decks": decks, "kappa": KAPPA["cloud"],
                      "cumulus": (dict(cu, npz=cumulus_build(cu, env, z0, int(sp.get("seed", 7)), cdir)) if cu else None)}
    if cu: plan["clouds"]["cumulus"]["light"] = puff_light(plan["clouds"]["cumulus"]["npz"], plan, KAPPA["cloud"], cdir)
    T = None
    if sp.get("terrain"):
        T = terrain_spec(sp["terrain"], scene_dir, env)
        if T: plan["terrain"] = {**T, **terrain_build(T, os.path.join(ROOT, "out", "far", plan["scene"]))}
    cnpz, cfires = city_build(sp.get("city"), scene_dir, T, env, int(sp.get("seed", 7)), os.path.join(ROOT, "out", "far", plan["scene"]))
    plan["city"] = {"npz": cnpz, "kits": [dict(CITY_KIT.get(c.get("kit", "hive"), CITY_KIT["hive"]), windows=float(c.get("windows", 0.35)))
                                          for c in (sp.get("city") if isinstance(sp.get("city"), list) else [sp.get("city")] if sp.get("city") else [])]}
    W = wind_model.resolve(sp.get("wind"), int(sp.get("seed", 7)))          # scene wind (core/light/wind.py): a description, a Beaufort number or the old vector
    plan["wind"] = W.to_dict(); wind = W.legacy_vector()                    # burning sites — records, smoke columns, fire, glow (the bend is given by a vector)
    fsp = dict(sp.get("fires") or {}); fsp["list"] = list(fsp.get("list", [])) + cfires   # burning houses of the city are fires too
    F, fnpz = fires_build(fsp if fsp.get("list") or fsp.get("n") else None, scene_dir, T, env, wind, int(sp.get("seed", 7)), os.path.join(ROOT, "out", "far", plan["scene"]))
    light_k = (1.0 + (NIGHT_BOOST - 1.0) * w_night) / NIGHT_BOOST           # fires (glow, lamps): at night as before, by day 800 times weaker
    plan["fires"] = {"list": F, "npz": fnpz, "wind": wind, "glow": float((sp.get("fires") or {}).get("glow", 1.0)), "kappa": KAPPA["smoke"],
                     "light_k": round(light_k, 6)}
    if fnpz: plan["fires"]["light"] = puff_light(fnpz, plan, KAPPA["smoke"], cdir)
    plan["city"]["win_k"] = round(win_k(sun["el"], w_night), 5)            # windows are light too (glow, cloud undersides)
    S_ = glow_sources(F, plan["fires"], cnpz, plan["city"]["kits"], plan["city"]["win_k"])
    plan["glow"] = {"sources": S_}
    if S_:                                                                   # the exposure knows about the fires and the city: at night the frame is lit by fire
        c0 = np.array([*env["center"], float(np.mean(env["z"]))]); e_glow = 0.0
        for f in S_:
            v = np.array(f["at"]) - c0; d = float(np.linalg.norm(v)); tau = 0.0
            for Ly in layers:
                rho = Ly["a"] * math.exp(float(np.clip(-(c0[2] - Ly["z0"]) / Ly["H"], -60, 3))); k = max(v[2] / max(d, 1) * d / Ly["H"], -40.0)
                tau += rho * d * (1.0 if abs(k) < 1e-5 else -math.expm1(-k) / k)
            e_glow += f["I"] / max(d * d, 1.0) * (1 - math.exp(-tau))
        e_all = e_key + 0.5 * e_glow
        g_ = 0.5 * e_glow / max(e_all, 1e-9); mood = mood * (1 - g_) + mood_fire * g_   # fire is a practical light: the frame is exposed by
        # it, and the night around stays dark (the old −3); the moon night is readable (night_ev). Without the mix a mountain pass with a burning city
        # turned into daytime (brightness 0.43–0.59), while night places were indecently dark. The night shift goes by the same fraction toward the full one: the glow
        # in the haze turns fully blue (at strength 0.5 the pass and the assault drowned in a ginger veil), the colors of places under the moon stay
        dec_, k_s = plan["sky"]["night_shift"]; plan["sky"]["night_shift"] = [dec_, round(k_s * (1 - g_) + g_, 3)]
        plan["light"].update(e_glow=round(e_glow, 6), ev=round(float(np.clip(math.log2(math.pi / e_all) + mood + st["ev"] + wt.get("ev", 0.0) + ev_c, -9.0, 11.0)), 3))
    plan["light"]["fire_k"] = round(2.0 ** (mood_fire - mood), 4)          # the scene's fires (bl_color lamps: burning objects, fire=) — under
    # the fire exposure: on screen as with the old night (−3), the moonlit surroundings are readable (night_ev); without the correction fire barrels
    # washed out a 5 m circle (chaos_squad); by day — 1
    plan["light"]["meter"] = {"mood_moon": round(mood_moon, 4), "mood_fire": round(mood_fire, 4),   # parts of the exposure formula: the frame
                              "off": round(st["ev"] + wt.get("ev", 0.0) + ev_c, 4)}                   # computes it from the visible glow (meter_ev)
    # frame flashes (explosion, lightning) — sources of glow and of light from below; the exposure does not know them (the camera does not adapt within a frame)
    plan["flash_list"] = flash_list(sp, light_k, z0, decks, scene_dir, T); plan["frame"] = frame
    plan["flashes"] = flashes_at(plan, frame)
    S_ = plan["glow"]["sources"] = S_ + [dict(at=f["at"], I=f["I"], rgb=f["rgb"], flash=True) for f in plan["flashes"]]
    if S_:
        c0 = np.array([*env["center"], float(np.mean(env["z"]))])
        plan["fires"]["glow_map"] = glow_map(plan, c0, os.path.join(ROOT, "out", "far", plan["scene"]))
        for i, d_ in enumerate(decks):                                       # fire and city light on the underside of the layers and puffs
            d_["under"] = under_map(plan, z0 + d_["alt_km"] * 1000.0, cdir)
        for key_ in (("clouds", "cumulus"), ("fires",)):
            o_ = plan[key_[0]] if len(key_) == 1 else plan[key_[0]].get(key_[1])
            if o_ and o_.get("npz"): o_["under"] = under_puffs(o_["npz"], plan, cdir, 300.0 if key_[0] == "fires" else 1.0)
    plan["hash"] = hashlib.sha1(json.dumps({k: plan[k] for k in ("sun", "moon", "sky", "light", "air", "fog")}, sort_keys=True).encode()).hexdigest()[:10]
    if T:                                                                    # whether the far terrain is visible: haze transmittance from the top of the envelope to the mountain crest
        zc = float(env["z"][1]); d = T["half"] + T["ramp_km"] * 1000.0
        plan["far_seen"] = {"crest_T": round(transmit(layers, zc, d, T["z0"] + T["height"] - zc), 4), "d_km": round(d / 1000.0, 1)}
    return plan


def transmit(layers, zc, d, dz):
    """Haze transmittance along a straight line from height zc over d m horizontally and dz m up (the same formula as for the frame haze)."""
    tau = 0.0
    for L_ in layers:
        k = max(dz / L_["H"], -40.0); f = (1.0 - math.exp(-k)) / k if abs(k) > 1e-6 else 1.0
        tau += L_["a"] * math.exp(-(zc - L_["z0"]) / L_["H"]) * math.hypot(d, dz) * f
    return math.exp(-tau)


def envelope(scene_dir):
    """Camera envelope: the scene's cameras (case A) and the site area (case B: the square of the near area × heights of 1.5–50 m);
    a fly-around (C) is cameras too, only farther. The center is the middle of the extent, the radius is the farthest point from the center; the far-terrain detail is set by it."""
    pts = []
    cp = os.path.join(scene_dir, "cameras.json")
    for c in (json.load(open(cp)) if os.path.exists(cp) else []): pts.append(np.array(c["matrix_world"])[:3, 3])
    lp = os.path.join(scene_dir, "location.json"); z0 = ground_level(scene_dir)
    if os.path.exists(lp):
        L = json.load(open(lp)); h = float(L["size"]) / 2; cx, cy = L.get("center", [0.0, 0.0])
        pts += [np.array([cx + sx * h, cy + sy * h, z0 + dz]) for sx in (-1, 1) for sy in (-1, 1) for dz in (1.5, 50.0)]
    P = np.array(pts) if pts else np.zeros((1, 3))
    c = P[:, :2].mean(0) if not os.path.exists(lp) else np.array(json.load(open(lp)).get("center", [0.0, 0.0]), float)
    R = float(np.max(np.hypot(P[:, 0] - c[0], P[:, 1] - c[1])))
    return {"center": c.round(3).tolist(), "radius": round(R, 1), "z": [round(float(P[:, 2].min()), 1), round(float(P[:, 2].max()), 1)],
            "points": P.round(2).tolist()[:64]}


BIOME = {   # bottom, middle (rock), snow — screen color; snow — a fraction of the mountain height, rock steepness — slope
    "temperate": dict(low=(0.33, 0.38, 0.22), mid=(0.45, 0.43, 0.40), snow=(0.93, 0.95, 0.98), snowline=0.62, rock=0.62),
    "alpine": dict(low=(0.30, 0.36, 0.22), mid=(0.47, 0.46, 0.45), snow=(0.94, 0.96, 0.99), snowline=0.45, rock=0.55),
    "desert": dict(low=(0.76, 0.63, 0.45), mid=(0.62, 0.47, 0.35), snow=(0.84, 0.74, 0.60), snowline=2.0, rock=0.50),
    "volcanic": dict(low=(0.24, 0.23, 0.23), mid=(0.33, 0.30, 0.29), snow=(0.86, 0.87, 0.89), snowline=0.80, rock=0.45),
    "wasteland": dict(low=(0.45, 0.40, 0.32), mid=(0.50, 0.45, 0.40), snow=(0.80, 0.80, 0.82), snowline=0.90, rock=0.55),
    "ice": dict(low=(0.86, 0.90, 0.95), mid=(0.62, 0.66, 0.72), snow=(0.95, 0.97, 1.0), snowline=0.0, rock=0.70)}
GROUND_BIOME = {"grass": "temperate", "soil": "temperate", "sand": "desert", "snow": "ice", "ice": "ice", "ash": "volcanic",
                "mud": "wasteland", "rock": "alpine"}
R_EARTH = 6371000.0 * 7 / 6                                          # Earth radius with refraction (horizon dip)


def battle_ground(scene_dir):
    """The ground of a battle scene (components/battle/battle.py: ground {half_x, y0, y1}): from the scene's battle.yaml, otherwise from the build referenced by
    source.json (battle_job.json next to the .blend) — views of the same scene (fly-around, 4K) have no battle.yaml of their own."""
    bp = os.path.join(scene_dir, "battle.yaml")
    if os.path.exists(bp):
        import yaml
        return (yaml.safe_load(open(bp)) or {}).get("ground") or {}
    sp = os.path.join(scene_dir, "source.json")
    bj = os.path.join(os.path.dirname(json.load(open(sp)).get("blend", "")), "battle_job.json") if os.path.exists(sp) else ""
    return (json.load(open(bj))["spec"].get("ground") or {}) if bj and os.path.exists(bj) else {}


def terrain_spec(spec, scene_dir, env):
    """Far terrain from the description (terrain:) and the site: the inner edge is the square of the site's mountain belt (location.json far.size) or
    the scene area; the height at the edge is the far-ground level; the biome comes from the site."""
    t = dict(spec or {})
    if t.get("kind", "ridged") == "none": return None
    lp = os.path.join(scene_dir, "location.json"); L = json.load(open(lp)) if os.path.exists(lp) else {}
    half = float(L.get("far", {}).get("size", 0)) / 2 if L else 0.0
    center = t.get("center") or env["center"]
    g = {} if L else battle_ground(scene_dir)                           # a large battle scene (components/battle/battle.py): the edge is its ground, otherwise a gap to the mountains
    if g:
        if {"half_x", "y0", "y1"} <= set(g):
            center = t.get("center") or [0.0, (g["y0"] + g["y1"]) / 2.0]; half = max(g["half_x"], (g["y1"] - g["y0"]) / 2.0)
    half = float(t.get("from_km", 0)) * 1000 or half or max(env["radius"] * 1.2, 800.0)
    biome = t.get("biome") or L.get("biome") or GROUND_BIOME.get(L.get("ground", "grass"), "temperate")
    real = {}
    fh = (L.get("far_heights") or {}).get("file") if L else None
    if fh and os.path.exists(os.path.join(scene_dir, fh)):            # the site's real mountains: the far terrain is the same window (site, far_heights.npz)
        p_ = os.path.join(scene_dir, fh); st_ = os.stat(p_); zz = np.load(p_)["z"]
        from core.surface.ground import snowline_of
        sl_ = snowline_of(L, biome)
        real = {"heights": p_, "heights_sig": f"{st_.st_size}:{int(st_.st_mtime)}", "belt_dir": scene_dir,
                "height": float(np.nanpercentile(zz, 99)), "snow_z": (ground_level(scene_dir) + sl_) if sl_ is not None else None}
    return {**{"kind": t.get("kind", "ridged"), "height": float(t.get("height", 1600.0)), "scale_km": float(t.get("scale_km", 7.0)),
            "ramp_km": float(t.get("ramp_km", 6.0)), "skirt_km": float(t.get("skirt_km", 160.0)), "to_km": float(t.get("to_km", 38.0)), "half": half, "center": [float(v) for v in center],
            "z0": ground_level(scene_dir), "biome": biome if biome in BIOME else "temperate", "seed": int(t.get("seed", 7)), "env_r": env["radius"],
            "sectors": t.get("sectors", []), "detail_m": float(t.get("detail_m", 8.0)), "k_px": float(t.get("k_px", 6.0))}, **real}


_REAL = {}


def _real(T):
    """The site's far terrain (far_heights.npz: heights above the valley floor, NaN — no data) and the belt edge (core.surface.ground) — loaded once."""
    if T["heights"] not in _REAL:
        from core.surface.ground import Ground
        z = np.load(T["heights"]); _REAL[T["heights"]] = (z["z"].astype(np.float64), *map(float, z["o"]), Ground(T["belt_dir"]))
    return _REAL[T["heights"]]


def terrain_height(T, x, y):
    """Height of the far terrain at a point (m): the rise from the belt edge, terrain.ridged crests, fbm fine detail, sectors, Earth curvature.
    The site's real mountains (T.heights): the far terrain is the same window, at the belt edge exactly its heights (a 400 m transition), beyond the map — noise."""
    if T.get("heights"): return _terrain_real(T, x, y)
    return _terrain_noise(T, x, y)


def _terrain_real(T, x, y):
    from core.surface.noise import smoothstep as ss
    Z, x0, y0, res, G = _real(T); cx, cy = T["center"]; r = np.hypot(x - cx, y - cy)
    fx, fy = (x - x0) / res, (y - y0) / res; j, i = np.floor(fx).astype(int), np.floor(fy).astype(int); tx, ty = fx - j, fy - i
    inside = (i >= 0) & (j >= 0) & (i < Z.shape[0] - 1) & (j < Z.shape[1] - 1); i, j = np.clip(i, 0, Z.shape[0] - 2), np.clip(j, 0, Z.shape[1] - 2)
    h = Z[i, j] * (1 - tx) * (1 - ty) + Z[i, j + 1] * tx * (1 - ty) + Z[i + 1, j] * (1 - tx) * ty + Z[i + 1, j + 1] * tx * ty
    edge = np.minimum(np.minimum(fx, Z.shape[1] - 1 - fx), np.minimum(fy, Z.shape[0] - 1 - fy)) * res   # to the map edge, m
    w = np.where(inside & np.isfinite(h), ss(0.0, 3000.0, edge), 0.0)   # toward the map edge and where there is no data — noise of the same height
    hn = _terrain_noise(T, x, y) - T["z0"] + r * r / (2 * R_EARTH)
    h = np.where(w > 0, np.nan_to_num(h) * w + hn * (1 - w), hn)
    d_in = np.maximum(np.abs(x - cx), np.abs(y - cy)) - T["half"]           # at the belt edge — exactly its heights (the meshes meet without a gap)
    hb = G.height(np.clip(x, cx - T["half"], cx + T["half"]), np.clip(y, cy - T["half"], cy + T["half"])) - T["z0"]
    h = hb + (h - hb) * ss(0.0, 400.0, d_in)
    return T["z0"] + h - r * r / (2 * R_EARTH)


def _terrain_noise(T, x, y):
    from core.surface.noise import ridged, fbm, smoothstep as ss
    cx, cy = T["center"]; dx, dy = x - cx, y - cy; r = np.hypot(dx, dy)
    d_in = np.maximum(np.abs(dx), np.abs(dy)) - T["half"]                     # from the edge of the belt square
    s = 1.0 / (T["scale_km"] * 1000.0)
    wx, wy = fbm(x * s * 0.5, y * s * 0.5, 3, T["seed"] + 5) * 0.35, fbm(x * s * 0.5 + 9.1, y * s * 0.5 - 4.3, 3, T["seed"] + 6) * 0.35
    base = ridged(x * s + wx, y * s + wy, 6, T["seed"] + 101) if T["kind"] == "ridged" else (fbm(x * s + wx, y * s + wy, 6, T["seed"] + 101) * 0.5 + 0.5)
    if T["kind"] == "mesa": base = np.clip((base - 0.35) * 3.0, 0, 1) ** 0.6
    k = np.ones_like(x)
    az = np.degrees(np.arctan2(dy, dx)) % 360
    for sc in T["sectors"]:                                                   # sectors: height by azimuth, soft edges ±15°
        a0, a1 = sc["az"]; mid = (a0 + ((a1 - a0) % 360) / 2) % 360; half_w = ((a1 - a0) % 360) / 2
        dd = np.abs((az - mid + 180) % 360 - 180)
        w = ss(half_w + 15, half_w - 15, dd); k = k * (1 - w) + w * float(sc.get("height", T["height"])) / T["height"]
    h = T["height"] * k * base * ss(0.0, T["ramp_km"] * 1000.0, d_in) + 25.0 * fbm(x / 400.0, y / 400.0, 4, T["seed"] + 13) * ss(0, 600, d_in)
    return T["z0"] + np.maximum(h, 0.0) - r * r / (2 * R_EARTH)


def _data_map(H, r, path):
    """Far-terrain data map for the mountain belts of the far shader (core/surface/bl_terrain_vista: R crests, G thalwegs, B foothills, A height above the
    valley floor) — like the site's relief_far.png, on the top-down color grid (row 0 — north). 1024² takes 2.4 s (drainage analysis)."""
    import cv2
    from PIL import Image
    from scipy.ndimage import minimum_filter, maximum_filter
    from core.surface import terrain_analysis as TA
    blur = lambda A, s_: cv2.GaussianBlur(np.asarray(A, np.float32), (0, 0), s_)
    gy, gx = np.gradient(H, r); sl = np.degrees(np.arctan(np.hypot(gx, gy)))
    lap = cv2.Laplacian(blur(H, 1.5), cv2.CV_32F) / (r * r); conv = np.clip(-lap / (2.5 * float(np.std(lap)) + 1e-6), 0, 1)
    F = TA.fill(H.astype(np.float64)); A = TA.accumulate(F, TA.receivers(F, r), r); thal = np.clip((np.log10(np.maximum(A, 1.0)) - 2.3) / 1.6, 0, 1)
    cliff = (sl > 45).astype(np.float32); foot = np.clip(blur(cliff, 2.5) * 2.5 * (1 - cliff), 0, 1)
    w = int(600 / r) | 1; lo, hi = blur(minimum_filter(H, size=w), 20 / r), blur(maximum_filter(H, size=w), 20 / r)
    hrel = np.clip((H - lo) / np.maximum(hi - lo, 30.0), 0, 1)
    D = np.stack([blur(conv, 1.0), blur(thal, 0.8), blur(foot, 1.0), blur(hrel, 1.0)], -1)
    Image.fromarray((np.clip(D, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA").save(path)


def terrain_build(T, cache_dir):
    """The far-terrain mesh (rays from the edge of the belt square to to_km, with a geometric step: the cells are almost square, at the edge detail_m) and two
    top-down color textures (near ±6 km and far) → npz + png in a cache keyed by the description hash. Returns paths and numbers."""
    key = hashlib.sha1(json.dumps(T, sort_keys=True).encode()).hexdigest()[:10]
    out = os.path.join(cache_dir, f"terrain_{key}"); os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "mesh.npz")): return {"dir": out, "key": key, **json.load(open(os.path.join(out, "info.json")))}
    from PIL import Image
    cx, cy = T["center"]; half, R = T["half"], T["to_km"] * 1000.0
    det = max(T["detail_m"], T["k_px"] * max(half - T.get("env_r", 0.0), 0.0) / 1280.0)   # at the edge — k pixels of a 2K frame at the edge's distance from the envelope
    nphi = int(math.ceil(2 * math.pi * half * 1.2 / det / 8) * 8)
    dphi = 2 * math.pi / nphi; nr = int(math.ceil(math.log(R / half) / dphi)) + 1
    phi = np.arange(nphi) * dphi; c_, s_ = np.cos(phi), np.sin(phi)
    rin = half / np.maximum(np.abs(c_), np.abs(s_))                          # the edge of the belt square on this ray
    t = np.linspace(0, 1, nr)[:, None]
    rr = rin[None, :] * (R / rin[None, :]) ** t                              # geometrically from the edge to to_km
    X, Y = cx + rr * c_[None, :], cy + rr * s_[None, :]
    Z = terrain_height(T, X, Y)
    if not T.get("heights"): Z[0] = T["z0"]                                  # the first ring is exactly the belt-edge level (the real ones — its heights)
    idx = np.arange(nr * nphi).reshape(nr, nphi); nxt = np.roll(idx, -1, axis=1)
    quads = np.stack([idx[:-1], nxt[:-1], nxt[1:], idx[1:]], -1).reshape(-1, 4)
    split = np.argmax(rr.max(1) > 6000.0) or nr - 1                          # the ring that divides the near and far textures
    V = np.stack([X, Y, Z], -1).reshape(-1, 3).astype(np.float32)
    parts = {}
    for name, r0, r1, H in (("near", 0, split, 6000.0 + half), ("far", split, nr - 1, R)):
        q = quads[(np.arange(nr - 1).repeat(nphi) >= r0) & (np.arange(nr - 1).repeat(nphi) < r1)]
        used = np.unique(q); rm = -np.ones(len(V), np.int64); rm[used] = np.arange(len(used))
        v = V[used]; uv = np.stack([(v[:, 0] - cx) / (2 * H) + 0.5, (v[:, 1] - cy) / (2 * H) + 0.5], -1).astype(np.float32)
        parts[name] = (v, rm[q].astype(np.int32), uv, H)
    B_ = BIOME[T["biome"]]
    for name, (_, _, _, H) in parts.items():                                 # top-down color: height, slope, snow, patches
        n = 2048 if name == "near" else 1024                                 # near ±7.5 km — ~7 m per pixel (measured: 4096 takes 76 s and 3.3 GB)
        g = ((np.arange(n) + 0.5) / n * 2 * H - H).astype(np.float32)
        GX, GY = np.meshgrid(np.float32(cx) + g, np.float32(cy) - g)          # row 0 — north (v = 1)
        GZ = terrain_height(T, GX, GY) - T["z0"] + (GX - cx) ** 2 / (2 * R_EARTH) + (GY - cy) ** 2 / (2 * R_EARTH)
        st = 2 * H / n; gy_, gx_ = np.gradient(GZ, st); slope = np.hypot(gx_, gy_)
        from core.surface.noise import fbm
        nz = fbm(GX / 900.0, GY / 900.0, 4, T["seed"] + 77) * 0.5
        hf = GZ / max(T["height"], 1.0)
        col = np.array(B_["low"])[None, None] * (1 - np.clip(hf * 1.6, 0, 1))[..., None] + np.array(B_["mid"])[None, None] * np.clip(hf * 1.6, 0, 1)[..., None]
        rock = np.clip((slope - B_["rock"] * 0.6) / 0.4, 0, 1)[..., None]; col = col * (1 - rock) + np.array(B_["mid"]) * rock
        snow = np.clip((hf - B_["snowline"] + 0.12 * nz) / 0.08, 0, 1) * np.clip(1.4 - slope, 0, 1)
        col = col * (1 - snow[..., None]) + np.array(B_["snow"]) * snow[..., None]
        col = col * (1 + 0.12 * nz[..., None])
        Image.fromarray((np.clip(col, 0, 1) * 255 + 0.5).astype(np.uint8)).save(os.path.join(out, f"{name}.png"))
        if name == "near" and T.get("heights") and T.get("snow_z") is not None:   # the site's mountains: the data map — height belts for the far shader
            _data_map(GZ[::2, ::2], st * 2, os.path.join(out, "near_data.png"))
    if T.get("skirt_km", 0) > T["to_km"]:                                   # a skirt out to the horizon of a high camera: coarse, 15 m lower, overlapping
        from core.surface.noise import ridged
        Rs = T["skirt_km"] * 1000.0; nps, nrs = 512, 48
        ph = np.arange(nps) * 2 * math.pi / nps; tt = np.linspace(0, 1, nrs)[:, None]
        rs = (R * 0.95) * (Rs / (R * 0.95)) ** tt + 0 * ph[None, :]
        SX, SY = cx + rs * np.cos(ph)[None, :], cy + rs * np.sin(ph)[None, :]
        sz = T["z0"] + 0.55 * T["height"] * ridged(SX / 25000.0, SY / 25000.0, 4, T["seed"] + 211) - 15.0 - (rs ** 2) / (2 * R_EARTH)
        ii = np.arange(nrs * nps).reshape(nrs, nps); nx_ = np.roll(ii, -1, axis=1)
        qs = np.stack([ii[:-1], nx_[:-1], nx_[1:], ii[1:]], -1).reshape(-1, 4).astype(np.int32)
        vs = np.stack([SX, SY, sz], -1).reshape(-1, 3).astype(np.float32)
        uvs = np.stack([(vs[:, 0] - cx) / (2 * Rs) + 0.5, (vs[:, 1] - cy) / (2 * Rs) + 0.5], -1).astype(np.float32)
        parts["skirt"] = (vs, qs, uvs, Rs)
        g = (np.arange(256) + 0.5) / 256 * 2 * Rs - Rs; GX, GY = np.meshgrid(cx + g, cy - g)
        hf = np.clip(0.55 * ridged(GX / 25000.0, GY / 25000.0, 4, T["seed"] + 211), 0, 1)
        col = np.array(B_["low"]) * (1 - hf[..., None]) + np.array(B_["mid"]) * hf[..., None]
        Image.fromarray((np.clip(col, 0, 1) * 255 + 0.5).astype(np.uint8)).save(os.path.join(out, "skirt.png"))
    np.savez_compressed(os.path.join(out, "mesh.npz"), **{f"{k}_{a}": parts[k][i] for k in parts for i, a in enumerate(("v", "f", "uv"))})
    info = {"parts": list(parts), "verts": int(sum(len(p[0]) for p in parts.values())), "faces": int(sum(len(p[1]) for p in parts.values())),
            "nphi": nphi, "nr": nr, "max_m": round(float(Z.max() - T["z0"]), 1)}
    json.dump(info, open(os.path.join(out, "info.json"), "w"))
    return {"dir": out, "key": key, **info}


CLOUDS = {   # clouds by weather: layers (height km, coverage, patch scale km, density) and cumulus (coverage, cloud size m, base height km)
    "clear": dict(decks=[], cumulus=dict(cover=0.10, size=900.0, base_km=1.4)),
    "cumulus": dict(decks=[dict(alt_km=9.0, cover=0.25, scale_km=14.0, density=0.35)], cumulus=dict(cover=0.38, size=1300.0, base_km=1.3)),
    "haze": dict(decks=[dict(alt_km=8.5, cover=0.35, scale_km=18.0, density=0.3)], cumulus=None),
    "overcast": dict(decks=[dict(alt_km=1.6, cover=0.97, scale_km=8.0, density=2.6)], cumulus=None),
    "storm": dict(decks=[dict(alt_km=1.1, cover=1.0, scale_km=6.0, density=4.0)], cumulus=None),
    "rain": dict(decks=[dict(alt_km=1.2, cover=1.0, scale_km=7.0, density=3.2)], cumulus=None),   # nimbostratus: low, overcast throughout
    "war_smoke": dict(decks=[], cumulus=None),
    "mist": dict(decks=[dict(alt_km=7.5, cover=0.3, scale_km=16.0, density=0.3)], cumulus=None)}


def clouds_spec(sp, wname):
    """The plan's clouds: from the weather or from the description `clouds: [{kind: deck|cumulus, …}]`."""
    base = CLOUDS.get(wname, CLOUDS["clear"]); decks = [dict(d) for d in base["decks"]]; cu = dict(base["cumulus"]) if base["cumulus"] else None
    if "clouds" in sp:
        decks = [dict(alt_km=c.get("alt_km", 2.0), cover=c.get("cover", 0.5), scale_km=c.get("scale_km", 10.0), density=c.get("density", 1.0))
                 for c in sp["clouds"] if c.get("kind", "deck") == "deck"]
        cs = [c for c in sp["clouds"] if c.get("kind") == "cumulus"]
        cu = dict(cover=cs[0].get("cover", 0.3), size=cs[0].get("size_m", 1200.0), base_km=cs[0].get("alt_km", 1.3)) if cs else None
    return decks, cu


def cumulus_build(cu, env, z0, seed, cache_dir):
    """Cumulus (Wang 2004: a cloud is a set of puff sprites): centers on a 3 km grid in the 2–40 km ring where the coarse coverage noise is above the
    threshold; a cloud has 8–16 puffs in a dome; clouds closer than 3 of their own sizes to the envelope points are dropped (no wall in front of the camera)."""
    key = hashlib.sha1(json.dumps([cu, env["center"], env["radius"], z0, seed, 2], sort_keys=True).encode()).hexdigest()[:10]
    os.makedirs(cache_dir, exist_ok=True); path = os.path.join(cache_dir, f"cumulus_{key}.npz")
    if os.path.exists(path): return path
    from core.surface.noise import fbm
    rng = np.random.default_rng(seed); cx, cy = env["center"]; step = 3000.0
    g = np.arange(-40000, 40001, step); GX, GY = np.meshgrid(g, g); GX = GX.ravel() + cx + rng.uniform(-0.45, 0.45, GX.size) * step
    GY = GY.ravel() + cy + rng.uniform(-0.45, 0.45, GY.size) * step
    r = np.hypot(GX - cx, GY - cy); n = fbm(GX / 15000.0, GY / 15000.0, 3, seed + 41) * 0.5 + 0.5
    keep = (r > 2000) & (r < 40000) & (n > 1 - cu["cover"] * 1.4) & (rng.random(GX.size) < 0.85)
    E = np.array(env["points"])[:, :2]
    P_, R_, C_, B_, T_ = [], [], [], [], []
    for x, y in zip(GX[keep], GY[keep]):
        W = cu["size"] * rng.uniform(0.6, 1.4); Hc = W * rng.uniform(0.45, 0.9); zb = z0 + cu["base_km"] * 1000.0 + rng.uniform(-150, 150)
        if np.min(np.hypot(E[:, 0] - x, E[:, 1] - y)) < 3 * W: continue
        ci = (C_[-1] + 1) if C_ else 0                                    # cloud index: light through neighbors — within it
        for k in range(int(rng.integers(8, 17))):                           # puffs: wider at the bottom, smaller at the top
            u = rng.random(); a = rng.uniform(0, 2 * math.pi); rr = W * 0.5 * math.sqrt(rng.random()) * (1 - 0.6 * u)
            P_.append((x + rr * math.cos(a), y + rr * math.sin(a), zb + Hc * u * 0.85)); R_.append(W * rng.uniform(0.22, 0.36) * (1 - 0.45 * u))
            C_.append(ci); B_.append(zb); T_.append(zb + Hc)
    np.savez_compressed(path, pos=np.array(P_, np.float32).reshape(-1, 3), rad=np.array(R_, np.float32), base=np.array(B_, np.float32),
                        top=np.array(T_, np.float32), cid=np.array(C_, np.int32))
    return path


def surface_z(scene_dir, T, x, y):
    """Ground height at a point: the site area and belt (ground.Ground), beyond that the plan's far terrain, otherwise the far level."""
    z0 = ground_level(scene_dir)
    if T is not None and max(abs(x - T["center"][0]), abs(y - T["center"][1])) > T["half"]:
        return float(terrain_height(T, np.array([x]), np.array([y]))[0])
    if os.path.exists(os.path.join(scene_dir, "relief.npz")):
        if scene_dir not in _GROUND:
            from core.surface.ground import Ground
            _GROUND[scene_dir] = Ground(scene_dir)
        return float(_GROUND[scene_dir].height(x, y))
    return z0


_GROUND = {}


def surface_zs(scene_dir, T, X, Y):
    """Ground heights for many points at once (see surface_z): the far terrain in one call, the site area through Ground."""
    Z = np.full(np.shape(X), ground_level(scene_dir), float)
    far_ = np.zeros(np.shape(X), bool) if T is None else np.maximum(np.abs(X - T["center"][0]), np.abs(Y - T["center"][1])) > T["half"]
    if far_.any(): Z[far_] = terrain_height(T, X[far_], Y[far_])
    if (~far_).any() and os.path.exists(os.path.join(scene_dir, "relief.npz")):
        if scene_dir not in _GROUND:
            from core.surface.ground import Ground
            _GROUND[scene_dir] = Ground(scene_dir)
        Z[~far_] = [float(_GROUND[scene_dir].height(x, y)) for x, y in zip(X[~far_], Y[~far_])]
    return Z


def fires_build(spec, scene_dir, T, env, wind, seed, cache_dir):
    """Burning sites: records — position, power, size, smoke; from the description `fires: {list: [{at: [x, y], power, size_m,
    smoke_m, albedo}], n, ring_km, sector, power}`. A smoke column — puffs along an axis bent by the wind, wider toward the top, lit from below;
    fire — flame sprites. Returns the records and the npz of puffs and flames (cached by description)."""
    if not spec: return [], None
    rng = np.random.default_rng(seed + 900); cx, cy = env["center"]; F = []
    for f in spec.get("list", []):
        x, y = f["at"]; F.append(dict(at=[x, y], power=f.get("power", 1.0), size=f.get("size_m", 25.0), smoke=f.get("smoke_m", 600.0), albedo=f.get("albedo", 0.18)))
    n = int(spec.get("n", 0)); a0, a1 = spec.get("sector", [0, 360]); r0, r1 = spec.get("ring_km", [2, 12])
    for _ in range(n):
        az = math.radians(a0 + rng.random() * ((a1 - a0) % 360 or 360)); r = 1000 * (r0 + (r1 - r0) * rng.random() ** 0.7)
        p = rng.uniform(*spec.get("power", [0.5, 2.0]))
        F.append(dict(at=[cx + r * math.cos(az), cy + r * math.sin(az)], power=p, size=18.0 * p ** 0.5 * rng.uniform(0.7, 1.4),
                      smoke=450.0 * p ** 0.6 * rng.uniform(0.7, 1.4), albedo=float(spec.get("albedo", 0.16))))
    for f in F: f["at"] = [f["at"][0], f["at"][1], surface_z(scene_dir, T, f["at"][0], f["at"][1])]
    key = hashlib.sha1(json.dumps([F, wind, seed, 2], sort_keys=True).encode()).hexdigest()[:10]
    os.makedirs(cache_dir, exist_ok=True); path = os.path.join(cache_dir, f"fires_{key}.npz")
    if not os.path.exists(path):
        sp_, sr, sb, st, sg, fp, fr, sc_ = [], [], [], [], [], [], [], []
        w = np.array([wind[0], wind[1], 0.0])
        for fi, f in enumerate(F):
            x, y, z = f["at"]; H = f["smoke"]; r = f["size"] * 0.8; h = 0.0
            while h < H:                                                     # column puffs: a step of ~0.55 radius, the axis drifts with the wind
                t = h / H; c = np.array([x, y, z + h]) + w * H * 0.6 * t ** 1.6 + rng.normal(0, r * 0.15, 3)
                sp_.append(c); sr.append(r * rng.uniform(0.85, 1.15)); sb.append(z); st.append(z + H); sg.append(math.exp(-h / (0.18 * H + 1))); sc_.append(fi)
                h += r * 0.55; r = f["size"] * 0.8 + h * 0.22
            for k in range(3):                                               # flames: 3 sprites at the base
                fp.append((x + rng.normal(0, f["size"] * 0.25), y + rng.normal(0, f["size"] * 0.25), z + f["size"] * (0.4 + 0.3 * k)))
                fr.append(f["size"] * (0.75 - 0.15 * k))
        np.savez_compressed(path, pos=np.array(sp_, np.float32).reshape(-1, 3), rad=np.array(sr, np.float32), base=np.array(sb, np.float32),
                            top=np.array(st, np.float32), glow=np.array(sg, np.float32), fpos=np.array(fp, np.float32).reshape(-1, 3),
                            frad=np.array(fr, np.float32), cid=np.array(sc_, np.int32))
    return F, path


CITY_KIT = {   # far-city kit: wall color, fraction of towers with tiers, spires, windows (step by height and width, m)
    "hive": dict(wall=(0.30, 0.28, 0.27), roof=(0.22, 0.21, 0.21), tiers=0.6, spire=0.04, win=(4.0, 3.2)),
    "gothic": dict(wall=(0.46, 0.43, 0.38), roof=(0.25, 0.22, 0.20), tiers=0.35, spire=0.08, win=(5.5, 4.0)),
    "ruins": dict(wall=(0.38, 0.35, 0.31), roof=(0.30, 0.28, 0.26), tiers=0.2, spire=0.0, win=(4.5, 3.5)),
    "block": dict(wall=(0.52, 0.52, 0.50), roof=(0.35, 0.35, 0.35), tiers=0.25, spire=0.0, win=(3.5, 3.0))}


def city_build(specs, scene_dir, T, env, seed, cache_dir):
    """The far city: houses by cells in a sector and ring, on the site's ground or the far terrain; steep slopes are skipped;
    heights follow a power law (many low, few tall), spires; prototypes: 0 — block, 1 — stepped tower, 2 — spire.
    The burning fraction — fire records on the roofs. Returns (npz of instances, fires)."""
    if not specs: return None, []
    specs = specs if isinstance(specs, list) else [specs]
    key = hashlib.sha1(json.dumps([specs, env["center"], seed, T and T.get("seed")], sort_keys=True, default=str).encode()).hexdigest()[:10]
    os.makedirs(cache_dir, exist_ok=True); path = os.path.join(cache_dir, f"city_{key}.npz"); fpath = path[:-4] + "_fires.json"
    if os.path.exists(path) and os.path.exists(fpath): return path, json.load(open(fpath))
    rng = np.random.default_rng(seed + 300); cx, cy = env["center"]; E = np.array(env["points"])[:, :2]
    P_, S_, I_, Y_, K_, fires = [], [], [], [], [], []
    for ci, c in enumerate(specs):
        kit = CITY_KIT.get(c.get("kit", "hive"), CITY_KIT["hive"]); cell = float(c.get("cell_m", 110.0))
        a0, a1 = c.get("sector", [0, 360]); r0, r1 = 1000 * c.get("from_km", 1.5), 1000 * c.get("to_km", 6.0); hmin, hmax = c.get("height", [40, 400])
        span = (a1 - a0) % 360 or 360; g = np.arange(-r1, r1 + cell, cell); GX, GY = np.meshgrid(g, g)
        X = cx + GX.ravel() + rng.uniform(-0.3, 0.3, GX.size) * cell; Y = cy + GY.ravel() + rng.uniform(-0.3, 0.3, GX.size) * cell
        r = np.hypot(X - cx, Y - cy); az = np.degrees(np.arctan2(Y - cy, X - cx)) % 360
        keep = (r >= r0) & (r <= r1) & ((az - a0) % 360 <= span) & (rng.random(X.size) <= c.get("density", 0.6))
        dE = np.min(np.hypot(X[:, None] - E[None, :, 0], Y[:, None] - E[None, :, 1]), 1); keep &= dE >= 150      # not on the shoot area
        X, Y = X[keep], Y[keep]
        if not len(X): continue
        Z = np.stack([surface_zs(scene_dir, T, X + dx, Y + dy) for dx, dy in ((-20, 0), (20, 0), (0, -20), (0, 20))], 1)
        flat = Z.max(1) - Z.min(1) <= 18                                               # steeper than ~25° — we do not build
        for x, y, zmin in zip(X[flat], Y[flat], Z[flat].min(1)):
            w, d = cell * rng.uniform(0.45, 0.8), cell * rng.uniform(0.45, 0.8)
            u = rng.random(); h = hmin + (hmax - hmin) * u ** 3.2
            kind = 2 if rng.random() < kit["spire"] else (1 if rng.random() < kit["tiers"] else 0)
            if kind == 2: w = d = cell * 0.35; h = min(hmax * 1.6, h * 2.5 + hmax * 0.5)
            P_.append((x, y, zmin - 2.0)); S_.append((w, d, h)); I_.append(kind); Y_.append(rng.uniform(-0.15, 0.15) + rng.integers(0, 4) * math.pi / 2); K_.append(ci)
            if rng.random() < c.get("burning", 0.0):
                fires.append(dict(at=[float(x), float(y)], power=float(0.4 + 1.2 * rng.random()), size_m=float(max(w, d) * 0.45), smoke_m=float(250 + h * 1.5), albedo=0.14))
    np.savez_compressed(path, pos=np.array(P_, np.float32).reshape(-1, 3), scl=np.array(S_, np.float32).reshape(-1, 3),
                        idx=np.array(I_, np.int32), yaw=np.array(Y_, np.float32), kit=np.array(K_, np.int32))
    json.dump(fires, open(fpath, "w")); return path, fires


def ground_level(scene_dir):
    """The site's ground level: far_z (the site from the plan), otherwise 0."""
    for p, key in ((os.path.join(scene_dir, "location.json"), "far_z"), (os.path.join(scene_dir, "world.json"), "far_z")):
        if os.path.exists(p):
            v = json.load(open(p)).get(key)
            if v is not None: return float(v)
    return 0.0


# ------------------------------------------------------------------------------------------------ AgX as in Blender 5.2
_LUT = None
M_XYZ_709 = np.array([[3.2409699, -1.5373832, -0.4986108], [-0.9692436, 1.8759675, 0.0415551], [0.0556301, -0.2039770, 1.0569715]])
M_XYZ_EG = np.array([[1.5250528, -0.3159135, -0.1226583], [-0.5091526, 1.3333274, 0.1382844], [0.0957153, 0.0508974, 0.7879558]])
M_709_EG = M_XYZ_EG @ np.linalg.inv(M_XYZ_709)
LG_MIN, LG_MAX = -12.47393, 12.5260688117


def _lut():
    global _LUT
    if _LUT is None:
        rows = [l.split() for l in open(os.path.join(OCIO, "luts", "AgX_Base_sRGB.cube")) if l[:1].isdigit() or l[:1] == "-"]
        a = np.array([[float(x) for x in r] for r in rows if len(r) == 3], np.float32); n = round(len(a) ** (1 / 3))
        _LUT = a.reshape(n, n, n, 3)                                   # .cube: red varies fastest → [b][g][r]
    return _LUT


CD_UNIT = 750.0                                                      # cd/m² per unit of plan brightness: the sun at the zenith ~145 units ≈ 110,000 lx [est.]


def night_shift(lin, P):
    """The blue shift of twilight vision (Jensen et al. 2000, "Night Rendering", §3.1): from the pixel's physical luminance Y (without the plan's ×800
    night boost) the photopic fraction s — a smooth transition over log10 Y from −2 to 0.6 (cd/m²); chromaticity goes toward (0.25, 0.25)
    with weight 1 − s, luminance Y' = 0.4468(1 − s)V + sY (V — scotopic luminance from XYZ, the Larson et al. 1997 formula [not verified]).
    Fire (photopic vision) stays warm, moonlight and shadows turn blue. There is no shift by day: s is no less than the scene key's s (the eye adapts to
    the frame, not to a shadow). The fire glow shifts like everything else: only fires, windows and the cores of bright halos stay warm (experiment on the assault:
    a fully or half warm glow is brown mud, 76 fires make an even veil)."""
    B = 1.0 + (NIGHT_BOOST - 1.0) * P["sky"]["w_night"]; dec, k_ = P["sky"].get("night_shift", (0.0, 1.0))   # style: STYLE night_shift
    def s_of(Y):
        t = np.clip((np.log10(np.maximum(Y, 1e-12)) + dec + 2.0) / 2.6, 0, 1); return 1 - k_ * (1 - (3 * t * t - 2 * t ** 3))
    s_key = float(s_of(P["light"]["e_key"] * 0.18 / math.pi * CD_UNIT / B))
    if s_key >= 0.999: return lin
    XYZ = lin @ np.linalg.inv(M_XYZ_709).T; X, Y, Z = XYZ[..., 0], XYZ[..., 1], XYZ[..., 2]
    s = np.maximum(s_of(Y * CD_UNIT / B), s_key)
    W = np.maximum(X + Y + Z, 1e-12); x, y = X / W, Y / W
    V = np.maximum(Y * (1.33 * (1 + (Y + Z) / np.maximum(X, 1e-6 * W)) - 1.68), 0.0)
    xn, yn = (1 - s) * 0.25 + s * x, (1 - s) * 0.25 + s * y; Yn = 0.4468 * (1 - s) * V + s * Y
    Xn = xn * Yn / np.maximum(yn, 1e-6); Zn = Xn / np.maximum(xn, 1e-6) - Xn - Yn
    return np.maximum(np.stack([Xn, Yn, Zn], -1) @ M_XYZ_709.T, 0.0).astype(np.float32)


_LOOKS = {}


def _look_lut(name):
    """A Blender AgX look (punchy, medium_high_contrast, …) — a table baked by OCIO itself in our log encoding (core/blender/bl_agx_bake.py;
    cache out/agx_looks, baked on first access, ~10 s)."""
    if name not in _LOOKS:
        d = os.path.join(ROOT, "out", "agx_looks"); p = os.path.join(d, name + ".npy")
        if not os.path.exists(p):
            if os.environ.get("SK_IN_BLENDER"):          # a nested Blender is forbidden (run.py, code 3) — the cache must exist beforehand
                raise SystemExit(f"no cache {p}: it cannot be baked inside Blender — run once outside Blender: "
                                 ".venv/bin/python -m core.light.far agx")
            import subprocess
            subprocess.run(["blender", "-b", "--factory-startup", "-P", _core_paths.RUN, "--", _core_paths.script("bl_agx_bake.py"), "--", d, "57"], capture_output=True, text=True, check=True)
        _LOOKS[name] = np.load(p)
    return _LOOKS[name]


def _tri(L, t):
    """Trilinear lookup in a 3D table [b][g][r] (t — coordinates in grid nodes); Blender itself interpolates tetrahedrally (the discrepancy is in runs/backdrop)."""
    n = L.shape[0]; i0 = np.minimum(t.astype(np.int32), n - 2); f = t - i0
    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]; fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
    c = lambda db, dg, dr: L[b0 + db, g0 + dg, r0 + dr]
    return (((c(0, 0, 0) * (1 - fr) + c(0, 0, 1) * fr) * (1 - fg) + (c(0, 1, 0) * (1 - fr) + c(0, 1, 1) * fr) * fg) * (1 - fb)
            + ((c(1, 0, 0) * (1 - fr) + c(1, 0, 1) * fr) * (1 - fg) + (c(1, 1, 0) * (1 - fr) + c(1, 1, 1) * fr) * fg) * fb)


def agx(rgb, ev=0.0, look=None):
    """Linear Rec.709 (scene) → screen sRGB 0..1, like the Blender AgX view (exposure ev — as view_settings.exposure); look —
    the AgX look (color by style) — an OCIO table baked in the same log encoding."""
    x = np.asarray(rgb, np.float32) * np.float32(2.0 ** ev)
    eg = x @ M_709_EG.T.astype(np.float32)
    t01 = np.clip((np.log2(np.maximum(eg, 2.0 ** LG_MIN)) - LG_MIN) / (LG_MAX - LG_MIN), 0, 1)
    if look and look != "base":
        L = _look_lut(look); return np.clip(_tri(L, t01 * (L.shape[0] - 1)), 0, 1)   # the table already returns screen sRGB
    L = _lut(); lin = np.clip(_tri(L, t01 * (L.shape[0] - 1)), 0, 1) ** 2.4          # Rec.1886 → linear display → sRGB
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)


def bloom(lin, ev, k=0.35, thr=0.8, weights=(0.6, 0.3, 0.1), sizes=(0.004, 0.02, 0.08)):
    """Glow of bright areas (fires, windows, the moon, searchlights, the sun): the part of brightness above the soft threshold thr (after exposure —
    near display white) is blurred by three Gaussians (σ as fractions of the frame height; the large ones on a downscaled frame) and added
    with strength k. Without a threshold (scattering "as in a lens") the halos of small fires are not visible, while a wide kernel lifts the black at night."""
    import cv2
    H, W_ = lin.shape[:2]; e = np.float32(2.0 ** ev); x = lin * e
    Y = x @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    br = x * (np.maximum(Y - thr, 0.0) / np.maximum(Y, 1e-6))[..., None]           # soft knee: the share above the threshold
    add = np.zeros_like(lin)
    for w, sz in zip(weights, sizes):
        sig = sz * H; kk = max(1, int(sig / 6))
        small = cv2.resize(br, (W_ // kk, H // kk), interpolation=cv2.INTER_AREA) if kk > 1 else br
        b = cv2.GaussianBlur(small, (0, 0), sig / kk)
        add += w * (cv2.resize(b, (W_, H), interpolation=cv2.INTER_LINEAR) if kk > 1 else b)
    return (lin + k * add / e).astype(np.float32)


# ------------------------------------------------------------------------------------------------ frame after render
def read_rgba(path):
    import OpenEXR
    f = OpenEXR.File(path); ch = {}
    for p in f.parts:
        for name, c in p.channels.items():
            px = np.asarray(c.pixels, np.float32)
            if px.ndim == 3:                                            # "RGBA" as one channel array
                if not (name in ("RGBA", "RGB") or name.endswith("Combined")): continue   # passes (surface color, AOV) are not the frame
                for i, k in enumerate("RGBA"[:px.shape[2]]): ch[k] = px[..., i]
            else: ch[name.split(".")[-1]] = px
    H, W = next(iter(ch.values())).shape
    rgb = np.stack([ch.get(k, np.zeros((H, W), np.float32)) for k in "RGB"], -1)
    return rgb, ch.get("A", np.ones((H, W), np.float32)), ch.get("Z")


def edge_depth(z, a):
    """Depth at a silhouette edge: in an antialiased frame the edge is partly sky (1e10 or a mix with it); take the nearest depth in a 3×3 neighborhood."""
    zz = np.where(np.isfinite(z) & (z < 1e9), z, np.inf).astype(np.float32)
    p = np.pad(zz, 1, constant_values=np.inf); H, W = zz.shape
    near = np.min(np.stack([p[i:i + H, j:j + W] for i in range(3) for j in range(3)]), 0)
    return np.where((a > 1e-3) & ~np.isfinite(zz), near, zz)


def hg(cos_t, g):
    """Henyey–Greenstein phase, normalized to 4π."""
    return (1 - g * g) / (4 * math.pi * np.power(np.maximum(1 + g * g - 2 * g * cos_t, 1e-6), 1.5))


def horizon_rgb(H, k=3.0, near=6.0, far_=24.0):
    """The color of the horizon ring without luminary disks. The ring is sampled at elevation el (1.5°): a low sun or moon got into it
    as a disk (the squad_take sunset: 1344 against a median of 0.86), and the air haze glowed with it along the whole azimuth column — a yellow column under the
    luminary across the whole frame. A peak above k medians of the neighbors (near…far_ degrees on both sides) is clipped to that level; the smooth
    halo around the luminary — the sky the mountain dissolves into — stays."""
    rgb = np.array(H["rgb"], float); n = len(rgb); step = 360.0 / n
    Y = rgb @ np.array([0.2126, 0.7152, 0.0722])
    lo, hi = max(1, int(round(near / step))), max(2, int(round(far_ / step)))
    off = np.r_[-hi:-lo + 1, lo:hi + 1]
    cap = k * np.array([np.median(Y[(i + off) % n]) for i in range(n)])
    return rgb * np.where(Y > cap, cap / np.maximum(Y, 1e-12), 1.0)[:, None]


def load_horizon(plan_path):
    """The sky at the horizon by azimuth (bl_far.render_horizon): the air haze color in the view direction (without luminary disks)."""
    H = json.load(open(os.path.splitext(plan_path)[0] + "_horizon.json"))
    return np.array(H["az"], float), horizon_rgb(H)


def _box(img, r):
    """Mean over a (2r + 1)² square — via cumulative sums (edges — over the available part)."""
    c = np.pad(img, ((r + 1, r), (r + 1, r), (0, 0)), mode="edge").cumsum(0).cumsum(1)
    H, W = img.shape[:2]; k = 2 * r + 1
    return (c[k:k + H, k:k + W] - c[:H, k:k + W] - c[k:k + H, :W] + c[:H, :W]) / (k * k)


def shaft_occlusion(a, cam, P, n=40, reach=0.6, q=4):
    """Light shafts through gaps (the Mitchell 2007 technique, GPU Gems 3, ch. 13 — scattering as a post-effect): for a pixel — the
    fraction of open sky (not blocked by geometry and clouds: the frame alpha) along the segment toward the luminary's screen point (no farther than reach
    of the diagonal; n samples on the frame downscaled by q). The haze in the shadow of a gap glows weaker — shafts from contrast, without adding
    energy. A luminary behind the camera — no shafts (None)."""
    import cv2
    if P["sun"]["el"] > -0.5 and max(P["sun"]["lamp_rgb"]) > 0: L = np.array(P["sun"]["dir"], float)
    elif P["moon"]["on"] and max(P["moon"]["lamp_rgb"]) > 0: L = np.array(P["moon"]["dir"], float)
    else: return None
    lx, ly, lz = cam.project(cam.pos + L * 1e7)
    if not lz > 0 or not np.isfinite(lx): return None
    H, W = a.shape; h, w = H // q, W // q
    occ = cv2.resize((1.0 - np.clip(a, 0, 1)).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32) + 0.5, np.arange(h, dtype=np.float32) + 0.5)
    dx, dy = float(lx) / q - X, float(ly) / q - Y; dist = np.hypot(dx, dy)
    f = np.minimum(1.0, reach * math.hypot(w, h) / np.maximum(dist, 1e-3)).astype(np.float32)
    acc = np.zeros((h, w), np.float32)
    for t in (np.arange(n) + 0.5) / n:
        acc += cv2.remap(occ, (X + t * f * dx).astype(np.float32), (Y + t * f * dy).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.resize(acc / n, (W, H), interpolation=cv2.INTER_LINEAR), L


def fog_terms(z, a, cam, P, hz, sky=None, shafts=0.0):
    """The plan's hazes by depth: transmittance and scattered light (linear) for geometry and for the sky.
    Layer: ρ(h) = a·exp(−(h − z₀)/H); τ up to distance d along a ray with vertical fraction v — ρ(z_c)·d·(1 − e^(−k))/k, k = v·d/H;
    to infinity — ρ(z_c)·H/v (v > 0). Light in the haze: air — the sky at the horizon by azimuth; the ground layer — the sun and moon with a
    Henyey–Greenstein phase + the sky, tinted by its own color; layers are mixed by thickness fractions. Air looking upward: the haze light smoothly
    turns from the horizon sky into the frame's sky in that direction (sky, without stars and moon — blurred): at sunset the upper sky does not get the orange
    veil of the horizon."""
    D = cam.ray_dirs(); v = D[..., 2].astype(np.float32); zc = float(cam.pos[2])
    d = np.where(np.isfinite(z), cam.ray_len(np.nan_to_num(z, posinf=0.0)), np.inf).astype(np.float32)
    geo = a > 1e-3
    az = np.degrees(np.arctan2(D[..., 1], D[..., 0])) % 360
    hz_rgb = np.stack([np.interp(az, hz[0], hz[1][:, i], period=360) for i in range(3)], -1).astype(np.float32)
    S = np.array(P["sun"]["dir"]); M = np.array(P["moon"]["dir"])
    sun_E, moon_E = np.array(P["sun"]["lamp_rgb"]), np.array(P["moon"]["lamp_rgb"])
    sky_E = np.array(P["light"]["sky_e_rgb"]) + np.array(P["light"].get("night_sky_e_rgb", [0, 0, 0]))
    sky_s = None if sky is None else np.minimum(_box(sky, 4), _box(sky, 12) * 1.5)       # without stars and the moon disk
    tg, ts = np.zeros(z.shape, np.float32), np.zeros(z.shape, np.float32)
    Lg, Ls = np.zeros(z.shape + (3,), np.float32), np.zeros(z.shape + (3,), np.float32)
    for Ly in P["fog"]["layers"]:
        rho = Ly["a"] * math.exp(float(np.clip(-(zc - Ly["z0"]) / Ly["H"], -60, 3)))   # density at the camera (deep in a valley — no more than e³·a)
        k = np.maximum(v * np.where(np.isfinite(d), d, 0) / Ly["H"], -40.0)        # clamped from below against overflow; do not clamp from above
        f = np.where(np.abs(k) < 1e-5, 1.0, -np.expm1(-k) / np.where(np.abs(k) < 1e-5, 1.0, k))
        tau_g = np.where(geo & np.isfinite(d), rho * np.where(np.isfinite(d), d, 0) * f, 0).astype(np.float32)
        if Ly["on_sky"]:                                               # sky: the path in the layer is finite — on a round Earth the ray leaves the layer (≤ D_SKY)
            ks = np.maximum(v * D_SKY / Ly["H"], -40.0); fs = np.where(np.abs(ks) < 1e-5, 1.0, -np.expm1(-ks) / np.where(np.abs(ks) < 1e-5, 1.0, ks))
            tau_s = (rho * D_SKY * fs).astype(np.float32)
        else: tau_s = np.zeros(z.shape, np.float32)
        if Ly["src"] == "horizon":
            L = hz_rgb
            if sky is not None:                                        # upward — smoothly toward the frame's sky in that direction
                wv = np.clip((v - 0.02) / 0.3, 0, 1)[..., None]
                L = hz_rgb * (1 - wv) + sky_s * wv
        else:
            alb = np.array(Ly["albedo"], np.float32)                   # single scattering (a halo toward the luminary) + multiple scattering (thick fog is white)
            iso = lambda E, d_: E * max(d_[2], 0.0) / math.pi
            L = alb * ((0.55 * hg(D @ S, Ly["g"]))[..., None] * sun_E + 0.45 * iso(sun_E, S) + (0.55 * hg(D @ M, Ly["g"]))[..., None] * moon_E
                       + 0.45 * iso(moon_E, M) + 0.5 * sky_E / math.pi + 0.5 * hz_rgb)
        tg += tau_g; ts += tau_s; Lg += tau_g[..., None] * L; Ls += tau_s[..., None] * L
    Tg, Ts = np.exp(-tg), np.exp(-ts)
    Lg = Lg / np.maximum(tg, 1e-9)[..., None] * (1 - Tg)[..., None]; Ls = Ls / np.maximum(ts, 1e-9)[..., None] * (1 - Ts)[..., None]
    so = shaft_occlusion(a, cam, P) if shafts > 0 else None
    if so is not None:                                                     # light shafts: the haze in the shadow of a gap is weaker, stronger toward the luminary (forward)
        O, L = so; k = shafts * np.maximum(D @ L, 0.0) ** 4
        Lg = Lg * (1 - k * (1 - O))[..., None]; Ls = Ls * (1 - k * (1 - O))[..., None]
    gg, gs = fire_glow(cam, D, tg, P)
    return Tg, Lg + gg, Ts, Ls + gs


def glow_map(P, c0, cache_dir, W=512, H=256):
    """Fire glow by direction from the envelope center (an equirectangular map, linear color) — for effect shaders: clouds and smoke drown in the same glow
    as the mountains and the sky (for those it is exact — per camera, fire_glow)."""
    import OpenEXR
    key = hashlib.sha1(json.dumps([(P.get("glow") or {}).get("sources"), P["fires"]["list"], P["fog"], list(c0), P["fires"]["glow"]], sort_keys=True, default=float).encode()).hexdigest()[:10]
    path = os.path.join(cache_dir, f"glow_{key}.exr")
    if os.path.exists(path): return path
    az = (np.arange(W) + 0.5) / W * 2 * math.pi - math.pi; el = (np.arange(H) + 0.5) / H * math.pi - math.pi / 2
    A, E = np.meshgrid(az, el[::-1])                                         # row 0 — zenith
    D = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1)
    class _C: pos = np.asarray(c0, float)
    _, gs = fire_glow(_C, D, np.zeros(D.shape[:2], np.float32), P)
    img = np.asarray(gs if not np.isscalar(gs) else np.zeros(D.shape), np.float32)
    OpenEXR.File({"type": OpenEXR.scanlineimage}, {"RGB": img}).write(path)
    return path


GLOW_K = 1.0e6                                                       # glow: illuminance from a fire of power 1 at distance d — GLOW_K / d²
FIRE_RGB = np.array([1.0, 0.42, 0.12], np.float32)
CITY_RGB = np.array([1.0, 0.72, 0.42], np.float32)                   # windows (as in the city shader)


FLASH = {"explosion": dict(rgb=(1.0, 0.72, 0.42), dur=6.0, k=15.0, size=30.0),        # k — how many times brighter than a fire of the same power (per frame;
                                                                                         # ×60 washed out the whole night assault in smoke)
         "lightning": dict(rgb=(0.75, 0.82, 1.0), dur=4.0, k=200.0, size=40.0)}


def flash_env(kind, t, dur):
    """Flash envelope t frames after its start: an explosion — a peak and a decay e^(−t/τ), τ = dur/3; lightning — three bursts within dur."""
    if t < 0: return 0.0
    if kind == "lightning":
        return float(sum(a * math.exp(-((t - tk) / 0.45) ** 2) for a, tk in ((1.0, 0.0), (0.55, dur * 0.4), (0.8, dur * 0.75))))
    return math.exp(-t / max(dur / 3.0, 1e-3))


def flash_list(sp, light_k, z0, decks, scene_dir, T):
    """All flashes of the description (`flashes: [{kind: explosion|lightning, at: [x, y(, z)], power, frame, dur, alt_km, size_m, color}]`)
    with position and peak strength I0 = GLOW_K · power · k of the kind · light_k (by day, as for a fire, without the night boost); lightning is in the cloud
    (alt_km, by default under the lowest layer, otherwise 2 km), an explosion is near the ground. The start frame and the duration are in frames."""
    out = []
    for f in sp.get("flashes", []) or []:
        kind = f.get("kind", "explosion"); D = FLASH.get(kind, FLASH["explosion"]); x, y = float(f["at"][0]), float(f["at"][1])
        if len(f["at"]) > 2: z = float(f["at"][2])
        elif kind == "lightning": z = z0 + 1000.0 * float(f.get("alt_km", min([d["alt_km"] for d in decks] or [2.2]) - 0.2))
        else: z = surface_z(scene_dir, T, x, y) + float(f.get("size_m", D["size"]))
        out.append(dict(kind=kind, at=[x, y, z], I0=GLOW_K * float(f.get("power", 1.0)) * D["k"] * light_k, frame0=float(f.get("frame", 0.0)),
                        dur=float(f.get("dur", D["dur"])), rgb=list(f.get("color", D["rgb"])), size=float(f.get("size_m", D["size"])),
                        ground=surface_z(scene_dir, T, x, y), seed=int(f.get("seed", 1))))
    return out


def battle_events(path, fps, light_k=1.0):
    """Blasts of the battle scene (components/battle/bl_battle.py → <build>/data/events.json: t — seconds, at, size, wall) — node events by frame
    (zero-based, as for a take): the glow in the haze and the light on the underside of clouds. The peak is the same as the scene's blast lamp (2.2·10⁶·(size/5)² W → I = E/4π),
    the duration is the same as its (0.45 s); the fireball and the lamp at a blast belong to the scene itself (own: the node does not place them)."""
    out = []
    for e in json.load(open(path)):
        x, y, z = (list(e["at"]) + [0.0])[:3]; s = float(e.get("size", 5.0))
        out.append(dict(kind="explosion", at=[x, y, z + 2.0], I0=2.2e6 * (s / 5.0) ** 2 / (4 * math.pi) * light_k, frame0=float(e["t"]) * fps,
                        dur=0.45 * fps, rgb=[1.0, 0.55, 0.22], size=s, ground=z, seed=1, own=True))
    return out


def flashes_at(P, frame):
    """Flashes burning in frame `frame` (zero-based): the description's flashes (flash_list) and the take's events (events — bl_take: explosions), strength —
    I0 × envelope (flash_env)."""
    out = []
    for f in (P.get("flash_list") or []) + (P.get("events") or []):
        e = flash_env(f["kind"], frame - f["frame0"], f["dur"])
        if e >= 1e-3: out.append(dict(f, env=round(e, 4), I=f["I0"] * e))
    return out


def frame_sources(P, frame):
    """Glow sources of the frame: the constant ones (fires and windows — fire flicker per frame) + the frame's flashes. For takes and fly-arounds."""
    out = []
    for i, s_ in enumerate(P["glow"]["sources"]):
        if s_.get("flash"): continue
        if s_.get("fire"):                                               # flicker: two sines with their own phase for each fire, ±20 %
            h = (i * 0.618) % 1.0; s_ = dict(s_, I=s_["I"] * (1.0 + 0.13 * math.sin(2 * math.pi * (frame * 0.37 + h)) + 0.07 * math.sin(2 * math.pi * (frame * 1.13 + 2 * h))))
        out.append(s_)
    return out + [dict(at=f["at"], I=f["I"], rgb=f["rgb"], flash=True) for f in flashes_at(P, frame)]


def win_k(sun_el, w_night):
    """Fraction of window brightness by time of day (the same as in the city shader bl_far._city_material): all of it at night, fading toward noon."""
    t = max(0.0, min(1.0, (sun_el + 6.0) / 16.0)); return max(w_night, 0.15 * (1 - t * t * (3 - 2 * t)))


def glow_sources(F, Fz, city_npz, kits, wk, cell=1500.0):
    """Sources of glow and of light from below: fires (strength I = GLOW_K · power · glow · light_k, a fire at the height of its size) and
    the city — windows by cells of cell m (the flux of lit windows π · 3.5 · window fraction · area, into a half-space: I = Φ / 2π);
    each has its own color."""
    S = [dict(at=[f["at"][0], f["at"][1], f["at"][2] + f["size"]], I=GLOW_K * f["power"] * Fz.get("glow", 1.0) * Fz.get("light_k", 1.0),
              rgb=FIRE_RGB.tolist(), fire=True) for f in F]
    if city_npz and wk > 1e-4 and os.path.exists(city_npz):
        D = np.load(city_npz); p, sc, ki = D["pos"].astype(float), D["scl"].astype(float), D["kit"]
        frac = np.array([k.get("windows", 0.35) for k in kits] or [0.35])[np.clip(ki, 0, max(len(kits) - 1, 0))]
        Phi = math.pi * 3.5 * wk * 2 * (sc[:, 0] + sc[:, 1]) * sc[:, 2] * frac * 0.225   # a window is 0.45 × 0.5 of a facade cell
        _, inv = np.unique(np.floor(p[:, :2] / cell).astype(np.int64), axis=0, return_inverse=True); inv = inv.ravel()
        for c in range(inv.max() + 1 if len(inv) else 0):
            m = inv == c; w = Phi[m]
            S.append(dict(at=[float(np.average(p[m, 0], weights=w)), float(np.average(p[m, 1], weights=w)), float(np.mean(p[m, 2] + 0.5 * sc[m, 2]))],
                          I=float(w.sum() / (2 * math.pi)), rgb=CITY_RGB.tolist()))
    return S


def light_below(pts, P, dmin=1.0):
    """Illuminance from below (onto a downward-facing surface) at points pts from the plan's glow sources, through the hazes — RGB.
    dmin — no closer than this (smoke above its own fire: a column's own fire is computed separately, pglow)."""
    pts = np.asarray(pts, float); E = np.zeros((len(pts), 3))
    for src in (P.get("glow") or {}).get("sources", []):
        a_ = np.asarray(src["at"], float); v = pts - a_; d = np.maximum(np.linalg.norm(v, axis=1), dmin)
        tau = np.zeros(len(pts))
        for Ly in P["fog"]["layers"]:
            k = np.maximum(v[:, 2] / Ly["H"], -40.0); f = np.where(np.abs(k) < 1e-5, 1.0, -np.expm1(-k) / np.where(np.abs(k) < 1e-5, 1.0, k))
            tau += Ly["a"] * math.exp(float(np.clip(-(a_[2] - Ly["z0"]) / Ly["H"], -60, 3))) * d * f
        E += (src["I"] * np.clip(v[:, 2] / d, 0, 1) / (d * d) * np.exp(-tau))[:, None] * np.asarray(src["rgb"])
    return E


def under_map(P, z, cache_dir, R=32000.0, N=128):
    """Fire and city light on the underside of the cloud layer at height z: an N×N map over ±R around the envelope center (EXR, linear)."""
    import OpenEXR
    cx, cy = P["envelope"]["center"]
    key = hashlib.sha1(json.dumps([P["glow"]["sources"], P["fog"], z, R, N, cx, cy], sort_keys=True, default=float).encode()).hexdigest()[:10]
    path = os.path.join(cache_dir, f"under_{key}.exr")
    if not os.path.exists(path):
        g = (np.arange(N) + 0.5) / N * 2 * R - R; X, Y = np.meshgrid(cx + g, cy + g)    # row = y (bottom to top in uv)
        E = light_below(np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1), P).reshape(N, N, 3)
        OpenEXR.File({"type": OpenEXR.scanlineimage}, {"RGB": E[::-1].astype(np.float32)}).write(path)
    return {"path": path, "center": [cx, cy], "R": R}


def under_puffs(npz, P, cache_dir, dmin=1.0):
    """Fire and city light from below onto cloud and smoke puffs (at the puff's bottom) — an npz with pfire (N×3)."""
    key = hashlib.sha1(json.dumps([npz, P["glow"]["sources"], P["fog"], dmin], sort_keys=True, default=float).encode()).hexdigest()[:10]
    path = os.path.join(cache_dir, f"underp_{key}.npz")
    if not os.path.exists(path):
        D = np.load(npz); pts = D["pos"].astype(float) - np.stack([np.zeros(len(D["rad"])), np.zeros(len(D["rad"])), D["rad"].astype(float)], 1)
        np.savez_compressed(path, pfire=light_below(pts, P, dmin).astype(np.float32))
    return path


def fire_glow(cam, D, tg, P):
    """Fire glow in the haze (a simplification of the point-source glow model in a medium, Sun et al. 2005): each fire gives a halo
    around the direction toward it (narrow and wide, Henyey–Greenstein 0.85 and 0.4), strength — power / d² × the haze thickness up to the fire;
    for geometry closer than the fire only its part of the path counts. At night an orange sky rises over a burning city."""
    Fz = P.get("fires") or {}
    S = P.get("_frame_sources", (P.get("glow") or {}).get("sources"))   # take frame: post puts the sources of its own frame here
    if S is None:                                                            # an older plan: fires only
        S = [dict(at=[f["at"][0], f["at"][1], f["at"][2] + f["size"]], I=GLOW_K * f["power"] * Fz.get("glow", 1.0) * Fz.get("light_k", 1.0),
                  rgb=FIRE_RGB.tolist()) for f in Fz.get("list") or []]
    if not S: return 0.0, 0.0
    zc = float(cam.pos[2]); gg = np.zeros(D.shape, np.float32); gs = np.zeros(D.shape, np.float32)
    for f in S:
        v = np.array(f["at"], float) - cam.pos; d = float(np.linalg.norm(v)); u = v / max(d, 1e-6)
        tau = 0.0
        for Ly in P["fog"]["layers"]:
            rho = Ly["a"] * math.exp(float(np.clip(-(zc - Ly["z0"]) / Ly["H"], -60, 3))); k = max(u[2] * d / Ly["H"], -40.0)
            tau += rho * d * (1.0 if abs(k) < 1e-5 else -math.expm1(-k) / k)
        E = f["I"] / max(d * d, 1.0)
        c = D @ u; kern = 0.85 * hg(c, 0.85) + 0.15 * hg(c, 0.4)
        full = (E * (1 - math.exp(-tau)) * kern)[..., None] * np.asarray(f["rgb"], np.float32)
        part = np.clip((1 - np.exp(-tg)) / max(1 - math.exp(-tau), 1e-9), 0, 1)[..., None]
        gg += full * part; gs += full
    return gg, gs


def cam_of(cam_dir, F):
    from core.outputs import maps
    return maps.Cam(F.get("camera") or json.load(open(os.path.join(cam_dir, "raw", "camera.json"))))


def depth_of(cam_dir):
    """Frame depth: the passes of the map bundle (raw/passes.exr, one sample — the same as for the maps), otherwise Depth from lit.exr."""
    from core.outputs import maps
    p = os.path.join(cam_dir, "raw", "passes.exr")
    if os.path.exists(p): return maps.read_exr(p)["z"]
    _, _, z = read_rgba(os.path.join(cam_dir, "lit.exr")); return z


def meter_ev(cam, a, Tg, F, P):
    """Frame exposure from the glow the camera sees (a static frame of a scene without its own light: far.json meter). The node computes
    the glow around and puts ½ e_glow into the exposure — "the camera looks at the fire" (the pass scene's cameras see 0.46–0.69 e_glow); a camera
    turned away from the fires sees a small fraction (far_fort — 0.06: the frame went black at 0.06 brightness) — the exposure uses the node's same formula
    on what is visible. It sees no less than ½ e_glow — the node's exposure. A battle with its own lamps, a take (one exposure per shot) and the panorama faces
    (seams) use the node's."""
    L = P["light"]; M = L.get("meter"); eg = float(L.get("e_glow", 0.0))
    S = [s for s in (P.get("glow") or {}).get("sources") or [] if not s.get("flash")]
    if not F.get("meter") or not M or eg <= 0 or not S: return F["ev"]
    s_ = (slice(None, None, 6), slice(None, None, 6)); a6 = a[s_]           # a grid every 6 px — a glow fraction, not a pattern
    gg, gs = fire_glow(cam, cam.ray_dirs()[s_], -np.log(np.clip(Tg[s_], 1e-4, 1.0)), dict(P, _frame_sources=S))
    Ef = math.pi * float(((gg * a6[..., None] + gs * (1 - a6)[..., None]) @ LUM).mean())
    v = min(Ef / (0.5 * eg), 1.0)
    if v >= 0.999: return F["ev"]
    e = L["e_key"] + 0.5 * eg * v; g = 0.5 * eg * v / e
    return round(float(np.clip(math.log2(math.pi / e) + M["mood_moon"] * (1 - g) + M["mood_fire"] * g + M["off"], -9.0, 11.0)), 3)


def post(cam_dirs):
    """A lit frame of the plan: geometry (lit.exr, premultiplied) and the camera sky (sky.exr) + hazes by depth → AgX → lit.png."""
    from PIL import Image
    done = []
    for d in cam_dirs:
        fj = os.path.join(d, "far.json")
        if not os.path.exists(fj) or not os.path.exists(os.path.join(d, "lit.exr")): continue
        F = json.load(open(fj)); P = json.load(open(F["plan"]))
        if F.get("frame") is not None and "glow" in P: P["_frame_sources"] = frame_sources(P, F["frame"])   # take: the frame's flashes and flicker
        rgb, a, _ = read_rgba(os.path.join(d, "lit.exr")); sky, _, _ = read_rgba(os.path.join(d, "sky.exr"))
        z = edge_depth(depth_of(d), a); ev = F["ev"]
        if "fog" in P:
            Tg, Lg, Ts, Ls = fog_terms(z, a, cam_of(d, F), P, load_horizon(F["plan"]), sky, P["sky"].get("shafts", 0.0))
            lin = rgb * Tg[..., None] + a[..., None] * Lg + (1 - a)[..., None] * (sky * Ts[..., None] + Ls); save_t(d, Tg)
            ev = meter_ev(cam_of(d, F), a, Tg, F, P)                         # exposure from the visible glow (a static frame without its own light)
            if F.get("meter"): json.dump(dict(F, ev_cam=ev), open(fj, "w"))   # which exposure the frame was composited with
        else: lin = rgb + sky * (1 - a)[..., None]
        sfx = os.path.join(d, "scene_fx.exr")
        if F.get("scene_fx") and os.path.exists(sfx):                      # scene effects over the haze (bl_far.render_scene_fx; the haze
            fr, fa, _ = read_rgba(sfx); lin = lin * (1 - np.clip(fa, 0, 1))[..., None] + fr   # for them is in their shader, bl_far.fx_fog)
        lin = night_shift(lin, P)                                           # night: a blue shift of twilight vision
        if P["sky"].get("bloom"): lin = bloom(lin, ev, P["sky"]["bloom"])     # glow of bright areas and the AgX view by style
        out = agx(lin, ev, P["sky"].get("look"))
        Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)).save(os.path.join(d, "lit.png")); done.append(d)
    return done


def save_t(cam_dir, Tg):
    """Haze transmittance on the geometry (fog_t.png): previs.py fades the outline by it — in haze the lines melt away together with the mountain."""
    from PIL import Image
    Image.fromarray((np.clip(Tg, 0, 1) * 255 + 0.5).astype(np.uint8)).save(os.path.join(cam_dir, "fog_t.png"))


def previs_far(cam_dir, c, a, F):
    """Workbench previs: the geometry (screen color after levels, c) on the plan's sky with hazes (in screen color, with the same numbers); on top — the effects layer
    (clouds, smoke, fire — Eevee, with the geometry held out in it: what is closer than the mountains lies on top, what is behind a mountain is cut off)."""
    P = json.load(open(F["plan"])); ev = F["ev"]
    sky, _, _ = read_rgba(os.path.join(cam_dir, "sky.exr"))
    fx = read_rgba(os.path.join(cam_dir, "fx.exr"))[:2] if F.get("fx") and os.path.exists(os.path.join(cam_dir, "fx.exr")) else None
    if sky.shape[:2] != c.shape[:2]: return c * a + agx(sky, ev)[:c.shape[0], :c.shape[1]] * (1 - a)
    sky_d = agx(sky, ev)
    if "fog" in P and os.path.exists(os.path.join(cam_dir, "raw", "passes.exr")):
        z = edge_depth(depth_of(cam_dir), a[..., 0])
        Tg, Lg, Ts, Ls = fog_terms(z, a[..., 0], cam_of(cam_dir, F), P, load_horizon(F["plan"]), sky); save_t(cam_dir, Tg)
        geo = c * Tg[..., None] + agx(Lg / np.maximum(1 - Tg, 1e-6)[..., None], ev) * (1 - Tg)[..., None]
        sk = sky_d * Ts[..., None] + agx(Ls / np.maximum(1 - Ts, 1e-6)[..., None], ev) * (1 - Ts)[..., None]
        base = geo * a + sk * (1 - a)
    else: base = c * a + sky_d * (1 - a)
    if fx is not None:                                                       # effects on top (premultiplied, in screen color)
        fr, fa = fx; fa3 = fa[..., None]; base = agx(fr / np.maximum(fa3, 1e-4), ev) * fa3 + base * (1 - fa3)
    return base


def sky_display(cam_dir, ev):
    """The camera sky (sky.exr) in screen color — the backdrop of the Workbench previs."""
    sky, _, _ = read_rgba(os.path.join(cam_dir, "sky.exr")); return agx(sky, ev)


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    kv = dict(x.split("=", 1) for x in args if "=" in x); pos = [x for x in args if "=" not in x]
    if cmd == "resolve":
        sd = pos[0]; spec = load_spec(sd, pos[1] if len(pos) > 1 else None)
        plan = resolve(spec, sd, kv)
        out = kv.get("out", os.path.join(ROOT, "out", "far", os.path.basename(os.path.abspath(sd)), "plan.json"))
        os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(plan, open(out, "w"), ensure_ascii=False, indent=1)
        print("__RESULT__" + json.dumps({"plan": out, "hash": plan["hash"], "sun": [plan["sun"]["az"], plan["sun"]["el"]],
                                         "ev": plan["light"]["ev"], "stars": plan["sky"]["stars"], "moon_on": plan["moon"]["on"],
                                         "far_seen": plan.get("far_seen")}, ensure_ascii=False))
        if plan.get("far_seen", {}).get("crest_T", 1.0) < 0.05:
            print(f"WARNING: the far terrain is hidden by haze — the transmittance to the crest at {plan['far_seen']['d_km']} km is {plan['far_seen']['crest_T']}; the mountains and the city "
                  "are not visible (except the fires). Visibility: air: {vis_km}, ground layer — ground_fog: {vis_km, height}", file=sys.stderr)
    elif cmd == "post":
        print("__RESULT__" + json.dumps({"lit": len(post(pos))}, ensure_ascii=False))
    elif cmd == "agx":                                   # AgX look tables — once per machine, outside Blender (a new install, a remote machine)
        _look_lut("base"); d = os.path.join(ROOT, "out", "agx_looks")
        print("__RESULT__" + json.dumps({"agx_looks": d, "n": len([f for f in os.listdir(d) if f.endswith(".npy")])}, ensure_ascii=False))
    else:
        sys.exit(f"unknown command {cmd}")

"""Scene wind — one model for all consumers: the description of the "light" section `far.yaml: wind` → the air speed at a point and a moment.
The parameters are the standard ones (main strength, turbulence, gusts: strength and frequency)
in physical units; strength in words follows the Beaufort scale (10-minute mean, RMetS); near the ground a logarithmic profile. Deterministic by
seed; no bpy — read by both plain Python and Blender. The response to wind belongs to the consumers: smoke (Effects), cloth (`components/cloth`), foliage.

    wind: {from: 250, beaufort: 4, gusts: 0.35, turbulence: 0.15, gust_period: 8, roughness: 0.05, seed: 7}
      from — where the wind blows from, ° on the compass (0 — from the north, 90 — from the east; world: +Y is north, +X is east); beaufort — a number 0–12
      or speed — m/s at a height of 10 m; gusts — a gust above the mean (0.35 — a gust 1.35 times stronger); turbulence — speed noise
      (0.1 open terrain … 0.25 city and forest); gust_period — mean interval between gusts, s; roughness — site roughness z0, m
      (0.03 meadow, 0.1 shrubs, 0.5–1 forest and city); seed — the seed.
    wind: [x, y] — the old form: where it blows to, length × 10 m/s at 10 m, no gusts (this is how Effects read it; far smoke columns — the bend).
    wind: 0 — calm. No entry — the old default [0.35, 0.12] (3.7 m/s), so that old scenes do not change."""
import math

import numpy as np

BEAUFORT = [   # force: (m/s at 10 m — from, to), what is seen on land (RMetS / Met Office, abridged)
    (0.0, 0.2, "calm: smoke rises vertically"),
    (0.3, 1.5, "smoke drifts, the weather vane stays still"),
    (1.6, 3.3, "wind felt on the face, leaves rustle, the weather vane turns"),
    (3.4, 5.4, "leaves and small twigs in motion, light flags extended"),
    (5.5, 7.9, "dust and paper are lifted, small branches sway"),
    (8.0, 10.7, "small leafy trees sway"),
    (10.8, 13.8, "large branches sway, wires hum, an umbrella is hard to hold"),
    (13.9, 17.1, "whole trees sway, walking against the wind is hard"),
    (17.2, 20.7, "twigs break off, walking is almost impossible"),
    (20.8, 24.4, "roof tiles are torn off, light structures are damaged"),
    (24.5, 28.4, "trees uprooted, buildings destroyed"),
    (28.5, 32.6, "devastation everywhere"),
    (32.7, 40.0, "hurricane"),
]
LEGACY = [0.35, 0.12]   # the light node's old default: where it blows to, × 10 m/s


def beaufort_of(u10):
    """Beaufort number from the speed at 10 m."""
    return next((b for b, (lo, hi, _) in enumerate(BEAUFORT) if u10 <= hi + 0.05), 12)


class Wind:
    """Scene wind: the mean speed at 10 m (u10), where it blows from (from_deg), gusts, turbulence, roughness, seed."""

    def __init__(self, u10=0.0, from_deg=0.0, gusts=0.0, turbulence=0.0, gust_period=8.0, roughness=0.05, seed=0, legacy=False):
        self.u10, self.from_deg, self.gusts, self.turbulence = float(u10), float(from_deg) % 360, float(gusts), float(turbulence)
        self.gust_period, self.roughness, self.seed, self.legacy = max(1.0, float(gust_period)), max(1e-3, float(roughness)), int(seed), legacy
        a = math.radians(self.from_deg); self.dir_to = (-math.sin(a), -math.cos(a))   # where it blows to (unit vector, world xy)
        rng = np.random.default_rng(self.seed + 4242)
        self._ph = rng.uniform(0, 2 * math.pi, (2, 3))                 # phases of the speed and direction noise
        self._fq = np.array([0.071, 0.173, 0.413])                      # Hz, incommensurable: noise with no visible repetition
        n = int(600 / self.gust_period) + 2                             # gusts for the next 10 min: intervals ±40 %, strength 0.6–1
        self._gt = np.cumsum(self.gust_period * rng.uniform(0.6, 1.4, n)) - self.gust_period * rng.uniform(0, 1)
        self._ga = rng.uniform(0.6, 1.0, n)

    @property
    def beaufort(self):
        return beaufort_of(self.u10)

    def describe(self):
        return BEAUFORT[self.beaufort][2]

    def profile(self, z):
        """Ratio of the speed at height z above the ground to the speed at 10 m (logarithmic, roughness z0)."""
        z0 = self.roughness
        return math.log(max(float(z), z0 * math.e) / z0) / math.log(10.0 / z0)

    def _noise(self, t, k):
        s = np.sin(2 * math.pi * self._fq * float(t) + self._ph[k])
        return float(s.sum() / math.sqrt(1.5))                          # a sum of three sines — unit standard deviation

    def _pulse(self, t):
        """Gust at time t (0…1): a smooth bump 0.4 of an interval long around each gust."""
        w = 0.4 * self.gust_period; d = np.abs(float(t) - self._gt); m = d < w / 2
        return float((self._ga[m] * 0.5 * (1 + np.cos(2 * math.pi * d[m] / w))).max()) if m.any() else 0.0

    def speed(self, t=0.0, z=10.0):
        """Speed, m/s, at height z at time t: mean × (1 + noise + gust − mean gust) × profile."""
        if self.u10 <= 0: return 0.0
        mean_p = 0.2 * 0.8                                              # fraction of time in a gust × mean strength
        f = 1.0 + self.turbulence * self._noise(t, 0) + self.gusts * (self._pulse(t) - mean_p)
        return max(0.0, self.u10 * f * self.profile(z))

    def direction(self, t=0.0):
        """Where it blows to at time t (unit xy): the direction swings by ±15° × (turbulence / 0.15)."""
        a = math.radians(self.from_deg + 15.0 * (self.turbulence / 0.15) * self._noise(t, 1) / 1.7)
        return (-math.sin(a), -math.cos(a))

    def at(self, t=0.0, z=10.0):
        """Air velocity (ux, uy), m/s, at height z at time t."""
        u = self.speed(t, z); d = self.direction(t)
        return (u * d[0], u * d[1])

    def mean_at(self, z=10.0):
        """Mean wind (ux, uy), m/s, at height z — without gusts and noise."""
        u = self.u10 * self.profile(z)
        return (u * self.dir_to[0], u * self.dir_to[1])

    def legacy_vector(self):
        """The light node's old vector [x, y] (where it blows to, length × 10 m/s) — the bend of far columns, the direction of clouds."""
        return [round(self.u10 / 10.0 * self.dir_to[0], 6) + 0.0, round(self.u10 / 10.0 * self.dir_to[1], 6) + 0.0]   # + 0.0: calm without −0

    def to_dict(self):
        return {"speed": round(self.u10, 3), "from": round(self.from_deg, 2), "beaufort": self.beaufort, "gusts": self.gusts,
                "turbulence": self.turbulence, "gust_period": self.gust_period, "roughness": self.roughness, "seed": self.seed,
                "legacy": self.legacy, "seen": self.describe()}


def resolve(spec, seed=0):
    """The `far.yaml: wind` description (a dict, the old vector [x, y], 0 — calm, None — the old default) → Wind."""
    if spec is None: spec = LEGACY
    if isinstance(spec, (int, float)) and not spec: return Wind(0.0, 0.0, seed=seed)
    if isinstance(spec, (list, tuple)):                                 # the old form: where it blows to, length × 10 m/s; no gusts
        x, y = float(spec[0]), float(spec[1]); u = 10.0 * math.hypot(x, y)
        return Wind(u, math.degrees(math.atan2(-x, -y)) % 360 if u > 0 else 0.0, seed=seed, legacy=True)
    s = dict(spec)
    if "speed" in s: u = float(s["speed"])
    else:
        lo, hi, _ = BEAUFORT[int(np.clip(int(s.get("beaufort", 3)), 0, 12))]; u = 0.5 * (lo + hi) if int(s.get("beaufort", 3)) else 0.0
    return Wind(u, float(s.get("from", 270.0)), float(s.get("gusts", 0.35)), float(s.get("turbulence", 0.15)), float(s.get("gust_period", 8.0)),
                float(s.get("roughness", 0.05)), int(s.get("seed", seed)))


def from_plan(P):
    """Wind from the light node's plan (the "light" step, provides light_plan): `wind` is the parsed description; if absent — the old fire vector."""
    w = P.get("wind") if P else None
    if isinstance(w, dict) and "speed" in w:
        return Wind(w["speed"], w["from"], w.get("gusts", 0.0), w.get("turbulence", 0.0), w.get("gust_period", 8.0),
                    w.get("roughness", 0.05), w.get("seed", 0), w.get("legacy", False))
    return resolve(((P or {}).get("fires") or {}).get("wind"), int((P or {}).get("seed", 0) or 0))


def of_scene(scene_dir):
    """The scene's wind from its description (the light node: far.yaml | far: in battle.yaml | the node default) — for steps before the "light" step
    (cloth, plants): the same parsing as in the light plan (far.resolve), without the sky computation."""
    from core.light import far
    sp = far.load_spec(scene_dir) or {}
    return resolve(sp.get("wind"), int(sp.get("seed", 7)))

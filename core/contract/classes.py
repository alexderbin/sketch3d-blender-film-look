"""Object classes for the maps: the sketch25 object kind (kind) → class, segmentation color, material tone, notan role.
Pure Python: read by both the Blender scripts and the numpy part. The segmentation palette is our own, not ADE20K: for
ControlNet-seg it must be translated separately by a verified table."""

# kind (sketch25) → class
KIND_CLASS = {
    "ground": "ground", "craters": "pit", "wall": "wall", "siege_tower": "siege", "tower": "tower", "ladders": "ladder",
    "box": "prop", "figure": "figure", "member": "member", "crowd": "member", "tree": "tree", "bush": "tree",
    "rocks": "rock", "rock": "rock", "ridge": "hill", "water": "water", "road": "ground", "mud": "ground",
    "scorch": "ground", "ledge": "wall", "stairs": "wall", "trench": "pit", "crater": "pit", "skyline": "hill",
}
DEFAULT_CLASS = "prop"

# class → (R, G, B) in the segmentation map; sky is separate
SEG_COLOR = {
    "sky": (120, 200, 240), "ground": (150, 120, 70), "pit": (90, 70, 50), "wall": (170, 170, 170), "tower": (200, 130, 130), "siege": (190, 110, 80),
    "ladder": (200, 160, 80), "prop": (230, 230, 60), "figure": (220, 40, 60), "member": (160, 20, 120), "tree": (40, 170, 60),
    "rock": (110, 110, 100), "hill": (120, 150, 110), "water": (50, 110, 230),
}
CLASS_ORDER = ["sky"] + [c for c in SEG_COLOR if c != "sky"]

# class → material tone (sRGB 0..1) for Eevee/Workbench, as in sketch25
ALBEDO = {"ground": 0.36, "pit": 0.50, "wall": 0.58, "tower": 0.58, "siege": 0.38, "ladder": 0.40, "prop": 0.50, "figure": 0.55,
          "member": 0.55, "tree": 0.40, "rock": 0.45, "hill": 0.45, "water": 0.30}

# notan role (sketch25.views.notan_design): 0 figure, 1 plant, 2 ground, 3 building
ROLE = {"figure": 0, "member": 0, "tree": 1, "ground": 2}      # everything else is a building (3)


def class_of(kind: str) -> str:
    return KIND_CLASS.get(kind, DEFAULT_CLASS)


def canon_name(obj_name: str) -> str:
    """OBJ object name → sketch25 id: figure__X → _X, ground_* → ground, the .001 suffix is dropped."""
    n = obj_name.split(".")[0]
    if n.startswith("figure_"):
        n = n[len("figure_"):]
    if n.startswith("ground"):
        n = "ground"
    return n

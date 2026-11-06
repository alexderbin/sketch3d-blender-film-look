"""The object contract: shared tags by which the steps understand each other. There is a single table, here; the copies in the
components' code (until they take it from here) are checked by check().
.venv/bin/python -m core.contract.contract check     → the copies' discrepancies; the exit code = their number
Pure Python: read by both Blender and .venv."""
import ast, os, sys

# object
SK_LAYER, SK_ID, SK_KIND = "sk_layer", "sk_id", "sk_kind"          # section, a stable id from the model, the class for maps
SK_STAND_IN = "sk_stand_in"                                          # 1 — a sketch stand-in within the model's extents (core/blender/stand_in.py), sk_id — the model's
# face (FACE INT)
SK_ROLE, SK_STONE, SK_SPACE, SK_SIDE = "sk_role", "sk_stone", "sk_space", "sk_side"
ROLES = {"wall": 0, "finish": 1, "roof": 2, "wood": 3, "metal": 4, "glass": 5, "mortar": 6, "rubble": 7,   # 0–7 — masonry
         "floor": 8, "opening": 9, "water": 10}                                                                       # 8–10 — architecture
ROLE_NAME = {v: k for k, v in ROLES.items()}
# collections by model section (build: a step that "adds" puts its objects into its own section's collection)
COLLECTIONS = ("site", "buildings", "constructions", "layout", "actors", "light", "fx")
MAX_SAMPLERS = 16                                                    # textures per shader — the Metal limit ("too many samplers (17)")
HUMAN_M = 1.8                                                        # the scale canon: a human is 1.8 m

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# where the copies of the roles table live today: file → dict name (code → name or name → code)
COPIES = {"core/surface/bl_matlib.py": "SK_ROLE"}                      # bl_arch.py and bl_stones.py import ROLES from here


def _dict_in(path, name):
    for node in ast.walk(ast.parse(open(path).read())):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    return None


def check():
    """The roles table copies in the components against the contract: every (name, code) pair of a copy is the same as in ROLES."""
    out = []
    for rel, name in COPIES.items():
        d = _dict_in(os.path.join(ROOT, rel), name)
        if d is None:
            out.append(f"{rel}: no {name}"); continue
        pairs = {(v, k) if isinstance(k, int) else (k, v) for k, v in d.items()}
        bad = [p for p in pairs if ROLES.get(p[0]) != p[1]]
        if bad: out.append(f"{rel}: {name} differs from the contract — {sorted(bad, key=lambda p: p[1])}")
    return out


if __name__ == "__main__":
    if sys.argv[1:2] != ["check"]:
        print(__doc__); sys.exit(2)
    R = check()
    print("\n".join(R) or f"contract: the role copies ({len(COPIES)}) match")
    sys.exit(len(R))

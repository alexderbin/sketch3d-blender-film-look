"""A scene copy for a run — page (page.py), shot (shot.py), site sheet (loc_sheet.py), take (take.py): links to the files
of the scene description as a single list. Previously each tool had its own list, and the copy lost materials and weather
(materials.yaml), the sky node (far.yaml, battle.yaml), the site (location.json — the node by biome, relief.npz — relief and water map,
relief_w.npz and surface.json — ground at the camera): the scene description did not reach the frame. The caller places the cameras itself.
The list now comes from the core's scene model (core/model/scene_model.py, files(copy=True)): the model sections and scene.yaml; FILES is
the previous list, for those who read it directly."""
import os, sys

from core.model import scene_model  # noqa: E402

FILES = ("scene.obj", "world.json", "figkit.json", "instances.json", "lib.json", "source.json", "edits.json",   # scene
         "far.yaml", "battle.yaml", "materials.yaml",                                                             # light, sky, weather
         "location.json", "relief.npz", "relief_w.npz", "surface.json")                                          # site from the plan


def link(src, dst, real=False):
    """Links dst/<file> → src/<file> for the description files the scene has; old links are redone. real — a link to the real
    path (through links: a copy of a copy). → the number of links."""
    os.makedirs(dst, exist_ok=True)
    n = 0
    have = scene_model.files(scene_model.load(src), copy=True)
    for f in sorted(set(FILES) | set(have) | {"scene.yaml"}):            # remove old links to files the scene does not have
        s, d = os.path.join(src, f), os.path.join(dst, f)
        if os.path.lexists(d):
            os.remove(d)
        if f in have and os.path.exists(s):
            os.symlink(os.path.realpath(s) if real else s, d)
            n += 1
    return n

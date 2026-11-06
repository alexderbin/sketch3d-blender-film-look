"""Blender's AgX looks (OCIO: GradingPrimary, GradingTone, CDL in the "AgX Log" space) — baked by OCIO itself from Blender into 3D
tables in our domain: a grid node is the log encoding of core/light/far.py agx (Rec.709 → E-gamut, log2 in [−12.47; 12.53]), we
convert it back to linear Rec.709 and run it through "scene_linear → sRGB display, AgX view, look" — the table is consistent
with our AgX (OCIO's own "AgX Log" encoding includes a Rec.2020 brightness compensation and does not match ours: the 99th
percentile of the difference is 48 levels).
blender -b --factory-startup -P core/blender/run.py -- core/blender/bl_agx_bake.py -- <folder> [N=57]   → <folder>/<look>.npy ([b][g][r][3], display 0…1) and __RESULT__"""
import json, os, sys
import numpy as np
import PyOpenColorIO as ocio
a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
dst = a[0]; N = int(a[1]) if len(a) > 1 else 57; os.makedirs(dst, exist_ok=True)
M_XYZ_709 = np.array([[3.2409699, -1.5373832, -0.4986108], [-0.9692436, 1.8759675, 0.0415551], [0.0556301, -0.2039770, 1.0569715]])
M_XYZ_EG = np.array([[1.5250528, -0.3159135, -0.1226583], [-0.5091526, 1.3333274, 0.1382844], [0.0957153, 0.0508974, 0.7879558]])
M_EG_709 = np.linalg.inv(M_XYZ_EG @ np.linalg.inv(M_XYZ_709)); LG_MIN, LG_MAX = -12.47393, 12.5260688117   # as in far.py
import bpy                                                   # the OCIO config of this Blender — on any system (not a macOS path)
cfg = ocio.Config.CreateFromFile(os.path.join(bpy.utils.system_resource("DATAFILES", path="colormanagement"), "config.ocio"))
g = np.linspace(0.0, 1.0, N); B, G, R = np.meshgrid(g, g, g, indexing="ij")
lin = (2.0 ** (np.stack([R, G, B], -1).reshape(-1, 3) * (LG_MAX - LG_MIN) + LG_MIN)) @ M_EG_709.T
out = {}
for look in [""] + [l.getName() for l in cfg.getLooks() if l.getName().startswith("AgX")]:
    lv = ocio.LegacyViewingPipeline(); lv.setDisplayViewTransform(ocio.DisplayViewTransform(src="scene_linear", display="sRGB", view="AgX"))
    lv.setLooksOverrideEnabled(True); lv.setLooksOverride(look)
    img = np.ascontiguousarray(lin, np.float32); lv.getProcessor(cfg).getDefaultCPUProcessor().applyRGB(img)
    name = (look or "AgX - Base").replace("AgX - ", "").replace(" ", "_").lower()
    np.save(os.path.join(dst, name + ".npy"), img.reshape(N, N, N, 3).astype(np.float32)); out[name] = look
print("__RESULT__" + json.dumps({"N": N, "looks": out}, ensure_ascii=False))

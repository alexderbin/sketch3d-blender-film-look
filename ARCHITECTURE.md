# Architecture

A scene is staged in 3D once and rendered many times: panels, frames, map packs. The code is a set of working
mechanisms; the choice of *what* to build is written in a recipe, never hidden in code.

## Folders

- `core/` — the core:
  - `manager/` — runs recipes (`pipeline.py`), sample recipes in `pipelines/`, passports of the core outputs (`outputs.yaml`);
  - `model/` — the scene model: files of a scene folder and their checks;
  - `blender/` — the Blender entry (`run.py` runs a script inside Blender, `build.py` runs one step over the `.blend`),
    shared Blender building blocks (`bl_kit.py`), stand-ins for sketches, a persistent "warm" Blender (`bl_serve.py`);
  - `outputs/` — frames, map packs (depth, normals, ids, pose skeletons), contact sheets, video, cut;
  - `light/` — the light engine: sky, sun, haze, the distant terrain belt, AgX, wind;
  - `surface/` — ground and surface materials, near and far terrain shading;
  - `layers.py` — the layer contract between components; `services/` — resource guards and tools.
- `components/<name>/` — one component: passport `component.yaml`, steps `steps.py` (plan and build), public entry
  `api.py` when neighbours need its mechanism, knowledge in YAML, benches in `bench/`. A component changes on its own,
  without edits to the core or its neighbours.
- `formats/<name>/` — thin steps of an output format (for example, a video shot) over the components.
- `library/` — catalogs of assets and knowledge data; downloaded files stay out of git.
- `bench/` — sample scenes and stands used by the tests.

## Recipe and run

A recipe is YAML: `pipeline, scene, profile (sketch | preview | final), build: stepwise, steps: [step: {keys}],
outputs: [frames: {…}, cut: {…}]`. Run:

```sh
.venv/bin/python -m core.manager.pipeline <recipe.yaml> [--scene bench/scenes/X] [--profile sketch|preview|final]
    [--dry] [--fresh] [--only step[,step]] [--from step] [--until step]
```

For each step: `plan` (Python, data only) → `build` (Blender over the `.blend` of the previous step) → a journal line:
what was created, which layers were written, what was skipped. The cache key of a step is its code, keys, input
`.blend` and the part of the scene it reads, so an unchanged recipe replays from cache. Heavy work runs under
`core.services.guard` (one Blender per machine, disk and memory checks; `memguard` caps and pauses processes instead
of letting the machine swap to death; `executor` runs independent jobs in parallel as memory allows). The frames
output records the cost of every frame (seconds per frame; the heaviest collections by triangles and volumes) and
writes a contact sheet.

**Profiles.** `sketch` builds stand-ins of every component and renders with Workbench — cameras and timing are cheapest
to decide here; approved cameras are pinned (`core.services.stage pin`) and do not move on `preview` and `final`.
A passport can set per-profile keys: `profiles: {profile: {keys} | plan | off}` (plan only, no Blender | step skipped);
a recipe step can override them with `stages:`. Each profile has its own run folder.

## Step passport (`component.yaml`)

```yaml
steps:
  step_name:
    plan: steps.py:plan_x          # Python: data
    build: steps.py:build_x        # Blender: objects in the step collection, tagged sk_*
    keys: {key: default}
    reads: [layers]                # what the step needs from the scene model
    writes: [own layer]
    # optional: may_read, gives, scene_files, cost, profiles, code
hooks: {point: module:function}   # extension points of the core (core/hooks.py)
```

Hooks let the core call a component without importing it: `light` (after the scene light), `maps_before` /
`maps_after` (map packs), `frame_near`, `frame_materials`, `weather` (weather presets).

## Contract between components

A component reads scene layers and the core, writes only its own layer, and never reads the plans or internal modules of
its neighbours. Layers (formats — in the header of `core/layers.py`):

- relief (`location.json`, `relief.npz`) — written by Site (Buildings refine it through Site's `api`); read through
  `core.surface.ground`;
- occupancy (`occupancy.json`: houses, plots, streets, yards, walls, towers, road, palette) — Buildings;
- anchors (`anchors.json` plus anchors of the site: `castle.gate`, `town.end.N`, `x.ford`) — Buildings, Site;
- masonry surfaces (`surfaces.json`, strips `core.surface.strip`) — Buildings; canopy (`canopy.json`) — Plants;
- settlement influence raster (`influence.npz`) — Buildings; cameras (`cameras.json`) — Camera.

A mechanism has one owner; neighbours and formats call its public entry `components/<name>/api.py`. Whatever many
components need lives in the core: Blender building blocks (`core/blender/bl_kit.py`: node helpers, points,
prototypes, keys, shot camera), the cut as the `cut` output. The core never imports components.

## Adding a component

1. Copy `core/templates/component` to `components/<name>/`; fill the passport: steps, keys with defaults, `reads`/`writes`.
2. `steps.py`: `plan` returns data (Python), `build` creates objects in the step collection with `sk_*` tags (Blender).
3. Check it: a recipe with `--until <step>`, a stand in `bench/`, and the frames themselves (contact sheet).

## Checks

`.venv/bin/python -m pytest tests` — compile, static import check, scene model and manager stands; no Blender.
Component benches (`components/<name>/bench/`) render reference sheets and compare numbers with stored references.

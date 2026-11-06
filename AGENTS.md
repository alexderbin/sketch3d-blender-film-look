# AGENTS.md — how an AI agent drives this repository

This repository is built to be operated by an AI coding agent (or a script) from the command line. Blender runs
headless; everything the agent needs is in files and in one-line JSON answers. Run all commands from the repository
root.

## Setup (once)

```sh
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m core.services.venv_root      # makes the repository root importable for every script
.venv/bin/python -m core.light.far agx           # bakes the AgX look tables (runs Blender for ~10 s)
.venv/bin/python -m pytest tests                 # smoke tests, no Blender
```

Blender 5.2 must be on `PATH` as `blender`. Downloadable assets (CC0 scans and textures) are fetched by the commands in
the component READMEs.

## Build a scene

1. **Write a recipe** — or start from `core/manager/pipelines/*.yaml`. A recipe may use the steps of the components
   present under `components/` (a standalone component repository has fewer of them; its README shows a recipe). With
   all components (the full repository) a recipe looks like this:

   ```yaml
   pipeline: my_scene
   scene: bench/scenes/castle_rock          # the scene folder the run copies
   profile: sketch                          # sketch | preview | final
   build: stepwise
   steps:
     - relief: {preset: fortress_hill, seed: 7}
     - camera: {view: hill, azimuth: 200, elevation: 10, distance: 170, res: 640x360}
     - light: {time: day}
   outputs:
     - frames: {video: false}
   ```

2. **Find the keys.** Every step is described in its component passport `components/<name>/component.yaml`:
   `steps.<step>.keys` (defaults), `values` (types and allowed values), `description` (what each key does), `reads` and
   `writes` (which scene layers it needs and fills), `hint` (where the step goes in a recipe).
3. **Check without Blender:** `.venv/bin/python -m core.manager.pipeline <recipe.yaml> --dry` — step order, the layer
   contract and the keys; nothing is built.
4. **Sketch:** `.venv/bin/python -m core.manager.pipeline <recipe.yaml> --profile sketch` — stand-ins and Workbench frames,
   about 0.1 s per frame. Look at the contact sheet before spending render time.
5. **Iterate:** change keys and re-run; unchanged steps come from the cache. Partial runs: `--only step[,step]`,
   `--from step`, `--until step`; `--fresh` ignores the cache; `--out <folder>` sets the run folder.
6. **Pin the approved cameras** so later profiles keep them:
   `.venv/bin/python -m core.services.stage pin <run folder> --names cam_,orbit_`.
7. **Render:** `--profile preview` (Eevee) or `--profile final`.

## Read the results

- The last line of a run is `__RESULT__{"pipeline", "out", "failures", "done", "cached", …}`; the exit code is the
  number of failed steps.
- The run folder (`out` in the result) holds `journal.md` (the run summary) and `journal.jsonl` (one line per step:
  what was created, which layers were written, what was skipped, seconds), one folder per step with its checkpoint, the
  frames with `contact.jpg`, the cost of every frame and, with maps on, depth / normal / id / pose maps per camera.
- Blender scripts started directly print the same kind of `__RESULT__{…}` line with their answer.

## Look at a scene quickly

```sh
# a picture of a scene folder or a .blend in a fraction of a second: through camera K, the viewport or a frame strip
blender -b --factory-startup -P core/blender/run.py -- core/blender/bl_view.py -- <scene folder | file.blend> <out.png> camera 1
# a warm Blender that keeps the scene in memory: look, inspect, run Python, stop
.venv/bin/python -m core.blender.bl_call <scene folder> look 1
.venv/bin/python -m core.blender.bl_call <scene folder> py '<code>'
.venv/bin/python -m core.blender.bl_call <scene folder> stop
```

## Rules

- **One heavy job at a time.** Wrap long renders and bakes: `.venv/bin/python -m core.services.guard heavy -- <command>`
  (waits for the shared lock, checks disk space and memory). `.venv/bin/python -m core.services.guard check` shows the
  state of the machine.
- **Choices go into the recipe.** What to build is written in keys; components decide how and record why in the
  journal. A new behaviour is a new key in the passport, not a hidden switch in code.
- **Layers are the contract.** A component writes only its own layer and reads the others through scene layers and the
  public `api.py` of a neighbour; the core never imports components. See [ARCHITECTURE.md](ARCHITECTURE.md).
- **Check after changes:** `.venv/bin/python -m pytest tests` (compile, imports, scene model, dry runs of all recipes).

# Post — film look and color grading for renders in Python: depth of field, halation, palette transfer, grain, letterbox

Post is a film-look and color-grading tool for rendered frames, written in Python. It grades the frames of a finished
run, or any PNG: lens effects by depth (depth of field, aerial haze, chromatic aberration, bloom, vignette), film
response (halation, palette transfer from a reference, tone, sun glow, grain, dithering) and a letterbox. It is pure
Python with NumPy and OpenCV — no Blender, no GPU, about 1.6 s per 1280×720 frame. `post` is an *output*, not a scene
step: it reads the frames of the run's last `frames` output and writes graded copies; the scene, the `.blend` and the
map packs (depth, ids, lines) stay as they were, so changing the preset re-runs only the grade.

**Use it to:**

- Give rendered frames a cinematic trailer look — a reference color palette, bloom, halation, vignette and a 2.39:1
  letterbox — in one line: `post: {preset: trailer}`.
- Make an anime-style composite: a softer palette, bloom, a painterly distance, a sun glow and no letterbox —
  `post: {preset: anime}`.
- Add depth of field and aerial haze to a finished render from its depth pass —
  `post: {fstop: 2.8, focus_m: 40, haze_km: 12}`.
- Match the colors of a reference video or an image folder without storing its pixels (palette transfer) —
  `components.post.palettes add <name> <video | image folder>`, then `post: {palette: <name>, mix: 0.6}`.
- Grade any PNG from the command line, or re-grade a finished run with another preset without re-rendering —
  `components.post.look image.png out.png --preset trailer`, `components.post.steps <run folder> preset=anime`.
- Add film grain for a long 16:9 video; dithering against banding in dark gradients is already in every preset —
  `post: {grain: 0.03}`.

## Features

- **Physical order**, fixed in code (`look.ORDER`): haze → painterly distance → depth of field → chromatic aberration →
  bloom → vignette → halation → palette → grade → sun glow → grain → dither → letterbox. Blurs and light mixing run in
  linear colour, whatever order a preset lists its operations in.
- **Aerial perspective** by Koschmieder's law: haze share `1 − exp(−3.912·d/V)` for visibility `V` in km; the haze colour
  is sampled at the horizon, the sky is left alone.
- **Depth of field from the frame's real lens** (focal length and sensor from the camera's `far.json`): circle of
  confusion `c = f²/(N(s − f))·|d − s|/d`, gathered in layers so the far blur never bleeds over the near;
  focus defaults to the median depth of the frame centre; f-numbers below f/0.95 are clamped to avoid the
  miniature (tilt-shift) look unless asked for explicitly.
- **Painterly distance**: a Kuwahara filter beyond a distance with a soft transition, as in anime compositing.
- **Bloom** at three widths (0.4, 2 and 8 % of the frame width); **halation** — a narrow red-orange halo around the
  brightest highlights, the light reflected by the film base.
- **Palette transfer** from a reference by the MKL linear map (Pitié & Kokaram 2007), matching colour mean and
  covariance. One transform for the whole run, so frames do not flicker; fire and flashes are protected by a
  saturation × value mask. A reference is stored as numbers only (mean, covariance, brightness, saturation, warm−cool):
  seven are bundled (CG trailers, stylised CG, CG short films, a live-action trailer), and you can measure your own
  from a video or an image folder.
- **Sun glow** (the "para" layer of anime compositing): a soft screen-blended gradient from the sun's position in frame.
- **Grain** per frame, strongest in the mid-tones; **dither** — triangular noise of ±1 code value before 8 bits (no
  banding in dark gradients); **letterbox** to an aspect ratio, skipped for frames that are already as tall (9:16).
- **Presets**: `trailer` (default), `anime` (half-strength palette, soft bloom, painterly distance, sun glow, no
  letterbox or aberration), `crush` (crushed shadows, clipped highlights, near-monochrome warm copper), `clean` (dither
  only — for comparisons and for images handed to image models).
- A before/after contact sheet, `post.json` with the operations applied, an optional video of the graded frames.

## How it works

`steps.py:plan_post` runs in Python only. It takes the camera folders `NN_frames/cam<K>/` of the run's last frames output
(`lit.png`, depth from the raw passes, `far.json` with lens, sensor and sun), merges the preset with the step keys
(`look.resolve`), computes the palette transform once from the colour statistics of all frames and calls `look.apply`
per frame → `cam<K>.png`. Haze and depth of field are off in the presets and are switched on by the `haze_km` and
`fstop` keys; without a depth pass they are skipped and the reason is noted in the step result. Knowledge:
`presets.yaml` (preset → operation → knobs; `false` turns an operation off) and `palettes.yaml` (reference statistics).

## Steps and keys

| Step | Reads → writes | Keys (defaults) |
|---|---|---|
| `post` (output, last in `outputs:`) | frames of the run → `cam<K>.png`, `contact.jpg`, `post.json`, video; nothing in the scene | `preset` (trailer), `palette` (preset's), `mix` (preset's, 0–1), `haze_km` (off), `fstop` (off), `focus_m` (frame centre), `grain` (off, 0–0.1), `letterbox` (2.39 in `trailer`; 0 — none), `video` (false), `fps` (12) |

The `sketch` profile skips `post` (sketch frames carry no depth). All keys with descriptions:
[component.yaml](/components/post/component.yaml)

## Usage

```yaml
outputs:
  - frames: {video: false}
  - post: {preset: trailer}     # {preset: anime}; {preset: trailer, palette: dark_cool, mix: 0.5, letterbox: 0}
```

```sh
# another preset on a finished run → <run>/post_<preset>/ (the pipeline's own output is not touched)
.venv/bin/python -m components.post.steps <run folder> preset=anime
# one camera folder or any image (+ depth and camera for the depth-based operations)
.venv/bin/python -m components.post.look <run>/NN_frames/cam1 out.png --preset trailer
.venv/bin/python -m components.post.look image.png out.png --preset trailer --depth depth.exr --camera far.json --set dof.fstop=2 palette.mix=0.5
# a reference palette of your own: 16 frames between 20 and 80 % of a video, or an image folder
.venv/bin/python -m components.post.palettes add <name> <video.mp4 | image folder> --note "what it is"
.venv/bin/python -m components.post.palettes list
# bench: 7 checks on a run folder with NN_frames; a sheet of all presets in bench_out/ next to it
.venv/bin/python -m components.post.bench.bench <run folder>
```

## Driving it from an AI agent

An agent adds `post` as the last output of a recipe, after `frames`:

```yaml
outputs:
  - frames: {video: false}
  - post: {preset: anime, palette: dark_cool, mix: 0.5, fstop: 2.8, focus_m: 40, haze_km: 12}
```

- Check without Blender: `.venv/bin/python -m core.manager.pipeline <recipe.yaml> --dry`; a wrong `preset` or
  `palette` is refused with the list of allowed names. All keys with defaults and descriptions:
  `components/post/component.yaml`.
- A run prints one `__RESULT__{…}` JSON line (`out` is the run folder) and leaves `journal.md` there; the `NN_post/`
  folder has `cam<K>.png`, a before/after `contact.jpg`, `post.json` (the operations applied) and, with `video: true`, a
  clip. The step result notes what was skipped, for example `dof: no depth ×4`.
- Request → keys: "a cold, vivid look" → `palette: cold_vivid` (warm−cool −17.7, saturation 0.557); the command
  `.venv/bin/python -m components.post.palettes list` prints these numbers for all seven palettes. "Blurred
  background" → `fstop`, `focus_m`; "misty distance" → `haze_km`; "no black bars" → `letterbox: 0`.

## Files

| File | Role |
|---|---|
| `look.py` | operations and their order (`ORDER`), preset resolution, single-frame CLI |
| `steps.py` | the `post` output: run frames → graded frames, contact sheet, `post.json`, video; re-grading a finished run |
| `presets.yaml` | presets `trailer`, `anime`, `crush`, `clean`: operation → knobs |
| `palettes.yaml`, `palettes.py` | reference palettes (colour mean and covariance only) and the tool that measures a new one |
| `bench/bench.py` | `clean` keeps the frame, the palette moves mean and saturation toward the reference, depth of field blurs only off focus, Kuwahara touches only the distance, exact letterbox height, cost per frame |

## Assets

None. Reference palettes are colour statistics measured once from public videos; no pixels are stored or downloaded.

## Limitations

- Grades display-referred 8-bit frames (`lit.png`, after the AgX view transform); grading before the tone curve needs a
  linear output of the light pass, which is not exposed yet.
- The core `cut` montage does not pick up graded frames yet; use `video: true` for a video of the graded frames.
- Event-driven light flashes are not part of the grade (they belong to the light component).
- No grain in the presets: low-bitrate platform encoders (vertical shorts) erase it and pay for it with shadow detail —
  set `grain` only for long 16:9 video at a high bitrate.
- All frames of a run are held in memory at once (one palette transform for the run).
- Checked on stills and short runs, not yet on a long edited video.

## FAQ

**How do I add a cinematic film look — halation, bloom, vignette, grain, letterbox — to rendered frames in Python?** Put
`post: {preset: trailer}` after `frames` in a recipe. For any PNG, run the grade directly; `--set operation.knob=value`
changes a knob:

```sh
.venv/bin/python -m components.post.look image.png out.png --preset trailer --set halation.strength=0.2 grain=0.03
```

**Can I use it without Blender or without the rest of sketch3d?** Yes: it is NumPy and OpenCV only and ships as its own
repository, `sketch3d-blender-film-look`. `components.post.look` grades any PNG; haze and depth of field need a depth map
(`--depth depth.exr` or `.npy`, with `--camera far.json`) and are skipped without one.

**Does it need a GPU, and how fast is it?** No GPU: it runs on the CPU at about 1.6 s per 1280×720 frame. All frames of
a run are held in memory at once, because the palette transform is computed once for the whole run.

**How do I copy the color mood of a reference video or a set of images?** Measure it once; only the color mean and
covariance of 16 frames are stored in `components/post/palettes.yaml`, no pixels:

```sh
.venv/bin/python -m components.post.palettes add <name> <video.mp4 | image folder> --note "what it is"
```

Then use `post: {palette: <name>, mix: 0.7}`. Seven palettes are bundled.

**How do I add depth of field and haze to a finished render?** Give `post` an `fstop` (and `focus_m`; by default the
focus is the median depth of the frame centre) and `haze_km`: the blur follows the frame's real lens and its depth
pass, and both operations are skipped, with a note in the step result, when there is no depth pass.

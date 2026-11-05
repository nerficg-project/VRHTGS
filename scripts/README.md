# VRHTGS scripts

Helper scripts for the VRHTGS performance evaluation and VR perceptual study. All scripts need to be copied to 
the script directory of the nerficg repository root and are run using 
`./scripts/<scriptname>.py` (or `python ./scripts/<scriptname>.py` on Windows):

```bash
# From the repository root
cp ./src/Methods/VRHTGS/scripts/* ./scripts
```
```bash
./scripts/<scriptname>.py
```

| Script | Purpose |
| --- | --- |
| `vrhtgs_create_experiment_config.py` | Generate shuffled per-participant experiment config CSV files for the perceptual study. |
| `vrhtgs_export_model.py` | Extract a trained model into a `.ply` + optional background model for use in our SIBR Viewer fork. |
| `vrhtgs_export_trajectory.py` | Render trajectory images to disk for every row of an evaluation manifest. |
| `vrhtgs_fvvdp_eval.py` | Score exported trajectories against a reference row with the foveated metric FovVideoVDP. |
| `vrhtgs_performance_eval.py` | Benchmark rendering performance for every row of an evaluation manifest. |
| `vrhtgs_utils.py` | Shared helpers (manifest handling, `get_stereo_pair`, `get_vr_views`) used by the eval scripts. |

All rendered outputs, benchmarks, and experiment configs land under
`<repo>/output/`.

## Evaluation manifests

`vrhtgs_export_trajectory.py` and `vrhtgs_performance_eval.py` both take a `--manifest`
describing one comparison: each row is a trained model, and all rows are measured over an
identical trajectory so their numbers can be put side by side. See `example_manifest.yaml`
next to this file, and `../benchmarks/manifests/` for the manifests used in the paper.

Paths (including the `--manifest` argument and the `run_dir` entries inside it) resolve against the
current working directory, so run the scripts from the repository root. A `run_dir` may be a glob
pattern, in which case the most recent matching run directory is used (based on directory name).

By default, a row's render method comes from its own run directory's `training_config.yaml`, so
HTGS and VRHTGS runs can appear in one manifest without extra configuration. The optional `method`
entry overrides it: the model is still loaded exactly as trained, but rendered with that method's
renderer. This is how the paper compares HTGS and Ours (FR) on the same model. Rows need
to be trained on the same scene.

Results are collected into `output/<manifest_stem>_<timestamp>/`, and the path is printed when the
script finishes. Every row writes its config, including method type, renderer class, background
model as loaded from the checkpoint, AAA mode, GPU, primitive count, view count and resolution.

---

## vrhtgs_create_experiment_config.py

Creates shuffled experiment configs for the perceptual study. Each generated CSV
file is a per-participant list of conditions: any number of calibration
conditions (sampled from a calibration directory) followed by every experiment
condition (from an experiment directory).

### Usage

```bash
# Generate 10 experiment configs (participants 0..9)
python ./scripts/vrhtgs_create_experiment_config.py

# Generate 5 experiment configs
python ./scripts/vrhtgs_create_experiment_config.py --num-experiments 5

# Also generate the calibration mode config (participant -1)
python ./scripts/vrhtgs_create_experiment_config.py --calibration-config
```

Arguments:

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--num-experiments` | int | `10` | Number of experiment configs (`experiment_config_0.csv` .. `experiment_config_{n-1}.csv`). |
| `--calibration-config` | flag | off | Also write the calibration mode config as participant `-1` (`experiment_config_-1.csv`). |

### What it generates

Without `--calibration-config`, one file is written per experiment participant id
`0 .. num-experiments - 1`. Each file starts with sampled calibration conditions
and is followed, after shuffling, by all experiment conditions.

*Calibration conditions* (sampled, one each, from `output/vr-scenes/calibration/`):
- 1 HTGS condition (from `htgs/`)
- 1 foveated VRHTGS condition (from `htgs/` since it uses the same model as the HTGS condition)
- 1 anti-aliased VRHTGS condition (from `htgs-fov/`)
- 1 anti-aliased + background VRHTGS condition (from `htgs-bgm/`)

*Experiment conditions* (all, from `output/vr-scenes/experiment/`, then shuffled):
- all HTGS conditions (from `htgs/`)
- all foveated VRHTGS conditions (from `htgs/` since it uses the same models as the HTGS conditions)
- all anti-aliased VRHTGS conditions (from `htgs-fov/`)
- all anti-aliased + background VRHTGS conditions (from `htgs-bgm/`)

With `--calibration-config`, an additional file is written for participant `-1`
containing only HTGS conditions, collected from both
`output/vr-scenes/calibration/htgs/` and `output/vr-scenes/experiment/htgs/` (not
shuffled). This is intended for the calibration mode of the study, used to
determine initial camera parameters (rotation, position, and scaling) to
align the scene with the real-world VR calibration.

### Expected directory structure

The script reads model directories under `output/`; each subdirectory of the
listed method directories becomes one condition.

```
output/
  vr-scenes/
    calibration/
      htgs/          # HTGS + foveated VRHTGS models (calibration)
      htgs-fov/      # anti-aliased VRHTGS models (calibration)
      htgs-bgm/      # anti-aliased + background VRHTGS models (calibration)
    experiment/
      htgs/          # HTGS + foveated VRHTGS models (experiment)
      htgs-fov/      # anti-aliased VRHTGS models (experiment)
      htgs-bgm/      # anti-aliased + background VRHTGS models (experiment)
```

### Output

Files are written to `output/`:

```
output/
  experiment_config_0.csv
  experiment_config_1.csv
  ...
  experiment_config_{n-1}.csv
  experiment_config_-1.csv   # only with --calibration-config
```

Each file contains `<calibration> + <experiment>` condition rows (one per line).
The script prints the number of conditions written to each file.

### Calibration scenes

The calibration block shows each variant (HTGS, foveated, +AA, +AA+BGM) once,
each in a different scene. One model is drawn per variant from `calibration/<variant>/`, 
skipping any model whose directory name begins with the same scene token (the
part before the first _) as an earlier draw. Name the calibration models `<scene>_<anything>`
(e.g. `garden_2026-01-01-12-00-00`) and provide at least four distinct scenes,
otherwise sampling fails. Experiment conditions are not filtered: every model
in `experiment/<variant>/` is used.

---

## vrhtgs_export_model.py

Extracts a trained model into a `.ply` file for use in our SIBR Viewer fork.
The model is loaded from the run directory (`training_config.yaml` +
`checkpoints/final.pt`) and serialized via its `get_ply_dict()`. If the model
has a background model, its data is exported as well; `-b` skips the background
export.

### Usage

```bash
# From the repository root
python ./scripts/vrhtgs_export_model.py -d output/VRHTGS/<model_dir>

# Export several runs at once, without the background model
python ./scripts/vrhtgs_export_model.py -d output/VRHTGS/<model_dir1> output/VRHTGS/<model_dir2> -b
```

Arguments:

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `-d`, `--dir` | path(s) | required | One or more directories containing the outputs of a completed training. |
| `-b`, `--ignore_background_model` | flag | off | Do not export the background model (SH coefficients or environment texture) even if present. |

### Output

For each run directory, `final.ply` is written into the run directory itself.
If the model has a background model (and `-b` was not passed), its data is
written next to it depending on its type: `background_sh_coefficients.bin`
(raw float32 SH coefficients) or `background_texture.exr` (half precision RGB
OpenEXR, tagged as a latitude-longitude environment map, at the resolution given
by `MODEL.ENVIRONMENT_MAP_WIDTH` and `MODEL.ENVIRONMENT_MAP_HEIGHT`). For the
`TEX` background model the texture stores pre-activation values, so the viewer
applies the sigmoid output activation after bilinear interpolation; the baked
`MLP` texture is already post-activation.

---

## vrhtgs_export_trajectory.py

Renders trajectory images to disk for every row of a manifest. Each frame is rendered
with the VR emulation parameters of the Meta Quest Pro headset (eye distance,
per-eye projection centers, focal length, and resolution from the camera
configuration), as left/right eye pairs with `--vr` or as single flat images.

The script computes no quality metrics. The framework's uniform full-image
metrics (PSNR / SSIM / LPIPS) do not account for foveated rendering and would penalize the
periphery for being deliberately approximated. Perceptual comparison is done separately with the
foveated metric FovVideoVDP (`vrhtgs_fvvdp_eval.py`), over the image sequences this script produces.
All rows share one trajectory, so frame `i` of any row corresponds to frame `i` of any other.

### Usage

```bash
# From the repository root
python ./scripts/vrhtgs_export_trajectory.py --manifest output/my_comparison.yaml -i 10 --vr

# Use a different camera configuration
python ./scripts/vrhtgs_export_trajectory.py --manifest output/my_comparison.yaml --camera-config ./scripts/<other_headset>.yaml
```

Arguments:

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--manifest` | path | required | Evaluation manifest listing the rows to export. |
| `-i`, `--interpolate-frames` | int | `-1` | If set to a value > 0, interpolates n poses between each set of ground truth poses. |
| `--vr` | flag | off | Renders left/right eye pairs instead of single flat images. |
| `--camera-config` | path | `meta_quest_pro.yaml` next to the script | YAML file with the VR camera configuration. |
| `--testset-only` | flag | off | Only considers views from the test set for interpolation. |
| `--no-visibility-mask` | flag | off | Renders the full image instead of only the area visible through the headset lenses (visibility mask index `-1`). Use it when comparing against renderers without a lens mask, e.g. unfoveated HTGS, where the masked-out corners would otherwise count as errors. |

### Output

`output/<manifest_stem>_<timestamp>/` containing one directory per row (`<name>/left/` and
`<name>/right/` with `--vr`, flat numbered images otherwise), a `<name>/config.txt` per row, and an
`index.txt` describing the layout.

---

## vrhtgs_fvvdp_eval.py

Scores the rows of a `vrhtgs_export_trajectory.py` output directory against one reference row with
[FovVideoVDP](https://github.com/gfxdisp/FovVideoVDP) in foveated mode. The trajectory consists of one
segment of `1 + n` frames per dataset view (`-i n`); every segment is scored as a short video, and a
row's JOD is the mean over all segments. The display model emulates the Meta Quest Pro
(diagonal field of view 111.24°, 3 m viewing distance, 100 cd/m² peak luminance, contrast 3000:1).
Without `--gaze`, the fixation point is the image centre, which is also where the foveated renderer
places the fovea when no gaze is given. Each eye of a stereo export is scored separately.

pyfvvdp is not a framework dependency; install it into the environment first:

```bash
pip install pyfvvdp==1.2.2
```

### Usage

```bash
# From the repository root: export the reference and the foveated renderer along the same trajectory ...
python ./scripts/vrhtgs_export_trajectory.py --manifest output/my_comparison.yaml --vr --testset-only -i 30 --no-visibility-mask
# ... then score every row against the row named "Reference"
python ./scripts/vrhtgs_fvvdp_eval.py output/my_comparison_<timestamp>
```

Arguments:

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `export_dir` | path | required | Output directory of `vrhtgs_export_trajectory.py`. |
| `--reference` | str | `Reference` | Name of the reference row. |
| `--rows` | str | all other rows | Comma-separated rows to score. |
| `--eyes` | str | all exported eyes | Comma-separated eyes to score for stereo exports, e.g. `left`. |
| `--window` | int | `31` | Frames per scored segment, i.e. `1 + n` for an export with `-i n`. Also used as the frame rate. |
| `--gaze` | int int | image centre | Fixation point in pixels. |

### Output

`fvvdp_table.txt` (mean JOD per row and eye) and `fvvdp_results.json` (per-segment JODs, resolution,
fixation point, pixels per degree, pyfvvdp version), both written into `export_dir`. The exported
images are large (about 2 × (1 + n) × #views PNGs per row), so delete them once scored.

---

## vrhtgs_performance_eval.py

Benchmarks rendering performance for every row of a manifest, with VR emulation parameters.
Each frame is rendered for both eyes with the stereo offsets of the Meta Quest
Pro headset (eye distance, per-eye projection centers, focal length, and
resolution from the camera configuration).

Every row is measured with each render strategy its renderer supports: `per_eye`, which
renders one eye at a time, and `stereo`, the fused both-eyes-in-one-pass path, where the
renderer provides one. Both strategies run in the same process with the same warmup, views and
GPU state, so drift between runs does not affect the stereo speedup.

### Usage

```bash
# From the repository root
python ./scripts/vrhtgs_performance_eval.py --manifest output/my_comparison.yaml -i 10 --repetitions 4

# Use a different camera configuration
python ./scripts/vrhtgs_performance_eval.py --manifest output/my_comparison.yaml --camera-config ./scripts/<other_headset>.yaml
```

Arguments:

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--manifest` | path | required | Evaluation manifest listing the rows to benchmark. |
| `-i`, `--interpolate-frames` | int | `-1` | If set to a value > 0, interpolates n poses between each set of ground truth poses. |
| `--repetitions` | int | `1` | Number of repetitions for rendering each view to obtain better averages. |
| `--camera-config` | path | `meta_quest_pro.yaml` next to the script | YAML file with the VR camera configuration. |

### Camera configuration

`meta_quest_pro.yaml` ships next to the script and contains the VR emulation
parameters of the Meta Quest Pro headset: `resolution`, `focal_length`,
`center_left`, `center_right`, and `eye_distance`. Pass a different file via
`--camera-config` to benchmark other headsets.

### Conditions

Each condition is a separate manifest row. There are no flags to switch off anti-aliasing or the
background model at render time: both affect training, so disabling them at render time would
measure a model under conditions it was never fit for. To benchmark "no background model",
train with `MODEL.BACKGROUND_MODEL: NONE` and give that run its own row. Conditions that only
change the renderer, such as foveated vs. regular rendering, reuse one trained model via the
row's `method` override.

With the `per_eye` strategy, each eye is rendered with its own visibility mask (left: `0`,
right: `1`), matching the `stereo` path; the renderer's `FOVEATION_VISIBILITY_MASK_INDEX` setting
only applies to single-view rendering, e.g., in the GUI.

### Output

`output/<manifest_stem>_<timestamp>/` containing:

- `performance_table.txt`: one line per row, with mean and 99th-percentile frame time per
  render strategy, plus primitive count
- `<name>.txt` per row: the config block (model, renderer and camera settings), then full frame
  time and FPS percentiles per strategy
- `<name>.result.json` per row: the same data in machine-readable form

The output path is printed when the script finishes.

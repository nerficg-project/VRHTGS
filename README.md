# Fast Eye-Tracked 3D Gaussian Rendering for Virtual Reality
Timon Scholz, Florian Hahlbohm, Martin Eisemann, Susana Castillo, Marcus Magnor<br>
| [Project Page](https://www.timonscholz.dev/paper/vr-htgs) | [Paper / Supplemental](https://diglib.eg.org/items/0155e618-7d14-4dd0-b864-1dbbf431ec1b) | Sample Models (Coming Soon) | VR Viewer (Coming Soon) |<br>

[//]: # (TODO: Add missing links)

## Overview
This repository contains the official implementation of "Fast Eye-Tracked 3D Gaussian Rendering for Virtual Reality".
It is provided as an extension to the [NeRFICG](https://github.com/nerficg-project) framework. Make sure you grab
at least version `v2.3` of the framework, as we rely on recent features including *Foveation Emulation*.

On this page, we provide scripts used for our experiments and user study. You can find additional
details on reproducing our experiments below. If you would like to try out our method in VR, please check out our 
[SIBR Viewer fork](https://github.com/nerficg-project/FoveatedSIBR) (coming soon).

## Getting Started

### Our Setup
All our tests as well as experiments for the paper within the nerficg framework were conducted using the following setup:
- Operating System: Arch Linux
- GPU: Nvidia GeForce RTX 5090
- CUDA Driver Version: 595.71.05
- CUDA Toolkit Version: 12.8.1
- Python Version: 3.11
- PyTorch Version: 2.8.0

Older / newer versions may work as well, but we cannot provide support for any versions 
other than the ones listed.

### Setup

As a preparatory step, the [NeRFICG framework](https://github.com/nerficg-project/nerficg) needs to be set up.
Make sure you grab at least version `v2.3`. Some of our evaluation scripts reference GUI functionality to avoid
code duplication. If you want to run these scripts, we recommend against using one of the headless conda environment
files as otherwise some scripts will fail to run. Since the scripts never create a window, most headless environments 
should still work with the default (non-headless) environment as long as the correct dependencies are installed.

<details>
<summary><span style="font-weight: bold;">TL;DR NeRFICG Setup</span></summary>

- Install a CUDA SDK (we recommend [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive)) and a compatible C++ compiler.

- Clone the NeRFICG repository and its submodules:
	```shell
	git clone git@github.com:nerficg-project/nerficg.git --recursive && cd nerficg
	```

- Set up a Conda environment that contains all dependencies using:
	```shell
    SET DISTUTILS_USE_SDK=1  # Windows only
    conda env create -f ./environments/py311_cu128.yaml
    conda activate nerficg
	```

</details>
<br>

Now, you can directly add this project as an additional method:

- Clone this repository into the `src/Methods/` directory:
	```shell
	git clone git@github.com:nerficg-project/VRHTGS.git src/Methods/VRHTGS
	```

- VRHTGS reuses the model and trainer of the [HTGS](https://github.com/nerficg-project/HTGS) method, extended with
  anti-aliasing, view-space bounding and background models. These extensions live on the `vrhtgs-training` branch,
  so clone exactly that branch into `src/Methods/HTGS` (the `main` branch will not work):
	```shell
	git clone -b vrhtgs-training git@github.com:nerficg-project/HTGS.git src/Methods/HTGS
	```

- Install all method-specific dependencies and CUDA extensions using:
	```shell
    # Linux
	./scripts/install.py -m VRHTGS
    # Windows
	python ./scripts/install.py -m VRHTGS
	```
 
- Under Windows, the tiny-cuda-nn compile may fail. In that case,
  setting `set CL=/bigobj` before running the install-script may help.

## Training and Inference

*For detailed instructions, please refer to the [NeRFICG framework repository](https://github.com/nerficg-project/nerficg).*

The VR-HTGS method is mostly compatible with the NeRFICG scripts in the `scripts/` directory.  This includes config file generation via `create_config.py`,
training via `train.py`, 
inference via `inference.py`,
and live rendering via `gui.py`.

When running under Windows, all of our scripts must be prepended with `python`. For example, to train a model, run:
```shell
python ./scripts/train.py -c configs/vrhtgs_garden.yaml
```
instead of
```
./scripts/train.py -c configs/vrhtgs_garden.yaml
```

Note that the test set metrics computed at the end of `train.py` are not meaningful when `GLOBAL.METHOD_TYPE` is
`VRHTGS`, as the uniform full-image metrics (PSNR / SSIM / LPIPS) penalize the deliberately lower-res periphery
of our foveated renderer. Training itself is unaffected: VRHTGS only replaces the renderer used for inference,
while model and training are those of HTGS. Therefore, all our benchmark configurations train with
`METHOD_TYPE: HTGS`, which yields the correct (unfoveated) reconstruction metrics, and the evaluation scripts below
render the same models with the foveated renderer (see [Reproducing Our Experiments](#reproducing-our-experiments)).

For our performance evaluation and the export of VR trajectories, we provide additional scripts in the `scripts/`
directory of this repository, which we used for our experiments in the paper. This includes the following:
- `vrhtgs_performance_eval.py`: Used to benchmark VR performance, i.e., rendering both views in a single pass.
- `vrhtgs_export_trajectory.py`: Used to export VR trajectory images for quality evaluation.
- `vrhtgs_fvvdp_eval.py`: Used to compute FovVideoVDP scores for perceptual quality evaluation.
- `vrhtgs_export_model.py`: Exports the final model (including background model if enabled) for use in our SIBR Viewer fork.
- `vrhtgs_create_experiment_config.py`: Randomization script for our user study trials, producing a list of conditions for use in our SIBR Viewer fork.

To use these scripts, copy them to the `scripts/` directory of your NeRFICG installation:
```bash
# From the nerficg repository root
cp ./src/Methods/VRHTGS/scripts/* ./scripts/
```

The configuration files `vrhtgs_garden.yaml` and `vrhtgs_playground.yaml` in this directory are meant for trying out
the method (e.g., with live GUI rendering during training). For reproducing the paper, use the configurations in
`benchmarks/` described below.

## Reproducing Our Experiments

The `benchmarks/` directory contains one training configuration for every model in the paper and its supplemental,
plus evaluation manifests per scene for the performance and perceptual quality comparisons:

```
benchmarks
└───configs
│   └───quality       # Tab. 8 of the supplemental (training ablations on Mip-NeRF360), <ablation>_<scene>.yaml
│   └───performance   # Tabs. 2/4/6 of the supplemental (VR performance), <model>_<scene>.yaml
└───manifests         # one manifest per scene for vrhtgs_performance_eval.py, <scene>.yaml
    └───fvvdp         # Tabs. 3/5/7 of the supplemental (FovVideoVDP), fvvdp_<scene>.yaml
```

The quality and performance models are trained separately and are not interchangeable. The quality
models follow the established evaluation protocol (outdoor Mip-NeRF360 scenes at 1/4, indoor scenes at 1/2
resolution), while the performance models are trained at the highest resolution we could afford per scene (mostly
full resolution) to maximize quality in VR. The respective `DATASET.IMAGE_SCALE_FACTOR` is set in each configuration
file.

All benchmark configurations train with `GLOBAL.METHOD_TYPE: HTGS` (see above) and use the Mip-NeRF360 loader,
except for the Deep Blending scenes, which use the COLMAP loader as `DrJohnson` was calibrated with two cameras.

### Datasets

The configurations expect the following scene directories, each in COLMAP format (`images/`, optionally `images_N/`,
and `sparse/0/`):

- `dataset/mipnerf360/<scene>`: the nine scenes of [Mip-NeRF360](https://jonbarron.info/mipnerf360/), including the
  provided downscaled `images_2/` and `images_4/` directories.
- `dataset/tanks_and_temples/<scene>`: the ten scenes `barn`, `family`, `francis`, `horse`, `lighthouse`, `m60`,
  `panther`, `playground`, `train`, and `truck` of [Tanks and Temples](https://www.tanksandtemples.org/) at full
  resolution, using [the COLMAP calibration provided by HTGS](https://cloud.tu-braunschweig.de/s/J5xYLLEdMnRwYPc).
- `dataset/deep_blending/DrJohnson` and `dataset/deep_blending/Playroom`: the `colmap/` directory of the official
  [Deep Blending](https://www-sop.inria.fr/reves/publis/2018/HPPFDB18/datasets.html) reconstructions
  ([DrJohnson](https://repo-sam.inria.fr/fungraph/deep-blending/data/DrJohnson_Reconstruction_Inputs_Outputs.zip),
  [Playroom](https://repo-sam.inria.fr/fungraph/deep-blending/data/Playroom_Reconstruction_Inputs_Outputs.zip)), with the
  contents of `sparse/` moved into `sparse/0/`.

### Quality Evaluation (Tab. 8)

The training ablations map to the following configuration parameters:

| Ablation | `RENDERER.USE_AAA_MODE` | `MODEL.BACKGROUND_MODEL` | `TRAINING.LOSS.LAMBDA_BCE` / `LAMBDA_BG_SEPARATION` |
|---|---|---|---|
| `base` | `false` | `NONE` | `0.0` / `0.0` |
| `+shbg` | `false` | `SH` | `0.0` / `0.0` |
| `+shbg,+reg` | `false` | `SH` | `0.001` / `0.01` |
| `+aa` | `true` | `NONE` | `0.0` / `0.0` |
| `+aa,+shbg` | `true` | `SH` | `0.0` / `0.0` |
| `+aa,+shbg,+reg` | `true` | `SH` | `0.001` / `0.01` |

Here, `aa` denotes anti-aliasing together with view-space bounding, `shbg` our spherical harmonics background model,
and `reg` the alpha entropy and background separation losses described in the supplemental (the sky masks for the
latter are computed automatically with SegFormer before training).

Train all models with:
```shell
./scripts/sequential_train.py -d src/Methods/VRHTGS/benchmarks/configs/quality
```
Each run directory `output/HTGS/quality_<ablation>_<scene>_<timestamp>/` then contains all reported values:
- PSNR / SSIM / LPIPS: `Mean` column of `test_30000/metrics_8bit.txt`
- Training time: the `Time` entry (in seconds) at the end of `timings.txt`
- Peak VRAM (allocated / reserved): `vram_stats.txt`

### Performance Evaluation (Tabs. 2, 4 and 6)

For each scene, three models are trained, each of which is benchmarked with both the HTGS and our foveated renderer:

| Configuration | Training settings | Benchmarked as |
|---|---|---|
| `htgs_<scene>.yaml` | `base` | HTGS (HTGS renderer), Ours (FR) (foveated renderer) |
| `aa_<scene>.yaml` | `+aa` | AA+VSP (HTGS renderer), Ours (NoBG) (foveated renderer) |
| `full_<scene>.yaml` | `+aa,+shbg,+reg` | Ours (Full) (foveated renderer) |

Most performance models are trained at full resolution. Their configurations preload the training images into RAM
(`TRAINING.DATA.PRELOADING_LEVEL: 1`), which requires up to ~30 GB of free RAM for the outdoor Mip-NeRF360 scenes;
set it to `0` if you have less. They also skip rendering the test set after training
(`TRAINING.BACKUP.RENDER_TESTSET: false`), as computing metrics at full resolution exceeds 32 GB of VRAM and the
quality of these models is not reported.

Train all models, copy our scripts to the framework (see above), and benchmark each scene with its manifest:
```shell
./scripts/sequential_train.py -d src/Methods/VRHTGS/benchmarks/configs/performance
cp ./src/Methods/VRHTGS/scripts/* ./scripts/
./scripts/vrhtgs_performance_eval.py --manifest src/Methods/VRHTGS/benchmarks/manifests/<scene>.yaml -i 10 --repetitions 4
```
This renders a stereo trajectory with the Meta Quest Pro parameters (`scripts/meta_quest_pro.yaml`) through all
dataset views, interpolating 10 poses between each pair of consecutive views. The resulting `performance_table.txt`
lists mean and 99th percentile frame times per render strategy. In the paper, we report `per_eye` for HTGS and
AA+VSP and `stereo` (both eyes in one pass) for all our rows. Model sizes correspond to the `primitives` column.
Since frame times are sensitive to other GPU workloads, make sure no other application uses the GPU during benchmarking.

### Perceptual Quality (Tabs. 3, 5 and 7)

The FovVideoVDP scores compare Ours (FR) against the same `htgs_<scene>.yaml` performance model rendered with the
HTGS renderer, along an identical trajectory. Install [FovVideoVDP](https://github.com/gfxdisp/FovVideoVDP)
(`pip install pyfvvdp==1.2.2`), then export and score each scene:
```shell
./scripts/vrhtgs_export_trajectory.py --manifest src/Methods/VRHTGS/benchmarks/manifests/fvvdp/fvvdp_<scene>.yaml --vr --testset-only -i 30 --no-visibility-mask
./scripts/vrhtgs_fvvdp_eval.py output/fvvdp_<scene>_<timestamp> --reference HTGS --eyes left
```
This renders the Meta Quest Pro stereo trajectory through the test views only, with 30 interpolated poses between
consecutive views. `--no-visibility-mask` renders the full image, since the HTGS reference has no lens mask. The
scoring script evaluates every 31-frame segment as a short video with the fixation point at the image center and 
reports the mean JOD over all segments in `fvvdp_table.txt`. In the paper, we report the left eye. The exported 
images take up to ~50 GB per scene, so delete them once scored.

### Using Custom Data

While this method is compatible with most of the dataset loaders provided with the [NeRFICG framework](https://github.com/nerficg-project/nerficg),
we recommend using the Mip-NeRF360 loader (`src/Datasets/MipNeRF360.py`) for single-camera captures and the COLMAP
loader (`src/Datasets/Colmap.py`) for multi-camera captures. Both expect the COLMAP format:

```
custom_scene
└───images
│   │   00000.jpg
│   │   ...
│   
└───sparse/0
│   │   cameras.bin
│   │   images.bin
│   │   points3D.bin
│
└───images_2  (optional)
│   │   00000.jpg
│   │   ...
```

To use it, just modify `DATASET.PATH` near the bottom of your configuration file. Furthermore, you may want to modify the following dataset configuration parameters:

- `DATASET.IMAGE_SCALE_FACTOR`: Set this to `null` for using the original resolution or a value between zero and one to train on downscaled images.
  The Mip-NeRF360 loader has special support for the factors `0.5`/`0.25`/`0.125`, where it will always load images from the directories `images_2`/`images_4`/`images_8`, respectively.
  We recommend using this feature and downscaling manually via, e.g., `mogrify -resize 50% *.jpg` for the best results.
- `TRAINING.DATA.PRELOADING_LEVEL`: Set this depending on the available RAM/VRAM as well as the size of your dataset (`2`: store training images in VRAM, `1`: in RAM, `0`: no preloading).
- `DATASET.BACKGROUND_COLOR`: Will be ignored, as the rasterizer only supports a black background (see the "Background color blending" note in the [HTGS README](https://github.com/nerficg-project/HTGS#additional-notes)); use `MODEL.BACKGROUND_MODEL` instead.
- `DATASET.NEAR_PLANE`: When `RENDERER.USE_AAA_MODE` is disabled, this must not be too small to avoid precision issues. We used `0.2` for those benchmark configurations. With `USE_AAA_MODE: true`, the precision issues are mitigated, so we used `0.01` in that case.
- `DATASET.FAR_PLANE`: Set this generously, i.e., not too tight for your scene to avoid clipping. We used `1000.0` for all scenes.
- `DATASET.TEST_STEP`: Set to `8` for the established evaluation protocol. Set to `0` to use all images for training.
- `DATASET.APPLY_PCA`: Tries to align the world space so that the up-axis is parallel to the direction of gravity using principal component analysis. 
  Although it does not always work, we recommend setting this to `true` if you want to view the final model inside a GUI or our VR viewer.

While we recommend setting `DATASET.APPLY_PCA_RESCALE` to `false`, it can be turned on to scale the scene so that all camera poses are inside the \[-1, 1\] cube.

If using your custom data fails, you have two options:

1. (Easy) Re-calibrate using, e.g., `./scripts/colmap.py -i <path/to/your/scene> --camera_mode single` and add `-u` at the end if your images are distorted.

2. (Advanced) Check the NeRFICG instructions for using custom data [here](https://github.com/nerficg-project/nerficg?tab=readme-ov-file#training-on-custom-image-sequences) and optionally dive into the NeRFICG code to extend one of the dataloaders to handle your data.

## License and Citation
This project is licensed under the MIT license (see [LICENSE](LICENSE)).

If you use this code for your research projects, please consider a citation:
```bibtex
@inproceedings{scholz2026vrhtgs,
    title = {{Fast Eye-Tracked 3D Gaussian Rendering for Virtual Reality}},
    author = {Scholz, Timon and Hahlbohm, Florian and Eisemann, Martin and Castillo, Susana and Magnor, Marcus},
    booktitle = {Vision, Modeling, and Visualization (VMV)},
    publisher = {The Eurographics Association},
    editor = {Leimkühler, Thomas and Weinkauf, Tino and Campen, Marcel},
    year = {2026},
    DOI = {10.2312/vmv.20261013},
    ISBN = {978-3-03868-318-6}
}
```

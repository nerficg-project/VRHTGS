"""vrhtgs_utils.py: Shared setup, VR view construction, and manifest handling for the VRHTGS evaluation scripts."""

import enum
import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable, Literal

import torch
import yaml

import utils
with utils.DiscoverSourcePath():
    import Framework
    from Datasets.Base import BaseDataset
    from Datasets.utils import View
    from ICGui.Controls.Camera import FlyingControls, OrbitalControls
    from Implementations import Datasets as DI
    from Implementations import Methods as MI
    from Logging import Logger
    from Methods.Base.Model import BaseModel
    from Methods.Base.Renderer import BaseRenderer


DEFAULT_CAMERA_CONFIG = Path(__file__).resolve().parent / 'meta_quest_pro.yaml'


class RenderStrategy(enum.Enum):
    """How a stereo frame is produced: one call per eye, or a single fused call for both."""
    PER_EYE = 'per_eye'
    STEREO = 'stereo'


@dataclass
class VRCameraConfig:
    resolution: list[int]
    focal_length: list[float]
    center_left: list[float]
    center_right: list[float]
    eye_distance: float


@dataclass
class ExperimentRow:
    """One row of an evaluation manifest: a trained model, rendered by the method its run directory declares
    unless `method` overrides it (e.g., to render an HTGS-trained model with the foveated VRHTGS renderer)."""
    name: str
    run_dir: Path
    checkpoint: str = 'final.pt'
    method: str | None = None


def load_camera_config(path: Path) -> VRCameraConfig:
    with open(path) as f:
        config = yaml.safe_load(f)
    return VRCameraConfig(
        resolution=config['resolution'],
        focal_length=config['focal_length'],
        center_left=config['center_left'],
        center_right=config['center_right'],
        eye_distance=config['eye_distance'],
    )


def resolve_run_dir(pattern: str, row_name: str) -> Path:
    """Resolves a manifest run_dir. Glob patterns (e.g. `output/HTGS/perf_base_garden_*`) select the most
    recent matching run, which works because run directory names end in a sortable timestamp."""
    if not any(c in pattern for c in '*?['):
        return Path(pattern)
    matches = sorted(p for p in Path().glob(pattern) if (p / 'training_config.yaml').is_file())
    if not matches:
        raise Framework.InferenceError(f'manifest row "{row_name}": no run directory matches {pattern}')
    if len(matches) > 1:
        Logger.log_info(f'manifest row "{row_name}": {len(matches)} runs match {pattern}, using {matches[-1]}')
    return matches[-1]


def load_manifest(path: Path) -> list[ExperimentRow]:
    """Loads an evaluation manifest. Row paths are resolved against the current working directory."""
    with open(path) as f:
        manifest = yaml.safe_load(f)
    rows, scenes = [], {}
    for entry in manifest['rows']:
        run_dir = resolve_run_dir(str(entry['run_dir']), entry['name'])
        config_path = run_dir / 'training_config.yaml'
        if not config_path.is_file():
            raise Framework.InferenceError(f'manifest row "{entry["name"]}": no training_config.yaml in {run_dir}')
        with open(config_path) as f:
            scenes[entry['name']] = yaml.safe_load(f)['DATASET']['PATH']
        rows.append(ExperimentRow(
            name=entry['name'],
            run_dir=run_dir,
            checkpoint=entry.get('checkpoint', 'final.pt'),
            method=entry.get('method'),
        ))
    if not rows:
        raise Framework.InferenceError(f'manifest contains no rows: {path}')
    # rows are compared against each other, so a scene mismatch makes the whole table meaningless
    if len(set(scenes.values())) > 1:
        Logger.log_warning(f'manifest rows were trained on different scenes, results are not comparable: {scenes}')
    return rows


def create_collection_directory(manifest_path: Path) -> Path:
    """Creates the timestamped directory collecting the results of one manifest run."""
    output_dir = Framework.Directories.OUTPUT_DIR / f'{manifest_path.stem}_{datetime.now():%Y-%m-%d-%H-%M-%S}'
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def setup(
    run_dir: Path,
    checkpoint_name: str,
    purpose: str,
    testset_only: bool = False,
    method: str | None = None,
) -> tuple[BaseDataset, BaseModel, BaseRenderer]:
    """Loads the dataset, model and renderer declared by a run directory's own training config.
    If `method` is given, the model is still loaded as trained, but rendered with that method's renderer."""
    Framework.setup(
        config_path=str(run_dir / 'training_config.yaml'),
        require_custom_config=True
    )
    Logger.log_info(f'Starting {purpose} for model trained in: {run_dir}')

    # Load dataset
    if not testset_only:
        Framework.config.DATASET.TEST_STEP = 0  # use all dataset images
    dataset = DI.get_dataset(
        dataset_type=Framework.config.GLOBAL.DATASET_TYPE,
        path=Framework.config.DATASET.PATH
    )
    if testset_only:
        dataset.test()
        if len(dataset) == 0:
            raise Framework.InferenceError('no testset images found, cannot perform testset-only evaluation')

    # Load model checkpoint
    model = MI.get_model(
        method=Framework.config.GLOBAL.METHOD_TYPE,
        checkpoint=str(run_dir / 'checkpoints' / checkpoint_name),
    ).eval()

    # Prepare renderer, optionally overriding the method the model was trained with
    if method is not None and method != Framework.config.GLOBAL.METHOD_TYPE:
        Logger.log_info(f'rendering {Framework.config.GLOBAL.METHOD_TYPE} model with the {method} renderer')
        Framework.config.GLOBAL.METHOD_TYPE = method
    renderer = MI.get_renderer(
        method=Framework.config.GLOBAL.METHOD_TYPE,
        model=model
    )
    return dataset, model, renderer


def available_strategies(renderer: BaseRenderer) -> list[RenderStrategy]:
    """Returns the render strategies this renderer supports. Every renderer can render one eye at a time."""
    strategies = [RenderStrategy.PER_EYE]
    if hasattr(renderer, 'render_image_stereo'):
        strategies.append(RenderStrategy.STEREO)
    return strategies


def make_render_fn(renderer: BaseRenderer, strategy: RenderStrategy) -> Callable[[View, View], dict[str, torch.Tensor]]:
    """Returns a callable rendering one stereo frame using the given strategy."""
    if strategy is RenderStrategy.STEREO:
        return lambda left, right: renderer.render_image_stereo(left, right, True)
    if not hasattr(renderer, 'FOVEATION_VISIBILITY_MASK_INDEX'):
        return lambda left, right: {
            'rgb_left': renderer.render_image(left, benchmark=True)['rgb'],
            'rgb_right': renderer.render_image(right, benchmark=True)['rgb'],
        }

    # Single-view foveated rendering takes its visibility mask from the renderer config (a GUI setting),
    # so select each eye's mask explicitly, matching what the stereo path does via camera_index.
    def render_per_eye(left: View, right: View) -> dict[str, torch.Tensor]:
        default_index = renderer.FOVEATION_VISIBILITY_MASK_INDEX
        try:
            renderer.FOVEATION_VISIBILITY_MASK_INDEX = left.camera_index
            rgb_left = renderer.render_image(left, benchmark=True)['rgb']
            renderer.FOVEATION_VISIBILITY_MASK_INDEX = right.camera_index
            rgb_right = renderer.render_image(right, benchmark=True)['rgb']
        finally:
            renderer.FOVEATION_VISIBILITY_MASK_INDEX = default_index
        return {'rgb_left': rgb_left, 'rgb_right': rgb_right}
    return render_per_eye


def collect_config(
    row: ExperimentRow,
    model: BaseModel,
    renderer: BaseRenderer,
    camera_config: Path,
    num_views: int,
    resolution: tuple[int, int],
) -> dict[str, Any]:
    """Records everything that changes what a measurement means, so a result file is self-describing."""
    return {
        'name': row.name,
        'run_dir': str(row.run_dir),
        'checkpoint': row.checkpoint,
        'method_type': Framework.config.GLOBAL.METHOD_TYPE,
        'method_override': row.method,
        'renderer_class': type(renderer).__name__,
        'iterations_trained': model.num_iterations_trained,
        # read off the loaded model, not the training config: BaseModel.load restores MODEL.* from the checkpoint
        'background_model': getattr(model, 'BACKGROUND_MODEL', None),
        'aaa_mode': getattr(renderer, 'AAA_MODE', getattr(renderer, 'USE_AAA_MODE', None)),
        'disable_bg_model_inference': getattr(renderer, 'DISABLE_BG_MODEL_INFERENCE', None),
        'primitives': model.gaussians.get_positions.shape[0] if hasattr(model, 'gaussians') else None,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'camera_config': str(camera_config),
        'num_views': num_views,
        'resolution': f'{resolution[0]}x{resolution[1]}',
    }


def write_config(path: Path, config: dict[str, Any]) -> None:
    """Writes the config block that heads every per-row result file."""
    with open(path, 'a') as f:
        for key, value in config.items():
            f.write(f'{key}: {value}\n')
        f.write('\n')


def run_manifest(
    rows: list[ExperimentRow],
    worker: Callable,
    shared_kwargs: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Runs one worker subprocess per manifest row, so no row's measurement is shaped by the rows before it.

    The worker hands its result back through a file rather than a queue: a queue read deadlocks
    the parent if the child dies before writing, and a row that fails should not stall the matrix.

    Children are spawned rather than forked, because a forked child cannot initialize CUDA once
    the parent has. Spawning also gives each row a fresh interpreter, which is the point: a row's
    frame times should not depend on the allocator state left behind by the rows before it.
    """
    context = get_context('spawn')
    results = []
    for index, row in enumerate(rows, start=1):
        Logger.log(f'[{index}/{len(rows)}] {row.name}')
        result_path = output_dir / f'{row.name}.result.json'
        process = context.Process(target=worker, args=(row, shared_kwargs, output_dir, result_path))
        process.start()
        process.join()
        if process.exitcode != 0 or not result_path.is_file():
            Logger.log_error(f'row "{row.name}" failed with exit code {process.exitcode}, skipping')
            continue
        with open(result_path) as f:
            results.append(json.load(f))
    return results


def write_result(path: Path, result: dict[str, Any]) -> None:
    """Hands a worker's result back to the parent process."""
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)


NO_VISIBILITY_MASK_INDEX = -1  # MaskTables.get_visibility_mask returns an all-ones mask for negative indices


def disable_visibility_mask(views: list[tuple[View, ...]], renderer: BaseRenderer) -> None:
    """Renders the full image instead of only the area visible through the headset lenses.
    Stereo and per-eye rendering select the mask via each view's camera_index, single-view rendering
    via the renderer's FOVEATION_VISIBILITY_MASK_INDEX, so both are set to the all-ones index."""
    for frame in views:
        for view in frame:
            view.camera_index = NO_VISIBILITY_MASK_INDEX
    if hasattr(renderer, 'FOVEATION_VISIBILITY_MASK_INDEX'):
        renderer.FOVEATION_VISIBILITY_MASK_INDEX = NO_VISIBILITY_MASK_INDEX


def get_stereo_pair(
    view: View,
    center_left: list[float] | None = None,
    center_right: list[float] | None = None,
    eye_distance: float | None = None,
) -> tuple[View, View]:
    """Default eye_distance corresponds to the reported value from the Meta Quest Pro headset."""
    if eye_distance is None:
        eye_distance = 0.0746
    left_c2w = view.c2w_numpy.copy()
    right_c2w = view.c2w_numpy.copy()
    left_c2w[:3, 3] -= (eye_distance / 2.0) * left_c2w[:3, 0]
    right_c2w[:3, 3] += (eye_distance / 2.0) * right_c2w[:3, 0]

    left_view = deepcopy(view)
    left_view.c2w = left_c2w
    left_view.camera_index = 0  # left eye
    if center_left is not None:
        left_view.camera.center_x = center_left[0]
        left_view.camera.center_y = center_left[1]

    right_view = deepcopy(view)
    right_view.c2w = right_c2w
    right_view.camera_index = 1  # right eye
    if center_right is not None:
        right_view.camera.center_x = center_right[0]
        right_view.camera.center_y = center_right[1]
    elif center_left is not None:
        right_view.camera.center_x = center_left[0]
        right_view.camera.center_y = center_left[1]

    return left_view, right_view


def get_vr_views(
    dataset: BaseDataset,
    resolution: list[int] | None,
    focal_length: list[float] | None,
    center_left: list[float] | None,
    center_right: list[float] | None,
    interpolation: Literal['flying', 'orbital'] = 'flying',
    interp_frames: int = -1,
    vr_stereo: bool = False,
    eye_distance: float | None = None,
) -> list[tuple[View, ...]]:
    """Prepares a list of views for VR benchmarking / video export. If recordings are provided,
    extracts views from them, otherwise uses interpolated dataset views."""
    dataset.test()
    # If no test images are available, use the training set
    if len(dataset) == 0:
        dataset.data['test'] = dataset.data['train']
        if len(dataset) == 0:
            raise Framework.InferenceError('No images found for benchmarking.')

    # Prepare views
    if interp_frames >= 0:
        views = []
        controller = FlyingControls(initial_c2w=dataset[0].c2w_numpy) \
            if interpolation == 'flying' \
            else OrbitalControls(initial_c2w=dataset[0].c2w_numpy)

        for i, view in enumerate(dataset):
            controller.c2w = view.c2w_numpy
            controller.ease_to(
                to_c2w=dataset[(i + 1) % len(dataset)].c2w_numpy,
                duration=1.0,
                skip_animations=False,
                easing_function=lambda t: t  # linear
            )
            view_copy = deepcopy(view)
            if resolution is not None:
                view_copy.camera.width = resolution[0]
                view_copy.camera.height = resolution[1]
            if focal_length is not None:
                view_copy.camera.focal_x = focal_length[0]
                view_copy.camera.focal_y = focal_length[1]
            if center_left is not None:
                view_copy.camera.center_x = center_left[0]
                view_copy.camera.center_y = center_left[1]
            views.append(get_stereo_pair(view_copy, center_left, center_right, eye_distance) if vr_stereo else (view_copy,))
            step = 1 / (interp_frames + 1)
            for t_idx in range(interp_frames):
                controller.run_animation(step, skip_animations=False)
                interp_view = deepcopy(view_copy)
                interp_view.c2w = controller.c2w
                views.append(get_stereo_pair(interp_view, center_left, center_right, eye_distance) if vr_stereo else (interp_view,))
    else:
        views = [get_stereo_pair(view, eye_distance=eye_distance) if vr_stereo else (view,) for view in dataset]

    Logger.log_info(f'Prepared {len(views)} {"stereo" if vr_stereo else "monocular"} views for benchmarking.')
    return views

#! /usr/bin/env python3

"""vrhtgs_export_trajectory.py: Exports rendered trajectory images for every row of an evaluation manifest.

This script only renders and writes images. It deliberately computes no quality metrics: the
framework's uniform full-image metrics (PSNR / SSIM / LPIPS) do not account for foveated
rendering and would penalize the periphery for being deliberately approximated. Perceptual
comparison is done separately with the foveated metric FovVideoVDP (vrhtgs_fvvdp_eval.py), over
the image sequences this script produces. All rows are rendered along an identical trajectory so they can
be compared frame for frame.
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Literal

import utils
from vrhtgs_utils import (
    DEFAULT_CAMERA_CONFIG, ExperimentRow, RenderStrategy, available_strategies, collect_config,
    create_collection_directory, disable_visibility_mask, get_vr_views, load_camera_config, load_manifest, make_render_fn, run_manifest,
    setup, write_config, write_result,
)
with utils.DiscoverSourcePath():
    from Datasets.utils import save_image
    from Logging import Logger


def export_row(row: ExperimentRow, shared: dict[str, Any], output_dir: Path, result_path: Path) -> None:
    """Renders and writes one manifest row's image sequence in its own process."""
    vr_camera = load_camera_config(shared['camera_config'])
    dataset, model, renderer = setup(
        row.run_dir, row.checkpoint, purpose='trajectory image export',
        testset_only=shared['testset_only'], method=row.method,
    )
    views = get_vr_views(
        dataset,
        resolution=vr_camera.resolution,
        focal_length=vr_camera.focal_length,
        center_left=vr_camera.center_left,
        center_right=vr_camera.center_right,
        interpolation=shared['interpolation'],
        interp_frames=shared['interp_frames'],
        eye_distance=vr_camera.eye_distance,
        vr_stereo=shared['vr_mode'],
    )
    if not shared['visibility_mask']:
        disable_visibility_mask(views, renderer)

    row_dir = output_dir / row.name
    row_dir.mkdir(parents=True, exist_ok=True)
    config = collect_config(
        row, model, renderer, shared['camera_config'], len(views),
        (views[0][0].camera.width, views[0][0].camera.height),
    )
    config['visibility_mask'] = shared['visibility_mask']
    write_config(row_dir / 'config.txt', config)

    Logger.log_info(f'writing {len(views)} frames for {row.name}')
    if shared['vr_mode']:
        # export is about the images, not the timing, so use the fused stereo path where the
        # renderer has one and fall back to per-eye where it does not
        strategy = RenderStrategy.STEREO if RenderStrategy.STEREO in available_strategies(renderer) \
            else RenderStrategy.PER_EYE
        render_stereo = make_render_fn(renderer, strategy)
        (row_dir / 'left').mkdir(parents=True, exist_ok=True)
        (row_dir / 'right').mkdir(parents=True, exist_ok=True)
        for i, view in Logger.log_progress(enumerate(views), total=len(views), leave=False, desc='Writing Images'):
            out = render_stereo(view[0], view[1])
            save_image(row_dir / f'left/{i:05d}.png', out['rgb_left'])
            save_image(row_dir / f'right/{i:05d}.png', out['rgb_right'])
    else:
        for i, view in Logger.log_progress(enumerate(views), total=len(views), leave=False, desc='Writing Images'):
            out = renderer.render_image(view[0], benchmark=True)
            save_image(row_dir / f'{i:05d}.png', out['rgb'])

    write_result(result_path, {'config': config, 'frames': len(views)})


def write_index(path: Path, results: list[dict[str, Any]], vr_mode: bool) -> None:
    """Writes an index of the exported sequences, so the external comparison step is self-describing."""
    lines = [
        f'{"name":<28}{"frames":>8}{"resolution":>14}{"primitives":>14}  images',
        '-' * 92,
    ]
    for result in results:
        config = result['config']
        layout = f'{config["name"]}/{{left,right}}/' if vr_mode else f'{config["name"]}/'
        lines.append(
            f'{config["name"]:<28}{result["frames"]:>8}{config["resolution"]:>14}'
            f'{config["primitives"] or 0:>14,}  {layout}'
        )
    lines += [
        '',
        'All rows share one trajectory, so frame i of any row corresponds to frame i of any other.',
        'No metrics are computed here; score them with FovVideoVDP via vrhtgs_fvvdp_eval.py <this directory>.',
        'Per-row settings are in <name>/config.txt.',
    ]
    path.write_text('\n'.join(lines) + '\n')


def main(
    *,
    manifest_path: Path, vr_mode: bool, camera_config: Path, interp_frames: int,
    interpolation: Literal['flying', 'orbital'] = 'flying', testset_only: bool = False,
    visibility_mask: bool = True,
) -> None:
    rows = load_manifest(manifest_path)
    output_dir = create_collection_directory(manifest_path)
    Logger.log(f'exporting {len(rows)} rows from {manifest_path}')
    results = run_manifest(
        rows, export_row,
        {
            'camera_config': camera_config,
            'interp_frames': interp_frames,
            'interpolation': interpolation,
            'vr_mode': vr_mode,
            'testset_only': testset_only,
            'visibility_mask': visibility_mask,
        },
        output_dir,
    )
    index_path = output_dir / 'index.txt'
    write_index(index_path, results, vr_mode)
    Logger.log(f'images written to: {output_dir}')
    Logger.log(f'index: {index_path}')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--manifest', action='store', dest='manifest_path', type=Path, required=True,
        metavar='path/to/manifest.yaml',
        help='Evaluation manifest listing the rows to export. Paths inside it resolve against the '
             'current working directory, so run this from the repository root.'
    )
    parser.add_argument(
        '-i', '--interpolate-frames', action='store', dest='interp_frames',
        default=-1, metavar='n', required=False, type=int,
        help='If set to a value > 0, interpolates n poses between each set of ground truth poses.'
    )
    parser.add_argument(
        '--vr', action='store_true', dest='vr_mode',
        help='Renders left/right eye pairs instead of single flat images.'
    )
    parser.add_argument(
        '--camera-config', action='store', dest='camera_config', default=DEFAULT_CAMERA_CONFIG, type=Path,
        help='YAML file with the VR camera configuration (resolution, focal length, projection centers, eye distance).'
    )
    parser.add_argument(
        '--testset-only', action='store_true', dest='testset_only',
        help='If set, only considers views from the test set for interpolation.'
    )

    parser.add_argument(
        '--no-visibility-mask', action='store_false', dest='visibility_mask',
        help='Renders the full image instead of only the area visible through the headset lenses. '
             'Use this when comparing against renderers without a lens mask, e.g. for FovVideoVDP '
             'against unfoveated HTGS, where the masked-out corners would otherwise count as errors.'
    )

    args, _ = parser.parse_known_args()
    Logger.set_mode(Logger.MODE_VERBOSE)
    return {
        'manifest_path': args.manifest_path,
        'interp_frames': args.interp_frames,
        'vr_mode': args.vr_mode,
        'camera_config': args.camera_config,
        'testset_only': args.testset_only,
        'visibility_mask': args.visibility_mask,
    }


if __name__ == '__main__':
    main(**parse_args())

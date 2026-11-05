#! /usr/bin/env python3

"""vrhtgs_performance_eval.py: Benchmarks VR rendering performance for every row of an evaluation manifest."""

import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch

import utils
from vrhtgs_utils import (
    DEFAULT_CAMERA_CONFIG, ExperimentRow, RenderStrategy, available_strategies, collect_config,
    create_collection_directory, get_vr_views, load_camera_config, load_manifest, make_render_fn, run_manifest,
    setup, write_config, write_result,
)
with utils.DiscoverSourcePath():
    from Datasets.utils import View
    from Logging import Logger


def benchmark_views(
    render_fn: Callable[[View, View], dict[str, torch.Tensor]],
    views: list[tuple[View, ...]],
    n_reps: int,
) -> tuple[np.ndarray, float]:
    # Warmup
    for view in Logger.log_progress(views, total=len(views), leave=False, desc='Warming Up'):
        render_fn(view[0], view[1])

    # Timed repetitions
    torch.cuda.synchronize()
    frame_times = np.empty((n_reps, len(views)), dtype=np.float32)
    start_time = time.perf_counter()
    for rep in Logger.log_progress(range(n_reps), leave=False, desc='Benchmarking Performance'):
        for i, view in enumerate(views):
            frame_start = time.perf_counter()
            render_fn(view[0], view[1])
            frame_end = time.perf_counter()
            frame_times[rep, i] = frame_end - frame_start
    torch.cuda.synchronize()
    return frame_times, time.perf_counter() - start_time


def summarize(frame_times: np.ndarray, total_time: float, n_views: int, n_reps: int) -> dict[str, float]:
    """Reduces raw frame times to the statistics reported in the paper, plus the tails."""
    frame_times = frame_times.flatten() * 1000.0
    fps_values = 1000.0 / frame_times
    total_images = n_reps * n_views * 2
    return {
        'mean_frametime_ms': float(np.mean(frame_times)),
        'p50_frametime_ms': float(np.percentile(frame_times, 50)),
        'p95_frametime_ms': float(np.percentile(frame_times, 95)),
        'p99_frametime_ms': float(np.percentile(frame_times, 99)),
        'p99_9_frametime_ms': float(np.percentile(frame_times, 99.9)),
        'max_frametime_ms': float(np.max(frame_times)),
        'p50_fps': float(np.percentile(fps_values, 50)),
        'low_5_fps': float(np.percentile(fps_values, 5)),
        'low_1_fps': float(np.percentile(fps_values, 1)),
        'low_0_1_fps': float(np.percentile(fps_values, 0.1)),
        'min_fps': float(np.min(fps_values)),
        'avg_fps_both_eyes': float(total_images / total_time / 2.0),
    }


def benchmark_row(row: ExperimentRow, shared: dict[str, Any], output_dir: Path, result_path: Path) -> None:
    """Benchmarks one manifest row in its own process, so allocator state does not carry between rows."""
    vr_camera = load_camera_config(shared['camera_config'])
    dataset, model, renderer = setup(row.run_dir, row.checkpoint, purpose='performance evaluation', method=row.method)
    views = get_vr_views(
        dataset,
        resolution=vr_camera.resolution,
        focal_length=vr_camera.focal_length,
        center_left=vr_camera.center_left,
        center_right=vr_camera.center_right,
        interpolation=shared['interpolation'],
        interp_frames=shared['interp_frames'],
        eye_distance=vr_camera.eye_distance,
        vr_stereo=True,
    )

    config = collect_config(
        row, model, renderer, shared['camera_config'], len(views),
        (views[0][0].camera.width, views[0][0].camera.height),
    )
    detail_path = output_dir / f'{row.name}.txt'
    write_config(detail_path, config)

    measurements = {}
    for strategy in available_strategies(renderer):
        Logger.log_info(f'benchmarking {row.name} [{strategy.value}]')
        frame_times, total_time = benchmark_views(
            make_render_fn(renderer, strategy), views, shared['n_reps']
        )
        stats = summarize(frame_times, total_time, len(views), shared['n_reps'])
        measurements[strategy.value] = stats
        with open(detail_path, 'a') as f:
            f.write(f'[{strategy.value}]\n')
            for key, value in stats.items():
                f.write(f'  {key}: {value:.4f}\n')
            f.write('\n')
        Logger.log_info(
            f'{row.name} [{strategy.value}]: {stats["mean_frametime_ms"]:.2f} ms mean, '
            f'{stats["p99_frametime_ms"]:.2f} ms 99th%'
        )

    write_result(result_path, {'config': config, 'measurements': measurements})


def write_table(path: Path, results: list[dict[str, Any]]) -> None:
    """Writes the cross-row comparison table: one row per model, one column pair per render strategy."""
    strategies = [strategy.value for strategy in RenderStrategy]
    header = f'{"name":<28}{"primitives":>12}'
    for strategy in strategies:
        header += f'{strategy + " mean":>18}{strategy + " 99%":>18}'
    lines = [header, '-' * len(header)]
    for result in results:
        line = f'{result["config"]["name"]:<28}{result["config"]["primitives"] or 0:>12,}'
        for strategy in strategies:
            stats = result['measurements'].get(strategy)
            line += f'{stats["mean_frametime_ms"]:>18.2f}{stats["p99_frametime_ms"]:>18.2f}' if stats \
                else f'{"-":>18}{"-":>18}'
        lines.append(line)
    lines.append('')
    lines.append('frame times in ms, lower is better; per-row config and detail in <name>.txt')
    path.write_text('\n'.join(lines) + '\n')


def main(
    *,
    manifest_path: Path, n_reps: int, camera_config: Path, interp_frames: int,
    interpolation: Literal['flying', 'orbital'] = 'flying',
) -> None:
    rows = load_manifest(manifest_path)
    output_dir = create_collection_directory(manifest_path)
    Logger.log(f'benchmarking {len(rows)} rows from {manifest_path}')
    results = run_manifest(
        rows, benchmark_row,
        {
            'camera_config': camera_config,
            'interp_frames': interp_frames,
            'interpolation': interpolation,
            'n_reps': n_reps,
        },
        output_dir,
    )
    table_path = output_dir / 'performance_table.txt'
    write_table(table_path, results)
    Logger.log(f'results written to: {output_dir}')
    Logger.log(f'comparison table: {table_path}')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--manifest', action='store', dest='manifest_path', type=Path, required=True,
        metavar='path/to/manifest.yaml',
        help='Evaluation manifest listing the rows to benchmark. Paths inside it resolve against the '
             'current working directory, so run this from the repository root.'
    )
    parser.add_argument(
        '-i', '--interpolate-frames', action='store', dest='interp_frames',
        default=-1, metavar='n', required=False, type=int,
        help='If set to a value > 0, interpolates n poses between each set of ground truth poses.'
    )
    parser.add_argument(
        '--repetitions', action='store', dest='n_reps', default=1,
        help='Number of repetitions for rendering each view to obtain better averages.', type=int
    )
    parser.add_argument(
        '--camera-config', action='store', dest='camera_config', default=DEFAULT_CAMERA_CONFIG, type=Path,
        help='YAML file with the VR camera configuration (resolution, focal length, projection centers, eye distance).'
    )

    args, _ = parser.parse_known_args()
    Logger.set_mode(Logger.MODE_VERBOSE)
    return {
        'manifest_path': args.manifest_path,
        'interp_frames': args.interp_frames,
        'n_reps': args.n_reps,
        'camera_config': args.camera_config,
    }


if __name__ == '__main__':
    main(**parse_args())

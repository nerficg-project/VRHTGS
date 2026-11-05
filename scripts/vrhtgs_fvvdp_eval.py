#! /usr/bin/env python3

"""vrhtgs_fvvdp_eval.py: Scores trajectory exports with the foveated video metric FovVideoVDP.

Takes a directory written by vrhtgs_export_trajectory.py and compares every row against one reference row,
frame for frame, with FovVideoVDP (pyfvvdp) in foveated mode. The trajectory consists of segments of
`1 + interpolated frames` frames per dataset view, so each segment is scored as one short video and the JOD
of a row is the mean over all segments. The display model emulates a Meta Quest Pro, matching the default
camera configuration of the export script.

Requires pyfvvdp, which is not a framework dependency: `pip install pyfvvdp==1.2.2`.
"""

import json
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torchvision import io

# display model: Meta Quest Pro emulation used for the paper's evaluation
FOV_DIAGONAL = 111.24  # degrees
DISTANCE_M = 3.0
Y_PEAK = 100.0  # cd/m^2
CONTRAST = 3000


def load_image(path: Path) -> torch.Tensor:
    """Loads an image as CHW float in [0, 1]."""
    image = io.decode_image(str(path), mode=io.ImageReadMode.UNCHANGED)
    return (image.float() / (65535 if image.dtype == torch.uint16 else 255))[:3]


def frame_dirs(row_dir: Path) -> dict[str, Path]:
    """Returns the image directories of an exported row: one per eye for stereo exports, one otherwise."""
    eyes = {eye: row_dir / eye for eye in ('left', 'right') if (row_dir / eye).is_dir()}
    return eyes or {'mono': row_dir}


def create_metric(width: int, height: int):
    import pyfvvdp
    geometry = pyfvvdp.fvvdp_display_geometry(
        resolution=(width, height), fov_diagonal=FOV_DIAGONAL, distance_m=DISTANCE_M,
    )
    photometry = pyfvvdp.fvvdp_display_photo_eotf(Y_peak=Y_PEAK, contrast=CONTRAST)
    # the display name only selects defaults; geometry and photometry are both given explicitly
    return pyfvvdp.fvvdp(
        display_name='htc_vive_pro', display_geometry=geometry, display_photometry=photometry,
        foveated=True, quiet=True,
    )


def score_sequence(reference: list[Path], test: list[Path], window: int, gaze: tuple[int, int] | None) -> dict:
    """Scores one image sequence against its reference in non-overlapping windows of `window` frames."""
    if len(reference) != len(test):
        raise RuntimeError(f'frame counts differ: {len(reference)} (reference) vs. {len(test)} (test)')
    if not reference or len(reference) % window:
        raise RuntimeError(f'{len(reference)} frames is not a positive multiple of the window size {window}')
    _, height, width = load_image(reference[0]).shape
    fixation = np.array(gaze if gaze is not None else (width // 2, height // 2))
    metric = create_metric(width, height)
    per_window = []
    pool = ThreadPoolExecutor(8)  # PNG decoding dominates otherwise
    for start in range(0, len(reference), window):
        ref = torch.stack(list(pool.map(load_image, reference[start:start + window]))).numpy()
        tst = torch.stack(list(pool.map(load_image, test[start:start + window]))).numpy()
        with torch.no_grad():
            jod, _ = metric.predict(tst, ref, dim_order='FCHW', fixation_point=fixation, frames_per_second=window)
        per_window.append(float(jod))
        print(f'  window {len(per_window)}/{len(reference) // window}: {per_window[-1]:.4f}', flush=True)
        torch.cuda.empty_cache()
    return {
        'jod': float(np.mean(per_window)),
        'per_window': per_window,
        'frames': len(reference),
        'resolution': [width, height],
        'fixation_point': fixation.tolist(),
        'ppd': metric.display_geometry.get_ppd(),
    }


def main(*, export_dir: Path, reference: str, rows: list[str] | None, eyes: list[str] | None,
         window: int, gaze: tuple[int, int] | None) -> None:
    try:
        import pyfvvdp  # noqa: F401
    except ImportError:
        raise SystemExit('pyfvvdp is not installed, run: pip install pyfvvdp==1.2.2')
    ref_dir = export_dir / reference
    if not ref_dir.is_dir():
        raise SystemExit(f'reference row "{reference}" not found in {export_dir}')
    if rows is None:
        rows = sorted(p.name for p in export_dir.iterdir() if p.is_dir() and p.name != reference)
    ref_eyes = frame_dirs(ref_dir)
    eyes = eyes or list(ref_eyes)

    results = {
        'pyfvvdp': version('pyfvvdp'),
        'reference': reference,
        'window': window,
        'display': {'fov_diagonal': FOV_DIAGONAL, 'distance_m': DISTANCE_M, 'Y_peak': Y_PEAK, 'contrast': CONTRAST},
        'rows': {},
    }
    for row in rows:
        row_eyes = frame_dirs(export_dir / row)
        results['rows'][row] = {}
        for eye in eyes:
            if eye not in ref_eyes or eye not in row_eyes:
                raise SystemExit(f'"{eye}" images missing for row "{row}" or the reference')
            print(f'{row} ({eye})', flush=True)
            results['rows'][row][eye] = score_sequence(
                sorted(ref_eyes[eye].glob('*.png')), sorted(row_eyes[eye].glob('*.png')), window, gaze,
            )

    lines = [f'FovVideoVDP (pyfvvdp {results["pyfvvdp"]}), mean JOD over {window}-frame windows, '
             f'reference: {reference}', '']
    lines.append(f'{"row":<28}' + ''.join(f'{eye:>10}' for eye in eyes))
    for row, scores in results['rows'].items():
        lines.append(f'{row:<28}' + ''.join(f'{scores[eye]["jod"]:>10.4f}' for eye in eyes))
    table = '\n'.join(lines) + '\n'
    (export_dir / 'fvvdp_results.json').write_text(json.dumps(results, indent=1))
    (export_dir / 'fvvdp_table.txt').write_text(table)
    print('\n' + table)
    print(f'results written to: {export_dir / "fvvdp_table.txt"} and fvvdp_results.json')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'export_dir', type=Path,
        help='Output directory of vrhtgs_export_trajectory.py, containing one subdirectory per manifest row.'
    )
    parser.add_argument(
        '--reference', default='Reference',
        help='Name of the row used as the reference (default: Reference).'
    )
    parser.add_argument(
        '--rows', default=None, type=lambda s: s.split(','),
        help='Comma-separated rows to score (default: all rows except the reference).'
    )
    parser.add_argument(
        '--eyes', default=None, type=lambda s: s.split(','),
        help='Comma-separated eyes to score for stereo exports, e.g. "left" (default: all exported eyes).'
    )
    parser.add_argument(
        '--window', default=31, type=int,
        help='Frames per scored window, i.e. 1 + the -i value of the export (default: 31 for -i 30). '
             'Also used as frames_per_second.'
    )
    parser.add_argument(
        '--gaze', default=None, type=int, nargs=2, metavar=('X', 'Y'),
        help='Fixation point in pixels (default: image centre, which is also where the foveated renderer '
             'places the fovea when no gaze is given).'
    )
    args = parser.parse_args()
    return {
        'export_dir': args.export_dir,
        'reference': args.reference,
        'rows': args.rows,
        'eyes': args.eyes,
        'window': args.window,
        'gaze': tuple(args.gaze) if args.gaze else None,
    }


if __name__ == '__main__':
    main(**parse_args())

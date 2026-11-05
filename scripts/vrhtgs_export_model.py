#! /usr/bin/env python3

"""export_model.py: Extracts a trained model into a .ply and optionally
background coefficients / texture file using a method-specific format."""

import struct
from argparse import ArgumentParser
from pathlib import Path

from plyfile import PlyData, PlyElement
import numpy as np
import torch

import utils
with utils.DiscoverSourcePath():
    import Framework
    from Datasets.utils import save_image
    from Logging import Logger
    from Implementations import Methods as MI


def save_as_ply(output_path: Path, ply_data_dict: dict[str, np.ndarray | list[str]], use_ascii: bool) -> None:
    elements = [PlyElement.describe(data, key) for key, data in ply_data_dict.items() if key != 'comments']  # noqa
    comments = ply_data_dict.get('comments', [])
    PlyData(elements=elements, text=use_ascii, comments=comments).write(output_path)


def save_environment_texture(output_path: Path, texture: torch.Tensor) -> None:
    """Writes a (3, height, width) texture as a half precision latitude-longitude environment map"""
    try:
        import OpenEXR
    except ImportError:
        Logger.log_error('OpenEXR is required to save environment textures. Please install it via "pip install OpenEXR"')
        exit(1)

    texture = texture.detach().half().cpu().contiguous()
    header = {
        'compression': OpenEXR.ZIP_COMPRESSION,
        'type': OpenEXR.scanlineimage,
        'envmap': OpenEXR.Envmap(OpenEXR.ENVMAP_LATLONG),
    }
    channels = {name: plane.numpy() for name, plane in zip('RGB', texture)}
    with OpenEXR.File(header, channels) as exr_file:
        exr_file.write(str(output_path))


def main(*, base_dir: Path, ignore_background_model: bool = False) -> None:
    # setup framework
    Framework.setup(config_path=str(base_dir / 'training_config.yaml'), require_custom_config=True)
    # load model
    model = MI.get_model(
        method=Framework.config.GLOBAL.METHOD_TYPE,
        checkpoint=str(base_dir / 'checkpoints' / 'final.pt'),
    ).eval()

    Logger.log_info('starting ply export')

    # extract data from model
    ply_data_dict = model.get_ply_dict()

    if ply_data_dict:
        # create and write .ply file
        save_as_ply(base_dir / 'final.ply', ply_data_dict, use_ascii=False)
        Logger.log_info('done')
    else:
        Logger.log_warning('model does not support ply export')

    if ignore_background_model:
        return

    background_model_type = getattr(model, 'BACKGROUND_MODEL', None)
    if background_model_type is None:
        Logger.log_warning('model does not have a background model')
        return
    if background_model_type == 'NONE':
        Logger.log_info('model was trained without a background model, skipping background export')
        return

    Logger.log_info('starting background model export')
    if background_model_type == 'SH':
        coeffs = model.background_model.get_sh_coefficients.cpu()
        with open(base_dir / 'background_sh_coefficients.bin', 'wb') as f:
            f.write(struct.pack(f'{coeffs.numel()}f', *coeffs.flatten().numpy()))
    elif background_model_type == 'TEX':
        # raw texture, i.e. storing pre-activation values: the viewer applies the sigmoid after interpolation
        save_environment_texture(base_dir / 'background_texture.exr', model.background_model.get_texture)
    elif background_model_type == 'MLP':
        # the baked texture already stores post-activation values, so it can be quantized to a png directly
        save_image(base_dir / 'background_texture.png', model.background_model.get_baked_texture)
    else:
        Logger.log_warning(f'background model type {background_model_type} is not supported')
    Logger.log_info('done')


if __name__ == '__main__':
    parser = ArgumentParser(
        prog='export_model.py',
        description='Extracts a trained model into a .ply file using a method-specific format. Also includes background'
                    'model as SH coefficients / environment texture if present.'
    )
    parser.add_argument(
        '-d', '--dir', nargs='+', dest='base_dir', default=None,
        metavar='path/to/output/directory', required=True,
        help='A directory containing the outputs of a completed training.'
    )
    parser.add_argument(
        '-b', '--ignore_background_model', action='store_true', default=False,
        help='Do not save background model data (SH coefficients or environment texture) even if present.'
    )
    args, _ = parser.parse_known_args()
    Logger.set_mode(Logger.MODE_VERBOSE)
    for base_dir in args.base_dir:
        main(
            base_dir=Path(base_dir),
            ignore_background_model=args.ignore_background_model
        )

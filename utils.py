"""VRHTGS/utils.py: Utility functions for the VRHTGS method, including visibility / fovea mask generation."""

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import Framework
from Cameras.Perspective import PerspectiveCamera
from Datasets.utils import View
from Logging import Logger
from Methods.VRHTGS.VRHTGSCudaBackend import RasterizerSettings
from Methods.VRHTGS.VRHTGSCudaBackend.VRHTGSCudaBackend.torch_bindings.rasterization import \
    BackgroundModel


@dataclass
class VisibilityMask:
    size: tuple[int, int]
    mask: torch.Tensor
    mask_sat: torch.Tensor  # Summed Area Table


class MaskTables:
    # IMPORTANT: Constants must match config.h in the CUDA backend!
    TILE_SIZE_SMALL = 8
    TILE_SIZE_LARGE = 16
    FOVEAL_TILES_OUTER_RADIUS = (375 - 125) // TILE_SIZE_LARGE
    BLENDED_TILES_OUTER_RADIUS = (375 + TILE_SIZE_LARGE - 1) // TILE_SIZE_LARGE  # >375 pixel radius
    visibility_mask_cache: dict[int, VisibilityMask] = {}
    fovea_mask_sat_cache: torch.Tensor | None = None

    @classmethod
    def _generate_ones_mask(cls, tiles_x: int, tiles_y: int) -> torch.Tensor:
        return torch.ones((tiles_y, tiles_x), dtype=torch.uint8, device='cuda')

    @classmethod
    def _generate_ellipsoid_mask(cls, tiles_x: int, tiles_y: int) -> torch.Tensor:
        rx = tiles_x / 2
        ry = tiles_y / 2
        y, x = torch.meshgrid(
            torch.linspace(-ry + 0.5, ry - 0.5, tiles_y, dtype=torch.float32, device='cuda') ** 2 / (ry * ry),
            torch.linspace(-rx + 0.5, rx - 0.5, tiles_x, dtype=torch.float32, device='cuda') ** 2 / (rx * rx),
            indexing='ij',
        )
        return torch.where(
            x + y > 1.0,
            torch.tensor(0, dtype=torch.uint8, device='cuda'),
            torch.tensor(1, dtype=torch.uint8, device='cuda')
        )  # Ellipsoid mask

    @classmethod
    def _mask_from_file(cls, path: Path, width: int, height: int, tiles_x: int, tiles_y: int) -> torch.Tensor:
        # Load visibility mask points
        visibility_mask = path.read_bytes()

        # Parse polygon points
        num_points = len(visibility_mask) // struct.calcsize('ff')
        points = np.empty((num_points, 2), dtype=np.float32)
        for i in range(num_points):
            points[i] = struct.unpack_from('ff', visibility_mask, offset=i * struct.calcsize('ff'))
        v1 = torch.from_numpy(points).to(device='cuda')
        v2 = torch.roll(v1, -1, dims=0)
        v2v1 = (v2 - v1)[None, None, :, :]

        # Create tile coordinate grid
        y, x = torch.meshgrid(
            torch.arange(0.5, tiles_y + 0.5, 1.0, dtype=torch.float32, device='cuda') * cls.TILE_SIZE_LARGE / height,
            torch.arange(0.5, tiles_x + 0.5, 1.0, dtype=torch.float32, device='cuda') * cls.TILE_SIZE_LARGE / width,
            indexing='ij',
        )
        p = torch.stack((x, y), dim=-1)

        # Check if points are on the same side of all edges
        pv1 = p[:, :, None, :] - v1[None, None, :, :]
        cross = v2v1[..., 0] * pv1[..., 1] - v2v1[..., 1] * pv1[..., 0]
        sign = cross > 0
        inside = torch.all(sign, dim=2) | torch.all(~sign, dim=2)
        mask = inside.to(torch.uint8)

        return mask

    @classmethod
    def get_visibility_mask(cls, idx: int, width: int, height: int) -> tuple[torch.Tensor, torch.Tensor]:
        cache_entry: VisibilityMask | None = cls.visibility_mask_cache.get(idx)
        cache_size = cache_entry.size if cache_entry is not None else (-1, -1)
        tiles_y = (height + cls.TILE_SIZE_LARGE - 1) // cls.TILE_SIZE_LARGE
        tiles_x = ( width + cls.TILE_SIZE_LARGE - 1) // cls.TILE_SIZE_LARGE
        if cache_size == (tiles_y, tiles_x):
            return cache_entry.mask, cache_entry.mask_sat

        # Try loading visibility mask points
        mask_dir = Path(__file__).parent / 'visibility_masks'
        mask_paths = list(sorted(mask_dir.glob('*.bin'))) if mask_dir.is_dir() else []
        if idx < 0:
            raw_mask = cls._generate_ones_mask(tiles_x, tiles_y)
        elif idx < len(mask_paths):
            mask_path = mask_paths[idx]
            raw_mask = cls._mask_from_file(mask_path, width, height, tiles_x, tiles_y)
        else:
            raw_mask = cls._generate_ellipsoid_mask(tiles_x, tiles_y)

        # Calculate summed area table, needs to be 1 element larger in each dimension to allow indexing using
        #   the *inclusive* range [0, tiles_x] and [0, tiles_y]
        mask_sum = torch.zeros((tiles_y + 1, tiles_x + 1), dtype=torch.uint32, device='cuda')
        mask_sum[1:, 1:] = torch.cumsum(torch.cumsum(raw_mask, dim=0), dim=1)

        packed_mask_np = np.packbits(raw_mask.flatten().cpu().numpy(), axis=-1, bitorder='little')
        packed_mask = torch.zeros(((packed_mask_np.shape[0] + 3) // 4 * 4,), dtype=torch.uint8)
        packed_mask[:packed_mask_np.shape[0]] = torch.from_numpy(packed_mask_np)

        cache_entry = VisibilityMask(
            mask=packed_mask.view(dtype=torch.uint32).to(device='cuda'),
            mask_sat=mask_sum,
            size=(tiles_y, tiles_x),
        )
        cls.visibility_mask_cache[idx] = cache_entry

        return cache_entry.mask, cache_entry.mask_sat


    @classmethod
    def get_fovea_mask_sat(cls) -> torch.Tensor:
        if cls.fovea_mask_sat_cache is not None:
            return cls.fovea_mask_sat_cache

        # Create tile mask
        grid_size = MaskTables.BLENDED_TILES_OUTER_RADIUS * 2
        fovea_mask = 3 * MaskTables._generate_ellipsoid_mask(grid_size, grid_size)

        # Calculate summed area table, needs to be 1 element larger in each dimension to allow indexing using
        #   the *inclusive* range [0, tiles_x] and [0, tiles_y]
        cls.fovea_mask_sat_cache = torch.zeros((grid_size + 1, grid_size + 1), dtype=torch.uint32, device='cuda')
        cls.fovea_mask_sat_cache[1:, 1:] = torch.cumsum(torch.cumsum(fovea_mask, dim=0), dim=1)
        return cls.fovea_mask_sat_cache


def extract_settings(
        view: View,
        K: int,
        active_sh_bases: int,
        scale_modifier: float,
        background_model: torch.nn.Module,
        background_model_type: str = None,
        visibility_mask_index: int = 0,
) -> RasterizerSettings:
    if not isinstance(view.camera, PerspectiveCamera):
        raise Framework.RendererError('HTGS renderer only supports perspective cameras')
    if view.camera.distortion is not None:
        Logger.log_warning('found distortion parameters that will be ignored by the rasterizer')
    M = view.w2c
    R = torch.eye(4, dtype=M.dtype, device=M.device)
    R[:3, :3] = view.rotation
    if background_model_type == 'SH':
        background_model_type = BackgroundModel.SH
        background_model_data = background_model.get_sh_coefficients
    elif background_model_type == 'TEX':
        background_model_type = BackgroundModel.TEXTURE_RAW  # pre-activation
        background_model_data = background_model.get_texture
    elif background_model_type == 'MLP':
        background_model_type = BackgroundModel.TEXTURE
        background_model_data = background_model.get_baked_texture
    else:
        background_model_type = BackgroundModel.NONE
        background_model_data = torch.empty(0, dtype=torch.float32, device='cuda')

    focal_x = view.camera.focal_x
    focal_y = view.camera.focal_y
    near_plane = view.camera.near_plane
    far_plane = view.camera.far_plane
    z_range = far_plane - near_plane
    VP = torch.tensor([
        [view.camera.focal_x, 0.0, view.camera.center_x - 0.5, 0.0],
        [0.0, view.camera.focal_y, view.camera.center_y - 0.5, 0.0],
        [0.0, 0.0, (far_plane + near_plane) / z_range, -2.0 * far_plane * near_plane / z_range],
        [0.0, 0.0, 1.0, 0.0]
    ], dtype=torch.float32, device=M.device)
    visibility_mask, visibility_mask_sat = MaskTables.get_visibility_mask(visibility_mask_index, view.camera.width, view.camera.height)
    fovea_mask_sum = MaskTables.get_fovea_mask_sat()

    return RasterizerSettings(
        M=M,
        VPM=VP @ M,
        VPR_inv=R @ torch.inverse(VP),
        cam_position=view.position,
        gaze_position=torch.from_numpy(view.camera.gaze_position),
        K=K,
        active_sh_bases=active_sh_bases,
        width=view.camera.width,
        height=view.camera.height,
        focal_x=focal_x,
        focal_y=focal_y,
        center_x=view.camera.center_x,
        center_y=view.camera.center_y,
        near_plane=near_plane,
        far_plane=far_plane,
        scale_modifier=scale_modifier,
        visibility_mask=visibility_mask,
        visibility_mask_area_table=visibility_mask_sat,
        fovea_mask_area_table=fovea_mask_sum,
        background_model=background_model_type,
        background_model_data=background_model_data,
    )

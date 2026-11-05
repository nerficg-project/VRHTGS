"""VRHTGS/Renderer.py: Defines the foveated renderer for the VR-HTGS method."""

from copy import deepcopy

import torch

import Framework
from Datasets.utils import View
from Logging import Logger
from Methods.Base.Renderer import BaseModel
from Methods.HTGS.Renderer import HTGSRenderer
from Methods.VRHTGS.VRHTGSCudaBackend import VRHTGSRasterizer, VRHTGSVisualizer
from Methods.VRHTGS.VRHTGSCudaBackend.VRHTGSCudaBackend.torch_bindings.rasterization import \
    GazeVisualizationType
from Methods.VRHTGS.utils import extract_settings, MaskTables


@Framework.Configurable.configure(
    BLUR_PERIPHERY=True,
    FOVEATION_ENABLED=True,
    FOVEATION_VISUALIZE_GAZE=False,
    FOVEATION_VISUALIZE_TILE_GRID=False,
    FOVEATION_VISUALIZATION_TYPE=0,
    FOVEATION_VISUALIZATION_SIZE=32,
    FOVEATION_VISUALIZATION_COLOR=[0.8, 0.475, 0.655],
    FOVEATION_VISIBILITY_MASK_INDEX=0,
)
class VRHTGSRenderer(HTGSRenderer):
    """Foveated renderer for the VR-HTGS method."""

    def __init__(self, model: 'BaseModel') -> None:
        super().__init__(model)
        self.foveated_rasterizer = VRHTGSRasterizer()
        self.foveated_visualizer = VRHTGSVisualizer()
        if self.BLEND_MODE != 0:
            Logger.log_warning(f'Unsupported blend mode {self.BLEND_MODE} for foveated rasterization. '
                               f'Training using a different blend mode will result in *significantly* worse results.')
        if self.K not in [1, 2, 4, 8, 16, 32]:
            Logger.log_warning(f'unsupported K value may lead to undefined behavior: {self.K}')
        if self.model.BACKGROUND_DISCARD_BELOW != 0.1:
            Logger.log_warning(f'the foveated rasterizer hardcodes the background discard at 0.1, '
                               f'ignoring MODEL.BACKGROUND_DISCARD_BELOW: {self.model.BACKGROUND_DISCARD_BELOW}')
        self._icgui_debug_force_stereo: bool = False

    @torch.no_grad()
    def render_image_inference(self, view: View, to_chw: bool) -> dict[str, torch.Tensor]:
        """Renders an image for a given view during inference."""
        if not self.FOVEATION_ENABLED:
            return super().render_image_inference(view, to_chw)
        if self._icgui_debug_force_stereo:
            return self._debug_render_stereo(view, to_chw)

        # modify sh features for visualization
        sh_0 = self.model.gaussians.get_sh_0
        if self.DISABLE_SH0:
            sh_0 = torch.zeros_like(sh_0)
        sh_rest = self.model.gaussians.get_sh_rest
        if self.DISABLE_SH1 or self.DISABLE_SH2 or self.DISABLE_SH3:
            sh_rest = sh_rest.clone()
        if self.DISABLE_SH1:
            sh_rest[:, 0:3].zero_()
        if self.DISABLE_SH2:
            sh_rest[:, 3:8].zero_()
        if self.DISABLE_SH3:
            sh_rest[:, 8:15].zero_()

        bg_model_type = self.model.BACKGROUND_MODEL
        if self.DISABLE_BG_MODEL_INFERENCE:
            bg_model_type = 'NONE'

        settings = extract_settings(
            view=view,
            K=self.K,
            active_sh_bases=self.model.gaussians.active_sh_bases,
            scale_modifier=self.SCALE_MODIFIER,
            background_model=self.model.background_model,
            background_model_type=bg_model_type,
            visibility_mask_index=self.FOVEATION_VISIBILITY_MASK_INDEX,
        )
        rgb = self.foveated_rasterizer.render(
            positions=self.model.gaussians.get_positions,
            scales=self.model.gaussians.get_scales_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_scales,
            rotations=self.model.gaussians.get_rotations,
            opacities=self.model.gaussians.get_opacities_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_opacities,
            sh_0=sh_0,
            sh_rest=sh_rest,
            settings=settings,
            to_chw=to_chw,
            blur_periphery=self.BLUR_PERIPHERY,
            aaa_mode=self.USE_AAA_MODE,
        )

        if self.FOVEATION_VISUALIZE_GAZE:
            gaze_type = GazeVisualizationType(self.FOVEATION_VISUALIZATION_TYPE)
            self.foveated_visualizer.visualize_gaze(
                image=rgb,
                gaze_position=torch.from_numpy(view.camera.gaze_position),
                gaze_color=torch.tensor(self.FOVEATION_VISUALIZATION_COLOR, device='cpu'),
                visualization_size=self.FOVEATION_VISUALIZATION_SIZE,
                visualization_type=gaze_type,
                to_chw=to_chw,
            )
        if self.FOVEATION_VISUALIZE_TILE_GRID:
            self.foveated_visualizer.visualize_tile_boundaries(
                image=rgb,
                visibility_mask=MaskTables.get_visibility_mask(self.FOVEATION_VISIBILITY_MASK_INDEX, view.camera.width, view.camera.height)[0],
                gaze_position=torch.from_numpy(view.camera.gaze_position),
                to_chw=to_chw,
            )

        return { 'rgb': rgb }

    @torch.inference_mode()
    def render_image_benchmark(self, view: View, to_chw: bool) -> dict[str, torch.Tensor]:
        """Renders only the rgb image "as fast as possible"."""
        bg_model_type = self.model.BACKGROUND_MODEL
        if self.DISABLE_BG_MODEL_INFERENCE:
            bg_model_type = 'NONE'

        settings = extract_settings(
            view=view,
            K=self.K,
            active_sh_bases=self.model.gaussians.active_sh_bases,
            scale_modifier=self.SCALE_MODIFIER,
            background_model=self.model.background_model,
            background_model_type=bg_model_type,
            visibility_mask_index=self.FOVEATION_VISIBILITY_MASK_INDEX,
        )
        rgb = self.foveated_rasterizer.benchmark(
            positions=self.model.gaussians.get_positions,
            scales=self.model.gaussians.get_scales_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_scales,
            rotations=self.model.gaussians.get_rotations,
            opacities=self.model.gaussians.get_opacities_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_opacities,
            sh_0=self.model.gaussians.get_sh_0,
            sh_rest=self.model.gaussians.get_sh_rest,
            settings=settings,
            to_chw=to_chw,
            blur_periphery=self.BLUR_PERIPHERY,
            aaa_mode=self.USE_AAA_MODE,
        )
        return { 'rgb': rgb }

    @torch.inference_mode()
    def render_image_stereo(self, view_left: View, view_right: View, to_chw: bool) -> dict[str, torch.Tensor]:
        """Renders only the rgb image "as fast as possible" for two views."""
        bg_model_type = self.model.BACKGROUND_MODEL
        if self.DISABLE_BG_MODEL_INFERENCE:
            bg_model_type = 'NONE'

        settings_left = extract_settings(
            view=view_left,
            K=self.K,
            active_sh_bases=self.model.gaussians.active_sh_bases,
            scale_modifier=self.SCALE_MODIFIER,
            background_model=self.model.background_model,
            background_model_type=bg_model_type,
            visibility_mask_index=view_left.camera_index,
        )
        settings_right = extract_settings(
            view=view_right,
            K=self.K,
            active_sh_bases=self.model.gaussians.active_sh_bases,
            scale_modifier=self.SCALE_MODIFIER,
            background_model=self.model.background_model,
            background_model_type=bg_model_type,
            visibility_mask_index=view_right.camera_index,
        )

        rgb_left, rgb_right = self.foveated_rasterizer.render_stereo(
            positions=self.model.gaussians.get_positions,
            scales=self.model.gaussians.get_scales_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_scales,
            rotations=self.model.gaussians.get_rotations,
            opacities=self.model.gaussians.get_opacities_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_opacities,
            sh_0=self.model.gaussians.get_sh_0,
            sh_rest=self.model.gaussians.get_sh_rest,
            settings_left=settings_left,
            settings_right=settings_right,
            to_chw=to_chw,
            blur_periphery=self.BLUR_PERIPHERY,
            aaa_mode=self.USE_AAA_MODE,
        )
        return { 'rgb_left': rgb_left, 'rgb_right': rgb_right }

    def _debug_render_stereo(self, view: View, to_chw: bool) -> dict[str, torch.Tensor]:
        """Renders the same view twice, for debugging purposes."""
        view1 = deepcopy(view)
        view2 = deepcopy(view)
        view1.camera_index = 0
        view2.camera_index = 1
        view1.camera.width //= 2
        view2.camera.width //= 2
        if hasattr(view.camera, 'center_x'):
            view1.camera.center_x *= view1.camera.width / view.camera.width
            view2.camera.center_x *= view2.camera.width / view.camera.width
        # the gaze is given in full-width coordinates, so it has to be mapped into the half-width views
        gaze_position = view.camera.gaze_position
        # wrap once(!) at the border between the two sides
        # this avoids jumps when moving the mouse outside the window bounds,
        #   as wrapping is performed in the backend code!
        if gaze_position[0] >= view1.camera.width:
            gaze_position[0] -= view1.camera.width
        view1.camera.gaze_position = gaze_position
        view2.camera.gaze_position = gaze_position

        res = self.render_image_stereo(view1, view2, to_chw)
        if self.FOVEATION_VISUALIZE_GAZE:
            gaze_type = GazeVisualizationType(self.FOVEATION_VISUALIZATION_TYPE)
            self.foveated_visualizer.visualize_gaze(
                image=res['rgb_left'],
                gaze_position=torch.from_numpy(view1.camera.gaze_position),
                gaze_color=torch.tensor(self.FOVEATION_VISUALIZATION_COLOR, device='cpu'),
                visualization_size=self.FOVEATION_VISUALIZATION_SIZE,
                visualization_type=gaze_type,
                to_chw=to_chw,
            )
            self.foveated_visualizer.visualize_gaze(
                image=res['rgb_right'],
                gaze_position=torch.from_numpy(view2.camera.gaze_position),
                gaze_color=torch.tensor(self.FOVEATION_VISUALIZATION_COLOR, device='cpu'),
                visualization_size=self.FOVEATION_VISUALIZATION_SIZE,
                visualization_type=gaze_type,
                to_chw=to_chw,
            )
        if self.FOVEATION_VISUALIZE_TILE_GRID:
            self.foveated_visualizer.visualize_tile_boundaries(
                image=res['rgb_left'],
                visibility_mask=
                MaskTables.get_visibility_mask(view1.camera_index, view1.camera.width, view1.camera.height)[0],
                gaze_position=torch.from_numpy(view1.camera.gaze_position),
                to_chw=to_chw,
            )
            self.foveated_visualizer.visualize_tile_boundaries(
                image=res['rgb_right'],
                visibility_mask=
                MaskTables.get_visibility_mask(view2.camera_index, view2.camera.width, view2.camera.height)[0],
                gaze_position=torch.from_numpy(view2.camera.gaze_position),
                to_chw=to_chw,
            )

        return {
            'rgb': torch.cat([res['rgb_left'], res['rgb_right']], dim=0 if to_chw else 1),
        }

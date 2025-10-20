#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
from .renderer import *
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from internal.utils.sh_utils import eval_sh
from pathlib import Path
from internal.models.sh_core import compute_distance, compute_nadir_angle, d_to_u, phi_to_v, apply_adaptive_sh_weights, GateLUT, GateMLP
import torch.nn as nn
import torch
from log import print_to


class CigaRenderer(Renderer):
    def __init__(self, compute_cov3D_python: bool = False, convert_SHs_python: bool = False):
        super().__init__()

        self.compute_cov3D_python = compute_cov3D_python
        self.convert_SHs_python = convert_SHs_python

    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            override_color=None,
            render_types: list = None
    ):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        if render_types is None:
            render_types = ["rgb"]
        assert len(render_types) == 1, "Only single type is allowed currently"

        rendered_image_key = "render"
        if "depth" in render_types:
            rendered_image_key = "depth"
            w2c = viewpoint_camera.world_to_camera  # already transposed
            means3D_in_camera_space = torch.matmul(pc.get_xyz, w2c[:3, :3]) + w2c[3, :3]
            depth = means3D_in_camera_space[:, 2:]
            # bg_color = torch.ones_like(bg_color) * depth.max()
            bg_color = torch.zeros_like(bg_color)
            override_color = depth.repeat(1, 3)

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True,
                                              device=bg_color.device) + 0

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.fov_x * 0.5)
        tanfovy = math.tan(viewpoint_camera.fov_y * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_to_camera,
            projmatrix=viewpoint_camera.full_projection,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if self.compute_cov3D_python is True:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if self.convert_SHs_python is True:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = override_color

        # 추가 
        shs = self.shs_weight_MLP(
            rasterizer, 
            shs, 
            viewpoint_camera, 
            means3D, 
            L=pc.active_sh_degree
            )

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            rendered_image_key: rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

    def set_mlp(self, mlp: nn.Module):
        self.mlp_model = mlp

    @staticmethod
    def render(
            means3D: torch.Tensor,  # xyz
            opacity: torch.Tensor,
            scales: Optional[torch.Tensor],
            rotations: Optional[torch.Tensor],
            features: Optional[torch.Tensor],  # shs
            active_sh_degree: int,
            viewpoint_camera,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            colors_precomp: Optional[torch.Tensor] = None,
            cov3D_precomp: Optional[torch.Tensor] = None,
    ):
        if colors_precomp is not None:
            assert features is None
        if cov3D_precomp is not None:
            assert scales is None
            assert rotations is None

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(
            means3D,
            dtype=means3D.dtype,
            requires_grad=True,
            device=means3D.device,
        )

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.fov_x * 0.5)
        tanfovy = math.tan(viewpoint_camera.fov_y * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_to_camera,
            projmatrix=viewpoint_camera.full_projection,
            sh_degree=active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means2D = screenspace_points

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rasterize_result = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=features,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
        if len(rasterize_result) == 2:
            rendered_image, radii = rasterize_result
            depth_image = None
        else:
            rendered_image, radii, depth_image = rasterize_result

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "render": rendered_image,
            "depth": depth_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

    def get_available_outputs(self) -> Dict:
        return {
            "rgb": RendererOutputInfo("render"),
            "depth": RendererOutputInfo("depth", RendererOutputTypes.GRAY),
        }

    def shs_weight_MLP(self, rasterizer, shs, VC, means3D, L):
            
        import torch

        if shs is None:
            return shs

        N, K, C = shs.shape
        device, dtype = shs.device, shs.dtype
        assert C == 3 and K == (L + 1) ** 2, f"got shs={shs.shape}, L={L}"

        if hasattr(rasterizer, "markVisible"):
            with torch.no_grad():
                vis_mask = rasterizer.markVisible(means3D)  # -> (N,) bool
            vis_idx = torch.where(vis_mask)[0]
            if vis_idx.numel() == 0:
                return shs
            means3D_vis = means3D[vis_idx]
        else:
            vis_idx = torch.arange(N, device=device)
            means3D_vis = means3D


        cam = getattr(VC, "camera_center", None)
        if cam is None:
            cam = getattr(VC, "cam_pos", None)
        if cam is None:
            raise RuntimeError("Viewer/Camera object has no camera_center/cam_pos")
        if not torch.is_tensor(cam):
            cam = torch.as_tensor(cam, device=device, dtype=dtype)
        else:
            cam = cam.to(device=device, dtype=dtype)

 
        vec = means3D_vis - cam   
        dis = torch.linalg.norm(vec, dim=1, keepdim=True).clamp_min(1e-8) 
        dir = vec / dis                

        try:
            u = d_to_u(dis)                        
        except TypeError:
        
            u = d_to_u(dis, 5.0, 0.2, 80.0)
        u = u.to(device=device, dtype=dtype)         

        x = torch.cat([u, dir], dim=1)            
      
        use_mlp = hasattr(self, "mlp_model") and self.mlp_model is not None
        if use_mlp:
            band_w = self.mlp_model(x)              
           
            band_w[:, 0] = 1.0
        else:
            M = means3D_vis.shape[0]
            band_w = torch.ones(M, L + 1, device=device, dtype=dtype)
            

        coeff_w = self.band_flatten(band_w, L)      
        coeff_w = coeff_w.unsqueeze(-1)           

        shs_out = shs.clone()
        shs_out[vis_idx] = shs[vis_idx] * coeff_w.to(dtype=dtype, device=device)
        return shs_out


        


    def band_flatten(self, band_weights, L):
        """
        band_weights : [N, L+1]
        return       : [N, K=(L+1)^2]  (계수별 스칼라 가중치)
        """
        import torch
        N, B = band_weights.shape
        assert B == L + 1, f"got {band_weights.shape}, expected [N, {L+1}]"

        band_sizes = torch.tensor([2 * l + 1 for l in range(L + 1)],
                                device=band_weights.device)
        band_of_coeff = torch.repeat_interleave(
            torch.arange(L + 1, device=band_weights.device),
            band_sizes
        )  # [K]
        coeff_weights = band_weights.index_select(dim=1, index=band_of_coeff)  # [N,K]
        return coeff_weights

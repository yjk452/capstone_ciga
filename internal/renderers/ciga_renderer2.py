

import math

from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .renderer import *
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from internal.utils.sh_utils import eval_sh

from internal.models.sh_core import (
    compute_distance, compute_nadir_angle,
    d_to_u, phi_to_v, apply_adaptive_sh_weights,
    GateLUT, GateMLP, band_flatten
)

from log import print_to 


class CigaRenderer(Renderer):
    def __init__(
        self,
        compute_cov3D_python: bool = False,
        convert_SHs_python: bool = False,
        mlp_cfg: dict = None,
        sh_max_degree: int = None,
        **kwargs
    ):
        super().__init__()

        self.compute_cov3D_python = compute_cov3D_python
        self.convert_SHs_python = convert_SHs_python
        self.mlp_cfg = mlp_cfg or {}
        self.mlp_model: Optional[nn.Module] = None
        self._last_band_w = None  

    
    def training_setup(self, pl_module):
      
        L_max = int(getattr(pl_module.gaussian_model, "max_sh_degree", 3))
        in_dim = self.mlp_cfg.get("in_dim", 4)  
        hidden = self.mlp_cfg.get("hidden", 32)
        self.mlp_model = GateMLP(in_dim=in_dim, L_max=L_max, hidden=hidden).to(pl_module.device)
        return None, None  

   
    def get_adaptive_parameters(self):
        return [] if self.mlp_model is None else self.mlp_model.parameters()

    def set_mlp(self, mlp: nn.Module):
        self.mlp_model = mlp

    def get_available_outputs(self) -> Dict:
        return {
            "rgb": RendererOutputInfo("render"),
            "depth": RendererOutputInfo("depth", RendererOutputTypes.GRAY),
        }

    def forward(
        self,
        viewpoint_camera: Camera,
        pc: GaussianModel,
        bg_color: torch.Tensor,
        scaling_modifier: float = 1.0,
        override_color: Optional[torch.Tensor] = None,
        render_types: Optional[list] = None,
    ):
    
        if render_types is None:
            render_types = ["rgb"]
        assert len(render_types) == 1, "CigaRenderer currently supports single render type at a time."

        rendered_image_key = "render"
        if "depth" in render_types:
            rendered_image_key = "depth"
            w2c = viewpoint_camera.world_to_camera  # already transposed
            means3D_in_camera_space = torch.matmul(pc.get_xyz, w2c[:3, :3]) + w2c[3, :3]
            depth = means3D_in_camera_space[:, 2:]

            bg_color = torch.zeros_like(bg_color)
            override_color = depth.repeat(1, 3)

        # screen-space points for grads
        screenspace_points = torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=bg_color.device
        ) + 0

        # raster settings
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
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        # covariance
        scales = rotations = cov3D_precomp = None
        if self.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        # SH / colors
        shs = None
        colors_precomp = None
        if override_color is None:
            if self.convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = override_color

       
        shs_before = shs
        shs_after = self.shs_weight_MLP(
            rasterizer=rasterizer,
            shs=shs,
            VC=viewpoint_camera,
            means3D=means3D,
            L=pc.active_sh_degree,
        )

        # render
        rasterize_result = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs_after,
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

        cam = viewpoint_camera.camera_center.to(means3D.device, means3D.dtype)
        vec = means3D - cam
        distances = torch.linalg.norm(vec, dim=1)  # [N]
        down = torch.tensor([0.0, 0.0, -1.0], device=means3D.device, dtype=means3D.dtype)
        dirs = vec / (distances.unsqueeze(1) + 1e-8)
        cosang = torch.clamp(dirs @ down, -1.0, 1.0)
        nadir_angles = torch.acos(cosang)  # [N]

        # ===== recover per-band weights across all N =====
        sh_weights = None
        if self._last_band_w is not None:
            band_w_full, vis_idx, L_full = self._last_band_w
            N = means3D.shape[0]
            sh_weights = torch.ones(N, L_full + 1, device=means3D.device, dtype=means3D.dtype)
            sh_weights[vis_idx] = band_w_full

        return {
            rendered_image_key: rendered_image,
            "depth": depth_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,

            # Adaptive SH training
            "shs": shs_after if shs_after is not None else shs_before,
            "sh_weights": sh_weights,  # [N, L_full+1] or None
            "adaptive_sh_info": {
                "distances": distances,        # [N]
                "nadir_angles": nadir_angles,  # [N]
            },
        }

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


        screenspace_points = torch.zeros_like(
            means3D, dtype=means3D.dtype, requires_grad=True, device=means3D.device
        )


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
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means2D = screenspace_points

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


        return {
            "render": rendered_image,
            "depth": depth_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

    
    
    def shs_weight_MLP(self, rasterizer, shs, VC, means3D, L=None):
        if shs is None:
            return shs
        assert shs.ndim == 3
        device, dtype = shs.device, shs.dtype


        orig_is_NKC = (shs.shape[2] == 3)
        if orig_is_NKC:
            N, K, C = shs.shape
            assert C == 3
            shs_nkc = shs
        else:
            N, C, K = shs.shape
            assert C == 3
            shs_nkc = shs.permute(0, 2, 1).contiguous()  # -> NKC

        r = int(round(math.sqrt(K)))
        assert r * r == K, f"K={K} is not a perfect square"
        L_full = r - 1

        if L is None or L < 0:
       
       
            L_active = getattr(self.gaussian_model, "active_sh_degree", L_full)
        else:
            L_active = L
        L_active = int(min(int(L_active), L_full))

        # visible subset
        if hasattr(rasterizer, "markVisible"):
            with torch.no_grad():
                vis_mask = rasterizer.markVisible(means3D)
            vis_idx = torch.where(vis_mask)[0]
            if vis_idx.numel() == 0:
                self._last_band_w = (torch.ones(0, L_full+1, device=device, dtype=dtype), vis_idx, L_full)
                return shs if orig_is_NKC else shs_nkc.permute(0, 2, 1).contiguous()
            means3D_vis = means3D[vis_idx]
        else:
            vis_idx = torch.arange(N, device=device)
            means3D_vis = means3D

        # camera center
        cam = getattr(VC, "camera_center", None) or getattr(VC, "cam_pos", None)
        if cam is None:
            raise RuntimeError("Viewer/Camera object has no camera_center/cam_pos")
        cam = cam.to(device=device, dtype=dtype) if torch.is_tensor(cam) else torch.as_tensor(cam, device=device, dtype=dtype)
        if cam.ndim == 2 and cam.shape[0] == 1:
            cam = cam.squeeze(0)
        elif cam.ndim != 1:
            cam = cam.view(-1)[:3]

        # features: [u, dir(3)]  (in_dim=4 기본)
        vec = means3D_vis - cam
        dis = torch.linalg.norm(vec, dim=1, keepdim=True).clamp_min(1e-8)
        dirv = vec / dis
        try:
            u = d_to_u(dis)
        except TypeError:
           
            u = d_to_u(dis, 0.2, 80.0, 5.0, 0.5)
        u = u.to(device=device, dtype=dtype)
        x = torch.cat([u, dirv], dim=1)  # [M,4]

        # predict band weights
        use_mlp = (self.mlp_model is not None)
        if use_mlp and L_active >= 0:
            band_w = self.mlp_model(x)  
            # DC 고정
            band_w[:, 0] = 1.0
           
            if band_w.shape[1] < (L_full + 1):
                pad_cols = (L_full + 1) - band_w.shape[1]
                pad = torch.ones(band_w.size(0), pad_cols, device=device, dtype=dtype)
                band_w_full = torch.cat([band_w, pad], dim=1)
            else:
                band_w_full = band_w[:, :L_full + 1]
        else:
            M = means3D_vis.shape[0]
            band_w_full = torch.ones(M, L_full + 1, device=device, dtype=dtype)

        coeff_w = band_flatten(band_w_full, L_full).unsqueeze(-1)  # [M,K,1]

        shs_out_nkc = shs_nkc.clone()
        shs_out_nkc[vis_idx] = shs_out_nkc[vis_idx] * coeff_w.to(dtype=dtype, device=device)

        self._last_band_w = (band_w_full.detach(), vis_idx, L_full)

        return shs_out_nkc if orig_is_NKC else shs_out_nkc.permute(0, 2, 1).contiguous()

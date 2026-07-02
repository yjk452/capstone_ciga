

import math
from typing import Optional, Dict

import torch
import torch.nn as nn

from .renderer import *
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from internal.utils.sh_utils import eval_sh

from internal.models.sh_core import (
    build_gate_input,
    compute_distance,
    d_to_u,
    apply_adaptive_sh_weights,
    GateMLP,
    band_flatten,
    compute_nadir_cos_angle,
    cos_to_v,
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

        L_max = sh_max_degree if sh_max_degree is not None else 3

        in_dim = self.mlp_cfg.get("in_dim", 2)
        hidden = self.mlp_cfg.get("hidden", 64)

        self.use_gate_mlp = self.mlp_cfg.get("use_gate_mlp", True)

        if self.use_gate_mlp:
            self.mlp_model = GateMLP(
                in_dim=in_dim,
                L_max=L_max,
                hidden=hidden,
            )
        else:
            self.mlp_model = None

        self._last_band_w = None

    def training_setup(self, pl_module):
        if self.mlp_model is not None:
            self.mlp_model.to(pl_module.device)
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

        screenspace_points = torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=bg_color.device
        ) + 0

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

        scales = rotations = cov3D_precomp = None
        if self.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            if callable(scales):
                scales = scales()
            rotations = pc.get_rotation

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
            scales=scales,  # 지금은 내부에서 안 씀(호환용)
            L=pc.active_sh_degree,
        )

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
        distances = torch.linalg.norm(vec, dim=1)

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
            "shs_raw": shs_before,
            "shs_gated": shs_after,
            "sh_weights": sh_weights,
            "adaptive_sh_info": {
                "distances": distances,
            },
        }

    @staticmethod
    def render(
        means3D: torch.Tensor,
        opacity: torch.Tensor,
        scales: Optional[torch.Tensor],
        rotations: Optional[torch.Tensor],
        features: Optional[torch.Tensor],
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

    def shs_weight_MLP(self, rasterizer, shs, VC, means3D, scales=None, L=None):
        if shs is None:
            return shs
        assert shs.ndim == 3
        device, dtype = shs.device, shs.dtype

        # NKC/NCK 정규화
        orig_is_NKC = (shs.shape[2] == 3)
        if orig_is_NKC:
            N, K, C = shs.shape
            assert C == 3
            shs_nkc = shs
        else:
            N, C, K = shs.shape
            assert C == 3
            shs_nkc = shs.permute(0, 2, 1).contiguous()

        r = int(round(math.sqrt(K)))
        assert r * r == K, f"K={K} is not a perfect square"
        L_full = r - 1

        if L is None or L < 0:
            L_active = getattr(self.gaussian_model, "active_sh_degree", L_full)
        else:
            L_active = L
        L_active = int(min(int(L_active), L_full))

        # 보이는 가우시안만
        if hasattr(rasterizer, "markVisible"):
            with torch.no_grad():
                vis_mask = rasterizer.markVisible(means3D)
            vis_idx = torch.where(vis_mask)[0]
            if vis_idx.numel() == 0:
                self._last_band_w = (torch.ones(0, L_full + 1, device=device, dtype=dtype), vis_idx, L_full)
                return shs if orig_is_NKC else shs_nkc.permute(0, 2, 1).contiguous()
            means3D_vis = means3D[vis_idx]
        else:
            vis_idx = torch.arange(N, device=device)
            means3D_vis = means3D

        # 카메라 위치
        cam = getattr(VC, "camera_center", None)
        if cam is None:
            cam = getattr(VC, "cam_pos", None)
        if cam is None:
            raise RuntimeError("Viewer/Camera object has no camera_center/cam_pos")

        if not torch.is_tensor(cam):
            cam = torch.as_tensor(cam, device=device, dtype=dtype)
        else:
            cam = cam.to(device=device, dtype=dtype)

        if cam.ndim == 2 and cam.shape[0] == 1:
            cam = cam.squeeze(0)
        elif cam.ndim != 1:
            cam = cam.view(-1)[:3]

        
        d = compute_distance(cam, means3D_vis)                 # [M]
        cos_angle = compute_nadir_cos_angle(cam, means3D_vis)  # [M]
        x = build_gate_input(d, cos_angle, d_low=0.2, d_high=30.0, d0=5.0, dw=0.5)  # [M,2]

        use_mlp = (self.mlp_model is not None)
        if use_mlp and L_active >= 0:
            band_w = self.mlp_model(x)  # [M, L'+1]

            # DC=1 (인플레이스 금지)
            if band_w.shape[1] >= 1:
                ones_dc = torch.ones_like(band_w[:, :1])
                band_w = torch.cat([ones_dc, band_w[:, 1:]], dim=1)

            # L_full 크기로 맞추기 (모자라면 1로 패딩)
            if band_w.shape[1] < (L_full + 1):
                pad_cols = (L_full + 1) - band_w.shape[1]
                pad = torch.ones(band_w.size(0), pad_cols, device=device, dtype=dtype)
                band_w_full = torch.cat([band_w, pad], dim=1)
            else:
                band_w_full = band_w[:, : L_full + 1]
        else:
            M = means3D_vis.shape[0]
            band_w_full = torch.ones(M, L_full + 1, device=device, dtype=dtype)

        if not torch.isfinite(band_w_full).all():
            band_w_full = torch.where(torch.isfinite(band_w_full), band_w_full, torch.ones_like(band_w_full))

        if torch.rand(1).item() < 0.001:
            with torch.no_grad():
                ho = band_w_full[:, 1:]
                print(
                    "[GateMLP] high-order mean={:.4f} std={:.4f} min={:.4f} max={:.4f}".format(
                        ho.mean().item(),
                        ho.std().item(),
                        ho.min().item(),
                        ho.max().item(),
                    )
                )

        coeff_w = band_flatten(band_w_full, L_full).unsqueeze(-1)  # [M,K,1]
        w_safe = torch.where(torch.isfinite(coeff_w), coeff_w, torch.ones_like(coeff_w))

        shs_out_nkc = shs_nkc.clone()
        shs_out_nkc[vis_idx] = shs_out_nkc[vis_idx] * w_safe.to(dtype=dtype, device=device)

        self._last_band_w = (band_w_full, vis_idx, L_full)
        return shs_out_nkc if orig_is_NKC else shs_out_nkc.permute(0, 2, 1).contiguous()




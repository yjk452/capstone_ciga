from typing import Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .renderer import Renderer, RendererOutputTypes, RendererOutputInfo
from ..cameras import Camera
from ..models.gaussian import GaussianModel

from diff_trim_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from internal.models.sh_core import (
    GateMLP,
    build_gate_input,
    compute_distance,
    band_flatten,
)


class SepDepthTrim2DGSRenderer(Renderer):
    """
    v2-style renderer (diff_trim_surfel_rasterization) + (optional) Ciga SH gating via GateMLP.
    입력(in_dim=7):
      x = [u(d), cam_center(3), cam_forward(3)]
    """

    def __init__(
        self,
        depth_ratio: float = 0.0,
        K: int = 5,
        v_pow: float = 0.1,
        prune_ratio: float = 0.1,
        contribution_prune_from_iter: int = 1000,
        contribution_prune_interval: int = 500,
        start_prune_ratio: float = 0.0,
        diable_start_trimming: bool = False,
        diable_trimming: bool = False,
        mlp_cfg: dict = None,
        sh_max_degree: int = None,
        **kwargs,
    ):
        super().__init__()

        # trimming hyper-parameters
        self.depth_ratio = depth_ratio
        self.K = K
        self.v_pow = v_pow
        self.prune_ratio = prune_ratio
        self.contribution_prune_from_iter = contribution_prune_from_iter
        self.contribution_prune_interval = contribution_prune_interval
        self.start_prune_ratio = start_prune_ratio
        self.diable_start_trimming = diable_start_trimming
        self.diable_trimming = diable_trimming

        # gating config
        self.mlp_cfg = mlp_cfg or {}
        self.use_gate_mlp = self.mlp_cfg.get("use_gate_mlp", False)

        self.L_max = int(sh_max_degree if sh_max_degree is not None else self.mlp_cfg.get("sh_max_degree", 3))
        in_dim = int(self.mlp_cfg.get("in_dim", 7))   # <- in_dim=7
        hidden = int(self.mlp_cfg.get("hidden", 64))

        if self.use_gate_mlp:
            self.mlp_model = GateMLP(in_dim=in_dim, L_max=self.L_max, hidden=hidden)
        else:
            self.mlp_model = None

        # cache for visualization/debug
        self._last_band_w = None

    def training_setup(self, pl_module):
        if self.mlp_model is not None:
            self.mlp_model.to(pl_module.device)
        return None, None

    def get_adaptive_parameters(self):
        return [] if self.mlp_model is None else self.mlp_model.parameters()

    @staticmethod
    def _camera_forward_world(VC, device, dtype) -> torch.Tensor:
        """
        world_to_camera: R_wc (world->camera)
        R_cw = R_wc^T
        camera forward를 camera -Z로 가정:
          f_world = R_cw @ [0,0,-1]
        만약 네 컨벤션이 +Z forward면, 아래 f_cam만 [0,0,1]로 바꾸면 됨.
        """
        W2C = VC.world_to_camera.to(device=device, dtype=dtype)
        R_wc = W2C[:3, :3]
        R_cw = R_wc.transpose(0, 1)
        f_cam = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=dtype)
        f_world = R_cw @ f_cam
        return F.normalize(f_world, dim=0)

    def forward(
        self,
        viewpoint_camera: Camera,
        pc: GaussianModel,
        bg_color: torch.Tensor,
        scaling_modifier: float = 1.0,
        record_transmittance: bool = False,
        **kwargs,
    ):
        """
        Render the scene.
        Background tensor (bg_color) must be on GPU!
        """

        screenspace_points = torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=bg_color.device
        ) + 0
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass

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
            record_transmittance=record_transmittance,
            debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        cov3D_precomp = None
        scales = pc.get_scaling[..., :2]  # v2 surfel-style
        rotations = pc.get_rotation

        # SH features
        shs = pc.get_features

        shs_before = shs
        shs_after = self.shs_weight_MLP(
            rasterizer=rasterizer,
            shs=shs,
            VC=viewpoint_camera,
            means3D=means3D,
            L=pc.active_sh_degree,
        )

        output = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs_after,
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        if record_transmittance:
            transmittance_sum, num_covered_pixels, radii = output
            transmittance = transmittance_sum / (num_covered_pixels + 1e-6)
            return transmittance

        rendered_image, radii, allmap = output

        rets = {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

        render_alpha = allmap[1:2]

        render_normal = allmap[2:5]
        render_normal = (
            (render_normal.permute(1, 2, 0) @ (viewpoint_camera.world_to_camera[:3, :3].T)).permute(2, 0, 1)
        )

        render_depth_median = allmap[5:6]
        render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

        render_depth_expected = allmap[0:1]
        render_depth_expected = (render_depth_expected / render_alpha)
        render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

        render_dist = allmap[6:7]

        surf_depth = render_depth_expected * (1 - self.depth_ratio) + (self.depth_ratio) * render_depth_median

        surf_normal = self.depth_to_normal(viewpoint_camera, surf_depth)
        surf_normal = surf_normal.permute(2, 0, 1)
        surf_normal = surf_normal * (render_alpha).detach()

        rets.update(
            {
                "rend_alpha": render_alpha,
                "rend_normal": render_normal,
                "view_normal": -allmap[2:5],
                "rend_dist": render_dist,
                "surf_depth": surf_depth,
                "surf_normal": surf_normal,
            }
        )

        # for logging/loss
        cam_center = viewpoint_camera.camera_center.to(means3D.device, means3D.dtype).view(-1)[:3]
        distances = torch.linalg.norm(means3D - cam_center, dim=1)
        cam_forward = self._camera_forward_world(viewpoint_camera, means3D.device, means3D.dtype)

        sh_weights = None
        if self._last_band_w is not None:
            band_w_full, vis_idx, L_full = self._last_band_w
            N = means3D.shape[0]
            sh_weights = torch.ones(N, L_full + 1, device=means3D.device, dtype=means3D.dtype)
            if vis_idx.numel() > 0:
                sh_weights[vis_idx] = band_w_full

        rets.update(
            {
                "shs_raw": shs_before,
                "shs_gated": shs_after,
                "sh_weights": sh_weights,
                "adaptive_sh_info": {
                    "distances": distances,
                    "cam_center": cam_center,
                    "cam_forward": cam_forward,
                },
            }
        )

        return rets

    def before_training_step(self, step: int, module):
        if step != 1 or self.diable_trimming or self.diable_start_trimming:
            return
        cameras = module.trainer.datamodule.dataparser_outputs.train_set.cameras
        device = module.gaussian_model.get_xyz.device
        top_list = [None] * self.K
        with torch.no_grad():
            print("Trimming...")
            for i in range(len(cameras)):
                camera = cameras[i].to_device(device)
                trans = self(
                    camera,
                    module.gaussian_model,
                    bg_color=module._fixed_background_color().to(device),
                    record_transmittance=True,
                )
                if top_list[0] is not None:
                    m = trans > top_list[0]
                    if m.any():
                        for j in range(self.K - 1):
                            top_list[self.K - 1 - j][m] = top_list[self.K - 2 - j][m]
                        top_list[0][m] = trans[m]
                else:
                    top_list = [trans.clone() for _ in range(self.K)]

            contribution = torch.stack(top_list, dim=-1).mean(-1)
            tile = torch.quantile(contribution, self.start_prune_ratio)
            prune_mask = contribution <= tile
            module.density_controller._prune_points(prune_mask, module.gaussian_model, module.gaussian_optimizers)
            print("Trimming done.")
        torch.cuda.empty_cache()

    def after_training_step(self, step: int, module):
        cameras = module.trainer.datamodule.dataparser_outputs.train_set.cameras
        if self.diable_trimming or (step > module.density_controller.config.densify_until_iter) \
           or (step < self.contribution_prune_from_iter) \
           or (step % self.contribution_prune_interval != 0):
            return

        device = module.gaussian_model.get_xyz.device

        top_list = [None] * self.K
        with torch.no_grad():
            print("Trimming...")
            for i in range(len(cameras)):
                camera = cameras[i].to_device(device)
                trans = self(
                    camera,
                    module.gaussian_model,
                    bg_color=module._fixed_background_color().to(device),
                    record_transmittance=True,
                )
                if top_list[0] is not None:
                    m = trans > top_list[0]
                    if m.any():
                        for j in range(self.K - 1):
                            top_list[self.K - 1 - j][m] = top_list[self.K - 2 - j][m]
                        top_list[0][m] = trans[m]
                else:
                    top_list = [trans.clone() for _ in range(self.K)]

            contribution = torch.stack(top_list, dim=-1).mean(-1)
            tile = torch.quantile(contribution, self.prune_ratio)
            prune_mask = (contribution <= tile)
            module.density_controller._prune_points(prune_mask, module.gaussian_model, module.gaussian_optimizers)
            print("Trimming done.")
        torch.cuda.empty_cache()

    @staticmethod
    def depths_to_points(view, depthmap):
        device = view.world_to_camera.device
        dtype = view.world_to_camera.dtype

        c2w = (view.world_to_camera.T).inverse()
        W, H = int(view.width), int(view.height)

        ndc2pix = torch.tensor(
            [
                [W / 2, 0, 0, W / 2],
                [0, H / 2, 0, H / 2],
                [0, 0, 0, 1],
            ],
            device=device,
            dtype=dtype,
        ).T

        projection_matrix = c2w.T @ view.full_projection
        intrins = (projection_matrix @ ndc2pix)[:3, :3].T

        grid_x, grid_y = torch.meshgrid(
            torch.arange(W, device=device, dtype=dtype),
            torch.arange(H, device=device, dtype=dtype),
            indexing="xy",
        )
        points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
        rays_d = points @ intrins.inverse().T @ c2w[:3, :3].T
        rays_o = c2w[:3, 3]
        points = depthmap.reshape(-1, 1) * rays_d + rays_o
        return points

    @classmethod
    def depth_to_normal(cls, view, depth):
        points = cls.depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
        output = torch.zeros_like(points)
        dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
        dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
        normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
        output[1:-1, 1:-1, :] = normal_map
        return output

    def get_available_outputs(self) -> Dict:
        return {
            "rgb": RendererOutputInfo("render"),
            "render_alpha": RendererOutputInfo("rend_alpha", type=RendererOutputTypes.GRAY),
            "render_normal": RendererOutputInfo("rend_normal", type=RendererOutputTypes.NORMAL_MAP),
            "view_normal": RendererOutputInfo("view_normal", type=RendererOutputTypes.NORMAL_MAP),
            "render_dist": RendererOutputInfo("rend_dist", type=RendererOutputTypes.GRAY),
            "surf_depth": RendererOutputInfo("surf_depth", type=RendererOutputTypes.GRAY),
            "surf_normal": RendererOutputInfo("surf_normal", type=RendererOutputTypes.NORMAL_MAP),
        }

    def shs_weight_MLP(self, rasterizer, shs, VC, means3D, L=None):
        """
        Apply per-band weights (l=0..L_full) to SH coefficients.
        in_dim=7: [u(d), cam_center(3), cam_forward(3)]
        """
        if shs is None:
            self._last_band_w = None
            return shs

        assert shs.ndim == 3
        device, dtype = shs.device, shs.dtype

        # normalize SH layout to [N,K,3]
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

        # visible-only (if supported)
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

        # camera center
        cam = getattr(VC, "camera_center", None)
        if cam is None:
            cam = getattr(VC, "cam_pos", None)
        if cam is None:
            raise RuntimeError("Camera has no camera_center/cam_pos")

        cam = cam.to(device=device, dtype=dtype).view(-1)[:3]  # [3]
        cam_forward = self._camera_forward_world(VC, device=device, dtype=dtype)  # [3]

        # distance
        d = compute_distance(cam, means3D_vis)  # [M]

        # cfg
        d_low = float(self.mlp_cfg.get("d_low", 0.2))
        d_high = float(self.mlp_cfg.get("d_high", 30.0))
        d0 = float(self.mlp_cfg.get("d0", 5.0))
        dw = float(self.mlp_cfg.get("dw", 0.5))
        cam_center_scale = float(self.mlp_cfg.get("cam_center_scale", 50.0))

        # build x: [M,7]
        x = build_gate_input(
            d,
            cam_center=cam,
            cam_forward=cam_forward,
            d_low=d_low,
            d_high=d_high,
            d0=d0,
            dw=dw,
            cam_center_scale=cam_center_scale,
        )

        # compute band weights
        if self.mlp_model is not None:
            band_w = self.mlp_model(x)  # [M, L_max+1] (sigmoid already)

            # DC must be 1
            if band_w.shape[1] >= 1:
                ones_dc = torch.ones_like(band_w[:, :1])
                band_w = torch.cat([ones_dc, band_w[:, 1:]], dim=1)

            # fit to L_full+1 (pad with ones if needed)
            if band_w.shape[1] < (L_full + 1):
                pad_cols = (L_full + 1) - band_w.shape[1]
                pad = torch.ones(band_w.size(0), pad_cols, device=device, dtype=dtype)
                band_w_full = torch.cat([band_w, pad], dim=1)
            else:
                band_w_full = band_w[:, : L_full + 1]
        else:
            M = means3D_vis.shape[0]
            band_w_full = torch.ones(M, L_full + 1, device=device, dtype=dtype)

        # sanitize
        if not torch.isfinite(band_w_full).all():
            band_w_full = torch.where(torch.isfinite(band_w_full), band_w_full, torch.ones_like(band_w_full))

        # band -> coeff weights
        coeff_w = band_flatten(band_w_full, L_full).unsqueeze(-1)  # [M,K,1]
        w_safe = torch.where(torch.isfinite(coeff_w), coeff_w, torch.ones_like(coeff_w))

        # apply (clone once)
        shs_out_nkc = shs_nkc.clone()
        shs_out_nkc[vis_idx] = shs_out_nkc[vis_idx] * w_safe.to(dtype=dtype, device=device)

        self._last_band_w = (band_w_full, vis_idx, L_full)
        return shs_out_nkc if orig_is_NKC else shs_out_nkc.permute(0, 2, 1).contiguous()

from typing import Dict, Tuple, Union, Callable, Optional, List

import lightning
import torch
import math
from .renderer import Renderer
from .renderer import RendererOutputTypes, RendererOutputInfo, Renderer
from ..cameras import Camera
from ..models.gaussian import GaussianModel

from diff_trim_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer

import torch.nn as nn
from log import print_to
import weakref

class SepDepthTrim2DGSRendererCiga(Renderer):
    def __init__(
            self,
            depth_ratio: float = 0.,
            K: int = 5,
            v_pow: float = 0.1,
            prune_ratio: float = 0.1,
            contribution_prune_from_iter : int = 1000,
            contribution_prune_interval: int = 500,
            start_prune_ratio: float = 0.0,
            diable_start_trimming: bool = False,
            diable_trimming: bool = False,
    ):
        super().__init__()
        self._gs_ref = None

        # hyper-parameters for trimming
        self.depth_ratio = depth_ratio

        self.K = K
        self.v_pow = v_pow
        self.prune_ratio = prune_ratio
        self.contribution_prune_from_iter = contribution_prune_from_iter
        self.contribution_prune_interval = contribution_prune_interval
        self.start_prune_ratio = start_prune_ratio
        self.diable_start_trimming = diable_start_trimming
        self.diable_trimming = diable_trimming

    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            record_transmittance=False,
            **kwargs,
    ):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True,
                                              device=bg_color.device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

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
            record_transmittance=record_transmittance,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        cov3D_precomp = None
        scales = pc.get_scaling[..., :2]
        rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = pc.get_features

        # partition 오류 수정을 위함
        gs = getattr(self, "gs", None)
        mlp_model = getattr(gs, "mlp_model", None) if gs is not None else None

        if mlp_model is not None:
            shs = self.shs_weight_MLP(
                rasterizer,
                shs,
                viewpoint_camera,
                means3D,
                L=pc.max_sh_degree
            )

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        output = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
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
        else:
            rendered_image, radii, allmap = output

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        rets = {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

        # additional regularizations
        render_alpha = allmap[1:2]

        # get normal map
        # transform normal from view space to world space
        render_normal = allmap[2:5]
        render_normal = (render_normal.permute(1, 2, 0) @ (viewpoint_camera.world_to_camera[:3, :3].T)).permute(2, 0, 1)

        # get median depth map
        render_depth_median = allmap[5:6]
        render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

        # get expected depth map
        render_depth_expected = allmap[0:1]
        render_depth_expected = (render_depth_expected / render_alpha)
        render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

        # get depth distortion map
        render_dist = allmap[6:7]

        # psedo surface attributes
        # surf depth is either median or expected by setting depth_ratio to 1 or 0
        # for bounded scene, use median depth, i.e., depth_ratio = 1;
        # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
        surf_depth = render_depth_expected * (1 - self.depth_ratio) + (self.depth_ratio) * render_depth_median

        # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
        surf_normal = self.depth_to_normal(viewpoint_camera, surf_depth)
        surf_normal = surf_normal.permute(2, 0, 1)
        # remember to multiply with accum_alpha since render_normal is unnormalized.
        surf_normal = surf_normal * (render_alpha).detach()

        rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'view_normal': -allmap[2:5],
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
        })

        return rets
    
    def before_training_step(
            self,
            step: int,
            module,
    ):
        if step != 1 or self.diable_trimming or self.diable_start_trimming:
            return
        cameras = module.trainer.datamodule.dataparser_outputs.train_set.cameras
        device =  module.gaussian_model.get_xyz.device
        top_list = [None, ] * self.K
        with torch.no_grad():
            print("Trimming...")
            for i in range(len(cameras)):
                camera = cameras[i].to_device(device)
                trans = self(
                    camera,
                    module.gaussian_model,
                    bg_color=module._fixed_background_color().to(device),
                    record_transmittance=True
                )
                if top_list[0] is not None:
                    m = trans > top_list[0]
                    if m.any():
                        for i in range(self.K - 1):
                            top_list[self.K - 1 - i][m] = top_list[self.K - 2 - i][m]
                            top_list[0][m] = trans[m]
                else:
                    top_list = [trans.clone() for _ in range(self.K)]

            contribution = torch.stack(top_list, dim=-1).mean(-1)
            tile = torch.quantile(contribution, self.start_prune_ratio)
            prune_mask = contribution <= tile
            module.density_controller._prune_points(prune_mask, module.gaussian_model, module.gaussian_optimizers)
            print("Trimming done.")
        torch.cuda.empty_cache()

    def after_training_step(
            self,
            step: int,
            module,
    ):
        cameras = module.trainer.datamodule.dataparser_outputs.train_set.cameras
        if self.diable_trimming or (step > module.density_controller.config.densify_until_iter) \
           or (step < self.contribution_prune_from_iter) \
           or (step % self.contribution_prune_interval != 0):
           return
        
        device =  module.gaussian_model.get_xyz.device

        top_list = [None, ] * self.K
        with torch.no_grad():
            print("Trimming...")
            for i in range(len(cameras)):
                camera = cameras[i].to_device(device)
                trans = self(
                    camera,
                    module.gaussian_model,
                    bg_color=module._fixed_background_color().to(device),
                    record_transmittance=True
                )
                if top_list[0] is not None:
                    m = trans > top_list[0]
                    if m.any():
                        for i in range(self.K - 1):
                            top_list[self.K - 1 - i][m] = top_list[self.K - 2 - i][m]
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
        c2w = (view.world_to_camera.T).inverse()
        W, H = view.width, view.height
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, W / 2],
            [0, H / 2, 0, H / 2],
            [0, 0, 0, 1]]).float().cuda().T
        projection_matrix = c2w.T @ view.full_projection
        intrins = (projection_matrix @ ndc2pix)[:3, :3].T

        grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
        points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
        rays_d = points @ intrins.inverse().T @ c2w[:3, :3].T
        rays_o = c2w[:3, 3]
        points = depthmap.reshape(-1, 1) * rays_d + rays_o
        return points

    @classmethod
    def depth_to_normal(cls, view, depth):
        """
            view: view camera
            depth: depthmap
        """
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
            'render_alpha': RendererOutputInfo("rend_alpha", type=RendererOutputTypes.GRAY),
            'render_normal': RendererOutputInfo("rend_normal", type=RendererOutputTypes.NORMAL_MAP),
            'view_normal': RendererOutputInfo("view_normal", type=RendererOutputTypes.NORMAL_MAP),
            'render_dist': RendererOutputInfo("rend_dist", type=RendererOutputTypes.GRAY),
            'surf_depth': RendererOutputInfo("surf_depth", type=RendererOutputTypes.GRAY),
            'surf_normal': RendererOutputInfo("surf_normal", type=RendererOutputTypes.NORMAL_MAP),
        }

    def shs_weight_MLP(self, rasterizer, shs, VC, means3D, L):
        """
        가시 가우시안들에 대해 MLP로 sh 가중치 예측. 이후 sh 계수에 가중치를 곱해 반환
        """
        if shs is None:
            return shs

        N, K, C = shs.shape
        device, dtype = shs.device, shs.dtype

        assert C == 3 and K == (L + 1) ** 2

        
        # sh 가중치(MLP 결과)를 담을 더미 텐서
        sh_weight = torch.zeros(N, L+1, C, device=device, dtype=dtype)

        x = self.gs.mlp_model.to_input(VC, means3D)
        print_to("R_input.txt", x[0:5])
        # if self.gs.logs['log']:
        #     log_weight_stats(self.gs.mlp_model, "R_mlp_weight.txt", self.gs.log_dir, step=self.gs.trainer.global_step)

        #x = torch.cat([d['cam_pos'], d['dis'], d['dir']], dim=1)
        if not self.gs.mlp_model.config.freeze:
            sh_weight = self.gs.mlp_model(x)
        else:
            with torch.no_grad():
                sh_weight = self.gs.mlp_model(x)
        
        print_to("R_output.txt", sh_weight[0:5])
        
        # 밴드별 가중치를 계수별 가중치로 변환
        sh_weight = self.band_flatten(sh_weight, L)
        # sh 계수와 가중치 곱 (MLP 출력 형태에 따라 연산 수정 필요)
        # in        : sh_weight shape   = [가시 가우시안 수, L+1] : L=sh_max_degree
        #         ▼ band_flatten
        # return    : sh_weight shape   = [gaussian 수, (L+1)^2, 3(RGB)]
        
        shs_out = shs.clone()
        shs_out = shs * sh_weight
        
        return shs_out

    def band_flatten(self, sh_weight, L):
        """
        sh_weight: [N, L+1]
        반환:      [N, (L+1)^2, 3]
        """
        import torch

        N, B = sh_weight.shape
        assert B == L + 1, f"got {sh_weight.shape}, expected [N,{L+1},3]"

        # 각 밴드의 계수 개수: 2l+1  (0차:1, 1차:3, ..., L차:2L+1)
        band_sizes = torch.tensor([2 * l + 1 for l in range(L + 1)],
                                device=sh_weight.device)

        # 길이 K=((L+1)^2) 벡터: 각 계수 k가 어느 밴드에 속하는지 (0..L)
        band_of_coeff = torch.repeat_interleave(
            torch.arange(L + 1, device=sh_weight.device),
            band_sizes
        )  # [K]

        # 밴드 축(dim=1) 기준으로 인덱싱 → [N, K, 3]
        weights_coeff = sh_weight.index_select(dim=1, index=band_of_coeff)
        
        weights_coeff = weights_coeff.unsqueeze(-1).expand(-1, -1, 3)

        return weights_coeff

    @property
    def gs(self):
        return self._gs_ref()
    
    def setup(self, gs):
        self._gs_ref = weakref.ref(gs)

    def __getstate__(self):
        state = self.__dict__.copy()
        if '_gs_ref' in state:
            del state['_gs_ref']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._gs_ref = None
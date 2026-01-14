# 입력 
# 정규화된 절대좌표([0,1]^3)에 대한 PE
# 카메라->가우시안 방향 벡터(dir)에 대한 PE
# 카메라<-> 가우시안 거리에 대한 로그값(log r)

# Layer
# 128 *3~4 
# residual
# 활성함수 : SiLU

# 출력
# 네트워크 출력은 가중치(w)에 대한 변화량(delta)
# 최종 가중치 구성 : w = exp(s*tanh(delta))
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Dict, Optional, Tuple, Union, List
from dataclasses import dataclass, field
from internal.models.mlp import MLP, MLPModel, OptimizerType, SchedulerType

from internal.optimizers import OptimizerConfig
from internal.schedulers import Scheduler

@dataclass
class MLPOptimizationConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0
    milestones: List[int] = field(default_factory=list)
    gamma: float = 0.5

    optimizer: OptimizerConfig = field(default_factory=lambda: {"class_path": "Adam"})
    lr_scheduler: Optional[Scheduler] = field(default_factory=lambda: {
        "class_path": "ExponentialDecayScheduler",
        "init_args": {
            "lr_final": 1e-5,
            "max_steps": 30_000,
        },
    })

@dataclass
class CigaMLPV2(MLP):
    hidden: int = 128
    sh_degree: int = 2
    s: float = 0.5
    pos_L: int = 8
    dir_L: int = 4
    include_input: bool = False
    eps: float = 1e-6
    train: bool = True
    optimization: MLPOptimizationConfig = field(default_factory=lambda: MLPOptimizationConfig())
    num_blocks: int = 3
    def instantiate(self, *args, **kwargs) -> "CigaMLPV2Model":
        model = CigaMLPV2Model(config=self, sh_degree=self.sh_degree)

        if not self.train:
            for p in model.parameters():
                p.requires_grad_(False)

        return model

class CigaMLPV2Model(MLPModel):
    def __init__(self, config: CigaMLPV2, sh_degree: int) -> None:
        super().__init__()
        self.config = config
        self.in_feature = 3 + (3 * config.pos_L * 2) + 3 + (3 * config.dir_L * 2) + 1 if config.include_input else (config.pos_L * 6) + (config.dir_L * 6) + 1
        self.sh_degree = sh_degree
        self.eps = float(config.eps)
        self.s = float(config.s)
        self.include_input = config.include_input

        self.register_buffer("pos_min", torch.zeros(3), persistent=True)
        self.register_buffer("pos_max", torch.ones(3), persistent=True)
        self.bbox_ready: bool = False

        self.fc_in = nn.Linear(self.in_feature, config.hidden)
        self.blocks = nn.ModuleList([ResidualMLPBlock(config.hidden) for _ in range(config.num_blocks)])
        self.fc_out = nn.Linear(config.hidden, self.sh_degree+1)
        self.mlp = nn.Sequential(
            nn.Linear(self.in_feature, config.hidden, bias=True),
            nn.ReLU(),
            nn.Linear(config.hidden, config.hidden, bias=True),
            nn.ReLU(),
            nn.Linear(config.hidden, config.hidden, bias=True),
            nn.ReLU(),
            nn.Linear(config.hidden, config.sh_degree+1, bias=True)
        )

    # def forward(self, viewpoint_camera, means3D) -> torch.Tensor:
    #     with torch.no_grad():
    #         x = self.to_input(viewpoint_camera, means3D)
    #     h = F.silu(self.fc_in(x))
    #     for blk in self.blocks:
    #         h = blk(h)
    #     delta = self.fc_out(h)
    #     w_band = torch.exp(self.s * torch.tanh(delta))
    #     return w_band
    def forward(self, x) -> torch.Tensor:
        delta = self.mlp(x)
        w_band = torch.exp(self.s * torch.tanh(delta))
        return w_band

    def to_input(self, camera, gaussian_pos: torch.Tensor, **kwargs):
        N = (gaussian_pos.shape)[0]

        cam_pos = camera.camera_center.to(device=gaussian_pos.device, dtype=gaussian_pos.dtype)
        cam_pos = cam_pos.unsqueeze(0).expand(N, -1)

        rel = gaussian_pos - cam_pos                          # [N,3]
        dis = torch.linalg.norm(rel, dim=-1, keepdim=True)    # [N,1]
        dis = dis.clamp_min(self.eps)
        dir = rel / dis                                       # [N,3]
        # 또는: dir = F.normalize(rel, dim=1)

        log_r = torch.log(dis)     

        # 가우시안 절대 위치 정규화
        denom = (self.pos_max - self.pos_min).clamp_min(self.eps)
        pos = (gaussian_pos - self.pos_min) / denom
        pos = pos.clamp(-0.01, 1.01)

        pos_PE = self.positional_encoding(pos, self.config.pos_L, self.include_input)
        dir_PE = self.positional_encoding(dir, self.config.dir_L, self.include_input)

        return torch.cat([pos_PE, dir_PE, log_r], dim=1)
    
    def training_setup(self, pl_module)-> Tuple[Optional[OptimizerType], Optional[SchedulerType]]:
        if not any(p.requires_grad for p in self.parameters()):
            return None, None

        opt_cfg = self.config.optimization
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=float(opt_cfg.lr),
            weight_decay=float(opt_cfg.weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(opt_cfg.milestones),
            gamma=float(opt_cfg.gamma),
        )
        return optimizer, scheduler
    
    def positional_encoding(self, x: torch.Tensor, L: int, include_input: bool = False):
        """
        x: [N, D]
        return: [N, D*(2L) (+D if include_input)]
        """
        if L <= 0:
            return x if include_input else x.new_zeros(x.shape[0], 0)

        # freq cache (device/dtype별로)
        key = (L, x.device, x.dtype)
        if not hasattr(self, "_freq_cache"):
            self._freq_cache = {}
        if key not in self._freq_cache:
            freqs = (2.0 ** torch.arange(L, device=x.device, dtype=x.dtype)) * math.pi  # [L]
            self._freq_cache[key] = freqs
        freqs = self._freq_cache[key]  # [L]

        # [N,D,1] * [L] -> [N,D,L]
        xb = x.unsqueeze(-1) * freqs  # already includes pi
        sin = torch.sin(xb)
        cos = torch.cos(xb)

        pe = torch.cat([sin, cos], dim=-1).reshape(x.shape[0], -1)  # [N, D*2L]
        return torch.cat([x, pe], dim=-1) if include_input else pe

    @torch.no_grad()
    def set_bbox(self, gaussian_pos):
        self.pos_min.copy_(gaussian_pos.quantile(0.001, dim=0))
        self.pos_max.copy_(gaussian_pos.quantile(0.999, dim=0))
        self.bbox_ready = True

    def apply_band_weights(self, shs: torch.Tensor, w_band: torch.Tensor):
        L = self.sh_degree
        N, K, C = shs.shape
        start = 0
        parts = []
        for l in range(L + 1):
            band_size = 2*l + 1
            end = start + band_size
            w = w_band[:, l].view(N, 1, 1)
            parts.append(shs[:, start:end, :] * w)  # out-of-place
            start = end
        return torch.cat(parts, dim=1)

class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.fc1(x))
        h = self.fc2(h)
        return F.silu(x + h)

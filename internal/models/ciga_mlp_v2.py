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
from log import print_to

@dataclass
class MLPOptimizationConfig:
    lr: float = 5e-3 # 1e-3
    weight_decay: float = 0.0
    milestones: List[int] = field(default_factory=list)
    gamma: float = 0.5

@dataclass
class CigaMLPV2(MLP):
    ckpt_path: Optional[str] = None
    step: int = 0
    hidden: int = 128
    sh_degree: int = 2
    s: float = 0.5
    pos_L: int = 8
    dir_L: int = 4
    include_input: bool = False
    eps: float = 1e-6
    freeze: bool = False
    optimization: MLPOptimizationConfig = field(default_factory=lambda: MLPOptimizationConfig())
    num_blocks: int = 3
    def instantiate(self, *args, **kwargs) -> "CigaMLPV2Model":
        model = CigaMLPV2Model(config=self, sh_degree=self.sh_degree)

        if self.ckpt_path:
            ckpt = torch.load(self.ckpt_path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)

            cleaned = {}
            for k, v in state.items():
                if k.startswith("mlp_model."):
                    cleaned[k[len("mlp_model."):]] = v
                elif k.startswith("model."):
                    cleaned[k[len("model."):]] = v
                else:
                    cleaned[k] = v

            missing, unexpected = model.load_state_dict(cleaned, strict=False)
            print(f"[MLP] loaded from {self.ckpt_path} (missing={len(missing)}, unexpected={len(unexpected)})")

        if self.freeze:
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
        self.step = config.step
        
        self.register_buffer("pos_min", torch.zeros(3), persistent=True)
        self.register_buffer("pos_max", torch.ones(3), persistent=True)
        #self.bbox_ready: bool = False
        self.register_buffer("bbox_ready", torch.tensor(False, dtype=torch.bool))

        # self.fc_in = nn.Linear(self.in_feature, config.hidden)
        # self.blocks = nn.ModuleList([ResidualMLPBlock(config.hidden) for _ in range(config.num_blocks)])
        # self.fc_out = nn.Linear(config.hidden, self.sh_degree+1)
        # self.mlp = nn.Sequential(
        #     nn.Linear(self.in_feature, config.hidden, bias=True),
        #     nn.ReLU(),
        #     nn.Linear(config.hidden, config.hidden, bias=True),
        #     nn.ReLU(),
        #     nn.Linear(config.hidden, config.hidden, bias=True),
        #     nn.ReLU(),
        #     nn.Linear(config.hidden, config.sh_degree+1, bias=True)
        # )

        self.is_active = False

        self.input_layer = nn.Sequential(
            nn.Linear(self.in_feature, config.hidden),
            nn.LayerNorm(config.hidden),
            nn.SiLU()
        )
        
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden, config.hidden),
                nn.LayerNorm(config.hidden),
                nn.SiLU(),
                nn.Linear(config.hidden, config.hidden),
                nn.LayerNorm(config.hidden)
            ) for _ in range(2)
        ])
        
        self.output_layer = nn.Linear(config.hidden, config.sh_degree+1)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    # def forward(self, x) -> torch.Tensor:
    #     delta = self.mlp(x)
    #     w_band = torch.exp(self.s * torch.tanh(delta))
    #     return w_band

    def forward(self, x) -> torch.Tensor:
        if self.is_active:
            h = self.input_layer(x)
            for block in self.res_blocks:
                h = h + block(h)
            delta = self.output_layer(h)
        else:
            with torch.no_grad():
                h = self.input_layer(x)
                for block in self.res_blocks:
                    h = h + block(h)
                delta = self.output_layer(h)
        return torch.exp(delta)

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
        print_to("MLP_bbox.txt", self.pos_min, self.pos_max)
        denom = (self.pos_max - self.pos_min).clamp_min(self.eps)
        pos = (gaussian_pos - self.pos_min) / denom
        pos = pos.clamp(-0.01, 1.01)
        print_to("MLP_before_PE.txt", f"pos: {pos}\ndir: {dir}\ndis (r): {dis}\n\n")
        pos_PE = self.positional_encoding(pos, self.config.pos_L, self.include_input)
        dir_PE = self.positional_encoding(dir, self.config.dir_L, self.include_input)

        return torch.cat([pos_PE, dir_PE, log_r], dim=1)
    
    def training_setup(self, pl_module)-> Tuple[Optional[OptimizerType], Optional[SchedulerType]]:
        if not any(p.requires_grad for p in self.parameters()):
            return None, None

        opt_cfg = self.config.optimization

        mlp_optimizer = torch.optim.Adam(
            self.parameters(),
            lr=float(opt_cfg.lr),
            weight_decay=float(opt_cfg.weight_decay),
        )

        mlp_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            mlp_optimizer,
            milestones=list(opt_cfg.milestones),
            gamma=float(opt_cfg.gamma),
        )

        scheduler_cfg = {
            "scheduler": mlp_scheduler,
            "interval": "step",   
            "frequency": 1,
        }

        return mlp_optimizer, scheduler_cfg
    
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
        self.bbox_ready.fill_(True)

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
    def set_active(self, active: bool):
        """
        active=True: 가중치 업데이트 활성화
        active=False: 가중치 업데이트 비활성화 (Frozen)
        """
        self.is_active = active
        for param in self.parameters():
            param.requires_grad = active
        
        status = "ENABLED" if active else "DISABLED"
        print(f"[CigaMLPV2] Training has been {status}.")

class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.fc1(x))
        h = self.fc2(h)
        return F.silu(x + h)

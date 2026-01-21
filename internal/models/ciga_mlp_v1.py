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
    lr: float = 5e-3
    weight_decay: float = 0.0
    milestones: List[int] = field(default_factory=list)
    gamma: float = 0.5

@dataclass
class CigaMLPV1(MLP):
    in_features: int = 7
    hidden: int = 64
    out_features: int = 4  
    sh_degree: int = 2
    optimization: MLPOptimizationConfig = field(default_factory=lambda: MLPOptimizationConfig())
    ckpt_path: Optional[str] = None
    freeze: bool = False

    def instantiate(self, *args, **kwargs) -> "CigaMLPV1Model":
        model = CigaMLPV1Model(config=self, sh_degree=self.sh_degree)

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

class CigaMLPV1Model(MLPModel):
    def __init__(self, config: CigaMLPV1, sh_degree: int) -> None:
        super().__init__()
        self.config = config
        self.sh_degree = sh_degree
        self.mlp = nn.Sequential(
            nn.Linear(self.config.in_features, self.config.hidden, bias=True),
            nn.ReLU(),
            nn.Linear(self.config.hidden, self.config.hidden, bias=True),
            nn.ReLU(),
            nn.Linear(self.config.hidden, self.sh_degree+1, bias=True)
        )
    def forward(self, x) -> torch.Tensor:
        output = self.mlp(x)
        print_to("MLP_output.txt", output[0:5])
        return output

    def to_input(self, camera, gaussian_pos: torch.Tensor, **kwargs):
        N = (gaussian_pos.shape)[0]
        # 카메라 위치
        cam_pos = camera.camera_center
        cam_pos = cam_pos.unsqueeze(0).expand(N, -1)
        # 카메라-가우시안 거리
        dis = (gaussian_pos - cam_pos).norm(dim=-1, keepdim=True)

        # 카메라-가우시안 방향
        dir = F.normalize(gaussian_pos - cam_pos, dim=1)
        return torch.cat([gaussian_pos, dis, dir], dim=1)
    
    def training_setup(self, pl_module)-> Tuple[Optional[OptimizerType], Optional[SchedulerType]]:
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
            "interval": "step",   # 중요: per-step으로 맞추기
            "frequency": 1,
        }

        return mlp_optimizer, scheduler_cfg
    
    def set_bbox(self, *args):
        pass
    




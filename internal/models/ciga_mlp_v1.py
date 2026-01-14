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
    optimizer: OptimizerConfig = field(default_factory=lambda: {"class_path": "Adam"})
    lr_scheduler: Optional[Scheduler] = field(default_factory=lambda: {
        "class_path": "ExponentialDecayScheduler",
        "init_args": {
            "lr_final": 1e-5,
            "max_steps": 30_000,
        },
    })

@dataclass
class CigaMLPV1(MLP):
    in_features: int = 7
    hidden: int = 64
    out_features: int = 4  
    sh_degree: int = 2
    train: bool = True
    optimization: MLPOptimizationConfig = field(default_factory=lambda: MLPOptimizationConfig())

    def instantiate(self, *args, **kwargs) -> "CigaMLPV1Model":
        model = CigaMLPV1Model(config=self, sh_degree=self.sh_degree)

        if not self.train:
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

        # OptimizerConfig가 내부적으로 instantiate를 제공한다고 가정(gaussian 쪽 패턴과 동일)
        # 만약 OptimizerConfig 구현이 다르면 여기만 프로젝트에 맞게 조정하면 됨.
        optimizer = opt_cfg.optimizer.instantiate(
            params=self.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
        ) if hasattr(opt_cfg.optimizer, "instantiate") else torch.optim.Adam(
            self.parameters(), lr=opt_cfg.lr, weight_decay=opt_cfg.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=opt_cfg.milestones,
            gamma=opt_cfg.gamma,
        )

        return optimizer, scheduler
    def set_bbox(self, *args):
        pass
    




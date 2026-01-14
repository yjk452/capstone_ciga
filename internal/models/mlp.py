from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union, List
import torch
from torch import nn
from internal.configs.instantiate_config import InstantiatableConfig


OptimizerType = Union[torch.optim.Optimizer, List[torch.optim.Optimizer]]
SchedulerType = Union[torch.optim.lr_scheduler.LRScheduler, List[torch.optim.lr_scheduler.LRScheduler]]



class MLPModel(nn.Module, ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def to_input(self, camera, gaussian_pos: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            camera
            gaussian_pos: [N,3] (world space gaussian centers)
        Returns:
            mlp input
        """
        raise NotImplementedError

    @abstractmethod
    def training_setup(self, pl_module) -> Tuple[Optional[OptimizerType], Optional[SchedulerType]]:
        """
        GaussianSplatting.configure_optimizers에서 호출되어 optimizer/scheduler를 등록하기 위한 훅.
        (현재 GS에서는 mlp_model.training_setup(self)로 호출하는 흐름이 존재) :contentReference[oaicite:8]{index=8}
        """
        raise NotImplementedError

class MLP(InstantiatableConfig, ABC):
    @abstractmethod
    def instantiate(self, *args, **kwargs) -> MLPModel:
        raise NotImplementedError()

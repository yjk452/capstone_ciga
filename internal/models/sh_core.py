import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math


def band_flatten(sh_weight: torch.Tensor, L: int) -> torch.Tensor:
    
    N, B = sh_weight.shape
    assert B == L + 1, f"got {sh_weight.shape}, expected [N,{L+1}]"

    dev = sh_weight.device
    band_sizes = torch.tensor([2*l + 1 for l in range(L + 1)],
                              device=dev, dtype=torch.long)       # [L+1]
    band_of_coeff = torch.repeat_interleave(
        torch.arange(L + 1, device=dev, dtype=torch.long),
        band_sizes
    )                                                              # [K]
    coeff_weights = sh_weight.index_select(dim=1, index=band_of_coeff)  # [N,K]
    return coeff_weights


class GateMLP(nn.Module):
    def __init__(self, in_dim=4, L_max=3, hidden=32): #[u(d),v_x​,v_y​,v_z​,cx​,cy​,cz]  dim =7 (최소)
        super().__init__()
        self.L_max = L_max
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, L_max+1)
        )
        # 초기 편향: 고차는 낮게 시작
        with torch.no_grad():
            last = self.net[-1]
            last.bias[:] = -2.2
 
    def forward(self, x):
        g = torch.sigmoid(self.net(x))  # [N, L+1]
        return g


def d_to_u(d, d_low, d_high, d0, w: float = 0.5):
    """거리를 [0,1]로 정규화 (선형+로그 혼합)"""
    d_clipped = d.clamp(d_low, d_high)
    u_lin = (d_clipped - d_low) / (d_high - d_low + 1e-8)
    u_log = torch.log1p(d_clipped / d0) / (math.log1p(d_high / d0) + 1e-8)
    return ((1 - w) * u_lin + w * u_log).clamp(0., 1.)

def phi_to_v(phi):
    """phi: [0, π] → v∈[0,1]"""
    return (phi / math.pi).clamp(0., 1.)




#각도 사용안한다면 끄기
class GateLUT(nn.Module):  
    def __init__(self, L_max=3, B_d=32, B_phi=32, ema=0.9, device="cuda"):
        super().__init__()
        self.L_max = L_max
        self.ema = ema
        self.B_d = B_d
        self.B_phi = B_phi
        table = torch.full((1, L_max+1, B_phi, B_d), 0.1, device=device)
        self.register_buffer("table", table)
 
    @torch.no_grad()
    def update_from_mlp(self, mlp: GateMLP, grid_u: torch.Tensor, grid_v: torch.Tensor):
        """MLP로부터 LUT 업데이트"""
        B_phi, B_d = grid_u.shape
        inp = torch.stack([grid_u.reshape(-1), grid_v.reshape(-1)], dim=-1)
        g = mlp(inp)  # [B_phi*B_d, L+1]
        g = g.reshape(B_phi, B_d, -1).permute(2, 0, 1).unsqueeze(0)
        # EMA 업데이트
        self.table.mul_(self.ema).add_(g*(1.0 - self.ema))
 
    def lookup(self, u, v):
        grid = torch.stack([u*2-1, v*2-1], dim=-1).unsqueeze(0).unsqueeze(0)
        grid = grid.to(self.table.device, dtype=self.table.dtype)
        out = F.grid_sample(self.table, grid, mode="bilinear", align_corners=True)
        g = out.squeeze(0).squeeze(1).transpose(0,1)  # [N, L+1]
        return g






class AdaptiveSHLoss(nn.Module):
    def __init__(self, lambda_sh=0.01, lambda_gate=0.001, lambda_tv=0.005, 
                 lambda_mono=0.01, aerial_threshold=math.pi/6):
        super().__init__()
        self.lambda_sh = lambda_sh
        self.lambda_gate = lambda_gate
        self.lambda_tv = lambda_tv
        self.lambda_mono = lambda_mono
        self.aerial_threshold = aerial_threshold
    



    def forward(self, base_l_rgb: torch.Tensor,
                shs: torch.Tensor, sh_weights: torch.Tensor,  # 변수명 통일
                nadir_angles: torch.Tensor, distances: torch.Tensor,
                gaussian_pos: Optional[torch.Tensor] = None) -> dict:  # 변수명 통일
    
        losses = {}
        
        L_RGB = base_l_rgb
        # L_RGB = F.mse_loss(pred_colors, gt_colors)
        losses['L_RGB'] = L_RGB
        
        L_SH_ratio = self._compute_sh_ratio_loss(shs, sh_weights, nadir_angles)
        losses['L_SH_ratio'] = L_SH_ratio
       
        L_gate = self._compute_gate_loss(sh_weights)
        losses['L_gate'] = L_gate
       
        L_TV = self._compute_tv_loss(sh_weights, gaussian_pos) if gaussian_pos is not None else torch.tensor(0.0, device=sh_weights.device)
        losses['L_TV'] = L_TV
       
        L_mono = self._compute_monotonicity_loss(sh_weights, distances, nadir_angles)
        losses['L_mono'] = L_mono
        
        total_loss = (L_RGB + 
                     self.lambda_sh * L_SH_ratio + 
                     self.lambda_gate * L_gate + 
                     self.lambda_tv * L_TV + 
                     self.lambda_mono * L_mono)
        
        losses['total_loss'] = total_loss
        return losses
    
    def _compute_sh_ratio_loss(self, shs: torch.Tensor, sh_weights: torch.Tensor, 
                              nadir_angles: torch.Tensor) -> torch.Tensor:
    
        device = shs.device
        assert shs.dim()==3 and shs.shape[2]==3, f"expect [N,K,3], got {shs.shape}"
        N, K, C = shs.shape
        L = int(math.isqrt(K)) - 1
        aerial_mask = (nadir_angles < self.aerial_threshold)
        if aerial_mask.sum() == 0:
            return torch.zeros((), device=device)

        #shs = shs.permute(0, 2, 1) # [N, 3, K]  계산 편의상 -> gs splatting에서 

        loss =  shs.new_zeros(())  # 스칼라 초기화
        
        shs_aerial = shs[aerial_mask] # [N_aerial, C, K]

        # l=1..L 밴드별로 손실 계산
        for l in range(1, L + 1):
            s = l * l
            e = s + (2 * l + 1)
            
            # 각 밴드 계수 크기 계산
            # [N_aerial, C, 2l+1] -> [N_aerial]
            curr_mag = shs_aerial[:, :, s:e].pow(2).mean(dim=(1, 2))

            # curr_mag가 작을수록 손실 커짐
            loss = loss + (1.0 - curr_mag).pow(2).mean()
        return loss / L # 밴드 수로 정규화
    

    
    def _compute_gate_loss(self, sh_weights: torch.Tensor) -> torch.Tensor:
        """게이팅 함수 부드러움 손실"""
        w = sh_weights[:, 1:]  # DC 제외
        weight_diffs = torch.diff(w, dim=-1)
        smoothness_loss = torch.mean(weight_diffs ** 2)
        
        monotonic_penalty = torch.tensor(0.0, device=sh_weights.device)
        for i in range(w.shape[-1] - 1):
            violation = torch.relu(w[:, i+1] - w[:, i])
            monotonic_penalty += torch.mean(violation ** 2)
        
        return smoothness_loss + 0.5 * monotonic_penalty
    
    def _compute_tv_loss(self, sh_weights: torch.Tensor, 
                        gaussian_pos: torch.Tensor) -> torch.Tensor:
        if gaussian_pos is None:
            return torch.tensor(0.0, device=sh_weights.device)
        
        N = gaussian_pos.shape[0]
        if N > 1000:  # 메모리 절약
            indices = torch.randperm(N, device=gaussian_pos.device)[:1000]
            positions = gaussian_pos[indices]
            weights = sh_weights[indices]
        else:
            positions = gaussian_pos
            weights = sh_weights
        
        N_sample = positions.shape[0]
        if N_sample <= 1:
            return torch.tensor(0.0, device=sh_weights.device)
        
        pos_diffs = positions.unsqueeze(1) - positions.unsqueeze(0)
        spatial_dists = torch.norm(pos_diffs, dim=-1)
        
        weight_diffs = weights.unsqueeze(1) - weights.unsqueeze(0)
        weight_dists = torch.norm(weight_diffs, dim=-1)
        
        spatial_weights = torch.exp(-spatial_dists / 0.1)
        tv_loss = torch.mean(spatial_weights * weight_dists)
        
        return tv_loss
    
    def _compute_monotonicity_loss(self, sh_weights: torch.Tensor, 
                                  distances: torch.Tensor, 
                                  nadir_angles: torch.Tensor) -> torch.Tensor:
        if sh_weights.shape[0] <= 1:
            return torch.tensor(0.0, device=sh_weights.device)
        
        sorted_indices = torch.argsort(distances)
        sorted_weights = sh_weights[sorted_indices]
        sorted_angles = nadir_angles[sorted_indices]
        
        mono_loss = sh_weights.new_zeros(())
        angle_factor = (1.0 - torch.cos(sorted_angles).clamp(-1,1)) * 0.5  # [0..1]
        for l in range(1, min(4, sorted_weights.shape[1])):
            diffs = torch.diff(sorted_weights[:, l])        # w_{i+1}-w_i
            af = angle_factor[1:]                     # align lengths
            violation = F.relu(diffs)                 # >0 means increasing with dist
            mono_loss += (af * violation).pow(2).mean()
        return mono_loss

    





def apply_adaptive_sh_weights(shs: torch.Tensor, sh_weights: torch.Tensor, 
                              L: int) -> torch.Tensor:

    assert shs.dim() == 3 and shs.shape[2] == 3, f"expect [N,K,3], got {shs.shape}"
    N, K, C = shs.shape
    assert K == (L + 1) ** 2 and C == 3, f"{K=} {(L+1)**2=} {C=}"


    sh_weights = sh_weights.clone()
    sh_weights[:, 0] = 1.0

    # [N,L+1] → [N,K] → [N,K,1]
    coeff_w = band_flatten(sh_weights, L).unsqueeze(-1)  # [N,K,1]

    return shs * coeff_w





def compute_nadir_angle(camera_pos: torch.Tensor, gaussian_pos: torch.Tensor) -> torch.Tensor:
    ''' Returns:
        [N] nadir 각도 [0, π] '''
    view_dirs = gaussian_pos - camera_pos.unsqueeze(0)
    view_dirs = F.normalize(view_dirs, dim=-1)
    
    down_dir = torch.tensor([0.0, 0.0, -1.0], device=view_dirs.device)
    cos_angle = torch.sum(view_dirs * down_dir.unsqueeze(0), dim=-1)
    nadir_angles = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
    
    return nadir_angles

def compute_distance(camera_pos: torch.Tensor, gaussian_pos: torch.Tensor) -> torch.Tensor:
    ''' Returns:
        [N] 거리  '''
    distances = torch.norm(gaussian_pos - camera_pos.unsqueeze(0), dim=-1)
    return distances


# def view_dir_from(camera_pos: torch.Tensor, gaussian_pos: torch.Tensor) -> torch.Tensor:
#     """Returns [N,3]"""
#     v = gaussian_pos - camera_pos.unsqueeze(0)
#     return F.normalize(v, dim=-1)

# def dir_to_v_from_down(view_dirs: torch.Tensor) -> torch.Tensor:
#     down_dir = torch.tensor([0.0, 0.0, -1.0], device=view_dirs.device)
#     cosang = torch.sum(view_dirs * down_dir, dim=-1).clamp(-1.0, 1.0)
#     return torch.acos(cosang) / math.pi  # [N]


def ensure_NKC(shs: torch.Tensor) -> torch.Tensor:
   
    if shs.dim() != 3:
        raise ValueError(f"expect 3D tensor, got {shs.shape}")
    N, A, B = shs.shape
    if B == 3:         # [N,K,3]
        return shs
    if A == 3:         # [N,3,K] -> [N,K,3]
        return shs.transpose(1, 2).contiguous()
    raise ValueError(f"expect [N,K,3] or [N,3,K], got {shs.shape}")

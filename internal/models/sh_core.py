
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def band_flatten(sh_weight: torch.Tensor, L: int) -> torch.Tensor:
    """
    sh_weight: [N, L+1]  (band weights for l=0..L)
    return:    [N, (L+1)^2] (per-coefficient weights)
    """
    N, B = sh_weight.shape
    assert B == L + 1, f"got {sh_weight.shape}, expected [N,{L+1}]"

    dev = sh_weight.device
    band_sizes = torch.tensor([2 * l + 1 for l in range(L + 1)], device=dev, dtype=torch.long)  # [L+1]
    band_of_coeff = torch.repeat_interleave(
        torch.arange(L + 1, device=dev, dtype=torch.long),
        band_sizes,
    )  # [K]
    coeff_weights = sh_weight.index_select(dim=1, index=band_of_coeff)  # [N,K]
    return coeff_weights


def d_to_u(d: torch.Tensor, d_low: float, d_high: float, d0: float, w: float = 1.0) -> torch.Tensor:
    """거리 d를 [0,1]로 정규화 (선형+로그 혼합)"""
    d_clipped = d.clamp(d_low, d_high)
    u_lin = (d_clipped - d_low) / (d_high - d_low + 1e-8)
    u_log = torch.log1p(d_clipped / d0) / (math.log1p(d_high / d0) + 1e-8)
    return ((1.0 - w) * u_lin + w * u_log).clamp(0.0, 1.0)


def cos_to_v(cos_angle: torch.Tensor) -> torch.Tensor:
    """cos_angle: [-1,1] -> v in [0,1]"""
    cos_angle = cos_angle.clamp(-1.0, 1.0)
    v = 0.5 * (1.0 - cos_angle)
    return v.clamp(0.0, 1.0)


def build_gate_input(
    d: torch.Tensor,
    cos_angle: torch.Tensor,
    *,
    d_low: float = 0.2,
    d_high: float = 30.0,  #80
    d0: float = 5.0,
    dw: float = 0.5,
) -> torch.Tensor:

    u = d_to_u(d, d_low=d_low, d_high=d_high, d0=d0, w=dw)  # [N]
    v = cos_to_v(cos_angle)                                  # [N]
    return torch.stack([u, v], dim=-1)                        # [N,2]


class GateMLP(nn.Module):
    def __init__(self, in_dim: int = 2, L_max: int = 2, hidden: int = 64):
        super().__init__()
        self.L_max = L_max
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, L_max + 1),
        )
        # 초기 편향: 고차는 낮게 시작
        with torch.no_grad():
            self.net[-1].bias[:] = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N,2] = [u, v]
        return torch.sigmoid(self.net(x))  # [N, L+1]


class GateLUT(nn.Module):
    def __init__(self, L_max: int = 2, B_d: int = 32, B_phi: int = 32, ema: float = 0.9, device: str = "cuda"):
        super().__init__()
        self.L_max = L_max
        self.ema = ema
        self.B_d = B_d
        self.B_phi = B_phi
        table = torch.full((1, L_max + 1, B_phi, B_d), 0.1, device=device)
        self.register_buffer("table", table)

    @torch.no_grad()
    def update_from_mlp(self, mlp: GateMLP, grid_u: torch.Tensor, grid_v: torch.Tensor):
        """MLP로부터 LUT 업데이트"""
        B_phi, B_d = grid_u.shape
        inp = torch.stack([grid_u.reshape(-1), grid_v.reshape(-1)], dim=-1)  # [B_phi*B_d, 2]
        g = mlp(inp)  # [B_phi*B_d, L+1]
        g = g.reshape(B_phi, B_d, -1).permute(2, 0, 1).unsqueeze(0)  # [1, L+1, B_phi, B_d]
        self.table.mul_(self.ema).add_(g * (1.0 - self.ema))

    def lookup(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        grid = torch.stack([u * 2 - 1, v * 2 - 1], dim=-1).unsqueeze(0).unsqueeze(0)
        grid = grid.to(self.table.device, dtype=self.table.dtype)
        out = F.grid_sample(self.table, grid, mode="bilinear", align_corners=True)
        g = out.squeeze(0).squeeze(1).transpose(0, 1)  # [N, L+1]
        return g


class AdaptiveSHLoss(nn.Module):
    def __init__(self, lambda_sh=0.1, lambda_gate=0.001, lambda_tv=0.0025, lambda_mono=0.2):
        super().__init__()
        self.lambda_sh = lambda_sh
        self.lambda_gate = lambda_gate
        self.lambda_tv = lambda_tv
        self.lambda_mono = lambda_mono

    def forward(
        self,
        base_l_rgb: torch.Tensor,
        shs_raw: torch.Tensor,
        sh_weights: torch.Tensor,
        distances: torch.Tensor,
        gaussian_pos: Optional[torch.Tensor] = None,
    ) -> dict:
        losses = {}

        L_RGB = base_l_rgb
        losses["L_RGB"] = L_RGB

        L_SH_ratio = self._compute_sh_ratio_loss(shs_raw, distances)
        losses["L_SH_ratio"] = L_SH_ratio

        if sh_weights is not None:
            L_gate = self._compute_gate_loss(sh_weights)
            losses["L_gate"] = L_gate
            L_TV = self._compute_tv_loss(sh_weights, gaussian_pos) if gaussian_pos is not None else sh_weights.new_zeros(())
            losses["L_TV"] = L_TV
            L_mono = self._compute_monotonicity_loss(sh_weights, distances)
            losses["L_mono"] = L_mono
        else:
            losses["L_gate"] = losses["L_TV"] = losses["L_mono"] = base_l_rgb.new_zeros(())

        for k in ["L_RGB", "L_SH_ratio", "L_gate", "L_TV", "L_mono"]:
            v = losses[k]
            if not torch.isfinite(v):
                losses[k] = torch.zeros((), device=v.device)

        total_loss = (
            losses["L_RGB"]
            + self.lambda_sh * torch.clamp(losses["L_SH_ratio"], min=0, max=1e3)
            + self.lambda_gate * torch.clamp(losses["L_gate"], min=0, max=1e3)
            + self.lambda_tv * torch.clamp(losses["L_TV"], min=0, max=1e3)
            + self.lambda_mono * torch.clamp(losses["L_mono"], min=0, max=1e3)
        )

        losses["total_loss"] = total_loss
        return losses

    def _compute_sh_ratio_loss(
        self,
        shs_raw: torch.Tensor,
        distances: Optional[torch.Tensor] = None,
        q_near: float = 0.3,
        q_far: float = 0.7,
    ) -> torch.Tensor:
        # shs_raw: [N,K,3]
        N, K, _ = shs_raw.shape
        L = int(math.isqrt(K)) - 1
        if L <= 1 or distances is None or distances.numel() != N:
            return shs_raw.new_zeros(())

        x2 = (shs_raw ** 2).mean(dim=2)  # [N,K]
        band_energy, idx = [], 0
        for l in range(L + 1):
            bw = 2 * l + 1
            band_energy.append(x2[:, idx : idx + bw].sum(dim=1))
            idx += bw
        E = torch.stack(band_energy, dim=1)  # [N,L+1]

        L0 = 1
        E_low = E[:, : L0 + 1].sum(dim=1) + 1e-6
        E_high = E[:, L0 + 1 :].sum(dim=1)
        ratio = E_high / E_low

        d = distances
        thr_near = torch.quantile(d, q_near)
        thr_far = torch.quantile(d, q_far)

        near_mask = d <= thr_near
        far_mask = d >= thr_far

        loss = shs_raw.new_zeros(())

        if near_mask.any():
            r_min_ground = 0.4
            ratio_near = ratio[near_mask]
            loss_near = F.relu(r_min_ground - ratio_near).pow(2).mean()
            loss = loss + loss_near

        if far_mask.any():
            r_max_aerial = 0.2
            ratio_far = ratio[far_mask]
            loss_far = F.relu(ratio_far - r_max_aerial).pow(2).mean()
            loss = loss + loss_far

        return loss

    def _compute_gate_loss(self, sh_weights: torch.Tensor) -> torch.Tensor:
        w = sh_weights[:, 1:]  # DC 제외
        weight_diffs = torch.diff(w, dim=-1)
        smoothness_loss = torch.mean(weight_diffs ** 2)

        monotonic_penalty = torch.tensor(0.0, device=sh_weights.device)
        for i in range(w.shape[-1] - 1):
            violation = torch.relu(w[:, i + 1] - w[:, i])
            monotonic_penalty += torch.mean(violation ** 2)

        return smoothness_loss + 0.5 * monotonic_penalty

    def _compute_tv_loss(self, sh_weights: torch.Tensor, gaussian_pos: torch.Tensor) -> torch.Tensor:
        if gaussian_pos is None:
            return torch.tensor(0.0, device=sh_weights.device)

        N = gaussian_pos.shape[0]
        if N > 1000:
            indices = torch.randperm(N, device=gaussian_pos.device)[:1000]
            positions = gaussian_pos[indices]
            weights = sh_weights[indices]
        else:
            positions = gaussian_pos
            weights = sh_weights

        if positions.shape[0] <= 1:
            return torch.tensor(0.0, device=sh_weights.device)

        pos_diffs = positions.unsqueeze(1) - positions.unsqueeze(0)
        spatial_dists = torch.norm(pos_diffs, dim=-1)

        weight_diffs = weights.unsqueeze(1) - weights.unsqueeze(0)
        weight_dists = torch.norm(weight_diffs, dim=-1)

        spatial_weights = torch.exp(-spatial_dists / 0.1)
        tv_loss = torch.mean(spatial_weights * weight_dists)
        return tv_loss

    def _compute_monotonicity_loss(
        self,
        sh_weights: torch.Tensor,
        distances: torch.Tensor,
        nbins: int = 16,
        max_order_for_mono: Optional[int] = None,
        p_high: float = 1.0,
    ) -> torch.Tensor:
        if sh_weights is None or sh_weights.shape[0] <= 1:
            return torch.tensor(0.0, device=sh_weights.device)

        d = (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)
        bins = torch.clamp((d * nbins).long(), 0, nbins - 1)

        Lp1 = sh_weights.shape[1]
        L_use = (Lp1 - 1) if max_order_for_mono is None else min(max_order_for_mono, Lp1 - 1)

        g_bin = sh_weights.new_zeros(nbins, Lp1)
        cnt = sh_weights.new_zeros(nbins, 1)
        g_bin.index_add_(0, bins, sh_weights)
        cnt.index_add_(0, bins, torch.ones_like(d).unsqueeze(1))
        g_bin = g_bin / (cnt + 1e-6)

        diffs = g_bin[1:, 1 : L_use + 1] - g_bin[:-1, 1 : L_use + 1]
        viol = F.relu(diffs)

        l_idx = torch.arange(1, L_use + 1, device=sh_weights.device, dtype=sh_weights.dtype)
        w_l = (l_idx ** p_high)
        w_l = w_l / (w_l.sum() + 1e-12)

        return (viol.pow(2) * w_l.unsqueeze(0)).mean()


def apply_adaptive_sh_weights(shs: torch.Tensor, sh_weights: torch.Tensor, L: int) -> torch.Tensor:
    assert shs.dim() == 3 and shs.shape[2] == 3, f"expect [N,K,3], got {shs.shape}"
    _, K, C = shs.shape
    assert K == (L + 1) ** 2 and C == 3, f"{K=} {(L+1)**2=} {C=}"

    sh_weights = sh_weights.clone()
    sh_weights[:, 0] = 1.0

    coeff_w = band_flatten(sh_weights, L).unsqueeze(-1)  # [N,K,1]
    return shs * coeff_w


def compute_nadir_angle(camera_pos: torch.Tensor, gaussian_pos: torch.Tensor) -> torch.Tensor:
    """Returns: [N] nadir angle in [0, pi]"""
    view_dirs = gaussian_pos - camera_pos.unsqueeze(0)
    view_dirs = F.normalize(view_dirs, dim=-1)

    down_dir = torch.tensor([0.0, 0.0, -1.0], device=view_dirs.device)
    cos_angle = torch.sum(view_dirs * down_dir.unsqueeze(0), dim=-1)
    nadir_angles = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
    return nadir_angles


def compute_nadir_cos_angle(camera_pos: torch.Tensor, gaussian_pos: torch.Tensor) -> torch.Tensor:
    """Returns: [N] cos(nadir_angle) in [-1,1]"""
    view_dirs = gaussian_pos - camera_pos.unsqueeze(0)
    view_dirs = F.normalize(view_dirs, dim=-1)

    down_dir = torch.tensor([0.0, 0.0, -1.0], device=view_dirs.device)
    cos_angle = torch.sum(view_dirs * down_dir.unsqueeze(0), dim=-1)
    return torch.clamp(cos_angle, -1.0, 1.0)


def compute_distance(camera_pos: torch.Tensor, gaussian_pos: torch.Tensor) -> torch.Tensor:
    """Returns: [N] Euclidean distance"""
    return torch.norm(gaussian_pos - camera_pos.unsqueeze(0), dim=-1)


def ensure_NKC(shs: torch.Tensor) -> torch.Tensor:
    """[N,K,3] or [N,3,K] -> [N,K,3]"""
    if shs.dim() != 3:
        raise ValueError(f"expect 3D tensor, got {shs.shape}")
    _, A, B = shs.shape
    if B == 3:
        return shs
    if A == 3:
        return shs.transpose(1, 2).contiguous()
    raise ValueError(f"expect [N,K,3] or [N,3,K], got {shs.shape}")



from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """Sliced isotropic Gaussian regularization used by LeJEPA and adopted by Laya."""

    def __init__(
        self,
        num_slices: int = 256,
        quadrature_points: int = 17,
        cf_t_max: float = 3.0,
        cf_bandwidth: float = 1.0,
        projection_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if num_slices <= 0:
            raise ValueError("num_slices must be positive")
        if quadrature_points < 2:
            raise ValueError("quadrature_points must be >= 2")
        if cf_t_max <= 0:
            raise ValueError("cf_t_max must be positive")
        if cf_bandwidth <= 0:
            raise ValueError("cf_bandwidth must be positive")

        self.num_slices = num_slices
        self.projection_seed = projection_seed
        t = torch.linspace(0.0, cf_t_max, quadrature_points, dtype=torch.float32)
        dt = cf_t_max / float(quadrature_points - 1)
        weights = torch.full((quadrature_points,), 2.0 * dt, dtype=torch.float32)
        weights[0] = dt
        weights[-1] = dt
        gaussian_cf = torch.exp(-0.5 * t.square())
        smoothing = torch.exp(-0.5 * (cf_bandwidth * t).square())
        self.register_buffer("t", t)
        self.register_buffer("reference_cf", gaussian_cf)
        self.register_buffer("weights", weights * smoothing)

    def _sample_directions(self, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        generator = None
        if self.projection_seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.projection_seed)
        directions = torch.randn(dim, self.num_slices, device=device, dtype=dtype, generator=generator)
        return directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 3:
            raise ValueError(f"Expected [views, batch, dim], got {tuple(z.shape)}")
        _, _, dim = z.shape
        directions = self._sample_directions(dim, z.device, z.dtype)
        projections = z @ directions
        scaled = projections.unsqueeze(-1) * self.t
        real_error = scaled.cos().mean(dim=1) - self.reference_cf
        imag_error = scaled.sin().mean(dim=1)
        return ((real_error.square() + imag_error.square()) @ self.weights).mean()

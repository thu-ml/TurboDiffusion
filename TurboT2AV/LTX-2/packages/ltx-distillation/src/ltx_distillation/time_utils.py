from __future__ import annotations

import torch


def shift_rf_time(u: torch.Tensor, shift: float) -> torch.Tensor:
    if shift <= 0:
        return u
    return shift * u / (1 + (shift - 1) * u)


def sigma_to_rf_time(sigma: torch.Tensor) -> torch.Tensor:
    return sigma / (sigma + 1)


def rf_to_sigma(rf_t: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(rf_t.dtype).eps
    rf_t = rf_t.clamp(min=0.0, max=1.0 - eps)
    return rf_t / (1 - rf_t)


def sigma_to_trig_time(sigma: torch.Tensor) -> torch.Tensor:
    return torch.atan(sigma)


def trig_to_sigma(trig_t: torch.Tensor) -> torch.Tensor:
    return torch.tan(trig_t)


def rf_to_trig_time(rf_t: torch.Tensor) -> torch.Tensor:
    return sigma_to_trig_time(rf_to_sigma(rf_t))


def trig_to_rf_time(trig_t: torch.Tensor) -> torch.Tensor:
    return sigma_to_rf_time(trig_to_sigma(trig_t))


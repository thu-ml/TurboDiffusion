import torch

from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationType
from ltx_core.model.transformer.model import output_modulate
from ltx_core.model.transformer.transformer import (
    gated_residual_from_ada,
    modulate_from_ada,
    modulated_rms_norm_from_ada,
)
from ltx_core.utils import rms_norm


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    x = torch.randn(2, 3, 4, dtype=torch.float32)
    table = torch.randn(6, 4, dtype=torch.float32)
    timestep = torch.randn(2, 3, 24, dtype=torch.float32)
    return x, table, timestep


def _ada_values(table: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    return table[None, None] + timestep.reshape(2, 3, 6, 4)


def test_modulated_rms_norm_from_ada_matches_original_formula() -> None:
    x, table, timestep = _inputs()
    ada = _ada_values(table, timestep)
    expected = rms_norm(x, eps=1e-6) * (1 + ada[:, :, 1]) + ada[:, :, 0]

    actual = modulated_rms_norm_from_ada(x, table, timestep, 1, 0, 6, 1e-6)

    torch.testing.assert_close(actual, expected)


def test_modulate_from_ada_matches_original_formula() -> None:
    x, table, timestep = _inputs()
    ada = _ada_values(table, timestep)
    expected = x * (1 + ada[:, :, 2]) + ada[:, :, 3]

    actual = modulate_from_ada(x, table, timestep, 2, 3, 6)

    torch.testing.assert_close(actual, expected)


def test_gated_residual_from_ada_matches_original_formula() -> None:
    x, table, timestep = _inputs()
    residual = torch.randn_like(x)
    ada = _ada_values(table, timestep)
    expected = x + residual * ada[:, :, 5]

    actual = gated_residual_from_ada(x, residual, table, timestep, 5, 6)

    torch.testing.assert_close(actual, expected)


def test_gated_residual_from_ada_applies_batch_mask() -> None:
    x, table, timestep = _inputs()
    residual = torch.randn_like(x)
    mask = torch.tensor([1.0, 0.0]).reshape(2, 1, 1)
    ada = _ada_values(table, timestep)
    expected = x + residual * ada[:, :, 5] * mask

    actual = gated_residual_from_ada(x, residual, table, timestep, 5, 6, mask)

    torch.testing.assert_close(actual, expected)


def test_output_modulate_matches_original_formula() -> None:
    torch.manual_seed(11)
    x = torch.randn(2, 3, 4)
    table = torch.randn(2, 4)
    timestep = torch.randn(2, 3, 4)
    values = table[None, None] + timestep[:, :, None]
    expected = x * (1 + values[:, :, 1]) + values[:, :, 0]

    actual = output_modulate(x, table, timestep)

    torch.testing.assert_close(actual, expected)


def test_empty_perturbations_use_scalar_identity_mask() -> None:
    perturbations = BatchedPerturbationConfig.empty(batch_size=2)
    values = torch.randn(2, 3, 4)

    mask = perturbations.mask_like(PerturbationType.SKIP_VIDEO_SELF_ATTN, block=0, values=values)

    assert mask == 1.0

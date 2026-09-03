"""Regression tests for the Wan image-conditioning projector.

The network module imports CUDA-only attention extensions at import time.  The
projector itself is device agnostic, so these tests load just its class to keep
the regression suite runnable on CPU-only builders as well.
"""

import ast
from pathlib import Path

import torch
from torch import nn

_NETWORK_SOURCE = (
    Path(__file__).parents[1] / "turbodiffusion" / "rcm" / "networks" / "wan2pt1_jvp.py"
)


def _load_projector_class():
    tree = ast.parse(_NETWORK_SOURCE.read_text(encoding="utf-8"))
    projector_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MLPProj"
    )
    namespace = {
        "torch": torch,
        "nn": nn,
        "FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER": 257 * 2,
    }
    module = ast.Module(body=[projector_node], type_ignores=[])
    exec(compile(module, str(_NETWORK_SOURCE), "exec"), namespace)  # noqa: S102
    return namespace["MLPProj"]


MLPProj = _load_projector_class()


def _canonical_projector(in_dim, out_dim):
    """Architecture used by the published Wan I2V projector."""
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, in_dim),
        nn.GELU(),
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
    )


def test_projector_uses_affine_layer_norms_and_checkpoint_keys():
    projector = MLPProj(8, 16)
    first_norm, last_norm = projector.proj[0], projector.proj[4]

    assert isinstance(first_norm, nn.LayerNorm)
    assert isinstance(last_norm, nn.LayerNorm)
    assert first_norm.elementwise_affine
    assert last_norm.elementwise_affine
    assert first_norm.eps == nn.LayerNorm(8).eps
    assert last_norm.eps == nn.LayerNorm(16).eps
    assert set(projector.state_dict()) == {
        "proj.0.weight",
        "proj.0.bias",
        "proj.1.weight",
        "proj.1.bias",
        "proj.3.weight",
        "proj.3.bias",
        "proj.4.weight",
        "proj.4.bias",
    }


def test_projector_matches_canonical_wan_forward_and_gradients():
    torch.manual_seed(7)
    projector = MLPProj(8, 16)
    reference = _canonical_projector(8, 16)
    reference.load_state_dict(projector.proj.state_dict())

    image_embeds = torch.randn(2, 5, 8, requires_grad=True)
    output = projector(image_embeds)
    expected = reference(image_embeds.detach().clone().requires_grad_())

    torch.testing.assert_close(output, expected)
    output.square().mean().backward()
    assert torch.isfinite(image_embeds.grad).all()
    assert projector.proj[0].weight.grad is not None
    assert torch.isfinite(projector.proj[0].weight.grad).all()


def test_first_last_frame_projector_keeps_two_frame_token_layout():
    projector = MLPProj(1280, 16, flf_pos_emb=True)
    # The first/last-frame path receives each frame as a separate batch item
    # and folds the pair into one sequence before adding positional features.
    image_embeds = torch.randn(2, 257, 1280)

    output = projector(image_embeds)

    assert output.shape == (1, 514, 16)
    assert projector.emb_pos.shape == (1, 514, 1280)

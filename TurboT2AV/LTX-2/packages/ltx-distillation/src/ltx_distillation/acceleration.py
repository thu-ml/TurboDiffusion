"""Optional SageAttention and FastNorm acceleration for LTX-2 inference."""

from __future__ import annotations

import importlib
import math
import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from ltx_core.model.transformer.attention import Attention

ATTENTION_TYPES = ("default", "sageattn", "sla", "sagesla")
ATTENTION_SCOPES = ("self", "video_self", "self_av")
QUANT_LINEAR_SCOPES = (
    "all",
    "transformer_blocks",
    "ffn",
    "video_ffn",
    "audio_ffn",
    "video_heavy",
    "non_attention",
)
QUANT_LINEAR_BACKENDS = ("turbodiffusion", "tilelang_postscale")
DEFAULT_SLA_TOPK = 1.0
_TD_W8A8_QUANT_WORKSPACES: dict[tuple[str, int | None, torch.dtype, int, int], tuple[torch.Tensor, torch.Tensor]] = {}
_TD_W8A8_OPS: tuple[
    Callable[..., object],
    Callable[..., object],
    Callable[..., object],
    Callable[..., object],
] | None = None


@dataclass(frozen=True)
class AccelerationReport:
    attention_type: str
    attention_scope: str
    replaced_attention: int = 0
    skipped_attention: int = 0
    replaced_linear: int = 0
    fused_attention_projection: int = 0
    quant_linear_scope: str = "none"
    quant_linear_backend: str = "none"
    sla_topk: float = DEFAULT_SLA_TOPK
    sla_topk_schedule: str = ""
    replaced_norm: int = 0
    replaced_functional_norm: int = 0

    def format(self) -> str:
        return (
            "[TurboT2AV][accel] "
            f"attention_type={self.attention_type} "
            f"attention_scope={self.attention_scope} "
            f"replaced_attention={self.replaced_attention} "
            f"skipped_attention={self.skipped_attention} "
            f"replaced_linear={self.replaced_linear} "
            f"fused_attention_projection={self.fused_attention_projection} "
            f"quant_linear_scope={self.quant_linear_scope} "
            f"quant_linear_backend={self.quant_linear_backend} "
            f"sla_topk={self.sla_topk:g} "
            f"sla_topk_schedule={self.sla_topk_schedule or 'none'} "
            f"replaced_norm={self.replaced_norm} "
            f"replaced_functional_norm={self.replaced_functional_norm}"
        )


def _ensure_turbodiffusion_path() -> None:
    """Make TurboDiffusion's local ops importable from the vendored TurboT2AV directory."""

    current = Path(__file__).resolve()
    for parent in current.parents:
        for candidate in (
            parent / "turbodiffusion",
            parent / "TurboDiffusion" / "turbodiffusion",
        ):
            if not (candidate / "ops").is_dir():
                continue
            for path in (candidate, candidate.parent):
                path_str = str(path)
                if path_str not in sys.path:
                    sys.path.insert(0, path_str)


def _import_turbodiffusion_attr(module_names: tuple[str, ...], attr_name: str) -> object:
    _ensure_turbodiffusion_path()
    errors = []
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, attr_name)
        except (ImportError, AttributeError) as exc:
            errors.append(f"{module_name}: {exc}")
    raise ImportError(
        f"Unable to import TurboDiffusion acceleration symbol {attr_name}. "
        "Install TurboDiffusion from source or place TurboT2AV inside the "
        "TurboDiffusion repository. Tried: " + "; ".join(errors)
    )


class SageAttentionCallable(torch.nn.Module):
    """Adapter from LTX attention tensors to SageAttention's HND layout."""

    def __init__(self) -> None:
        super().__init__()
        self._fn, self._fn_kwargs = self._load_sage_attention()

    @staticmethod
    def _load_sage_attention() -> tuple[Callable[..., torch.Tensor], dict[str, object]]:
        errors = []
        for module_name, attr_name in (
            ("sageattention", "sageattn"),
            ("sageattention.core", "sageattn"),
            ("sageattention", "sageattn_qk_int8_pv_fp16_cuda"),
            ("sageattention.core", "sageattn_qk_int8_pv_fp16_cuda"),
            ("sageattention", "sageattn_qk_int8_pv_fp16_triton"),
            ("sageattention.core", "sageattn_qk_int8_pv_fp16_triton"),
        ):
            try:
                module = importlib.import_module(module_name)
                fn = getattr(module, attr_name)
                kwargs: dict[str, object] = {}
                if attr_name == "sageattn_qk_int8_pv_fp16_cuda":
                    kwargs["qk_quant_gran"] = "per_warp"
                return fn, kwargs
            except (ImportError, AttributeError) as exc:
                errors.append(f"{module_name}.{attr_name}: {exc}")
        raise ImportError(
            "Unable to import SageAttention. Install `sageattention` or use "
            "--attention_type default. Tried: " + "; ".join(errors)
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is not None:
            raise NotImplementedError(
                "SageAttention is enabled only for unmasked LTX attention. "
                "Use --attention_scope self or self_av."
            )

        batch, _, inner_dim = q.shape
        head_dim = inner_dim // heads
        q, k, v = (
            tensor.view(batch, -1, heads, head_dim).transpose(1, 2).contiguous()
            for tensor in (q, k, v)
        )
        out = self._fn(
            q,
            k,
            v,
            tensor_layout="HND",
            is_causal=False,
            sm_scale=head_dim**-0.5,
            **self._fn_kwargs,
        )
        return out.transpose(1, 2).reshape(batch, -1, inner_dim)


class LTXSLAAttention(torch.nn.Module):
    """Adapter from LTX attention tensors to TurboDiffusion's vendored SLA."""

    def __init__(
        self,
        head_dim: int,
        topk: float,
        block_q: int,
        block_k: int,
        use_bf16: bool,
    ) -> None:
        super().__init__()
        _ensure_turbodiffusion_path()
        from SLA import SparseLinearAttention

        self.requested_topk = topk
        self.block_k = block_k
        self.local_attn = SparseLinearAttention(
            head_dim=head_dim,
            topk=topk,
            BLKQ=block_q,
            BLKK=block_k,
            use_bf16=use_bf16,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is not None:
            raise NotImplementedError(
                "SLA is enabled only for unmasked LTX self-attention. "
                "Use --attention_scope self."
            )

        batch, _, inner_dim = q.shape
        head_dim = inner_dim // heads
        q, k, v = (tensor.view(batch, -1, heads, head_dim).contiguous() for tensor in (q, k, v))
        key_blocks = max(1, math.ceil(k.shape[1] / self.block_k))
        effective_topk = max(self.requested_topk, 1.0 / key_blocks)
        original_topk = self.local_attn.topk
        self.local_attn.topk = effective_topk
        try:
            out = self.local_attn(q, k, v)
        finally:
            self.local_attn.topk = original_topk
        return out.reshape(batch, -1, inner_dim)


class LTXSageSLAAttention(torch.nn.Module):
    """Adapter from LTX attention tensors to TurboDiffusion's SageSLA path."""

    def __init__(self, head_dim: int, topk: float, use_bf16: bool) -> None:
        super().__init__()
        _ensure_turbodiffusion_path()
        from SLA import SageSparseLinearAttention

        self.requested_topk = topk
        self.local_attn = SageSparseLinearAttention(
            head_dim=head_dim,
            topk=topk,
            use_bf16=use_bf16,
        )
        self._skip_zero_linear: bool | None = None

    @staticmethod
    def _block_k_for_device(device: torch.device) -> int:
        if device.type == "cuda" and torch.cuda.get_device_capability(device) == (9, 0):
            return 128
        return 64

    @staticmethod
    def _skip_zero_linear_enabled() -> bool:
        return os.environ.get("TURBOT2AV_SLA_SKIP_ZERO_LINEAR", "1").lower() not in {"0", "false", "no"}

    def _should_skip_zero_linear(self) -> bool:
        if not self._skip_zero_linear_enabled():
            return False
        if self._skip_zero_linear is None:
            proj_l = self.local_attn.proj_l
            with torch.no_grad():
                has_weight = bool(torch.count_nonzero(proj_l.weight).item())
                has_bias = bool(torch.count_nonzero(proj_l.bias).item()) if proj_l.bias is not None else False
            self._skip_zero_linear = not (has_weight or has_bias)
        return self._skip_zero_linear

    def _sparse_only_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        import SLA.core as sla_core

        dtype = q.dtype
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        arch = sla_core.get_cuda_arch(q.device.index)
        if arch == "sm90":
            sparse_map, _, _ = sla_core.get_block_map(q, k, topk_ratio=self.local_attn.topk, BLKQ=64, BLKK=128)
        else:
            sparse_map, _, _ = sla_core.get_block_map(q, k, topk_ratio=self.local_attn.topk, BLKQ=128, BLKK=64)

        q = q.to(self.local_attn.dtype)
        k = k.to(self.local_attn.dtype)
        v = v.to(self.local_attn.dtype)

        km = k.mean(dim=-2, keepdim=True)
        head_dim = q.size(-1)
        if arch == "sm90":
            q_int8, q_scale, k_int8, k_scale = sla_core.get_vanilla_qk_quant(q, k, km, 64, 128)
        else:
            q_int8, q_scale, k_int8, k_scale = sla_core.get_vanilla_qk_quant(q, k, km, 128, 64)
        lut, valid_block_num = sla_core.block_map_lut_triton(sparse_map)
        scale = 1.0 / (head_dim**0.5)

        o_s = torch.empty_like(q)
        if arch in ("sm80", "sm86", "sm87"):
            pvthreshold = torch.full((q.shape[-3],), 1e6, dtype=torch.float32, device=q.device)
            v_fp16 = v.to(torch.float16)
            sla_core.qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(
                q_int8,
                k_int8,
                v_fp16,
                o_s,
                lut,
                valid_block_num,
                pvthreshold,
                q_scale,
                k_scale,
                1,
                False,
                1,
                scale,
                0,
            )
        else:
            batch, heads, kv_len, head_dim = v.shape
            padded_len = (kv_len + 127) // 128 * 128
            v_transposed_permutted = torch.empty(
                (batch, heads, head_dim, padded_len),
                dtype=v.dtype,
                device=v.device,
            )
            sla_core.fused.transpose_pad_permute_cuda(v, v_transposed_permutted, 1)
            v_fp8 = torch.empty(v_transposed_permutted.shape, dtype=torch.float8_e4m3fn, device=v.device)
            v_scale = torch.empty((batch, heads, head_dim), dtype=torch.float32, device=v.device)
            sla_core.fused.scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, v_scale, kv_len, 2.25, 1)

            if arch == "sm90":
                sla_core.qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_sm90(
                    q_int8,
                    k_int8,
                    v_fp8,
                    o_s,
                    lut,
                    valid_block_num,
                    q_scale,
                    k_scale,
                    v_scale,
                    1,
                    False,
                    1,
                    scale,
                )
            else:
                pvthreshold = torch.full((q.shape[-3],), 1e6, dtype=torch.float32, device=q.device)
                if sla_core.SAGE2PP_ENABLED:
                    sla_core.qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(
                        q_int8,
                        k_int8,
                        v_fp8,
                        o_s,
                        lut,
                        valid_block_num,
                        pvthreshold,
                        q_scale,
                        k_scale,
                        v_scale,
                        1,
                        False,
                        1,
                        scale,
                        0,
                    )
                else:
                    sla_core.qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(
                        q_int8,
                        k_int8,
                        v_fp8,
                        o_s,
                        lut,
                        valid_block_num,
                        pvthreshold,
                        q_scale,
                        k_scale,
                        v_scale,
                        1,
                        False,
                        1,
                        scale,
                        0,
                    )

        return o_s.to(dtype).transpose(1, 2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is not None:
            raise NotImplementedError(
                "SageSLA is enabled only for unmasked LTX self-attention. "
                "Use --attention_scope self."
            )

        batch, _, inner_dim = q.shape
        head_dim = inner_dim // heads
        q, k, v = (tensor.view(batch, -1, heads, head_dim).contiguous() for tensor in (q, k, v))
        key_blocks = max(1, math.ceil(k.shape[1] / self._block_k_for_device(k.device)))
        effective_topk = max(self.requested_topk, 1.0 / key_blocks)
        original_topk = self.local_attn.topk
        self.local_attn.topk = effective_topk
        try:
            if self._should_skip_zero_linear():
                out = self._sparse_only_forward(q, k, v)
            else:
                out = self.local_attn(q, k, v)
        finally:
            self.local_attn.topk = original_topk
        return out.reshape(batch, -1, inner_dim)


def _is_self_attention_name(name: str) -> bool:
    return name.endswith("attn1")


def _is_video_self_attention_name(name: str) -> bool:
    return name.endswith(".attn1") or name == "attn1"


def _is_av_cross_attention_name(name: str) -> bool:
    return name.endswith(("audio_to_video_attn", "video_to_audio_attn"))


def _attention_name_in_scope(name: str, attention_scope: str) -> bool:
    if attention_scope == "self":
        return _is_self_attention_name(name)
    if attention_scope == "video_self":
        return _is_video_self_attention_name(name)
    if attention_scope == "self_av":
        return _is_self_attention_name(name) or _is_av_cross_attention_name(name)
    raise ValueError(f"Unsupported attention_scope: {attention_scope}")


def _attention_supported_by_backend(name: str, attention_type: str) -> bool:
    if attention_type in {"sla", "sagesla"} and not _is_self_attention_name(name):
        return False
    return True


_TRANSFORMER_BLOCK_RE = re.compile(r"(?:^|\.)transformer_blocks\.(\d+)\.")
SlaTopkSchedule = tuple[tuple[int, int, float], ...]


def _parse_sla_topk_schedule(schedule: str | None) -> SlaTopkSchedule:
    """Parse layer ranges like ``0-15:0.35,16-47:0.3``."""

    if not schedule:
        return ()
    parsed: list[tuple[int, int, float]] = []
    for raw_item in schedule.split(","):
        item = raw_item.strip()
        if not item:
            continue
        try:
            layer_part, topk_part = item.split(":", 1)
        except ValueError as exc:
            raise ValueError(
                "--sla_topk_schedule entries must use START-END:TOPK, "
                f"got {item!r}"
            ) from exc
        layer_part = layer_part.strip()
        if "-" in layer_part:
            start_text, end_text = layer_part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
        else:
            start = end = int(layer_part)
        topk = float(topk_part)
        if start < 0 or end < 0 or end < start:
            raise ValueError(f"Invalid layer range in --sla_topk_schedule: {item!r}")
        if not (0.0 < topk <= 1.0):
            raise ValueError(f"SLA topk schedule values must be in (0, 1], got {topk!r}")
        parsed.append((start, end, topk))
    return tuple(parsed)


def _attention_layer_index(name: str) -> int | None:
    match = _TRANSFORMER_BLOCK_RE.search(name)
    if match is None:
        return None
    return int(match.group(1))


def _scheduled_sla_topk(name: str, default_topk: float, schedule: SlaTopkSchedule) -> float:
    layer_idx = _attention_layer_index(name)
    if layer_idx is None:
        return default_topk
    for start, end, topk in schedule:
        if start <= layer_idx <= end:
            return topk
    return default_topk


def _projection_device_dtype(projection: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
    """Return the runtime device/dtype for Linear or TurboDiffusion Int8Linear."""

    device: torch.device | None = None
    dtype: torch.dtype | None = None

    for attr_name in ("weight", "bias", "int8_weight", "scale"):
        tensor = getattr(projection, attr_name, None)
        if not isinstance(tensor, torch.Tensor) or tensor.device == torch.device("meta"):
            continue
        if device is None:
            device = tensor.device
        if dtype is None and tensor.is_floating_point() and tensor.dtype != torch.float32:
            dtype = tensor.dtype

    if device is None:
        for tensor in list(projection.parameters(recurse=False)) + list(projection.buffers(recurse=False)):
            if tensor.device != torch.device("meta"):
                device = tensor.device
                break

    if dtype is None:
        dtype = torch.bfloat16
    if device is None:
        device = torch.device("cpu")
    return device, dtype


def replace_ltx_attention(
    model: torch.nn.Module,
    attention_type: str,
    attention_scope: str = "self",
    sla_topk: float = DEFAULT_SLA_TOPK,
    sla_topk_schedule: str | None = None,
    sla_block_q: int = 128,
    sla_block_k: int = 64,
) -> tuple[int, int]:
    if attention_type not in ATTENTION_TYPES:
        raise ValueError(f"--attention_type must be one of {ATTENTION_TYPES}")
    if attention_scope not in ATTENTION_SCOPES:
        raise ValueError(f"--attention_scope must be one of {ATTENTION_SCOPES}")
    if attention_type == "default":
        return 0, 0
    if attention_type in {"sla", "sagesla"} and not (0.0 < sla_topk <= 1.0):
        raise ValueError(f"--sla_topk must be in (0, 1], got {sla_topk!r}")
    if attention_type == "sla" and (sla_block_q <= 0 or sla_block_k <= 0):
        raise ValueError("--sla_block_q and --sla_block_k must be positive")

    parsed_topk_schedule = _parse_sla_topk_schedule(sla_topk_schedule)
    replaced = 0
    skipped = 0
    for name, module in model.named_modules():
        if not isinstance(module, Attention) or not _attention_name_in_scope(name, attention_scope):
            continue
        if not _attention_supported_by_backend(name, attention_type):
            skipped += 1
            continue
        projection_device, projection_dtype = _projection_device_dtype(module.to_q)
        if attention_type == "sageattn":
            attention_callable = SageAttentionCallable().to(device=projection_device)
        elif attention_type == "sla":
            module_topk = _scheduled_sla_topk(name, sla_topk, parsed_topk_schedule)
            attention_callable = LTXSLAAttention(
                head_dim=module.dim_head,
                topk=module_topk,
                block_q=sla_block_q,
                block_k=sla_block_k,
                use_bf16=projection_dtype == torch.bfloat16,
            ).to(device=projection_device, dtype=projection_dtype)
        elif attention_type == "sagesla":
            module_topk = _scheduled_sla_topk(name, sla_topk, parsed_topk_schedule)
            attention_callable = LTXSageSLAAttention(
                head_dim=module.dim_head,
                topk=module_topk,
                use_bf16=projection_dtype == torch.bfloat16,
            ).to(device=projection_device, dtype=projection_dtype)
        else:
            raise ValueError(f"Unsupported attention_type={attention_type!r}")
        module.attention_function = attention_callable
        replaced += 1
    return replaced, skipped


def _set_child_module(root: torch.nn.Module, qualified_name: str, new_module: torch.nn.Module) -> None:
    parent = root
    parts = qualified_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _fast_rmsnorm_from_torch(
    original_rmsnorm: torch.nn.RMSNorm,
    fast_rmsnorm_cls: Callable[..., torch.nn.Module],
) -> torch.nn.Module:
    normalized_shape = original_rmsnorm.normalized_shape
    dim = normalized_shape[0] if isinstance(normalized_shape, tuple) else normalized_shape
    fast_rmsnorm = fast_rmsnorm_cls(dim=dim, eps=original_rmsnorm.eps)
    device = original_rmsnorm.weight.device if original_rmsnorm.weight is not None else torch.device("cpu")
    if original_rmsnorm.weight is not None and original_rmsnorm.weight.device != torch.device("meta"):
        fast_rmsnorm.weight.data.copy_(original_rmsnorm.weight.float().data)
    return fast_rmsnorm.to(device=device)


def _fast_layernorm_from_torch(
    original_layernorm: torch.nn.LayerNorm,
    fast_layernorm_cls: type[torch.nn.Module],
) -> torch.nn.Module:
    normalized_shape = original_layernorm.normalized_shape
    if not isinstance(normalized_shape, tuple) or len(normalized_shape) != 1:
        raise ValueError(
            "TurboDiffusion FastLayerNorm only supports 1D normalized_shape; "
            f"got {normalized_shape!r}"
        )
    fast_layernorm = fast_layernorm_cls.from_layernorm(original_layernorm)
    if original_layernorm.weight is not None:
        device = original_layernorm.weight.device
    elif original_layernorm.bias is not None:
        device = original_layernorm.bias.device
    else:
        device = torch.device("cpu")
    return fast_layernorm.to(device=device)


def replace_ltx_norms(model: torch.nn.Module) -> int:
    fast_rmsnorm_cls = _import_turbodiffusion_attr(("ops", "turbodiffusion.ops"), "FastRMSNorm")
    fast_layernorm_cls = _import_turbodiffusion_attr(("ops", "turbodiffusion.ops"), "FastLayerNorm")

    replacements: dict[str, torch.nn.Module] = {}
    for name, module in model.named_modules():
        if not name or ".attention_function." in name or name.endswith(".attention_function"):
            continue
        if isinstance(module, torch.nn.RMSNorm):
            replacements[name] = _fast_rmsnorm_from_torch(module, fast_rmsnorm_cls)
        elif isinstance(module, torch.nn.LayerNorm):
            replacements[name] = _fast_layernorm_from_torch(module, fast_layernorm_cls)

    for name, new_module in replacements.items():
        _set_child_module(model, name, new_module)
    return len(replacements)


def _is_ltx_ffn_linear_name(name: str) -> bool:
    return ".ff.net." in name or ".audio_ff.net." in name


def _is_ltx_video_ffn_linear_name(name: str) -> bool:
    return ".ff.net." in name and ".audio_ff.net." not in name


def _is_ltx_audio_ffn_linear_name(name: str) -> bool:
    return ".audio_ff.net." in name


def _is_ltx_attention_linear_name(name: str) -> bool:
    return any(
        marker in name
        for marker in (
            ".attn1.",
            ".attn2.",
            ".audio_attn1.",
            ".audio_attn2.",
            ".audio_to_video_attn.",
            ".video_to_audio_attn.",
        )
    )


def _is_ltx_video_self_attention_linear_name(name: str) -> bool:
    return ".attn1." in f".{name}."


def _is_ltx_transformer_block_linear_name(name: str) -> bool:
    scoped_name = f".{name}."
    return ".transformer_blocks." in scoped_name and "proj_l" not in name


def _linear_name_in_quant_scope(name: str, scope: str) -> bool:
    if scope == "all":
        return True
    if scope == "transformer_blocks":
        return _is_ltx_transformer_block_linear_name(name)
    if scope == "ffn":
        return _is_ltx_ffn_linear_name(name)
    if scope == "video_ffn":
        return _is_ltx_video_ffn_linear_name(name)
    if scope == "audio_ffn":
        return _is_ltx_audio_ffn_linear_name(name)
    if scope == "video_heavy":
        return _is_ltx_video_ffn_linear_name(name) or _is_ltx_video_self_attention_linear_name(name)
    if scope == "non_attention":
        return not _is_ltx_attention_linear_name(name)
    raise ValueError(f"Unsupported quant_linear_scope={scope!r}; expected one of {QUANT_LINEAR_SCOPES}")


def _td_w8a8_ops() -> tuple[Callable[..., object], Callable[..., object], Callable[..., object], Callable[..., object]]:
    global _TD_W8A8_OPS
    if _TD_W8A8_OPS is None:
        _ensure_turbodiffusion_path()
        td_ops = importlib.import_module("turbodiffusion.ops")
        cuda_ops = importlib.import_module("turbo_diffusion_ops")
        _TD_W8A8_OPS = (
            getattr(td_ops, "int8_quant"),
            getattr(cuda_ops, "quant_cuda"),
            getattr(cuda_ops, "gemm_cuda_swizzle"),
            getattr(cuda_ops, "gemm_cuda_swizzle_bias"),
        )
    return _TD_W8A8_OPS


def _td_w8a8_cdiv(a: int, b: int = 128) -> int:
    return (a + b - 1) // b


def _td_w8a8_quant_workspace(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.shape
    key = (x.device.type, x.device.index, x.dtype, m, k)
    workspace = _TD_W8A8_QUANT_WORKSPACES.get(key)
    if workspace is None:
        workspace = (
            torch.empty((m, k), dtype=torch.int8, device=x.device),
            torch.empty((_td_w8a8_cdiv(m), _td_w8a8_cdiv(k)), dtype=torch.float32, device=x.device),
        )
        _TD_W8A8_QUANT_WORKSPACES[key] = workspace
    return workspace


def _td_w8a8_swizzle(m: int, k: int, n: int) -> tuple[int, int]:
    override = os.environ.get("TURBOT2AV_TD_W8A8_SWIZZLE")
    if override:
        direction, log_size = override.split(",", maxsplit=1)
        return int(direction), int(log_size)

    if k == 4096 and n == 16384:
        return (0, 0) if m <= 8192 else (0, 4)
    if k == 16384 and n == 4096:
        return (1, 3) if m <= 8192 else (0, 2)
    return 1, 4


class _TurboDiffusionInt8Linear(torch.nn.Module):
    """TurboDiffusion W8A8 Linear with the same state dict layout and kernels."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("int8_weight", torch.empty((out_features, in_features), dtype=torch.int8))
        self.register_buffer(
            "scale",
            torch.empty((_td_w8a8_cdiv(out_features), _td_w8a8_cdiv(in_features)), dtype=torch.float32),
        )
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=dtype))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, quant_cuda, gemm_cuda_swizzle, gemm_cuda_swizzle_bias = _td_w8a8_ops()
        shape = x.shape
        x_2d = x.reshape(-1, shape[-1])
        m, k = x_2d.shape
        y = torch.empty((m, self.out_features), dtype=x_2d.dtype, device=x_2d.device)

        if os.environ.get("TURBOT2AV_TD_W8A8_PREALLOC_QUANT", "1") == "0":
            x_q, x_s = quant_cuda(x_2d, None, None)
        else:
            x_q_buf, x_s_buf = _td_w8a8_quant_workspace(x_2d)
            x_q, x_s = quant_cuda(x_2d, x_q_buf, x_s_buf)

        swizzle_dir, swizzle_log = _td_w8a8_swizzle(m, k, self.out_features)
        if self.bias is not None:
            gemm_cuda_swizzle_bias(x_q, x_s, self.int8_weight, self.scale, y, self.bias, swizzle_dir, swizzle_log)
        else:
            gemm_cuda_swizzle(x_q, x_s, self.int8_weight, self.scale, y, swizzle_dir, swizzle_log)
        return y.reshape(*shape[:-1], self.out_features)

    @classmethod
    def from_linear(cls, original_linear: torch.nn.Linear) -> "_TurboDiffusionInt8Linear":
        device = original_linear.weight.device
        if device.type != "cuda":
            raise ValueError("TurboDiffusion W8A8 requires Linear weights on CUDA")
        int8_quant, _, _, _ = _td_w8a8_ops()
        int8_layer = cls(
            original_linear.in_features,
            original_linear.out_features,
            bias=original_linear.bias is not None,
            dtype=original_linear.weight.dtype,
        ).to(device=device)
        int8_w, scale = int8_quant(original_linear.weight.detach().contiguous())
        int8_layer.int8_weight.copy_(int8_w)
        int8_layer.scale.copy_(scale)
        if original_linear.bias is not None:
            int8_layer.bias.copy_(original_linear.bias.detach())
        return int8_layer


def _replace_ltx_linears_turbodiffusion(model: torch.nn.Module, scope: str) -> int:
    """Replace loaded generator Linear layers with TurboDiffusion Int8Linear."""

    replacements: dict[str, torch.nn.Module] = {}
    for name, module in model.named_modules():
        if not name or ".attention_function." in name or name.endswith(".attention_function"):
            continue
        if not isinstance(module, torch.nn.Linear):
            continue
        if not _linear_name_in_quant_scope(name, scope):
            continue
        if module.weight.device == torch.device("meta"):
            continue
        int8_linear = _TurboDiffusionInt8Linear.from_linear(module)
        replacements[name] = int8_linear

    for name, new_module in replacements.items():
        _set_child_module(model, name, new_module)
    return len(replacements)


def _replace_ltx_linears_tilelang_postscale(model: torch.nn.Module, scope: str) -> int:
    """Replace Linear layers with TileLang post-scale W8A8 modules."""

    from ltx_distillation.tilelang_w8a8 import TileLangPostScaleInt8Linear

    replacements: dict[str, torch.nn.Module] = {}
    for name, module in model.named_modules():
        if not name or ".attention_function." in name or name.endswith(".attention_function"):
            continue
        if not isinstance(module, torch.nn.Linear):
            continue
        if not _linear_name_in_quant_scope(name, scope):
            continue
        if module.weight.device == torch.device("meta"):
            continue
        int8_linear = TileLangPostScaleInt8Linear.from_linear(module)
        replacements[name] = int8_linear

    for name, new_module in replacements.items():
        _set_child_module(model, name, new_module)
    return len(replacements)


def fuse_tilelang_attention_projections(model: torch.nn.Module) -> int:
    if os.environ.get("TURBOT2AV_TILELANG_W8A8_FUSE_QKV", "1").lower() in {"0", "false", "no"}:
        return 0
    from ltx_distillation.tilelang_w8a8 import TileLangPostScaleInt8Linear

    fused = 0
    for module in model.modules():
        if not isinstance(module, Attention):
            continue
        has_tilelang_qkv = all(
            isinstance(getattr(module, attr), TileLangPostScaleInt8Linear) for attr in ("to_q", "to_k", "to_v")
        )
        if has_tilelang_qkv and (
            module.to_q.in_features == module.to_k.in_features == module.to_v.in_features
        ):
            module.to_qkv = TileLangPostScaleInt8Linear.from_tilelang_linears(
                (module.to_q, module.to_k, module.to_v)
            )
            fused += 1
        if all(isinstance(getattr(module, attr), TileLangPostScaleInt8Linear) for attr in ("to_k", "to_v")) and (
            module.to_k.in_features == module.to_v.in_features
        ):
            module.to_kv = TileLangPostScaleInt8Linear.from_tilelang_linears((module.to_k, module.to_v))
            fused += 1
    return fused


def replace_ltx_linears(
    model: torch.nn.Module,
    scope: str = "all",
    backend: str = "turbodiffusion",
) -> int:
    if scope not in QUANT_LINEAR_SCOPES:
        raise ValueError(f"Unsupported quant_linear_scope={scope!r}; expected one of {QUANT_LINEAR_SCOPES}")
    if backend not in QUANT_LINEAR_BACKENDS:
        raise ValueError(f"Unsupported quant_linear_backend={backend!r}; expected one of {QUANT_LINEAR_BACKENDS}")
    if backend == "turbodiffusion":
        return _replace_ltx_linears_turbodiffusion(model, scope=scope)
    if backend == "tilelang_postscale":
        return _replace_ltx_linears_tilelang_postscale(model, scope=scope)
    raise ValueError(f"Unsupported quant_linear_backend={backend!r}")


def enable_fast_functional_rms_norm() -> int:
    """Patch LTX's imported rms_norm helper to TurboDiffusion's fused kernel."""

    fused_rmsnorm = _import_turbodiffusion_attr(("ops", "turbodiffusion.ops"), "rmsnorm")
    from ltx_distillation.fast_norm_kernels import fast_rms_norm as triton_fast_rms_norm

    utils_module = importlib.import_module("ltx_core.utils")
    attention_module = importlib.import_module("ltx_core.model.transformer.attention")
    model_module = importlib.import_module("ltx_core.model.transformer.model")
    transformer_module = importlib.import_module("ltx_core.model.transformer.transformer")
    rope_module = importlib.import_module("ltx_core.model.transformer.rope")
    original_rms_norm = utils_module.rms_norm
    if getattr(original_rms_norm, "_turbot2av_fast_functional_norm", False):
        return 0

    def fast_rms_norm(
        x: torch.Tensor,
        weight: torch.Tensor | None = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        if (
            not x.is_cuda
            or x.dim() not in {2, 3}
            or not x.is_contiguous()
            or x.dtype not in {torch.float16, torch.bfloat16, torch.float32}
        ):
            return original_rms_norm(x, weight=weight, eps=eps)

        try:
            if weight is None:
                return triton_fast_rms_norm(x, weight, eps, fallback=original_rms_norm)
            else:
                if weight.device == torch.device("meta") or tuple(weight.shape) != (x.shape[-1],):
                    return original_rms_norm(x, weight=weight, eps=eps)
                fused_weight = weight.to(device=x.device, dtype=torch.float32)

            return fused_rmsnorm(x.float(), fused_weight, eps).to(dtype=x.dtype)
        except (AssertionError, RuntimeError):
            return original_rms_norm(x, weight=weight, eps=eps)

    fast_rms_norm._turbot2av_fast_functional_norm = True
    fast_rms_norm._turbot2av_original_rms_norm = original_rms_norm

    patched = 0
    for module_name in (
        "ltx_core.utils",
        "ltx_core.model.transformer.transformer",
        "ltx_core.text_encoders.gemma.embeddings_connector",
    ):
        module = importlib.import_module(module_name)
        current = getattr(module, "rms_norm", None)
        if current is original_rms_norm:
            module.rms_norm = fast_rms_norm
            patched += 1

    original_modulated_rms_norm = getattr(transformer_module, "modulated_rms_norm", None)
    if original_modulated_rms_norm is not None and not getattr(
        original_modulated_rms_norm, "_turbot2av_fast_modulated_rms_norm", False
    ):
        from ltx_distillation.fast_norm_kernels import fast_modulated_rms_norm

        def fast_modulated_norm(
            x: torch.Tensor,
            scale: torch.Tensor,
            shift: torch.Tensor,
            eps: float,
        ) -> torch.Tensor:
            return fast_modulated_rms_norm(x, scale, shift, eps, fallback=original_modulated_rms_norm)

        fast_modulated_norm._turbot2av_fast_modulated_rms_norm = True
        fast_modulated_norm._turbot2av_original_modulated_rms_norm = original_modulated_rms_norm
        transformer_module.modulated_rms_norm = fast_modulated_norm
        patched += 1

    original_modulate = getattr(transformer_module, "modulate", None)
    if original_modulate is not None and not getattr(original_modulate, "_turbot2av_fast_modulate", False):
        from ltx_distillation.fast_norm_kernels import fast_modulate

        def fast_modulate_fn(
            x: torch.Tensor,
            scale: torch.Tensor,
            shift: torch.Tensor,
        ) -> torch.Tensor:
            return fast_modulate(x, scale, shift, fallback=original_modulate)

        fast_modulate_fn._turbot2av_fast_modulate = True
        fast_modulate_fn._turbot2av_original_modulate = original_modulate
        transformer_module.modulate = fast_modulate_fn
        patched += 1

    original_modulate_from_ada = getattr(transformer_module, "modulate_from_ada", None)
    if original_modulate_from_ada is not None and not getattr(
        original_modulate_from_ada,
        "_turbot2av_fast_modulate_from_ada",
        False,
    ):
        from ltx_distillation.fast_norm_kernels import fast_modulate_from_ada

        def fast_modulate_from_ada_fn(
            x: torch.Tensor,
            scale_shift_table: torch.Tensor,
            timestep: torch.Tensor,
            scale_index: int,
            shift_index: int,
            num_ada_params: int,
        ) -> torch.Tensor:
            return fast_modulate_from_ada(
                x,
                scale_shift_table,
                timestep,
                scale_index,
                shift_index,
                num_ada_params,
                fallback=original_modulate_from_ada,
            )

        fast_modulate_from_ada_fn._turbot2av_fast_modulate_from_ada = True
        fast_modulate_from_ada_fn._turbot2av_original_modulate_from_ada = original_modulate_from_ada
        transformer_module.modulate_from_ada = fast_modulate_from_ada_fn
        patched += 1

    original_output_modulate = getattr(model_module, "output_modulate", None)
    if original_output_modulate is not None and not getattr(
        original_output_modulate,
        "_turbot2av_fast_output_modulate",
        False,
    ):
        from ltx_distillation.fast_norm_kernels import fast_output_modulate

        def fast_output_modulate_fn(
            x: torch.Tensor,
            scale_shift_table: torch.Tensor,
            embedded_timestep: torch.Tensor,
        ) -> torch.Tensor:
            return fast_output_modulate(
                x,
                scale_shift_table,
                embedded_timestep,
                fallback=original_output_modulate,
            )

        fast_output_modulate_fn._turbot2av_fast_output_modulate = True
        fast_output_modulate_fn._turbot2av_original_output_modulate = original_output_modulate
        model_module.output_modulate = fast_output_modulate_fn
        patched += 1

    original_gated_residual = getattr(transformer_module, "gated_residual", None)
    if original_gated_residual is not None and not getattr(
        original_gated_residual, "_turbot2av_fast_gated_residual", False
    ):
        from ltx_distillation.fast_norm_kernels import fast_gated_residual

        def fast_gated_residual_fn(
            x: torch.Tensor,
            residual: torch.Tensor,
            gate: torch.Tensor,
            mask: torch.Tensor | float = 1.0,
        ) -> torch.Tensor:
            return fast_gated_residual(x, residual, gate, mask, fallback=original_gated_residual)

        fast_gated_residual_fn._turbot2av_fast_gated_residual = True
        fast_gated_residual_fn._turbot2av_original_gated_residual = original_gated_residual
        transformer_module.gated_residual = fast_gated_residual_fn
        patched += 1

    original_modulated_rms_norm_from_ada = getattr(transformer_module, "modulated_rms_norm_from_ada", None)
    if original_modulated_rms_norm_from_ada is not None and not getattr(
        original_modulated_rms_norm_from_ada,
        "_turbot2av_fast_modulated_rms_norm_from_ada",
        False,
    ):
        from ltx_distillation.fast_norm_kernels import fast_modulated_rms_norm_from_ada

        def fast_modulated_rms_norm_from_ada_fn(
            x: torch.Tensor,
            scale_shift_table: torch.Tensor,
            timestep: torch.Tensor,
            scale_index: int,
            shift_index: int,
            num_ada_params: int,
            eps: float,
        ) -> torch.Tensor:
            return fast_modulated_rms_norm_from_ada(
                x,
                scale_shift_table,
                timestep,
                scale_index,
                shift_index,
                num_ada_params,
                eps,
                fallback=original_modulated_rms_norm_from_ada,
            )

        fast_modulated_rms_norm_from_ada_fn._turbot2av_fast_modulated_rms_norm_from_ada = True
        fast_modulated_rms_norm_from_ada_fn._turbot2av_original_modulated_rms_norm_from_ada = (
            original_modulated_rms_norm_from_ada
        )
        transformer_module.modulated_rms_norm_from_ada = fast_modulated_rms_norm_from_ada_fn
        patched += 1

    original_gated_residual_from_ada = getattr(transformer_module, "gated_residual_from_ada", None)
    if original_gated_residual_from_ada is not None and not getattr(
        original_gated_residual_from_ada,
        "_turbot2av_fast_gated_residual_from_ada",
        False,
    ):
        from ltx_distillation.fast_norm_kernels import fast_gated_residual_from_ada

        def fast_gated_residual_from_ada_fn(
            x: torch.Tensor,
            residual: torch.Tensor,
            scale_shift_table: torch.Tensor,
            timestep: torch.Tensor,
            gate_index: int,
            num_ada_params: int,
            mask: torch.Tensor | float = 1.0,
        ) -> torch.Tensor:
            return fast_gated_residual_from_ada(
                x,
                residual,
                scale_shift_table,
                timestep,
                gate_index,
                num_ada_params,
                mask,
                fallback=original_gated_residual_from_ada,
            )

        fast_gated_residual_from_ada_fn._turbot2av_fast_gated_residual_from_ada = True
        fast_gated_residual_from_ada_fn._turbot2av_original_gated_residual_from_ada = original_gated_residual_from_ada
        transformer_module.gated_residual_from_ada = fast_gated_residual_from_ada_fn
        patched += 1

    original_apply_rotary_emb = getattr(attention_module, "apply_rotary_emb", None)
    if original_apply_rotary_emb is not None and not getattr(
        original_apply_rotary_emb, "_turbot2av_fast_split_rope", False
    ):
        from ltx_distillation.fast_norm_kernels import fast_split_rotary_emb

        def fast_apply_rotary_emb(
            input_tensor: torch.Tensor,
            freqs_cis: tuple[torch.Tensor, torch.Tensor],
            rope_type,
        ) -> torch.Tensor:
            if rope_type == rope_module.LTXRopeType.SPLIT:
                return fast_split_rotary_emb(
                    input_tensor,
                    freqs_cis[0],
                    freqs_cis[1],
                    fallback=rope_module.apply_split_rotary_emb,
                )
            return original_apply_rotary_emb(input_tensor, freqs_cis, rope_type)

        fast_apply_rotary_emb._turbot2av_fast_split_rope = True
        fast_apply_rotary_emb._turbot2av_original_apply_rotary_emb = original_apply_rotary_emb
        attention_module.apply_rotary_emb = fast_apply_rotary_emb
        patched += 1
    return patched


def apply_turbodiffusion_acceleration(
    model: torch.nn.Module,
    attention_type: str = "default",
    attention_scope: str = "self",
    fast_norm: bool = False,
    quant_linear: bool = False,
    quant_linear_scope: str = "all",
    quant_linear_backend: str = "turbodiffusion",
    sla_topk: float = DEFAULT_SLA_TOPK,
    sla_topk_schedule: str | None = None,
    sla_block_q: int = 128,
    sla_block_k: int = 64,
) -> AccelerationReport:
    if attention_type not in ATTENTION_TYPES:
        raise ValueError(f"Unsupported attention_type={attention_type!r}; expected one of {ATTENTION_TYPES}")
    if quant_linear_backend not in QUANT_LINEAR_BACKENDS:
        raise ValueError(
            f"Unsupported quant_linear_backend={quant_linear_backend!r}; expected one of {QUANT_LINEAR_BACKENDS}"
        )
    replaced_attention, skipped_attention = replace_ltx_attention(
        model=model,
        attention_type=attention_type,
        attention_scope=attention_scope,
        sla_topk=sla_topk,
        sla_topk_schedule=sla_topk_schedule,
        sla_block_q=sla_block_q,
        sla_block_k=sla_block_k,
    )
    replaced_linear = (
        replace_ltx_linears(
            model,
            scope=quant_linear_scope,
            backend=quant_linear_backend,
        )
        if quant_linear
        else 0
    )
    fused_attention_projection = (
        fuse_tilelang_attention_projections(model)
        if quant_linear and quant_linear_backend == "tilelang_postscale"
        else 0
    )
    replaced_norm = replace_ltx_norms(model) if fast_norm else 0
    replaced_functional_norm = enable_fast_functional_rms_norm() if fast_norm else 0
    return AccelerationReport(
        attention_type=attention_type,
        attention_scope=attention_scope,
        replaced_attention=replaced_attention,
        skipped_attention=skipped_attention,
        replaced_linear=replaced_linear,
        fused_attention_projection=fused_attention_projection,
        quant_linear_scope=quant_linear_scope if (quant_linear or replaced_linear) else "none",
        quant_linear_backend=quant_linear_backend if (quant_linear or replaced_linear) else "none",
        sla_topk=sla_topk,
        sla_topk_schedule=sla_topk_schedule or "",
        replaced_norm=replaced_norm,
        replaced_functional_norm=replaced_functional_norm,
    )

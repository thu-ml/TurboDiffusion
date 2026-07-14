from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(
    X,
    Y,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N
    x = tl.load(X + row * N + offsets, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(x * x, axis=0) / N
    y = x * tl.rsqrt(mean_square + EPS)
    tl.store(Y + row * N + offsets, y, mask=mask)


@triton.jit
def _modulated_rms_norm_kernel(
    X,
    SCALE,
    SHIFT,
    Y,
    N: tl.constexpr,
    SCALE_ROWS: tl.constexpr,
    ROWS_PER_SCALE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N
    x = tl.load(X + row * N + offsets, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(x * x, axis=0) / N
    inv_rms = tl.rsqrt(mean_square + EPS)

    if SCALE_ROWS == 1:
        scale_row = 0
    else:
        scale_row = row // ROWS_PER_SCALE

    scale = tl.load(SCALE + scale_row * N + offsets, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(SHIFT + scale_row * N + offsets, mask=mask, other=0.0).to(tl.float32)
    y = x * inv_rms * (1.0 + scale) + shift
    tl.store(Y + row * N + offsets, y, mask=mask)


def fast_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    fallback,
) -> torch.Tensor:
    if (
        weight is not None
        or not x.is_cuda
        or x.dim() not in {2, 3}
        or not x.is_contiguous()
        or x.dtype not in {torch.float16, torch.bfloat16}
    ):
        return fallback(x, weight=weight, eps=eps)
    try:
        x_2d = x.reshape(-1, x.shape[-1])
        m, n = x_2d.shape
        y = torch.empty_like(x_2d)
        block_n = triton.next_power_of_2(n)
        num_warps = 16 if block_n >= 4096 else 8
        _rms_norm_kernel[(m,)](
            x_2d,
            y,
            n,
            eps,
            block_n,
            num_warps=num_warps,
        )
        return y.reshape_as(x)
    except (AssertionError, RuntimeError):
        return fallback(x, weight=weight, eps=eps)


@triton.jit
def _modulation_kernel(
    X,
    SCALE,
    SHIFT,
    Y,
    N: tl.constexpr,
    SCALE_ROWS: tl.constexpr,
    ROWS_PER_SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N
    x = tl.load(X + row * N + offsets, mask=mask, other=0.0).to(tl.float32)
    if SCALE_ROWS == 1:
        scale_row = 0
    else:
        scale_row = row // ROWS_PER_SCALE
    scale = tl.load(SCALE + scale_row * N + offsets, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(SHIFT + scale_row * N + offsets, mask=mask, other=0.0).to(tl.float32)
    y = x * (1.0 + scale) + shift
    tl.store(Y + row * N + offsets, y, mask=mask)


def _reshape_modulation_inputs(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
):
    if (
        not x.is_cuda
        or x.dim() not in {2, 3}
        or not x.is_contiguous()
        or x.dtype not in {torch.float16, torch.bfloat16}
        or scale.shape[-1] != x.shape[-1]
        or shift.shape[-1] != x.shape[-1]
    ):
        return None
    x_2d = x.reshape(-1, x.shape[-1])
    m, n = x_2d.shape
    scale_2d = scale.reshape(-1, n)
    shift_2d = shift.reshape(-1, n)
    scale_rows = scale_2d.shape[0]
    if shift_2d.shape[0] != scale_rows:
        return None
    if scale_rows != 1 and m % scale_rows != 0:
        return None
    if not scale_2d.is_contiguous():
        scale_2d = scale_2d.contiguous()
    if not shift_2d.is_contiguous():
        shift_2d = shift_2d.contiguous()
    rows_per_scale = m if scale_rows == 1 else m // scale_rows
    return x_2d, scale_2d, shift_2d, m, n, scale_rows, rows_per_scale


def fast_modulated_rms_norm(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    fallback,
) -> torch.Tensor:
    try:
        inputs = _reshape_modulation_inputs(x, scale, shift)
        if inputs is None:
            return fallback(x, scale, shift, eps)
        x_2d, scale_2d, shift_2d, m, n, scale_rows, rows_per_scale = inputs

        y = torch.empty_like(x_2d)
        block_n = triton.next_power_of_2(n)
        num_warps = 16 if block_n >= 4096 else 8
        _modulated_rms_norm_kernel[(m,)](
            x_2d,
            scale_2d,
            shift_2d,
            y,
            n,
            scale_rows,
            rows_per_scale,
            eps,
            block_n,
            num_warps=num_warps,
        )
        return y.reshape_as(x)
    except (AssertionError, RuntimeError):
        return fallback(x, scale, shift, eps)


def fast_modulate(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    fallback,
) -> torch.Tensor:
    try:
        inputs = _reshape_modulation_inputs(x, scale, shift)
        if inputs is None:
            return fallback(x, scale, shift)
        x_2d, scale_2d, shift_2d, m, n, scale_rows, rows_per_scale = inputs
        y = torch.empty_like(x_2d)
        block_n = triton.next_power_of_2(n)
        num_warps = 16 if block_n >= 4096 else 8
        _modulation_kernel[(m,)](
            x_2d,
            scale_2d,
            shift_2d,
            y,
            n,
            scale_rows,
            rows_per_scale,
            block_n,
            num_warps=num_warps,
        )
        return y.reshape_as(x)
    except (AssertionError, RuntimeError):
        return fallback(x, scale, shift)


@triton.jit
def _gated_residual_kernel(
    X,
    RESIDUAL,
    GATE,
    Y,
    N: tl.constexpr,
    GATE_ROWS: tl.constexpr,
    ROWS_PER_GATE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N
    x = tl.load(X + row * N + offsets, mask=mask, other=0.0)
    residual = tl.load(RESIDUAL + row * N + offsets, mask=mask, other=0.0)
    if GATE_ROWS == 1:
        gate_row = 0
    else:
        gate_row = row // ROWS_PER_GATE
    gate = tl.load(GATE + gate_row * N + offsets, mask=mask, other=0.0)
    y = x + residual * gate
    tl.store(Y + row * N + offsets, y, mask=mask)


def fast_gated_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    mask: torch.Tensor | float,
    fallback,
) -> torch.Tensor:
    if not isinstance(mask, float) or mask != 1.0:
        return fallback(x, residual, gate, mask)
    try:
        if (
            not x.is_cuda
            or x.shape != residual.shape
            or x.dim() not in {2, 3}
            or not x.is_contiguous()
            or not residual.is_contiguous()
            or x.dtype not in {torch.float16, torch.bfloat16}
            or residual.dtype != x.dtype
            or gate.shape[-1] != x.shape[-1]
        ):
            return fallback(x, residual, gate, mask)

        x_2d = x.reshape(-1, x.shape[-1])
        residual_2d = residual.reshape(-1, residual.shape[-1])
        gate_2d = gate.reshape(-1, gate.shape[-1])
        m, n = x_2d.shape
        gate_rows = gate_2d.shape[0]
        if gate_rows != 1 and m % gate_rows != 0:
            return fallback(x, residual, gate, mask)
        if not gate_2d.is_contiguous():
            gate_2d = gate_2d.contiguous()

        y = torch.empty_like(x_2d)
        block_n = triton.next_power_of_2(n)
        num_warps = 16 if block_n >= 4096 else 8
        rows_per_gate = m if gate_rows == 1 else m // gate_rows
        _gated_residual_kernel[(m,)](
            x_2d,
            residual_2d,
            gate_2d,
            y,
            n,
            gate_rows,
            rows_per_gate,
            block_n,
            num_warps=num_warps,
        )
        return y.reshape_as(x)
    except (AssertionError, RuntimeError):
        return fallback(x, residual, gate, mask)


@triton.jit
def _split_rope_kernel(
    X,
    COS,
    SIN,
    Y,
    TOTAL: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    HALF_D: tl.constexpr,
    COS_STRIDE_B: tl.constexpr,
    COS_STRIDE_H: tl.constexpr,
    COS_STRIDE_T: tl.constexpr,
    COS_STRIDE_D: tl.constexpr,
    SIN_STRIDE_B: tl.constexpr,
    SIN_STRIDE_H: tl.constexpr,
    SIN_STRIDE_T: tl.constexpr,
    SIN_STRIDE_D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < TOTAL

    pair = offsets
    d = pair % HALF_D
    tmp = pair // HALF_D
    h = tmp % H
    tmp = tmp // H
    t = tmp % T
    b = tmp // T

    head_dim = HALF_D * 2
    base = ((b * T + t) * H + h) * head_dim + d
    x1 = tl.load(X + base, mask=mask, other=0.0)
    x2 = tl.load(X + base + HALF_D, mask=mask, other=0.0)

    cos = tl.load(
        COS + b * COS_STRIDE_B + h * COS_STRIDE_H + t * COS_STRIDE_T + d * COS_STRIDE_D,
        mask=mask,
        other=0.0,
    )
    sin = tl.load(
        SIN + b * SIN_STRIDE_B + h * SIN_STRIDE_H + t * SIN_STRIDE_T + d * SIN_STRIDE_D,
        mask=mask,
        other=0.0,
    )
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    tl.store(Y + base, y1, mask=mask)
    tl.store(Y + base + HALF_D, y2, mask=mask)


def fast_split_rotary_emb(
    input_tensor: torch.Tensor,
    cos_freqs: torch.Tensor,
    sin_freqs: torch.Tensor,
    fallback,
) -> torch.Tensor:
    try:
        if (
            not input_tensor.is_cuda
            or input_tensor.dim() != 3
            or cos_freqs.dim() != 4
            or sin_freqs.dim() != 4
            or not input_tensor.is_contiguous()
            or input_tensor.dtype not in {torch.float16, torch.bfloat16}
            or cos_freqs.shape != sin_freqs.shape
        ):
            return fallback(input_tensor, cos_freqs, sin_freqs)

        b, t, inner_dim = input_tensor.shape
        cos_b, h, cos_t, half_d = cos_freqs.shape
        if cos_b != b or cos_t != t or inner_dim != h * half_d * 2:
            return fallback(input_tensor, cos_freqs, sin_freqs)

        output = torch.empty_like(input_tensor)
        total = b * t * h * half_d
        block_size = 256
        grid = (triton.cdiv(total, block_size),)
        _split_rope_kernel[grid](
            input_tensor,
            cos_freqs,
            sin_freqs,
            output,
            total,
            t,
            h,
            half_d,
            cos_freqs.stride(0),
            cos_freqs.stride(1),
            cos_freqs.stride(2),
            cos_freqs.stride(3),
            sin_freqs.stride(0),
            sin_freqs.stride(1),
            sin_freqs.stride(2),
            sin_freqs.stride(3),
            block_size,
            num_warps=8,
        )
        return output
    except (AssertionError, RuntimeError):
        return fallback(input_tensor, cos_freqs, sin_freqs)


@triton.jit
def _modulated_rms_norm_ada_kernel(
    X,
    TABLE,
    TIMESTEP,
    Y,
    N: tl.constexpr,
    NUM_ADA_PARAMS: tl.constexpr,
    SCALE_INDEX: tl.constexpr,
    SHIFT_INDEX: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N
    x = tl.load(X + row * N + offsets, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(x * x, axis=0) / N
    inv_rms = tl.rsqrt(mean_square + EPS)

    row_base = row * NUM_ADA_PARAMS * N
    scale = (
        tl.load(TABLE + SCALE_INDEX * N + offsets, mask=mask, other=0.0).to(tl.float32)
        + tl.load(TIMESTEP + row_base + SCALE_INDEX * N + offsets, mask=mask, other=0.0).to(tl.float32)
    )
    shift = (
        tl.load(TABLE + SHIFT_INDEX * N + offsets, mask=mask, other=0.0).to(tl.float32)
        + tl.load(TIMESTEP + row_base + SHIFT_INDEX * N + offsets, mask=mask, other=0.0).to(tl.float32)
    )
    y = x * inv_rms * (1.0 + scale) + shift
    tl.store(Y + row * N + offsets, y, mask=mask)


@triton.jit
def _gated_residual_ada_kernel(
    X,
    RESIDUAL,
    TABLE,
    TIMESTEP,
    Y,
    N: tl.constexpr,
    NUM_ADA_PARAMS: tl.constexpr,
    GATE_INDEX: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N
    x = tl.load(X + row * N + offsets, mask=mask, other=0.0)
    residual = tl.load(RESIDUAL + row * N + offsets, mask=mask, other=0.0)
    row_base = row * NUM_ADA_PARAMS * N
    gate = (
        tl.load(TABLE + GATE_INDEX * N + offsets, mask=mask, other=0.0).to(tl.float32)
        + tl.load(TIMESTEP + row_base + GATE_INDEX * N + offsets, mask=mask, other=0.0).to(tl.float32)
    )
    y = x + residual * gate
    tl.store(Y + row * N + offsets, y, mask=mask)


@triton.jit
def _modulate_ada_kernel(
    X,
    TABLE,
    TIMESTEP,
    Y,
    N: tl.constexpr,
    NUM_ADA_PARAMS: tl.constexpr,
    SCALE_INDEX: tl.constexpr,
    SHIFT_INDEX: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N
    x = tl.load(X + row * N + offsets, mask=mask, other=0.0).to(tl.float32)
    row_base = row * NUM_ADA_PARAMS * N
    scale = (
        tl.load(TABLE + SCALE_INDEX * N + offsets, mask=mask, other=0.0).to(tl.float32)
        + tl.load(TIMESTEP + row_base + SCALE_INDEX * N + offsets, mask=mask, other=0.0).to(tl.float32)
    )
    shift = (
        tl.load(TABLE + SHIFT_INDEX * N + offsets, mask=mask, other=0.0).to(tl.float32)
        + tl.load(TIMESTEP + row_base + SHIFT_INDEX * N + offsets, mask=mask, other=0.0).to(tl.float32)
    )
    y = x * (1.0 + scale) + shift
    tl.store(Y + row * N + offsets, y, mask=mask)


@triton.jit
def _output_modulate_kernel(
    X,
    TABLE,
    TIMESTEP,
    Y,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N
    x = tl.load(X + row * N + offsets, mask=mask, other=0.0).to(tl.float32)
    timestep = tl.load(TIMESTEP + row * N + offsets, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(TABLE + offsets, mask=mask, other=0.0).to(tl.float32) + timestep
    scale = tl.load(TABLE + N + offsets, mask=mask, other=0.0).to(tl.float32) + timestep
    y = x * (1.0 + scale) + shift
    tl.store(Y + row * N + offsets, y, mask=mask)


def _reshape_ada_inputs(
    x: torch.Tensor,
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
    num_ada_params: int,
):
    if (
        not x.is_cuda
        or x.dim() != 3
        or not x.is_contiguous()
        or x.dtype not in {torch.float16, torch.bfloat16}
        or scale_shift_table.shape[0] < num_ada_params
        or scale_shift_table.shape[-1] != x.shape[-1]
        or timestep.shape[0] != x.shape[0]
        or timestep.shape[1] != x.shape[1]
    ):
        return None
    x_2d = x.reshape(-1, x.shape[-1])
    m, n = x_2d.shape
    if timestep.numel() != m * num_ada_params * n:
        return None
    table_2d = scale_shift_table.reshape(scale_shift_table.shape[0], n)
    timestep_2d = timestep.reshape(m, num_ada_params * n)
    if not table_2d.is_contiguous():
        table_2d = table_2d.contiguous()
    if not timestep_2d.is_contiguous():
        timestep_2d = timestep_2d.contiguous()
    return x_2d, table_2d, timestep_2d, m, n


def fast_modulated_rms_norm_from_ada(
    x: torch.Tensor,
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
    scale_index: int,
    shift_index: int,
    num_ada_params: int,
    eps: float,
    fallback,
) -> torch.Tensor:
    try:
        inputs = _reshape_ada_inputs(x, scale_shift_table, timestep, num_ada_params)
        if inputs is None:
            return fallback(x, scale_shift_table, timestep, scale_index, shift_index, num_ada_params, eps)
        x_2d, table_2d, timestep_2d, m, n = inputs
        y = torch.empty_like(x_2d)
        block_n = triton.next_power_of_2(n)
        num_warps = 16 if block_n >= 4096 else 8
        _modulated_rms_norm_ada_kernel[(m,)](
            x_2d,
            table_2d,
            timestep_2d,
            y,
            n,
            num_ada_params,
            scale_index,
            shift_index,
            eps,
            block_n,
            num_warps=num_warps,
        )
        return y.reshape_as(x)
    except (AssertionError, RuntimeError):
        return fallback(x, scale_shift_table, timestep, scale_index, shift_index, num_ada_params, eps)


def fast_modulate_from_ada(
    x: torch.Tensor,
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
    scale_index: int,
    shift_index: int,
    num_ada_params: int,
    fallback,
) -> torch.Tensor:
    try:
        inputs = _reshape_ada_inputs(x, scale_shift_table, timestep, num_ada_params)
        if inputs is None:
            return fallback(x, scale_shift_table, timestep, scale_index, shift_index, num_ada_params)
        x_2d, table_2d, timestep_2d, m, n = inputs
        y = torch.empty_like(x_2d)
        block_n = triton.next_power_of_2(n)
        num_warps = 16 if block_n >= 4096 else 8
        _modulate_ada_kernel[(m,)](
            x_2d,
            table_2d,
            timestep_2d,
            y,
            n,
            num_ada_params,
            scale_index,
            shift_index,
            block_n,
            num_warps=num_warps,
        )
        return y.reshape_as(x)
    except (AssertionError, RuntimeError):
        return fallback(x, scale_shift_table, timestep, scale_index, shift_index, num_ada_params)


def fast_output_modulate(
    x: torch.Tensor,
    scale_shift_table: torch.Tensor,
    embedded_timestep: torch.Tensor,
    fallback,
) -> torch.Tensor:
    try:
        if (
            not x.is_cuda
            or x.dim() != 3
            or not x.is_contiguous()
            or x.dtype not in {torch.float16, torch.bfloat16}
            or scale_shift_table.shape != (2, x.shape[-1])
            or embedded_timestep.shape != x.shape
        ):
            return fallback(x, scale_shift_table, embedded_timestep)
        x_2d = x.reshape(-1, x.shape[-1])
        timestep_2d = embedded_timestep.reshape(-1, x.shape[-1])
        table_2d = scale_shift_table.reshape(2, x.shape[-1])
        if not timestep_2d.is_contiguous():
            timestep_2d = timestep_2d.contiguous()
        if not table_2d.is_contiguous():
            table_2d = table_2d.contiguous()
        m, n = x_2d.shape
        y = torch.empty_like(x_2d)
        block_n = triton.next_power_of_2(n)
        num_warps = 16 if block_n >= 4096 else 8
        _output_modulate_kernel[(m,)](
            x_2d,
            table_2d,
            timestep_2d,
            y,
            n,
            block_n,
            num_warps=num_warps,
        )
        return y.reshape_as(x)
    except (AssertionError, RuntimeError):
        return fallback(x, scale_shift_table, embedded_timestep)


def fast_gated_residual_from_ada(
    x: torch.Tensor,
    residual: torch.Tensor,
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
    gate_index: int,
    num_ada_params: int,
    mask: torch.Tensor | float,
    fallback,
) -> torch.Tensor:
    if not isinstance(mask, float) or mask != 1.0:
        return fallback(x, residual, scale_shift_table, timestep, gate_index, num_ada_params, mask)
    try:
        inputs = _reshape_ada_inputs(x, scale_shift_table, timestep, num_ada_params)
        if inputs is None or residual.shape != x.shape or not residual.is_contiguous() or residual.dtype != x.dtype:
            return fallback(x, residual, scale_shift_table, timestep, gate_index, num_ada_params, mask)
        x_2d, table_2d, timestep_2d, m, n = inputs
        residual_2d = residual.reshape(-1, residual.shape[-1])
        y = torch.empty_like(x_2d)
        block_n = triton.next_power_of_2(n)
        num_warps = 16 if block_n >= 4096 else 8
        _gated_residual_ada_kernel[(m,)](
            x_2d,
            residual_2d,
            table_2d,
            timestep_2d,
            y,
            n,
            num_ada_params,
            gate_index,
            block_n,
            num_warps=num_warps,
        )
        return y.reshape_as(x)
    except (AssertionError, RuntimeError):
        return fallback(x, residual, scale_shift_table, timestep, gate_index, num_ada_params, mask)

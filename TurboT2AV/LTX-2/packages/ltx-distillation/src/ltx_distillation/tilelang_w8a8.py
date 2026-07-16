from __future__ import annotations

import os

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_ROW_QUANT_WORKSPACES: dict[tuple[str, int | None, torch.dtype, int, int], tuple[torch.Tensor, torch.Tensor]] = {}
_PAD_WORKSPACES: dict[tuple[str, int | None, torch.dtype, int, int], torch.Tensor] = {}


@triton.jit
def _row_quant_kernel(
    X,
    Q,
    S,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < K
    x = tl.load(X + row * K + offsets, mask=mask, other=0.0).to(tl.float32)
    abs_x = tl.abs(x)
    abs_x = tl.where(mask, abs_x, 0.0)
    amax = tl.max(abs_x, axis=0)
    scale = tl.maximum(amax, 1.0e-4) / 127.0
    scaled = x / scale
    rounded = tl.where(scaled >= 0.0, tl.floor(scaled + 0.5), tl.ceil(scaled - 0.5))
    clipped = tl.minimum(tl.maximum(rounded, -128.0), 127.0)
    tl.store(Q + row * K + offsets, clipped.to(tl.int8), mask=mask)
    tl.store(S + row, scale)


def row_quant_int8(
    x: torch.Tensor,
    q: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.dim() != 2 or not x.is_cuda or x.dtype != torch.bfloat16:
        raise ValueError("tilelang_postscale W8A8 currently expects a 2D CUDA bfloat16 tensor")
    m, k = x.shape
    q = torch.empty((m, k), device=x.device, dtype=torch.int8) if q is None else q
    scale = torch.empty((m,), device=x.device, dtype=torch.float32) if scale is None else scale
    block_k = triton.next_power_of_2(k)
    num_warps = 16 if block_k >= 16384 else 8
    _row_quant_kernel[(m,)](x, q, scale, k, block_k, num_warps=num_warps)
    return q, scale


def _row_quant_workspace(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.shape
    key = (x.device.type, x.device.index, x.dtype, m, k)
    workspace = _ROW_QUANT_WORKSPACES.get(key)
    if workspace is None:
        workspace = (
            torch.empty((m, k), dtype=torch.int8, device=x.device),
            torch.empty((m,), dtype=torch.float32, device=x.device),
        )
        _ROW_QUANT_WORKSPACES[key] = workspace
    return workspace


def _pad_m_workspace(x: torch.Tensor, padded_m: int) -> torch.Tensor:
    _, k = x.shape
    key = (x.device.type, x.device.index, x.dtype, padded_m, k)
    workspace = _PAD_WORKSPACES.get(key)
    if workspace is None:
        workspace = torch.empty((padded_m, k), dtype=x.dtype, device=x.device)
        _PAD_WORKSPACES[key] = workspace
    return workspace


@tilelang.jit
def _tl_gemm_int8_post_scale_bias(
    M: int,
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 128,
    threads: int = 256,
    num_stages: int = 4,
    k_pack: int = 2,
):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.int8),
        B: T.Tensor((N, K), T.int8),
        C: T.Tensor((M, N), T.bfloat16),
        scales_a: T.Tensor((M,), T.float32),
        scales_b: T.Tensor((N,), T.float32),
        bias: T.Tensor((N,), T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), T.int8)
            B_shared = T.alloc_shared((block_N, block_K), T.int8)
            C_local = T.alloc_fragment((block_M, block_N), T.int32)

            T.use_swizzle(panel_size=10)
            T.clear(C_local)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True, k_pack=k_pack)

            for i, j in T.Parallel(block_M, block_N):
                C[by * block_M + i, bx * block_N + j] = (
                    C_local[i, j] * scales_a[by * block_M + i] * scales_b[bx * block_N + j] + bias[bx * block_N + j]
                )

    return main


class TileLangPostScaleInt8Linear(torch.nn.Module):
    """Dynamic A8/static W8 Linear using per-row/per-output-channel scale.

    This is intentionally separate from TurboDiffusion's blockwise W8A8 path:
    it keeps the INT8 Tensor Core accumulation continuous across K and applies
    scale once in the epilogue.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        if dtype != torch.bfloat16:
            raise ValueError("tilelang_postscale currently supports bfloat16 Linear modules only")
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("int8_weight", torch.empty((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scale", torch.empty((out_features,), dtype=torch.float32))
        self.register_buffer("bias", torch.empty(out_features, dtype=torch.bfloat16))
        self.register_buffer("fp_weight", torch.empty((out_features, in_features), dtype=torch.bfloat16))
        self._had_bias = bias
        self._kernel_cache: dict[tuple[int, int, int], object] = {}

    @staticmethod
    def _block_params() -> tuple[int, int, int, int, int, int]:
        return (
            int(os.environ.get("TURBOT2AV_TILELANG_W8A8_BLOCK_M", "128")),
            int(os.environ.get("TURBOT2AV_TILELANG_W8A8_BLOCK_N", "256")),
            int(os.environ.get("TURBOT2AV_TILELANG_W8A8_BLOCK_K", "128")),
            int(os.environ.get("TURBOT2AV_TILELANG_W8A8_THREADS", "256")),
            int(os.environ.get("TURBOT2AV_TILELANG_W8A8_STAGES", "4")),
            int(os.environ.get("TURBOT2AV_TILELANG_W8A8_K_PACK", "2")),
        )

    def _kernel(self, m: int, k: int) -> object:
        block_m, block_n, block_k, threads, num_stages, k_pack = self._block_params()
        if m % block_m != 0:
            block_m = 128
        if self.out_features % block_n != 0:
            block_n = 128
        if k % block_k != 0:
            block_k = 128
        params = (block_m, block_n, block_k, threads, num_stages, k_pack)
        key = (m, self.out_features, k, *params)
        kernel = self._kernel_cache.get(key)
        if kernel is None:
            kernel = _tl_gemm_int8_post_scale_bias(m, self.out_features, k, *params)
            self._kernel_cache[key] = kernel
        return kernel

    @staticmethod
    def _pad_m_enabled() -> bool:
        return os.environ.get("TURBOT2AV_TILELANG_W8A8_PAD_M", "").lower() in {"1", "true", "yes"}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.bfloat16 or not x.is_cuda:
            bias = self.bias if self._had_bias else None
            return F.linear(x, self.fp_weight, bias)
        shape = x.shape
        x_2d = x.reshape(-1, shape[-1])
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        m, k = x_2d.shape
        original_m = m
        use_padded_m = False
        if (
            m % 128 != 0
            and k % 128 == 0
            and self.out_features % 128 == 0
            and self._pad_m_enabled()
        ):
            padded_m = ((m + 127) // 128) * 128
            x_pad = _pad_m_workspace(x_2d, padded_m)
            x_pad[:m].copy_(x_2d)
            x_pad[m:padded_m].zero_()
            x_2d = x_pad
            m = padded_m
            use_padded_m = True
        if m % 128 != 0 or k % 128 != 0 or self.out_features % 128 != 0:
            bias = self.bias if self._had_bias else None
            return F.linear(x, self.fp_weight, bias)
        x_q_buf, x_s_buf = _row_quant_workspace(x_2d)
        x_q, x_s = row_quant_int8(x_2d, x_q_buf, x_s_buf)
        y = torch.empty((m, self.out_features), dtype=torch.bfloat16, device=x_2d.device)
        self._kernel(m, k)(x_q, self.int8_weight, y, x_s, self.scale, self.bias)
        if use_padded_m:
            y = y[:original_m]
        return y.reshape(*shape[:-1], self.out_features)

    @classmethod
    def from_linear(cls, original_linear: torch.nn.Linear) -> "TileLangPostScaleInt8Linear":
        device = original_linear.weight.device
        if device.type != "cuda":
            raise ValueError("tilelang_postscale W8A8 requires Linear weights on CUDA")
        int8_layer = cls(
            original_linear.in_features,
            original_linear.out_features,
            bias=original_linear.bias is not None,
            dtype=original_linear.weight.dtype,
        ).to(device=device)
        original_weight = original_linear.weight.detach().contiguous()
        int8_w, scale = row_quant_int8(original_weight)
        int8_layer.int8_weight.copy_(int8_w)
        int8_layer.scale.copy_(scale)
        int8_layer.fp_weight.copy_(original_weight.to(torch.bfloat16))
        if original_linear.bias is not None:
            int8_layer.bias.copy_(original_linear.bias.detach().to(torch.bfloat16))
        else:
            int8_layer.bias.zero_()
        return int8_layer

    @classmethod
    def from_tilelang_linears(
        cls,
        linears: tuple["TileLangPostScaleInt8Linear", ...],
    ) -> "TileLangPostScaleInt8Linear":
        if not linears:
            raise ValueError("Expected at least one TileLangPostScaleInt8Linear")
        in_features = linears[0].in_features
        if any(linear.in_features != in_features for linear in linears):
            raise ValueError("Fused TileLang W8A8 linears must share in_features")
        device = linears[0].int8_weight.device
        if any(linear.int8_weight.device != device for linear in linears):
            raise ValueError("Fused TileLang W8A8 linears must share a device")
        fused = cls(
            in_features,
            sum(linear.out_features for linear in linears),
            bias=any(linear._had_bias for linear in linears),
            dtype=torch.bfloat16,
        ).to(device=device)
        fused.int8_weight.copy_(torch.cat([linear.int8_weight for linear in linears], dim=0))
        fused.scale.copy_(torch.cat([linear.scale for linear in linears], dim=0))
        fused.fp_weight.copy_(torch.cat([linear.fp_weight for linear in linears], dim=0))
        bias_parts = [
            linear.bias
            if linear._had_bias
            else torch.zeros(linear.out_features, device=linear.bias.device, dtype=linear.bias.dtype)
            for linear in linears
        ]
        fused.bias.copy_(torch.cat(bias_parts, dim=0))
        return fused

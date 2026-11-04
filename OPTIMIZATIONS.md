# Inference Hot-Path Optimizations

This document describes a set of performance optimizations applied to the TurboDiffusion inference
hot path. All changes are **numerically identical** (bit-for-bit) to the previous behavior, or
control-flow-only — no accuracy/quality trade-off is introduced. They were verified on CPU
(functional equivalence tests, cache-safety tests, and static source checks); see
[Verification](#verification).

The impact estimates below are computed for **Wan2.1-1.3B @ 480p, 81 frames** (transformer
sequence length L = 32,760 tokens — the 21×60×104 latent grid after the (1,2,2) patchify —
30 blocks × 4 sampling steps ≈ 120 block evaluations per generation).

---

## Summary

| # | Change | File(s) | Waste removed per generation |
|---|--------|---------|------------------------------|
| 1 | `torch.zeros` → `torch.empty` for the Int8 GEMM output buffer | `turbodiffusion/ops/core.py` | Full memset of every quantized-linear output immediately overwritten by the GEMM (≈180 GB of writes per 480p generation) |
| 2 | Norm kernels consume/produce the activation dtype natively (no external fp32 round-trip casts) | `turbodiffusion/ops/core.py` | ~2 extra full-tensor passes per call (≈0.6 GB saved per call at 480p) × ~500 norm calls |
| 3 | Weight/bias loads in the norm Triton kernels masked | `turbodiffusion/ops/core.py` | Out-of-bounds reads (up to `next_pow2(N) − N` floats past the weight buffer) |
| 4 | `km = k.mean(...)` computed once and shared with `get_block_map` | `turbodiffusion/SLA/core.py`, `turbodiffusion/SLA/utils.py` | One duplicate ~50 MB reduction over k × 240 SageSLA calls |
| 5 | RoPE cos/sin cached across attention calls | `turbodiffusion/rcm/networks/wan2pt1.py`, `wan2pt2.py` | ~470 redundant `cos`/`sin` kernel launches + allocations per generation |
| 6 | Position-embedding generation cached per (T, H, W) across sampling steps | `turbodiffusion/rcm/networks/wan2pt1.py`, `wan2pt2.py` | Repeat/cat/rearrange embedding chain recomputed on every step |
| 7 | Per-step `.item()` GPU sync removed from both I2V sampling loops | `turbodiffusion/inference/wan2.2_i2v_infer.py`, `turbodiffusion/serve/pipeline.py` | CPU launch pipelining serialized behind GPU drain on every sampling step |
| 8 | `get_cuda_arch` cached per device index | `turbodiffusion/SLA/utils.py` | Per-call device-capability query (240 calls per generation) |

---

## Details

### 1. Int8 GEMM output buffer: `zeros` → `empty` (`turbodiffusion/ops/core.py`)

`int8_linear()` allocated its output with `torch.zeros(...)`, paying a full memset of the
`[m, n]` output before the GEMM immediately overwrote every element. Verified from the kernel
source (`turbodiffusion/ops/gemm/kernel.hpp`): the accumulator is `clear()`ed (or set to a
constant for `FastInt2Float`), the kernel never reads `Dptr`, and every output element in
`[0, m) × [0, n)` is written exactly once (predicated stores for uneven tiles; no split-K
accumulation). Zero-initialization therefore has no effect and only costs bandwidth.

For the 1.3B model at 480p, each q/k/v/o projection output is ~100 MB of bf16 and each FFN
intermediate is ~590 MB — ≈180 GB of pure memset traffic per generation, a significant fraction
of E2E time.

*Note: `can_implement()` requires `k % BlockSize == 0` and `(n · sizeof(dtype)) % 16 == 0`,
which every Wan block linear satisfies (`k, n ∈ {1536, 5120, 8960, 13824}`); the wrapped layers
are unchanged.*

### 2. Native-dtype norm kernels (`turbodiffusion/ops/core.py`)

`FastRMSNorm` / `FastLayerNorm` previously forced the whole activation through fp32 twice:

```python
# before
return rmsnorm(x.float(), self.weight, self.eps).to(x.dtype)
```

The Triton kernels already upcast on load (`tl.load(...).to(tl.float32)`) and round once on
store (`y.to(Y.type.element_ty)`), so the external casts were pure memory traffic — roughly
0.8 GB of traffic per call instead of 0.2 GB at 480p, across ~500 calls per generation
(≈300 GB of redundant traffic per generation)
(`norm_q`/`norm_k`, `norm1`/`norm2`/`norm3` and the head norm, per block, per step):

```python
# after
return rmsnorm(x, self.weight, self.eps)
```

The output allocation in `layernorm_param` / `layernorm_noparam` changed from a hardcoded
fp32 `torch.empty_like(x, dtype=torch.float32)` to `torch.empty_like(x)`; the store path
(`x_hat.to(Y.type.element_ty)`) already handled any dtype. Because bf16 → fp32 is exact and
there is exactly one rounding to bf16 in both variants, results are **bit-identical**.

### 3. Masked weight/bias loads (hardening, `turbodiffusion/ops/core.py`)

`_rms_norm_fwd_fused` and `_layer_norm_param_fwd_fused` loaded `w`/`b` unmasked over
`cols = tl.arange(0, N2)` where `N2 = next_power_of_2(N)`. For non-power-of-2 dims (e.g.
N = 1536 → N2 = 2048) this reads up to 512 floats past the end of the weight buffer — it only
avoided faulting because of CUDA allocator padding. The loads are now masked exactly like the
adjacent `x` loads; masked lanes are never stored, so outputs are unchanged.

### 4. Shared `km` reduction in SageSLA (`turbodiffusion/SLA/core.py`, `SLA/utils.py`)

`SageSparseLinearAttention.forward` computed `km = k.mean(dim=-2, keepdim=True)` for the
int8 quantization, while `get_block_map` internally computed the *identical* reduction for the
smooth-k technique — two full passes over `k` per attention call. `get_block_map` now accepts
an optional `km=` argument (default behavior unchanged for existing callers, e.g. plain SLA),
and SageSLA computes `km` once and passes it to both consumers. In the bf16 pipeline the
intervening `.to(self.dtype)` casts are no-ops, so the shared `km` is bit-identical.

### 5. RoPE cos/sin cache (`wan2pt1.py`, `wan2pt2.py`)

`rope_apply` recomputed `torch.cos(freqs)` / `torch.sin(freqs)` on *every* attention call —
2 calls × 30 blocks × 4 steps = 240 recomputations of identical values per generation.
The new `_rope_cos_sin()` helper caches the pair keyed by the freqs tensor's
`(data_ptr, shape, device, dtype)`, with two safety properties:

- **Stale-hit impossibility:** each entry holds a `weakref` to the freqs tensor and the cached
  values are only returned when `ref() is freqs`; if a new tensor is allocated at a recycled
  address, the identity check fails and values are recomputed.
- **Bounded memory:** at most 4 entries; dead (weakref-cleared) entries are purged on insert.

The `freqs.view(seq_len, head_dim // 2)` is shape-preserving (it either matches or raises,
exactly as before), so computing cos/sin before the view is equivalent.

### 6. Position-embedding cache (`wan2pt1.py`, `wan2pt2.py`)

`WanModel.forward` called `rope_position_embedding.generate_embeddings(...)` on every forward —
i.e., once per sampling step — although the result depends only on `(T, H, W)`. The new
`WanModel._get_rope_freqs()` caches the generated tensor per `(T, H, W, device)` (bounded at 4
entries, so changing resolution or `num_frames` doesn't thrash). Consumers are read-only:
`rope_apply` only views it, and `split_inputs_cp` (context parallelism) copies via
view + `index_select`, so sharing the tensor across steps is safe. Under CP, each rank shards
the same cached tensor as before.

### 7. No per-step GPU sync in the I2V loops (`wan2.2_i2v_infer.py`, `serve/pipeline.py`)

The high→low noise model switch was decided with `t_cur.item()` inside the sampling loop — a
GPU→CPU sync at the top of every step that drains the GPU before the CPU can enqueue the next
step's kernels. The timestep values are now read once (`t_steps[:-1].tolist()`) before the loop
starts, while the GPU is still idle, and the boundary check compares Python floats:

```python
t_cur_values = t_steps[:-1].tolist()   # single sync up front
...
if t_cur_values[i] < args.boundary and not switched:   # no sync in the loop
```

Switch semantics are unchanged (same fp64 values, same comparison).

### 8. `get_cuda_arch` cache (`SLA/utils.py`)

`torch.cuda.get_device_capability()` was queried on every SageSLA forward (240×/generation).
The compute capability of a device never changes at runtime, so the result is now cached per
device index.

---

## Verification

Performed on CPU (the development machine's GPUs were occupied by an unrelated job):

- **Byte-compilation** of all modified files.
- **Functional tests against the real source** (functions extracted from the shipped files via
  AST and executed):
  - rope cache: hit path returns the identical cached tensors; cos computed exactly once;
    new tensors get fresh correct values; a tensor allocated at a *recycled address* never
    receives stale values; cache stays bounded with dead-entry purge.
  - position-embedding cache: no regeneration on repeated `(T, H, W)`; same tensor object
    returned; bounded at 4 entries; correct after eviction.
  - `get_block_map(km=...)`: outputs (`sparse_map`, `lut`, `real_topk`) bit-identical to the
    internal-mean version for both block configurations (64/128 and 128/64), default path
    unchanged.
  - norm plumbing: old (external fp32 casts) vs new (native bf16 out) paths bit-identical under
    a torch emulation of the Triton kernel math.
- **Static checks:** no `.item()` calls remain in either I2V sampling loop (AST-scanned);
  `int8_linear` allocates with `empty`; SageSLA passes `km=km` in both arch branches and
  computes the mean exactly once; rope/freqs caches wired in both network files.

**Not yet done: on-GPU benchmarking.** Suggested command once a GPU is available:

```bash
export PYTHONPATH=turbodiffusion
python turbodiffusion/inference/wan2.1_t2v_infer.py \
    --model Wan2.1-1.3B \
    --dit_path checkpoints/TurboWan2.1-T2V-1.3B-480P-quant.pth \
    --resolution 480p \
    --prompt "..." --num_samples 1 --num_steps 4 \
    --quant_linear --attention_type sagesla --sla_topk 0.1
```

Compare end-to-end sampling time (the tqdm "Sampling" wall time) before/after this change; the
outputs should be identical given the same seed, since no numerics changed.

---

## Deferred follow-up

Self-attention projects q/k/v with three separate `Int8Linear` layers that each re-quantize the
same activation (`int8_quant(x)` runs 3× per block). Fusing them into a single QKV linear would
remove 2 of the 3 quantization passes per block per step, but requires checkpoint key remapping
at load time (`q/k/v.weight → qkv.weight`). At 480p the duplicate-quant traffic is only ≈30 GB per
generation (≈1–2 % of E2E), so this is a modest win for its complexity.

---

## Remaining opportunities (not implemented)

Ranked by estimated impact at 480p; all need on-GPU validation before shipping.

1. **Fuse the fp32 modulation chains (or opt-in `torch.compile`).**
   `WanAttentionBlock.forward` computes `(self.norm1(x).float() * (1 + e[1]) + e[0]).type_as(x)`
   and the FFN equivalent as four separate eager kernels in fp32 over the full activation, and
   `Head` repeats the pattern (`self.head(self.norm(x) * (1 + e[1]) + e[0])`): ≈1.4 GB of traffic
   per site vs ≈0.2 GB if fused — ≈340 GB per generation at 480p, roughly 5–10 % of E2E.
   `torch.compile` fuses these chains automatically: the custom Int8-GEMM and SLA Triton kernels
   graph-break around them, but the pointwise chains in between still compile, and
   `torch._dynamo.config.suppress_errors = True` is already set in the inference scripts, so an
   opt-in `--torch_compile` flag degrades to eager on failure. A hand-written fused Triton kernel
   is the alternative, but its fma contraction rounds ±1 ulp differently from eager — not
   bit-identical, so it belongs in a separate change.
2. **CUDA graphs** (`torch.compile(mode="reduce-overhead")` or manual capture): the token count is
   static across the sampling steps, so the step is capturable in principle; the win is kernel
   launch overhead on the small 1.3B model. Needs capture-safe (pool-allocated) buffers in the
   quant/SLA path.
3. **I2V: skip the model offload ping-pong when VRAM allows.** The high→low noise switch does
   `model.cpu()` + `empty_cache()` + reload; on an 80 GB H100 both int8-quantized 14B models fit
   in VRAM at once, so guarding the offload with `torch.cuda.mem_get_info()` saves a ~2×14 GB
   PCIe round-trip (≈5–10 s) per I2V generation; on 24–32 GB cards the guard keeps the offload.
4. **Cross-attention K/V caching across steps.** The text/image context is constant across the
   sampling steps, so each block's cross-attn k/v (+ their int8 quants, ~6 kernels/block) could be
   computed once and reused (weakref-keyed like the rope cache). Only a few ms per generation —
   low priority.

Checked and already fine: the T5 text encoder runs in bf16; VAE decode is chunked with a feature
cache; no `.item()`/`.cpu()`/`synchronize` calls remain on the per-step path (AST-scanned).

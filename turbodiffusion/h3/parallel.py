from __future__ import annotations

"""Ulysses sequence-parallel state and head/sequence all-to-all.

The default path is native NCCL ``all_to_all_single``. Set ``H3_USE_NCCL_A2A=0``
to use ``all_gather`` (+ local slice) as a compatibility fallback with higher
communication cost. NCCL is not bundled here; see ``docs/h3_i2va.md`` for the
tested RTX 5090 PyTorch/NCCL combination and runtime library verification.

Two knobs cut the wire cost, both off unless enabled in ``dit.*`` config:

``use_quant_comm``
    Payload becomes symmetric INT8 plus one FP32 scale per (token, head), i.e.
    ~52% of the bf16 bytes at head_dim=128. This communication path is lossy.
``use_async_comm``
    Q/K/V all-to-alls are issued non-blocking so they pipeline with each other
    and overlap the next block's weight H2D.
"""

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class ParallelState:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    ulysses_degree: int = 1
    group: dist.ProcessGroup | None = None
    use_quant_comm: bool = False  # INT8 payload on the a2a wire (all_gather path too)
    use_async_comm: bool = False  # issue Q/K/V a2a non-blocking, overlap with H2D

    @property
    def sp_rank(self) -> int:
        return self.rank % max(self.ulysses_degree, 1)

    @property
    def sp_size(self) -> int:
        return self.ulysses_degree


_STATE: ParallelState | None = None


def get_parallel_state() -> ParallelState:
    global _STATE
    if _STATE is None:
        _STATE = ParallelState()
    return _STATE


def init_parallel(
    ulysses_degree: int = 1,
    use_quant_comm: bool = False,
    use_async_comm: bool = False,
    num_heads: int = 56,
) -> ParallelState:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if ulysses_degree < 1:
        raise ValueError("ulysses_degree must be >= 1")
    if world_size % ulysses_degree != 0:
        raise ValueError(f"world_size={world_size} not divisible by ulysses_degree={ulysses_degree}")
    if num_heads % ulysses_degree != 0:
        raise ValueError(f"num_heads={num_heads} not divisible by ulysses_degree={ulysses_degree}")

    if world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        # set_device only records the current device; it creates no context, so
        # without device_id NCCL cannot tell which GPU this rank owns and
        # bootstraps on cuda:0 — leaving a ~500MB context there on every rank.
        dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    group = None
    if ulysses_degree > 1 and world_size > 1:
        if ulysses_degree == world_size:
            group = dist.group.WORLD
        else:
            groups: list[dist.ProcessGroup] = []
            for start in range(0, world_size, ulysses_degree):
                ranks = list(range(start, start + ulysses_degree))
                groups.append(dist.new_group(ranks=ranks, backend="nccl"))
            group = groups[rank // ulysses_degree]

    state = ParallelState(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        ulysses_degree=ulysses_degree,
        group=group,
        use_quant_comm=use_quant_comm,
        use_async_comm=use_async_comm,
    )
    global _STATE
    _STATE = state
    if rank == 0:
        mode = "nccl_a2a" if os.environ.get("H3_USE_NCCL_A2A", "1") != "0" else "all_gather"
        print(
            f"[parallel] world={world_size} ulysses={ulysses_degree} "
            f"quant_comm={use_quant_comm} async_comm={use_async_comm} mode={mode}"
        )
    return state


def _all_gather_cat(x: torch.Tensor, group: dist.ProcessGroup, dim: int = 0) -> torch.Tensor:
    sp = dist.get_world_size(group)
    if get_parallel_state().use_quant_comm:
        # Same per-(token, head) quantizer as the a2a path, so both routes carry
        # the same error characteristics.
        q, s = _quantize_groups(x)
        qs = [torch.empty_like(q) for _ in range(sp)]
        ss = [torch.empty_like(s) for _ in range(sp)]
        dist.all_gather(qs, q.contiguous(), group=group)
        dist.all_gather(ss, s.contiguous(), group=group)
        parts = [
            _dequantize_groups(qs[i], ss[i], dtype=x.dtype) for i in range(sp)
        ]
        return torch.cat(parts, dim=dim)
    parts = [torch.empty_like(x) for _ in range(sp)]
    dist.all_gather(parts, x.contiguous(), group=group)
    return torch.cat(parts, dim=dim)


def _quantize_2d(flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amax = flat.abs().amax(dim=-1, keepdim=True).float()
    inv = torch.where(amax > 0, 127.0 / amax, torch.zeros_like(amax))
    q = (flat.float() * inv).round().clamp(-127, 127).to(torch.int8)
    return q, (amax / 127.0).squeeze(-1)


def _dequantize_2d(
    q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    return (q.to(torch.float32) * scale.unsqueeze(-1)).to(dtype)


# Eager quant/dequant move ~10x the bytes they need to (fp32 temporaries the size
# of the payload), which alone costs more than the comm it saves. Fusing them into
# one kernel each is what makes INT8 comm a win.
_FUSED: dict[str, object] = {}
_FUSE_OK = os.environ.get("H3_COMPILE_QUANT_COMM", "1") != "0"


def _fused(name: str, fn):
    global _FUSE_OK
    if not _FUSE_OK:
        return fn
    got = _FUSED.get(name)
    if got is None:
        try:
            got = torch.compile(fn, dynamic=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[parallel] quant-comm fusion unavailable ({exc}); using eager")
            _FUSE_OK = False
            return fn
        _FUSED[name] = got
    return got


def _quantize_groups(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric INT8 quantize with one scale per trailing-dim group (a head).

    Returns ``(q, scale)`` where ``q`` matches ``x``'s shape and ``scale`` drops
    the trailing dim. The stored scale already includes the ``/127``, so dequant
    is a plain multiply. ``amax == 0`` rows yield ``q = 0`` and ``scale = 0``.
    """
    global _FUSE_OK
    flat = x.reshape(-1, x.shape[-1])
    try:
        q, s = _fused("quant", _quantize_2d)(flat)
    except Exception as exc:  # noqa: BLE001
        print(f"[parallel] quant-comm fusion failed ({exc}); using eager")
        _FUSE_OK = False
        q, s = _quantize_2d(flat)
    return q.view_as(x), s.view(x.shape[:-1])


def _dequantize_groups(
    q: torch.Tensor, scale: torch.Tensor, *, dtype: torch.dtype
) -> torch.Tensor:
    global _FUSE_OK
    d = q.shape[-1]
    flat = q.reshape(-1, d)
    s = scale.reshape(-1)
    try:
        out = _fused("dequant", _dequantize_2d)(flat, s, dtype)
    except Exception as exc:  # noqa: BLE001
        print(f"[parallel] dequant-comm fusion failed ({exc}); using eager")
        _FUSE_OK = False
        out = _dequantize_2d(flat, s, dtype)
    return out.view_as(q)


class _PendingA2A:
    """In-flight all-to-all: holds the NCCL work handles and their buffers.

    ``wait()`` blocks on the collective(s) and returns the received tensor in the
    original dtype/shape (dequantizing first when the payload was INT8). Send
    buffers are pinned here until then, since NCCL reads them asynchronously.
    """

    __slots__ = ("_works", "_recv", "_recv_scale", "_shape", "_dtype", "_keep", "_done")

    def __init__(self, works, recv, recv_scale, shape, dtype, keep):
        self._works = works
        self._recv = recv
        self._recv_scale = recv_scale
        self._shape = shape
        self._dtype = dtype
        self._keep = keep
        self._done = False

    def wait(self) -> torch.Tensor:
        if self._done:
            raise RuntimeError("_PendingA2A already awaited")
        for w in self._works:
            if w is not None:
                w.wait()
        self._done = True
        if self._recv_scale is None:
            out = self._recv.view(self._shape)
        else:
            d = self._shape[-1]
            out = _dequantize_groups(
                self._recv.view(-1, d), self._recv_scale.reshape(-1), dtype=self._dtype
            ).view(self._shape)
        self._keep = None
        return out


def _a2a_launch(
    send: torch.Tensor, group: dist.ProcessGroup, *, quantize: bool
) -> _PendingA2A:
    """Issue an async all-to-all over ``send``'s dim 0 (equal split per rank).

    With ``quantize`` the payload becomes INT8 data plus FP32 per-head scales,
    which is ~52% of the bf16 bytes at head_dim=128. Everything outside the wire
    stays bf16.
    """
    sp = dist.get_world_size(group)
    send = send.contiguous()
    if quantize and send.dtype in (torch.bfloat16, torch.float16, torch.float32):
        q, s = _quantize_groups(send)
        send_q = q.view(sp, -1)
        send_s = s.view(sp, -1).contiguous()
        recv_q = torch.empty_like(send_q)
        recv_s = torch.empty_like(send_s)
        w1 = dist.all_to_all_single(recv_q, send_q, group=group, async_op=True)
        w2 = dist.all_to_all_single(recv_s, send_s, group=group, async_op=True)
        return _PendingA2A(
            [w1, w2], recv_q, recv_s, send.shape, send.dtype, (send_q, send_s)
        )
    recv = torch.empty_like(send)
    w = dist.all_to_all_single(recv, send, group=group, async_op=True)
    return _PendingA2A([w], recv, None, send.shape, send.dtype, (send,))


def _a2a_single(
    send: torch.Tensor, group: dist.ProcessGroup, *, quantize: bool = False
) -> torch.Tensor:
    return _a2a_launch(send, group, quantize=quantize).wait()


class _PendingPreAttn:
    """Q/K/V scatters in flight together; ``wait`` returns them in launch order."""

    __slots__ = ("_pending", "_ready", "_shape")

    def __init__(self, pending, ready, shape):
        self._pending = pending
        self._ready = ready
        self._shape = shape

    def wait(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._ready is not None:
            return self._ready
        return tuple(p.wait().view(self._shape) for p in self._pending)  # type: ignore[return-value]


def ulysses_pre_attn_launch(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> _PendingPreAttn:
    """Start the Q/K/V head scatters without blocking; pair with ``_wait``.

    All three all-to-alls are issued back to back so they pipeline on the NCCL
    stream and the caller can run unrelated work (e.g. the next block's weight
    H2D) before consuming the results.
    """
    state = get_parallel_state()
    if (
        state.group is None
        or state.ulysses_degree <= 1
        or os.environ.get("H3_USE_NCCL_A2A", "1") == "0"
    ):
        return _PendingPreAttn(None, ulysses_pre_attn(q, k, v), None)

    group = state.group
    sp = dist.get_world_size(group)
    t_local, h, d = q.shape
    if h % sp != 0:
        raise ValueError(f"heads {h} not divisible by ulysses {sp}")
    h_local = h // sp
    quantize = state.use_quant_comm
    pending = [
        _a2a_launch(
            x.view(t_local, sp, h_local, d).permute(1, 0, 2, 3),
            group,
            quantize=quantize,
        )
        for x in (q, k, v)
    ]
    return _PendingPreAttn(pending, None, (sp * t_local, h_local, d))


def ulysses_pre_attn_wait(
    pending: _PendingPreAttn,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return pending.wait()


def ulysses_pre_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """[T_local, H, D] -> [T, H_local, D]."""
    state = get_parallel_state()
    if state.group is None or state.ulysses_degree <= 1:
        return q, k, v

    group = state.group
    sp = dist.get_world_size(group)
    sp_rank = dist.get_rank(group)

    def _one(x: torch.Tensor) -> torch.Tensor:
        t_local, h, d = x.shape
        if h % sp != 0:
            raise ValueError(f"heads {h} not divisible by ulysses {sp}")
        h_local = h // sp
        if os.environ.get("H3_USE_NCCL_A2A", "1") != "0":
            send = x.view(t_local, sp, h_local, d).permute(1, 0, 2, 3).contiguous()
            recv = _a2a_single(send, group, quantize=state.use_quant_comm)
            return recv.contiguous().view(sp * t_local, h_local, d)
        # fallback: all_gather full sequence, then take local head shard
        full = _all_gather_cat(x, group, dim=0)  # [T, H, D]
        return full[:, sp_rank * h_local : (sp_rank + 1) * h_local, :].contiguous()

    return _one(q), _one(k), _one(v)


def ulysses_post_attn(x: torch.Tensor) -> torch.Tensor:
    """[T, H_local, D] -> [T_local, H, D]."""
    state = get_parallel_state()
    if state.group is None or state.ulysses_degree <= 1:
        return x

    group = state.group
    sp = dist.get_world_size(group)
    sp_rank = dist.get_rank(group)
    t, h_local, d = x.shape
    if t % sp != 0:
        raise ValueError(f"seq_len {t} not divisible by ulysses {sp}")
    t_local = t // sp

    if os.environ.get("H3_USE_NCCL_A2A", "1") != "0":
        send = x.view(sp, t_local, h_local, d)
        recv = _a2a_single(send, group, quantize=state.use_quant_comm)
        return recv.permute(1, 0, 2, 3).contiguous().view(t_local, sp * h_local, d)

    # fallback all_gather head shards
    full = _all_gather_cat(x, group, dim=1)  # [T, H, D]
    return full[sp_rank * t_local : (sp_rank + 1) * t_local].contiguous()


def shard_sequence(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    state = get_parallel_state()
    if state.ulysses_degree <= 1:
        return x
    sp = state.ulysses_degree
    if x.shape[dim] % sp != 0:
        raise ValueError(f"dim{dim} size {x.shape[dim]} not divisible by ulysses {sp}")
    chunk = x.shape[dim] // sp
    start = state.sp_rank * chunk
    return x.narrow(dim, start, chunk)

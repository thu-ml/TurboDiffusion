from __future__ import annotations

"""Vidu-style async block offload with double GPU buffers.

Source (offloaded) blocks stay on pinned CPU. Two resident GPU buffer modules
receive async H2D weight copies on a side stream while the other buffer
computes — overlapping PCIe with attention/MLP.
"""

from copy import deepcopy

import torch
import torch.nn as nn


def build_preload_map(num_blocks: int, offload_interlace: int) -> list[bool]:
    """True entries keep blocks resident on the GPU.

    - interlace == 0: first + last on GPU, middle offloaded
    - interlace  > 0: every (interlace+1)-th block on GPU
    - interlace  < 0: inverted pattern of abs(interlace)
    """
    if num_blocks <= 0:
        return []
    if offload_interlace == 0:
        preload = [True] + [False] * (num_blocks - 1)
        preload[-1] = True
        return preload
    if offload_interlace > 0:
        return [not bool(i % (offload_interlace + 1)) for i in range(num_blocks)]
    k = abs(offload_interlace)
    return [bool((i + 1) % (k + 1)) for i in range(num_blocks)]


def pin_block_cpu(block: nn.Module) -> None:
    """Keep block on CPU with pinned parameter/buffer storage for fast H2D."""
    block.to("cpu")
    for p in block.parameters(recurse=True):
        if p.device.type == "cpu" and not p.is_pinned():
            p.data = p.data.pin_memory()
    for b in block.buffers(recurse=True):
        if b.device.type == "cpu" and not b.is_pinned():
            b.data = b.data.pin_memory()


def copy_module_tensors(src: nn.Module, dst: nn.Module, *, non_blocking: bool = True) -> None:
    """Copy all parameters + buffers src→dst (CPU pin → GPU buf).

    INT8 weights live in ``named_buffers`` (not Parameters) — must copy
    integer buffers too, not only floating-point ones.
    """
    src_params = dict(src.named_parameters(recurse=True))
    dst_params = dict(dst.named_parameters(recurse=True))
    missing = [k for k in src_params if k not in dst_params]
    if missing:
        raise RuntimeError(f"offload buffer missing params: {missing[:5]}...")
    for k, sp in src_params.items():
        dst_params[k].data.copy_(sp.data, non_blocking=non_blocking)

    src_bufs = dict(src.named_buffers(recurse=True))
    dst_bufs = dict(dst.named_buffers(recurse=True))
    for k, sb in src_bufs.items():
        if k not in dst_bufs:
            continue
        db = dst_bufs[k]
        if db.shape != sb.shape:
            raise RuntimeError(f"offload buffer shape mismatch {k}: {tuple(db.shape)} vs {tuple(sb.shape)}")
        if db.dtype != sb.dtype:
            # e.g. unexpected cast — still try via .to
            db.data.copy_(sb.data.to(dtype=db.dtype), non_blocking=non_blocking)
        else:
            db.data.copy_(sb.data, non_blocking=non_blocking)


class OffloadHandle:
    """Deferred async H2D, launched mid-block before attention."""

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.event = torch.cuda.Event()
        self.kwargs = {**kwargs, "prev_event": self.event}

    def record(self) -> None:
        self.event.record(stream=torch.cuda.current_stream())

    def run(self):
        return self.func(*self.args, **self.kwargs)


class BlockOffloader:
    """Double-buffer asynchronous block offload controller."""

    def __init__(
        self,
        device: torch.device,
        preload_map: list[bool],
        *,
        buf_length: int = 2,
        cpu_only_submodule: str | None = None,
    ):
        self.device = device
        self.cpu_only_submodule = cpu_only_submodule
        self.preload_map = list(preload_map)
        self.offload_flags = [not p for p in self.preload_map]
        self.buf_length = int(buf_length)
        self.stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self.events = [torch.cuda.Event() for _ in range(self.buf_length)] if self.stream else []
        self.bufs: nn.ModuleList | None = None
        self.buf_block_ids: list[int] = [-1] * self.buf_length
        self.buf_id = 0
        self._blocks: nn.ModuleList | None = None

    @property
    def num_offloaded(self) -> int:
        return sum(1 for f in self.offload_flags if f)

    def pattern_str(self) -> str:
        return "".join("o" if p else "x" for p in self.preload_map)

    def install(self, blocks: nn.ModuleList) -> None:
        assert len(blocks) == len(self.preload_map)
        self._blocks = blocks
        for i, blk in enumerate(blocks):
            if self.preload_map[i]:
                blk.to(self.device)
                # Weights that get baked out later never need to reach the GPU here.
                if self.cpu_only_submodule:
                    sub = getattr(blk, self.cpu_only_submodule, None)
                    if sub is not None:
                        sub.to("cpu")
            else:
                pin_block_cpu(blk)

        if self.num_offloaded == 0 or self.device.type != "cuda":
            self.bufs = None
            return

        # GPU double buffers share the structure of a real block, including INT8 buffers.
        template_idx = next(i for i, f in enumerate(self.offload_flags) if f)
        template = blocks[template_idx]
        bufs = []
        for _ in range(self.buf_length):
            b = deepcopy(template).to(self.device)
            b.block_offload_id = -1
            bufs.append(b)
        self.bufs = nn.ModuleList(bufs)
        self.buf_block_ids = [-1] * self.buf_length
        self.buf_id = 0
        # Primed events so the first wait is a no-op
        for ev in self.events:
            ev.record(torch.cuda.current_stream(self.device))

    def release(self) -> None:
        """Drop the GPU buffers and the pinned-CPU block references.

        The offloader is what keeps the middle blocks pinned, so a caller that
        releases the DiT without this gets the weights back but not the ~40 GB of
        page-locked host memory.
        """
        if self.stream is not None:
            self.stream.synchronize()
        self.bufs = None
        self._blocks = None
        self.events = []
        self.stream = None
        self.buf_block_ids = [-1] * self.buf_length

    def _onload_async(
        self,
        src_block: nn.Module,
        dst_block: nn.Module,
        *,
        blk_id: int,
        buf_slot: int,
        prev_event: torch.cuda.Event | None = None,
    ):
        assert self.stream is not None and self.bufs is not None
        with torch.cuda.stream(self.stream):
            if prev_event is not None:
                self.stream.wait_event(prev_event)
            copy_module_tensors(src_block, dst_block, non_blocking=True)
            dst_block.block_offload_id = blk_id
            self.buf_block_ids[buf_slot] = blk_id
            self.events[buf_slot].record(self.stream)
        return self.stream, self.events[buf_slot]

    def _onload_sync(self, src_block: nn.Module, dst_block: nn.Module, *, blk_id: int, buf_slot: int) -> None:
        copy_module_tensors(src_block, dst_block, non_blocking=False)
        dst_block.block_offload_id = blk_id
        self.buf_block_ids[buf_slot] = blk_id
        if self.stream is not None:
            self.events[buf_slot].record(torch.cuda.current_stream(self.device))

    def begin_step(self) -> None:
        """If layer-0 is offloaded, sync-fill the current GPU buf before the loop."""
        if self.bufs is None or self._blocks is None or self.stream is None:
            return
        if self.offload_flags[0] and self.buf_block_ids[self.buf_id] != 0:
            self._onload_sync(self._blocks[0], self.bufs[self.buf_id], blk_id=0, buf_slot=self.buf_id)
            torch.cuda.current_stream(self.device).synchronize()

    def prepare(self, blk_id: int) -> None:
        """Sync-load ``blk_id`` into the current GPU buf (DBCache Fn→Bn jump)."""
        if (
            self.bufs is None
            or self._blocks is None
            or self.stream is None
            or not self.offload_flags[blk_id]
            or self.buf_block_ids[self.buf_id] == blk_id
        ):
            return
        self._onload_sync(
            self._blocks[blk_id], self.bufs[self.buf_id], blk_id=blk_id, buf_slot=self.buf_id
        )
        torch.cuda.current_stream(self.device).synchronize()

    def drop_block_submodule(self, name: str) -> None:
        """Drop a submodule from the GPU buffers after its weights were baked out.

        Source blocks lose the submodule first; the buffers must follow or they
        keep holding (and re-copying into) weights nobody reads anymore.
        """
        if self.bufs is None:
            return
        for buf in self.bufs:
            if getattr(buf, name, None) is not None:
                setattr(buf, name, None)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def acquire(
        self, blk_id: int, *, prefetch_id: int | None = -1
    ) -> tuple[nn.Module, OffloadHandle | None]:
        """Return compute module + optional deferred prefetch handle.

        ``prefetch_id``: ``-1`` → ``blk_id+1`` (wrap); ``None`` → no prefetch;
        ``>=0`` → that block id.
        """
        assert self._blocks is not None
        blocks = self._blocks
        n = len(blocks)

        if self.bufs is None or self.stream is None:
            return blocks[blk_id], None

        buf_id = self.buf_id
        buf_next = (buf_id + 1) % self.buf_length
        torch.cuda.current_stream(self.device).wait_event(self.events[buf_id])

        if self.offload_flags[blk_id]:
            if self.buf_block_ids[buf_id] != blk_id:
                self._onload_sync(
                    blocks[blk_id], self.bufs[buf_id], blk_id=blk_id, buf_slot=buf_id
                )
                torch.cuda.current_stream(self.device).synchronize()
            assert self.buf_block_ids[buf_id] == blk_id, (
                f"buf[{buf_id}] has block {self.buf_block_ids[buf_id]}, expected {blk_id}"
            )
            compute = self.bufs[buf_id]
        else:
            compute = blocks[blk_id]

        if prefetch_id is None:
            next_id = None
        elif prefetch_id < 0:
            next_id = (blk_id + 1) % n
        else:
            next_id = int(prefetch_id)

        handle = None
        if next_id is not None and self.offload_flags[next_id]:
            handle = OffloadHandle(
                self._onload_async,
                blocks[next_id],
                self.bufs[buf_next],
                blk_id=next_id,
                buf_slot=buf_next,
            )
            self.buf_id = buf_next
        return compute, handle

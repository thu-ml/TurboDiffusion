from __future__ import annotations

"""DBCache (Cache-DiT Dual Block Cache) for MiniMax-H3 DiT.

Skips middle transformer blocks when the Fn residual is close to the previous
step. All ranks all-reduce the decision so Ulysses stays synchronized.
"""

from dataclasses import dataclass, field, fields

import torch
import torch.distributed as dist

from turbodiffusion.h3.parallel import get_parallel_state


def rel_l1_diff(curr: torch.Tensor, prev: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    num = (curr.float() - prev.float()).abs().mean()
    den = prev.float().abs().mean().clamp_min(eps)
    return num / den


@dataclass
class DBCacheConfig:
    enabled: bool = False
    probe_only: bool = False  # record diffs, never skip
    Fn_compute_blocks: int = 8
    Bn_compute_blocks: int = 0
    max_warmup_steps: int = 8
    max_continuous_cached_steps: int = -1  # -1 = unlimited
    residual_diff_threshold: float = 0.08
    enable_taylorseer: bool = False
    taylorseer_order: int = 1

    @classmethod
    def from_dict(cls, raw: dict | None) -> DBCacheConfig:
        if not raw:
            return cls()
        known = {f.name for f in fields(cls)}
        kwargs = {}
        for k, v in raw.items():
            if k not in known:
                continue
            default = getattr(cls, k)
            kwargs[k] = type(default)(v) if default is not None else v
        return cls(**kwargs)

    def validate(self, num_layers: int) -> None:
        if self.Fn_compute_blocks < 1:
            raise ValueError("Fn_compute_blocks must be >= 1")
        if self.Bn_compute_blocks < 0:
            raise ValueError("Bn_compute_blocks must be >= 0")
        if self.Fn_compute_blocks + self.Bn_compute_blocks > num_layers:
            raise ValueError(
                f"Fn({self.Fn_compute_blocks})+Bn({self.Bn_compute_blocks}) "
                f"> num_layers({num_layers})"
            )
        if self.enable_taylorseer and self.taylorseer_order != 1:
            raise ValueError("only taylorseer_order=1 is implemented")


@dataclass
class DBCacheState:
    cfg: DBCacheConfig
    num_layers: int
    step: int = 0
    continuous_hits: int = 0
    prev_fn_residual: torch.Tensor | None = None
    cached_mid_residual: torch.Tensor | None = None
    prev_mid_residual: torch.Tensor | None = None  # TaylorSeer only
    diffs: list[float] = field(default_factory=list)
    hits: int = 0
    misses: int = 0

    def reset(self) -> None:
        self.step = 0
        self.continuous_hits = 0
        self.prev_fn_residual = None
        self.cached_mid_residual = None
        self.prev_mid_residual = None
        self.diffs.clear()
        self.hits = 0
        self.misses = 0

    def ranges(self) -> tuple[range, range, range]:
        fn_end = self.cfg.Fn_compute_blocks
        bn_start = self.num_layers - self.cfg.Bn_compute_blocks
        return range(0, fn_end), range(fn_end, bn_start), range(bn_start, self.num_layers)

    def decide_hit(self, fn_residual: torch.Tensor) -> bool:
        """Synced cache-hit decision. Always all-reduces (even on first step)."""
        prev = self.prev_fn_residual
        can_compare = prev is not None and prev.shape == fn_residual.shape
        packed = torch.empty(2, device=fn_residual.device, dtype=torch.float32)
        packed[0] = rel_l1_diff(fn_residual, prev) if can_compare else 0.0
        packed[1] = 1.0 if can_compare else 0.0

        state = get_parallel_state()
        if state.world_size > 1 and dist.is_initialized():
            dist.all_reduce(packed, op=dist.ReduceOp.AVG)

        if float(packed[1].item()) < 1.0 - 1e-6:
            return False

        diff = float(packed[0].item())
        self.diffs.append(diff)

        cfg = self.cfg
        if cfg.probe_only or self.step < cfg.max_warmup_steps:
            return False
        if self.cached_mid_residual is None:
            return False
        if 0 <= cfg.max_continuous_cached_steps <= self.continuous_hits:
            return False
        return diff < cfg.residual_diff_threshold

    def mid_residual_for_hit(self) -> torch.Tensor:
        assert self.cached_mid_residual is not None
        if not self.cfg.enable_taylorseer or self.prev_mid_residual is None:
            return self.cached_mid_residual
        # r̂ = r_{t-1} + (r_{t-1} - r_{t-2}); delta not refreshed on consecutive hits.
        return 2 * self.cached_mid_residual - self.prev_mid_residual

    def update_mid(self, mid_residual: torch.Tensor) -> None:
        if self.cfg.enable_taylorseer and self.cached_mid_residual is not None:
            self.prev_mid_residual = self.cached_mid_residual
        self.cached_mid_residual = mid_residual.detach()

    def finish_step(self, *, hit: bool, fn_residual: torch.Tensor) -> None:
        if hit:
            self.hits += 1
            self.continuous_hits += 1
        else:
            # Reference stays pinned to the step whose mid residual is cached, so
            # the diff measures drift since that step (cache-dit only refreshes
            # its Fn buffer on compute steps). Refreshing every step would turn
            # this into a per-step delta, which never accumulates and lets a
            # single stale residual be reused for the whole schedule.
            self.prev_fn_residual = fn_residual.detach()
            self.misses += 1
            self.continuous_hits = 0
        self.step += 1

    def summary(self) -> dict:
        n = self.hits + self.misses
        return {
            "Fn": self.cfg.Fn_compute_blocks,
            "Bn": self.cfg.Bn_compute_blocks,
            "threshold": self.cfg.residual_diff_threshold,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / n, 4) if n else 0.0,
            "avg_diff": round(sum(self.diffs) / len(self.diffs), 6) if self.diffs else None,
            "diffs": [round(x, 6) for x in self.diffs],
        }

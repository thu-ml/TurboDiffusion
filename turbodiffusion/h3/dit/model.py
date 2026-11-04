from __future__ import annotations

import math
from typing import Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from turbodiffusion.h3.attention import attention_fn
from turbodiffusion.h3.dit.config import ADALN_MODALITY_NUM, H3DiTConfig
from turbodiffusion.h3.dit.dbcache import DBCacheState
from turbodiffusion.h3.parallel import (
    get_parallel_state,
    shard_sequence,
    ulysses_post_attn,
    ulysses_pre_attn,
    ulysses_pre_attn_launch,
    ulysses_pre_attn_wait,
)


class _BlockCtx(NamedTuple):
    """Shared args for running transformer blocks in one forward pass."""

    adaln_input: torch.Tensor | None
    combined: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    block_modulations: list[tuple[torch.Tensor, ...]] | None
    offloader: Any
    sequence_length: int


def _rms_norm(size: int, eps: float) -> nn.RMSNorm:
    return nn.RMSNorm(size, eps=eps)


def _reorder_grouped_qkv_to_qkv(
    weight: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """FL2VA stores per-head [q,k,v]; runtime wants [q_all,k_all,v_all]."""
    per_head = 3 * head_dim
    grouped = weight.reshape(num_heads, per_head, *weight.shape[1:])
    q, k, v = grouped.split(head_dim, dim=1)
    return torch.cat(
        [
            q.reshape(num_heads * head_dim, *weight.shape[1:]),
            k.reshape(num_heads * head_dim, *weight.shape[1:]),
            v.reshape(num_heads * head_dim, *weight.shape[1:]),
        ],
        dim=0,
    )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [T, H, D], cos/sin: [T, rot_dim]
    rot_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    x_rot = x_rot * cos + _rotate_half(x_rot) * sin
    return torch.cat((x_rot, x_pass), dim=-1)


def _modulate(h: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return h * (1.0 + scale.index_select(0, idx)) + shift.index_select(0, idx)


def _gate(residual: torch.Tensor, gate: torch.Tensor, update: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return residual + gate.index_select(0, idx) * update


class MiniMaxH3Rope(nn.Module):
    def __init__(self, inv_freq_len: int = 16):
        super().__init__()
        self.register_buffer("inv_freq", torch.empty(inv_freq_len, dtype=torch.float32), persistent=True)

    def _apply(self, fn, recurse=True):
        frequencies = self.inv_freq
        super()._apply(fn, recurse=recurse)
        if frequencies.is_meta:
            self.inv_freq = torch.empty_like(self.inv_freq, dtype=torch.float32)
        else:
            # Preserve the original values across model-wide BF16/device casts.
            self.inv_freq = frequencies.to(device=self.inv_freq.device, dtype=torch.float32)
        return self

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """position_ids: [1, S, 3] or [S, 3] -> freqs [S, 96]."""
        if position_ids.dim() == 3:
            pos = position_ids[0]
        else:
            pos = position_ids
        pos = pos.to(torch.float32)
        per_axis = pos.unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        t_f, h_f, w_f = per_axis.unbind(dim=1)
        half = torch.cat((t_f, h_f, w_f), dim=-1)
        return torch.cat((half, half), dim=-1)


class MiniMaxH3TimeEmbedder(nn.Module):
    def __init__(self, cfg: H3DiTConfig):
        super().__init__()
        self.frequency_embedding_size = cfg.timestep_input_dim
        self.proj_in = nn.Linear(cfg.timestep_input_dim, cfg.time_embed_hidden_size, bias=True)
        self.proj_out = nn.Linear(cfg.time_embed_hidden_size, cfg.time_embed_dim, bias=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        # Official MiniMaxH3TimeEmbedder: embed raw timestep (1-σ), no ×1000
        args = t.float()[:, None] * freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # The released timestep MLP and its sinusoidal input remain FP32.
        t_freq = t_freq.to(dtype=self.proj_in.weight.dtype)
        hidden = F.silu(self.proj_in(t_freq))
        return self.proj_out(hidden)


class MiniMaxH3Attention(nn.Module):
    def __init__(
        self, cfg: H3DiTConfig, *, attn_backend: str = "sdpa", sequence_parallel: bool = True
    ):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.attention_head_dim
        self.inner_dim = self.num_heads * self.head_dim
        self.attn_backend = attn_backend
        self.sequence_parallel = sequence_parallel
        self.sparge_attn_topk = 0.5
        self.sparge_attn_pre_sink_length = 0
        self.sparge_attn_post_sink_length = 0
        self.qkv_proj = nn.Linear(cfg.hidden_size, 3 * self.inner_dim, bias=False)
        self.q_norm = _rms_norm(self.head_dim, cfg.qk_norm_eps)
        self.k_norm = _rms_norm(self.head_dim, cfg.qk_norm_eps)
        self.out_proj = nn.Linear(self.inner_dim, cfg.hidden_size, bias=False)

    def configure_sparge(
        self,
        *,
        topk: float = 0.5,
        pre_sink_length: int = 0,
        post_sink_length: int = 0,
    ) -> None:
        self.sparge_attn_topk = float(topk)
        self.sparge_attn_pre_sink_length = int(pre_sink_length)
        self.sparge_attn_post_sink_length = int(post_sink_length)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        use_rope: bool = True,
        offload_handle=None,
        sequence_length: int | None = None,
    ) -> torch.Tensor:
        # x: [T_local, hidden] under Ulysses (full T if sp=1)
        t = x.shape[0]
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.inner_dim, dim=-1)
        q = q.view(t, self.num_heads, self.head_dim)
        k = k.view(t, self.num_heads, self.head_dim)
        v = v.view(t, self.num_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if use_rope and cos is not None and sin is not None:
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)

        # Ulysses: [T_local, H, D] -> [T, H_local, D] -> attn -> reverse
        if self.sequence_parallel and get_parallel_state().use_async_comm:
            # Q/K/V all-to-alls in flight together: each one's quantize/dequantize
            # runs on the SMs while its siblings occupy the link.
            q, k, v = ulysses_pre_attn_wait(ulysses_pre_attn_launch(q, k, v))
        elif self.sequence_parallel:
            q, k, v = ulysses_pre_attn(q, k, v)
        # Overlap next-block H2D with attention. Do not
        # hoist this above the all-to-all: without NVLink both ride PCIe and the
        # contention costs more than the overlap saves.
        if offload_handle is not None:
            offload_handle.record()
            offload_handle.run()
        padded_length = q.shape[0]
        if sequence_length is not None:
            q, k, v = (tensor[:sequence_length] for tensor in (q, k, v))
        out = attention_fn(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
            backend=self.attn_backend,
            sparge_attn_topk=self.sparge_attn_topk,
            sparge_attn_pre_sink_length=self.sparge_attn_pre_sink_length,
            sparge_attn_post_sink_length=self.sparge_attn_post_sink_length,
        ).squeeze(0)
        if sequence_length is not None:
            out = F.pad(out, (0, 0, 0, 0, 0, padded_length - sequence_length))
        if self.sequence_parallel:
            out = ulysses_post_attn(out)

        out = out.reshape(t, self.inner_dim)
        return self.out_proj(out)


class MiniMaxH3MLP(nn.Module):
    def __init__(self, cfg: H3DiTConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden_size, 2 * cfg.ffn_hidden_size, bias=False)
        self.fc2 = nn.Linear(cfg.ffn_hidden_size, cfg.hidden_size, bias=False)
        # Preserve aliases used by existing H3 checkpoint layouts.
        self.w1 = self.fc1
        self.w2 = self.fc2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * up)


class MiniMaxH3AdalnProj(nn.Module):
    def __init__(self, cfg: H3DiTConfig, out_features: int, *, modality_num: int, expand_ratio: int):
        super().__init__()
        self.modality_num = modality_num
        self.expand_ratio = expand_ratio
        self.hidden_size = cfg.hidden_size
        self.linear = nn.Linear(cfg.time_embed_dim, out_features, bias=True)

    def forward(self, adaln_input: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.linear(adaln_input)
        m = x.shape[0]
        x = x.view(m * self.modality_num, self.expand_ratio * self.hidden_size)
        return tuple(x.chunk(self.expand_ratio, dim=-1))


class MiniMaxH3TokenRefinerBlock(nn.Module):
    def __init__(self, cfg: H3DiTConfig):
        super().__init__()
        self.norm1 = _rms_norm(cfg.hidden_size, cfg.norm_eps)
        # Text is replicated and not yet sequence-sharded.
        self.attn = MiniMaxH3Attention(cfg, attn_backend="sdpa", sequence_parallel=False)
        self.norm2 = _rms_norm(cfg.hidden_size, cfg.norm_eps)
        self.mlp = MiniMaxH3MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos=None, sin=None, use_rope=False)
        x = x + self.mlp(self.norm2(x))
        return x


class MiniMaxH3TokenRefiner(nn.Module):
    def __init__(self, cfg: H3DiTConfig):
        super().__init__()
        self.blocks = nn.ModuleList(
            [MiniMaxH3TokenRefinerBlock(cfg) for _ in range(cfg.token_refiner_num_layers)]
        )
        self.final_norm = _rms_norm(cfg.hidden_size, cfg.final_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.final_norm(x)


class MiniMaxH3DiTBlock(nn.Module):
    def __init__(self, cfg: H3DiTConfig, *, attn_backend: str, block_id: int = -1):
        super().__init__()
        self.block_id = block_id
        self.block_offload_id = block_id
        self.norm1 = _rms_norm(cfg.hidden_size, cfg.norm_eps)
        self.attn = MiniMaxH3Attention(cfg, attn_backend=attn_backend)
        self.norm2 = _rms_norm(cfg.hidden_size, cfg.norm_eps)
        self.mlp = MiniMaxH3MLP(cfg)
        self.adaln_proj = MiniMaxH3AdalnProj(
            cfg, cfg.adaln_out_features, modality_num=ADALN_MODALITY_NUM, expand_ratio=6
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        combined_indices: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        adaln_params: tuple[torch.Tensor, ...] | None = None,
        offload_handle=None,
        sequence_length: int | None = None,
    ) -> torch.Tensor:
        if adaln_params is None:
            adaln_params = self.adaln_proj(adaln_input)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaln_params

        residual = x
        h = _modulate(self.norm1(x), shift_msa, scale_msa, combined_indices)
        h = self.attn(
            h, cos=cos, sin=sin, use_rope=True, offload_handle=offload_handle,
            sequence_length=sequence_length,
        )
        x = _gate(residual, gate_msa, h, combined_indices)

        residual = x
        h = _modulate(self.norm2(x), shift_mlp, scale_mlp, combined_indices)
        h = self.mlp(h)
        return _gate(residual, gate_mlp, h, combined_indices)


class MiniMaxH3FinalLayer(nn.Module):
    def __init__(self, cfg: H3DiTConfig):
        super().__init__()
        self.norm = _rms_norm(cfg.hidden_size, cfg.final_norm_eps)
        self.adaln_proj = MiniMaxH3AdalnProj(
            cfg, cfg.final_adaln_out_features, modality_num=1, expand_ratio=2
        )
        self.video_out = nn.Linear(cfg.hidden_size, cfg.video_patch_dim, bias=True)
        self.audio_out = nn.Linear(cfg.hidden_size, cfg.audio_latents_dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        timestep_indices: torch.Tensor,
        video_mask: torch.Tensor,
        audio_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale = self.adaln_proj(adaln_input)
        h = self.norm(x)
        h = h * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(0, timestep_indices)
        video = self.video_out(h[video_mask].to(self.video_out.weight.dtype))
        audio = self.audio_out(h[audio_mask].to(self.audio_out.weight.dtype))
        return video, audio


class MiniMaxH3DiTModel(nn.Module):
    """Lean FL2VA-layout Omni-DiT (no SGLang / TP coupling)."""

    def __init__(self, cfg: H3DiTConfig | None = None, *, attn_backend: str = "sdpa"):
        super().__init__()
        self.cfg = cfg or H3DiTConfig()
        self.attn_backend = attn_backend
        self.video_patch_proj = nn.Linear(self.cfg.video_patch_dim, self.cfg.hidden_size, bias=True)
        self.audio_patch_proj = nn.Linear(self.cfg.audio_latents_dim, self.cfg.hidden_size, bias=True)
        self.condition_proj = nn.Linear(self.cfg.text_dim, self.cfg.hidden_size, bias=True)
        self.time_embedder = MiniMaxH3TimeEmbedder(self.cfg)
        self.rope = MiniMaxH3Rope(self.cfg.rope_inv_freq_len)
        self.token_refiner = MiniMaxH3TokenRefiner(self.cfg)
        self.blocks = nn.ModuleList(
            [
                MiniMaxH3DiTBlock(self.cfg, attn_backend=attn_backend, block_id=i)
                for i in range(self.cfg.num_layers)
            ]
        )
        self.final_layer = MiniMaxH3FinalLayer(self.cfg)

    def configure_sparge(
        self,
        *,
        topk: float = 0.5,
        pre_sink_length: int = 0,
        post_sink_length: int = 0,
    ) -> None:
        """Configure only the sequence-sharded main-block attention."""
        for mod in self.modules():
            if isinstance(mod, MiniMaxH3Attention) and mod.sequence_parallel:
                mod.configure_sparge(
                    topk=topk,
                    pre_sink_length=pre_sink_length,
                    post_sink_length=post_sink_length,
                )

    def forward(
        self,
        *,
        video_tokens: torch.Tensor,
        audio_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        position_ids: torch.Tensor,
        token_tags: torch.Tensor,
        timesteps: torch.Tensor,
        inverse_indices: torch.Tensor,
        video_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        text_mask: torch.Tensor,
        offloader=None,
        block_modulations: list[tuple[torch.Tensor, ...]] | None = None,
        final_modulation: tuple[torch.Tensor, torch.Tensor] | None = None,
        dbcache: DBCacheState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Packed forward for one denoise step.

        Args:
            video_tokens / audio_tokens: noisy latents already patchified where needed
            text_tokens: Qwen3-VL layer-50 embeds [Tt, text_dim]
            position_ids: [S, 3] for the full packed sequence (t,h,w)
            token_tags: [S] modality ids
            timesteps: unique timesteps [M]
            inverse_indices: [S] maps each token to unique timestep index
            video_mask / audio_mask / text_mask: boolean masks over packed S
            offloader: double-buffer asynchronous H2D controller
            block_modulations / final_modulation: optional AdaLN cache (official deploy
                precomputes these from the sigma schedule so AdaLN weights need not run
                every step — and can be omitted entirely when baked offline)
            dbcache: optional DBCache state (Cache-DiT Dual Block Cache)
        """
        device = video_tokens.device
        dtype = video_tokens.dtype

        v = self.video_patch_proj(video_tokens.to(self.video_patch_proj.weight.dtype)).to(dtype)
        a = self.audio_patch_proj(audio_tokens.to(self.audio_patch_proj.weight.dtype)).to(dtype)
        c = self.condition_proj(text_tokens.to(self.condition_proj.weight.dtype)).to(dtype)
        c = self.token_refiner(c)

        s = token_tags.shape[0]
        x = torch.zeros(s, self.cfg.hidden_size, device=device, dtype=dtype)
        x[video_mask] = v
        x[audio_mask] = a
        x[text_mask] = c

        # time embed (fp32 path) -> SiLU for AdaLN (skipped if modulations cached)
        if block_modulations is None or final_modulation is None:
            t_emb = self.time_embedder(timesteps.to(device))
            adaln_input = F.silu(t_emb).to(dtype)
        else:
            adaln_input = None

        freqs = self.rope(position_ids.to(device))
        cos = torch.cos(freqs).to(dtype)
        sin = torch.sin(freqs).to(dtype)
        combined = inverse_indices.to(device) * ADALN_MODALITY_NUM + token_tags.clamp(min=0).to(device)

        # Ulysses: shard packed sequence evenly (S must be % sp == 0; packer aligns to 64)
        x = shard_sequence(x, dim=0)
        cos = shard_sequence(cos, dim=0)
        sin = shard_sequence(sin, dim=0)
        combined = shard_sequence(combined, dim=0)
        local_video_mask = shard_sequence(video_mask, dim=0)
        local_audio_mask = shard_sequence(audio_mask, dim=0)
        local_inv = shard_sequence(inverse_indices.to(device), dim=0)

        if offloader is not None:
            offloader.begin_step()

        ctx = _BlockCtx(
            adaln_input=adaln_input,
            combined=combined,
            cos=cos,
            sin=sin,
            block_modulations=block_modulations,
            offloader=offloader,
            sequence_length=v.shape[0] + a.shape[0] + c.shape[0],
        )
        if dbcache is not None and dbcache.cfg.enabled:
            x = self._forward_blocks_dbcache(x, dbcache, ctx)
        else:
            x = self._run_blocks(range(len(self.blocks)), x, ctx, after_prefetch=-1)

        # final layer uses full-sequence masks on the local shard
        if final_modulation is not None:
            shift, scale = final_modulation
            h = self.final_layer.norm(x)
            h = h * (1.0 + scale.index_select(0, local_inv)) + shift.index_select(0, local_inv)
            video = self.final_layer.video_out(
                h[local_video_mask].to(self.final_layer.video_out.weight.dtype)
            )
            audio = self.final_layer.audio_out(
                h[local_audio_mask].to(self.final_layer.audio_out.weight.dtype)
            )
        else:
            video, audio = self.final_layer(
                x,
                adaln_input=adaln_input,
                timestep_indices=local_inv,
                video_mask=local_video_mask,
                audio_mask=local_audio_mask,
            )
        return video, audio

    def _run_block(
        self,
        i: int,
        x: torch.Tensor,
        ctx: _BlockCtx,
        *,
        prefetch_id: int | None = -1,
    ) -> torch.Tensor:
        handle = None
        if ctx.offloader is not None:
            blk, handle = ctx.offloader.acquire(i, prefetch_id=prefetch_id)
        else:
            blk = self.blocks[i]

        mod = None if ctx.block_modulations is None else ctx.block_modulations[i]
        x = blk(
            x,
            adaln_input=ctx.adaln_input,
            combined_indices=ctx.combined,
            cos=ctx.cos,
            sin=ctx.sin,
            adaln_params=mod,
            offload_handle=handle,
            sequence_length=ctx.sequence_length,
        )
        return x

    def _run_blocks(
        self,
        indices,
        x: torch.Tensor,
        ctx: _BlockCtx,
        *,
        after_prefetch: int | None = None,
    ) -> torch.Tensor:
        """Run blocks in order; prefetch next id, or ``after_prefetch`` on the last."""
        ids = list(indices)
        for j, i in enumerate(ids):
            pref = ids[j + 1] if j + 1 < len(ids) else after_prefetch
            x = self._run_block(i, x, ctx, prefetch_id=pref)
        return x

    def _forward_blocks_dbcache(
        self, x: torch.Tensor, dbcache: DBCacheState, ctx: _BlockCtx
    ) -> torch.Tensor:
        fn_ids, mid_ids, bn_ids = (list(r) for r in dbcache.ranges())
        offloader = ctx.offloader

        x0 = x.detach().clone()
        # Do not prefetch past Fn — hit/miss is decided from the Fn residual.
        x = self._run_blocks(fn_ids, x, ctx, after_prefetch=None)
        fn_residual = x - x0
        hit = dbcache.decide_hit(fn_residual)

        if hit:
            mid = dbcache.mid_residual_for_hit().to(dtype=x.dtype, device=x.device)
            x = x + mid
            if bn_ids:
                if offloader is not None:
                    offloader.prepare(bn_ids[0])
                x = self._run_blocks(bn_ids, x, ctx, after_prefetch=None)
        else:
            if mid_ids and offloader is not None:
                offloader.prepare(mid_ids[0])
            if mid_ids:
                x_mid0 = x.detach().clone()
                x = self._run_blocks(
                    mid_ids, x, ctx, after_prefetch=(bn_ids[0] if bn_ids else None)
                )
                dbcache.update_mid(x - x_mid0)
            x = self._run_blocks(bn_ids, x, ctx, after_prefetch=None)

        dbcache.finish_step(hit=hit, fn_residual=fn_residual)
        return x

    @torch.no_grad()
    def bake_and_drop_adaln(
        self,
        timesteps_per_step: list[torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> tuple[list[list[tuple[torch.Tensor, ...]]], list[tuple[torch.Tensor, torch.Tensor]]]:
        """Bake AdaLN modulations for the whole sigma schedule, then free the weights.

        AdaLN input depends only on the timestep, so the whole schedule can be
        resolved in a single sweep with blocks on the outer loop: each of the
        ~26GB of AdaLN weights is touched once instead of once per denoise step.
        The weights are dropped as we go, which also releases their pinned CPU
        copy under block offload. Blocks must afterwards be fed the returned
        modulations — they can no longer compute their own.
        """
        assert self.blocks is not None
        dtype = dtype or torch.bfloat16

        adaln_inputs = [
            F.silu(self.time_embedder(ts.to(device))).to(dtype) for ts in timesteps_per_step
        ]
        n_steps = len(adaln_inputs)
        block_mods: list[list[tuple[torch.Tensor, ...]]] = [[] for _ in range(n_steps)]

        for blk in self.blocks:
            proj = blk.adaln_proj
            if proj is None:
                raise RuntimeError("AdaLN already dropped; rebuild the model to re-bake")
            proj.to(device)
            for i, ain in enumerate(adaln_inputs):
                block_mods[i].append(proj(ain))
            blk.adaln_proj = None

        fin = self.final_layer.adaln_proj
        fin.to(device)
        final_mods = [fin(ain) for ain in adaln_inputs]
        self.final_layer.adaln_proj = None

        if device.type == "cuda":
            torch.cuda.empty_cache()
        return block_mods, final_mods

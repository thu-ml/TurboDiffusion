from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import torch
from tqdm import tqdm

from turbodiffusion.h3.dit.config import H3DiTConfig
from turbodiffusion.h3.dit.dbcache import DBCacheConfig, DBCacheState
from turbodiffusion.h3.dit.model import MiniMaxH3DiTModel
from turbodiffusion.h3.dit.pack import (
    H3_IMGVID_COND_TIMESTEP,
    PackedBatch,
    build_i2va_packed,
)
from turbodiffusion.h3.dit.weight_loader import load_fl2va_streaming, materialize_cpu_model
from turbodiffusion.h3.resolution import normalize_resolution
from turbodiffusion.h3.offload import BlockOffloader, build_preload_map
from turbodiffusion.h3.parallel import get_parallel_state, init_parallel
from turbodiffusion.h3.utils import barrier, collect_gpu_info


class H3DiTProcessor:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        dit_cfg = cfg.get("dit", {})
        self.model_dir = Path(dit_cfg["model_path"])
        self.turbo_int8_path = (
            Path(dit_cfg["turbo_int8_path"]) if dit_cfg.get("turbo_int8_path") else None
        )
        if self.turbo_int8_path and not dit_cfg.get("adaln_bake_path"):
            raise ValueError("turbo W8A8 checkpoint requires dit.adaln_bake_path")
        self.ulysses_degree = int(dit_cfg.get("ulysses_degree", 8))
        # INT8 communication is lossy and enabled explicitly by the shipped profiles.
        self.use_quant_comm = bool(dit_cfg.get("use_quant_comm", False))
        # Launch Q/K/V all-to-alls without waiting between them.
        self.use_async_comm = bool(dit_cfg.get("use_async_comm", True))
        self.attn_backend = dit_cfg.get("attn_backend", "sparge_attn")
        self.sparge_attn_topk = float(dit_cfg.get("sparge_attn_topk", 0.5))
        # "auto" → text+audio prefix as dense sinks (H3 pack is [text|audio|video|pad])
        self.sparge_attn_pre_sink = dit_cfg.get("sparge_attn_pre_sink_length", "auto")
        self.sparge_attn_post_sink = int(dit_cfg.get("sparge_attn_post_sink_length", 0))
        self.use_offload = bool(dit_cfg.get("use_offload", True))
        self.offload_interlace = int(dit_cfg.get("offload_interlace", 0))
        # Bake AdaLN modulations from the sigma schedule, then drop the weights.
        # One schedule per process — batch stages must keep steps/shifts fixed.
        self.bake_adaln = bool(dit_cfg.get("bake_adaln", False))
        # Pre-baked table shipped next to an AdaLN-free INT8 checkpoint. When set,
        # the weights are never loaded and the table is used as-is.
        self.adaln_bake_path = (
            Path(dit_cfg["adaln_bake_path"]) if dit_cfg.get("adaln_bake_path") else None
        )
        self._adaln_bake: tuple | None = None
        self.dbcache_cfg = DBCacheConfig.from_dict(dit_cfg.get("dbcache"))
        self.dbcache: DBCacheState | None = None
        self.dtype = torch.bfloat16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: MiniMaxH3DiTModel | None = None
        self.offloader: BlockOffloader | None = None

    def setup(self) -> None:
        arch = H3DiTConfig.from_json(str(self.model_dir / "config.json"))
        state = init_parallel(
            ulysses_degree=self.ulysses_degree,
            use_quant_comm=self.use_quant_comm,
            use_async_comm=self.use_async_comm,
            num_heads=arch.num_attention_heads,
        )
        self.device = torch.device(f"cuda:{state.local_rank}" if torch.cuda.is_available() else "cpu")
        gpu = collect_gpu_info(state.local_rank)

        if state.rank == 0:
            print(f"[dit] gpu={gpu} weights={'W8A8' if self.turbo_int8_path else 'BF16'}")
            extra = ""
            if self.attn_backend == "sparge_attn":
                extra = (
                    f" sparge_topk={self.sparge_attn_topk} "
                    f"pre_sink={self.sparge_attn_pre_sink} post_sink={self.sparge_attn_post_sink}"
                )
            print(
                f"[dit] attn={self.attn_backend}{extra} ulysses={self.ulysses_degree} "
                f"offload={self.use_offload} interlace={self.offload_interlace} "
                f"bake_adaln={self.bake_adaln}"
            )
            if self.dbcache_cfg.enabled:
                print(
                    f"[dit] dbcache enabled F{self.dbcache_cfg.Fn_compute_blocks}"
                    f"B{self.dbcache_cfg.Bn_compute_blocks} "
                    f"thr={self.dbcache_cfg.residual_diff_threshold} "
                    f"warmup={self.dbcache_cfg.max_warmup_steps} "
                    f"probe_only={self.dbcache_cfg.probe_only} "
                    f"taylorseer={self.dbcache_cfg.enable_taylorseer}",
                    flush=True,
                )

        # meta init → empty BF16 on CPU (FP32 empty would be ~132GB/rank and OOM×8)
        with torch.device("meta"):
            self.model = MiniMaxH3DiTModel(arch, attn_backend=self.attn_backend)

        if self.turbo_int8_path is not None:
            from turbodiffusion.h3.dit.turbo_int8 import apply_turbo_int8, load_fl2va_turbo_int8

            with torch.device("meta"):
                wrapped = apply_turbo_int8(self.model, quantize=False)
            self._load_adaln_bake(rank=state.rank)
            load_fl2va_turbo_int8(self.model, self.turbo_int8_path, rank=state.rank)
            if state.rank == 0:
                print(f"[dit] turbo W8A8 main-block Linears={wrapped}", flush=True)
            if state.world_size > 1:
                barrier()
        else:
            materialize_cpu_model(self.model, self.dtype)

            # BF16×8 ≈528GB; with ≥1Ti host RAM parallel load is fine.
            # Set H3_SERIAL_LOAD=1 to reduce simultaneous checkpoint reads.
            serial = os.environ.get("H3_SERIAL_LOAD", "0") == "1"
            if state.rank == 0:
                print(
                    f"[dit] streaming FL2VA safetensors into CPU BF16 "
                    f"({'serial' if serial else 'parallel'} load) ...",
                    flush=True,
                )
            if serial:
                for r in range(state.world_size):
                    if state.rank == r:
                        loaded, n_miss = load_fl2va_streaming(
                            self.model, self.model_dir, rank=state.rank
                        )
                        print(
                            f"[dit] rank{state.rank} loaded={loaded} missing≈{n_miss}",
                            flush=True,
                        )
                    if state.world_size > 1:
                        barrier()
            else:
                loaded, n_miss = load_fl2va_streaming(
                    self.model, self.model_dir, rank=state.rank
                )
                print(f"[dit] rank{state.rank} loaded={loaded} missing≈{n_miss}", flush=True)
                if state.world_size > 1:
                    barrier()

            if n_miss:
                raise RuntimeError(f"incomplete H3 checkpoint: {n_miss} tensors missing")
            # BF16 already paid to read AdaLN, but honouring the table here keeps the
            # config key meaningful on both paths and drops the weights immediately.
            if self.adaln_bake_path is not None:
                self._load_adaln_bake(rank=state.rank)

        self.model.eval()

        # Always keep embed / time / refiner / final on GPU
        for name in (
            "video_patch_proj",
            "audio_patch_proj",
            "condition_proj",
            "time_embedder",
            "rope",
            "token_refiner",
            "final_layer",
        ):
            getattr(self.model, name).to(self.device)

        if self.use_offload:
            preload = build_preload_map(arch.num_layers, self.offload_interlace)
            self.offloader = BlockOffloader(
                self.device,
                preload,
                buf_length=2,
                # AdaLN is baked out before the first forward, so resident blocks
                # must not waste VRAM on it during setup.
                cpu_only_submodule="adaln_proj" if self.bake_adaln else None,
            )
            self.offloader.install(self.model.blocks)
            if state.rank == 0:
                print(
                    f"[dit] offload pattern (o=GPU,x=CPU): {self.offloader.pattern_str()} "
                    f"({self.offloader.num_offloaded}/{arch.num_layers} offloaded, "
                    f"async double-buf={self.offloader.buf_length})"
                )
        else:
            if state.rank == 0:
                print("[dit] WARNING: use_offload=false; full DiT BF16 will OOM on 32GB 5090")
            self.model.to(self.device)

        # SpargeAttn defaults (pre_sink refined per-sample when set to "auto")
        if self.attn_backend == "sparge_attn":
            pre = 0 if self.sparge_attn_pre_sink == "auto" else int(self.sparge_attn_pre_sink)
            self.model.configure_sparge(
                topk=self.sparge_attn_topk,
                pre_sink_length=pre,
                post_sink_length=self.sparge_attn_post_sink,
            )

        if self.dbcache_cfg.enabled:
            self.dbcache_cfg.validate(arch.num_layers)
            self.dbcache = DBCacheState(cfg=self.dbcache_cfg, num_layers=arch.num_layers)

    def _load_adaln_bake(self, *, rank: int) -> None:
        """Install a pre-baked modulation table and drop the AdaLN modules."""
        from turbodiffusion.h3.dit.adaln_bake import load_adaln_bake

        assert self.model is not None
        t0 = time.perf_counter()
        key, block_mods, final_mods, meta = load_adaln_bake(
            self.adaln_bake_path, device=self.device, dtype=self.dtype
        )
        n_blocks = len(self.model.blocks)
        if meta["n_blocks"] != n_blocks:
            raise RuntimeError(
                f"adaln bake has {meta['n_blocks']} blocks, model has {n_blocks}"
            )
        for blk in self.model.blocks:
            blk.adaln_proj = None
        self.model.final_layer.adaln_proj = None
        self._adaln_bake = (key, block_mods, final_mods)
        self.bake_adaln = True
        if rank == 0:
            nbytes = sum(
                t.numel() * t.element_size() for mods in block_mods for m in mods for t in m
            )
            print(
                f"[dit] loaded AdaLN bake {meta['task']} steps={meta['steps']} "
                f"shift=({meta['video_shift']},{meta['audio_shift']}) "
                f"{meta['n_steps']} entries {nbytes / 1e9:.2f}GB in "
                f"{time.perf_counter() - t0:.1f}s",
                flush=True,
            )

    def dbcache_summary(self) -> dict | None:
        return None if self.dbcache is None else self.dbcache.summary()

    def _bake_adaln(self, timesteps_per_step: list[torch.Tensor]):
        """Resolve AdaLN for the whole schedule up front and free its ~26GB of weights.

        Returns ``(None, None)`` when disabled, which keeps AdaLN inline in the blocks.
        The table depends only on the sigma schedule, so consecutive samples that
        share steps/shifts/task reuse it; the weights are gone after the first bake,
        so a different schedule in the same process cannot be served.
        """
        assert self.model is not None
        if not self.bake_adaln:
            return None, None

        from turbodiffusion.h3.dit.adaln_bake import bake_key

        key = bake_key(timesteps_per_step)
        if self._adaln_bake is not None:
            cached_key, block_mods, final_mods = self._adaln_bake
            if cached_key == key:
                return block_mods, final_mods
            where = (
                f"the table at {self.adaln_bake_path}"
                if self.adaln_bake_path is not None
                else "an earlier sample in this process"
            )
            raise RuntimeError(
                f"bake_adaln: this sample's sigma schedule does not match {where}; "
                "keep task/steps/shifts fixed or bake a table for this schedule"
            )

        state = get_parallel_state()
        t0 = time.perf_counter()
        block_mods, final_mods = self.model.bake_and_drop_adaln(
            timesteps_per_step, device=self.device, dtype=self.dtype
        )
        if self.offloader is not None:
            self.offloader.drop_block_submodule("adaln_proj")
        self._adaln_bake = (key, block_mods, final_mods)
        if state.rank == 0:
            nbytes = sum(
                t.numel() * t.element_size() for mods in block_mods for m in mods for t in m
            )
            print(
                f"[dit] baked AdaLN for {len(timesteps_per_step)} steps in "
                f"{time.perf_counter() - t0:.1f}s; weights dropped, "
                f"modulation table {nbytes / 1e9:.2f}GB",
                flush=True,
            )
        return block_mods, final_mods

    def _apply_sparge_sinks(self, packed: PackedBatch) -> None:
        """Dense-attend conditioning tokens (text/audio, + keyframe for I2VA)."""
        if self.model is None or self.attn_backend != "sparge_attn":
            return
        if self.sparge_attn_pre_sink == "auto":
            # Dense-attend everything before target video rows.
            pre = int(packed.text_mask.sum().item() + packed.audio_mask.sum().item())
            pre += int(packed.n_cond_rows)
        else:
            pre = int(self.sparge_attn_pre_sink)
        self.model.configure_sparge(
            topk=self.sparge_attn_topk,
            pre_sink_length=pre,
            post_sink_length=self.sparge_attn_post_sink,
        )
        # Offload buffers are independent module copies; tensor transfers do not
        # propagate per-sample attention attributes.
        if self.offloader is not None and self.offloader.bufs is not None:
            for block in self.offloader.bufs:
                block.attn.configure_sparge(
                    topk=self.sparge_attn_topk,
                    pre_sink_length=pre,
                    post_sink_length=self.sparge_attn_post_sink,
                )
        state = get_parallel_state()
        if state.rank == 0:
            print(
                f"[dit] sparge sinks pre={pre} post={self.sparge_attn_post_sink} "
                f"topk={self.sparge_attn_topk} seq={packed.token_tags.numel()} task={packed.task}",
                flush=True,
            )


    def release(self) -> None:
        """Free the DiT: GPU weights, offload buffers, pinned host blocks, AdaLN table.

        Leaves the process group alone — the caller owns that and usually reuses it
        for the VAE decode that follows.
        """
        if self.model is None and self.offloader is None:
            return
        if self.offloader is not None:
            self.offloader.release()
        self.offloader = None
        self.model = None
        self._adaln_bake = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        state = get_parallel_state()
        if getattr(state, "rank", 0) == 0:
            alloc = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            print(f"[dit] released cuda_alloc≈{alloc:.1f}GB", flush=True)

    def _cfg_resolution(self) -> str | None:
        """Use the configured resolution, or H3's native canvas rule."""
        return normalize_resolution((self.cfg.get("sample") or {}).get("resolution"))


    @torch.no_grad()
    def sample_i2va(
        self,
        prompt_embeds: torch.Tensor,
        visual_cond_rows: torch.Tensor,
        *,
        duration_seconds: float = 5.0,
        short_edge: int = 768,
        aspect_ratio: str = "16:9",
        seed: int = 0,
        steps: int = 5,
        video_shift: float = 12.0,
        audio_shift: float = 3.0,
        imgvid_cond_timestep: float = H3_IMGVID_COND_TIMESTEP,
        keyframe_anchors: tuple[int, ...] = (0,),
        resolution: str | None = None,
        canvas_latent_hw: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """FL2VA: frozen keyframe cond + denoise target video/audio.

        ``keyframe_anchors`` says which end of the video each block of
        ``visual_cond_rows`` is: ``(0,)`` starts from a frame, ``(-1,)`` ends on one,
        ``(0, -1)`` does both.

        ``canvas_latent_hw`` is the canvas the keyframes were encoded onto, which the
        encode stage records next to them. Pass it: the keyframe rows have to land in a
        block sized from the same geometry, and rederiving it here from a ratio the
        caller typed is how the two stop agreeing.
        """
        assert self.model is not None
        packed: PackedBatch = build_i2va_packed(
            prompt_embeds=prompt_embeds,
            visual_cond_rows=visual_cond_rows,
            duration_seconds=duration_seconds,
            short_edge=short_edge,
            aspect_ratio=aspect_ratio,
            seed=seed,
            device=self.device,
            dtype=self.dtype,
            arch=self.model.cfg,
            keyframe_anchors=keyframe_anchors,
            resolution=resolution if resolution is not None else self._cfg_resolution(),
            canvas_latent_hw=canvas_latent_hw,
        )
        self._apply_sparge_sinks(packed)

        video_sigmas = _shifted_sigma_schedule(steps, video_shift, device=self.device)
        audio_sigmas = _shifted_sigma_schedule(steps, audio_shift, device=self.device)
        assert video_sigmas.numel() == audio_sigmas.numel()
        n_steps = int(video_sigmas.numel()) - 1

        target_rows = packed.patchify_video(packed.video_latents)
        assert target_rows.shape[0] == packed.n_target_rows
        cond_rows = packed.visual_cond_rows
        assert cond_rows is not None and packed.update_mask is not None
        # Fully-clean keyframe (noise_level=1.0); keep frozen across steps
        video_rows = torch.cat([cond_rows, target_rows], dim=0)
        audio_rows = packed.pack_audio_rows(packed.audio_latents)
        update = packed.update_mask

        schedule = [
            _unique_timesteps_i2va(
                t_video=1.0 - float(video_sigmas[i]),
                t_audio=1.0 - float(audio_sigmas[i]),
                t_cond=max(1.0 - float(video_sigmas[i]), float(imgvid_cond_timestep)),
                token_tags=packed.token_tags,
                update_mask=update,
                video_mask=packed.video_mask,
                device=packed.inverse_indices.device,
            )
            for i in range(n_steps)
        ]
        block_mods, final_mods = self._bake_adaln([ts for ts, _ in schedule])
        if self.dbcache is not None:
            self.dbcache.reset()

        rank = get_parallel_state().rank
        iterator = range(n_steps)
        if rank == 0:
            iterator = tqdm(iterator, desc="dit-fl2va")
            print(
                f"[dit] fl2va frame_count={packed.frame_count} "
                f"latent_t={packed.video_latents.shape[2]} "
                f"keyframes={_keyframe_names(keyframe_anchors)} "
                f"cond_rows={packed.n_cond_rows} target_rows={packed.n_target_rows} "
                f"seq={packed.token_tags.numel()}",
                flush=True,
            )

        for i in iterator:
            sigma_v = float(video_sigmas[i])
            sigma_a = float(audio_sigmas[i])
            sigma_v_next = float(video_sigmas[i + 1])
            sigma_a_next = float(audio_sigmas[i + 1])
            t_v = 1.0 - sigma_v
            t_a = 1.0 - sigma_a
            timesteps, inv = schedule[i]

            v_pred, a_pred = self.model(
                video_tokens=video_rows.to(dtype=self.dtype),
                audio_tokens=audio_rows.to(dtype=self.dtype),
                text_tokens=packed.text_tokens,
                position_ids=packed.position_ids,
                token_tags=packed.token_tags,
                timesteps=timesteps,
                inverse_indices=inv,
                video_mask=packed.video_mask,
                audio_mask=packed.audio_mask,
                text_mask=packed.text_mask,
                offloader=self.offloader,
                block_modulations=None if block_mods is None else block_mods[i],
                final_modulation=None if final_mods is None else final_mods[i],
                dbcache=self.dbcache,
            )

            v_pred = _gather_token_preds(
                v_pred, packed.video_mask, feat=int(self.model.cfg.video_patch_dim)
            )
            a_pred = _gather_token_preds(
                a_pred, packed.audio_mask, feat=int(self.model.cfg.audio_latents_dim)
            )

            # Only Euler-update target visual rows; keep cond frozen
            updated = _euler_eta0_rf_step(
                video_rows, v_pred.float(), sigma_curr=sigma_v, sigma_next=sigma_v_next, cond_t=t_v
            )
            video_rows = torch.where(update[:, None], updated, video_rows)
            audio_rows = _euler_eta0_rf_step(
                audio_rows, a_pred.float(), sigma_curr=sigma_a, sigma_next=sigma_a_next, cond_t=t_a
            )

        self._log_dbcache()
        target_only = video_rows[packed.n_cond_rows :]
        video_x = packed.unpatchify_video(target_only.to(dtype=self.dtype))
        audio_x = packed.unpack_audio_rows(audio_rows.to(dtype=self.dtype))
        return video_x, audio_x


    def _log_dbcache(self) -> None:
        if self.dbcache is None or get_parallel_state().rank != 0:
            return
        s = self.dbcache.summary()
        print(
            f"[dit] dbcache hit_rate={s['hit_rate']} hits={s['hits']} "
            f"misses={s['misses']} avg_diff={s['avg_diff']}",
            flush=True,
        )
        if self.dbcache.cfg.probe_only:
            # probe_only earns its keep by sizing residual_diff_threshold, and the
            # per-step curve is what one sizes it against — a mean hides both the
            # warmup spike and where in the schedule the run is actually cacheable.
            print(f"[dit] dbcache diffs={s['diffs']}", flush=True)


def _shifted_sigma(t: torch.Tensor | float, shift: float) -> torch.Tensor:
    """Map base t∈[0,1] → σ with H3 time shift: σ = shift·t / (1+(shift-1)·t)."""
    if not torch.is_tensor(t):
        t = torch.tensor(t)
    shift_t = torch.as_tensor(shift, device=t.device, dtype=t.dtype)
    return shift_t * t / (1.0 + (shift_t - 1.0) * t)


def _shifted_sigma_schedule(sigma_points: int, shift: float, *, device: torch.device) -> torch.Tensor:
    """Match official schedule: ``linspace(1,0,N)`` → shift → ensure trailing 0."""
    base = torch.linspace(1.0, 0.0, sigma_points, device=device, dtype=torch.float32)
    shifted = _shifted_sigma(base, shift)
    shifted = torch.unique_consecutive(shifted)
    if sigma_points > 1 and float(shifted[-1]) > 0.0:
        shifted = torch.cat([shifted, torch.zeros(1, device=device, dtype=torch.float32)])
    return shifted




def _keyframe_names(anchors: tuple[int, ...]) -> str:
    """``(0, -1)`` → ``"first+last"``, for a log line that says what was conditioned on."""
    return "+".join("first" if a == 0 else "last" for a in anchors) or "none"


def _unique_timesteps_i2va(
    *,
    t_video: float,
    t_audio: float,
    t_cond: float,
    token_tags: torch.Tensor,
    update_mask: torch.Tensor,
    video_mask: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign AdaLN timesteps: target-video / text → t_v, audio → t_a, keyframe cond → t_cond."""
    from turbodiffusion.h3.dit.config import MODALITY_AUDIO, MODALITY_TEXT

    candidates = torch.tensor([t_video, t_audio, t_cond], device=device, dtype=torch.float32)
    # ``t_cond`` already folds in imgvid_cond_timestep; adaln_candidates("i2va", ...)
    # derives the same triple offline for the bake table.
    unique, slot = torch.unique(candidates, sorted=True, return_inverse=True)
    inv = torch.full((token_tags.shape[0],), int(slot[0]), device=device, dtype=torch.long)
    inv[token_tags == MODALITY_AUDIO] = int(slot[1])
    inv[token_tags == MODALITY_TEXT] = int(slot[0])

    # Map packed visual positions → update_mask order (cond then target)
    vis_idx = torch.nonzero(video_mask, as_tuple=False).view(-1)
    assert vis_idx.numel() == update_mask.numel()
    cond_vis = vis_idx[~update_mask]
    tgt_vis = vis_idx[update_mask]
    inv[cond_vis] = int(slot[2])
    inv[tgt_vis] = int(slot[0])
    return unique, inv




def _rf_velocity_to_x0(state: torch.Tensor, velocity: torch.Tensor, cond_t: float) -> torch.Tensor:
    return state + (1.0 - float(cond_t)) * velocity


def _euler_eta0_rf_step(
    state: torch.Tensor,
    velocity: torch.Tensor,
    *,
    sigma_curr: float,
    sigma_next: float,
    cond_t: float,
) -> torch.Tensor:
    """Official RF velocity → x0 → eta=0 Euler: x' = (σ'/σ)·x + (1-σ'/σ)·x0."""
    if sigma_curr == 0.0:
        return state
    denoised = _rf_velocity_to_x0(state, velocity, cond_t)
    ratio = sigma_next / sigma_curr
    return ratio * state + (1.0 - ratio) * denoised


def _gather_token_preds(
    local: torch.Tensor, full_mask: torch.Tensor, *, feat: int
) -> torch.Tensor:
    """Reassemble per-rank modality preds into full-length token preds under Ulysses.

    Ranks with zero modality tokens in their sequence shard still participate in
    all_gather with an empty row slice — ``feat`` must be passed explicitly so
    empty locals do not collapse the feature dim to 0.
    """
    import torch.distributed as dist

    from turbodiffusion.h3.parallel import get_parallel_state

    state = get_parallel_state()
    if state.ulysses_degree <= 1 or state.group is None:
        return local

    sp = state.ulysses_degree
    s = full_mask.shape[0]
    assert s % sp == 0
    chunk = s // sp
    local_mask = full_mask[state.sp_rank * chunk : (state.sp_rank + 1) * chunk]
    n_local = int(local_mask.sum().item())
    dtype = local.dtype
    shard = torch.zeros(chunk, feat, device=full_mask.device, dtype=dtype)
    if n_local > 0:
        shard[local_mask] = local.reshape(n_local, feat)
    parts = [torch.empty_like(shard) for _ in range(sp)]
    dist.all_gather(parts, shard.contiguous(), group=state.group)
    full = torch.cat(parts, dim=0)
    return full[full_mask]

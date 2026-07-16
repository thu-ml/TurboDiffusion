from dataclasses import dataclass, replace

import torch

from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationType
from ltx_core.model.transformer.attention import Attention, AttentionCallable, AttentionFunction
from ltx_core.model.transformer.feed_forward import FeedForward
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.transformer_args import TransformerArgs
from ltx_core.utils import rms_norm


@dataclass
class TransformerConfig:
    dim: int
    heads: int
    d_head: int
    context_dim: int


def modulated_rms_norm(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return rms_norm(x, eps=eps) * (1 + scale) + shift


def modulate(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def gated_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    mask: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    if isinstance(mask, float):
        return torch.addcmul(x, residual, gate)
    return x + residual * gate * mask


def _ada_value(
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
    batch_size: int,
    index: int,
    num_ada_params: int,
) -> torch.Tensor:
    timestep_values = timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)
    table_value = scale_shift_table[index].unsqueeze(0).unsqueeze(0).to(device=timestep.device, dtype=timestep.dtype)
    return table_value + timestep_values[:, :, index, :]


def modulated_rms_norm_from_ada(
    x: torch.Tensor,
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
    scale_index: int,
    shift_index: int,
    num_ada_params: int,
    eps: float,
) -> torch.Tensor:
    scale = _ada_value(scale_shift_table, timestep, x.shape[0], scale_index, num_ada_params)
    shift = _ada_value(scale_shift_table, timestep, x.shape[0], shift_index, num_ada_params)
    return modulated_rms_norm(x, scale, shift, eps)


def modulate_from_ada(
    x: torch.Tensor,
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
    scale_index: int,
    shift_index: int,
    num_ada_params: int,
) -> torch.Tensor:
    scale = _ada_value(scale_shift_table, timestep, x.shape[0], scale_index, num_ada_params)
    shift = _ada_value(scale_shift_table, timestep, x.shape[0], shift_index, num_ada_params)
    return modulate(x, scale, shift)


def gated_residual_from_ada(
    x: torch.Tensor,
    residual: torch.Tensor,
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
    gate_index: int,
    num_ada_params: int,
    mask: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    gate = _ada_value(scale_shift_table, timestep, x.shape[0], gate_index, num_ada_params)
    return gated_residual(x, residual, gate, mask)


class BasicAVTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        idx: int,
        video: TransformerConfig | None = None,
        audio: TransformerConfig | None = None,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        norm_eps: float = 1e-6,
        attention_function: AttentionFunction | AttentionCallable = AttentionFunction.DEFAULT,
    ):
        super().__init__()

        self.idx = idx
        if video is not None:
            self.attn1 = Attention(
                query_dim=video.dim,
                heads=video.heads,
                dim_head=video.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
            )
            self.attn2 = Attention(
                query_dim=video.dim,
                context_dim=video.context_dim,
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
            )
            self.ff = FeedForward(video.dim, dim_out=video.dim)
            self.scale_shift_table = torch.nn.Parameter(torch.empty(6, video.dim))

        if audio is not None:
            self.audio_attn1 = Attention(
                query_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
            )
            self.audio_attn2 = Attention(
                query_dim=audio.dim,
                context_dim=audio.context_dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
            )
            self.audio_ff = FeedForward(audio.dim, dim_out=audio.dim)
            self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(6, audio.dim))

        if audio is not None and video is not None:
            # Q: Video, K,V: Audio
            self.audio_to_video_attn = Attention(
                query_dim=video.dim,
                context_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
            )

            # Q: Audio, K,V: Video
            self.video_to_audio_attn = Attention(
                query_dim=audio.dim,
                context_dim=video.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
            )

            self.scale_shift_table_a2v_ca_audio = torch.nn.Parameter(torch.empty(5, audio.dim))
            self.scale_shift_table_a2v_ca_video = torch.nn.Parameter(torch.empty(5, video.dim))

        self.norm_eps = norm_eps

    def get_ada_values(
        self, scale_shift_table: torch.Tensor, batch_size: int, timestep: torch.Tensor, indices: slice
    ) -> tuple[torch.Tensor, ...]:
        num_ada_params = scale_shift_table.shape[0]

        ada_values = (
            scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
        ).unbind(dim=2)
        return ada_values

    def get_av_ca_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        scale_shift_timestep: torch.Tensor,
        gate_timestep: torch.Tensor,
        num_scale_shift_values: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scale_shift_ada_values = self.get_ada_values(
            scale_shift_table[:num_scale_shift_values, :], batch_size, scale_shift_timestep, slice(None, None)
        )
        gate_ada_values = self.get_ada_values(
            scale_shift_table[num_scale_shift_values:, :], batch_size, gate_timestep, slice(None, None)
        )

        scale_shift_chunks = [t.squeeze(2) for t in scale_shift_ada_values]
        gate_ada_values = [t.squeeze(2) for t in gate_ada_values]

        return (*scale_shift_chunks, *gate_ada_values)

    def forward(  # noqa: PLR0915
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig | None = None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        if video is None and audio is None:
            raise ValueError("At least one of video or audio must be provided")

        batch_size = (video or audio).x.shape[0]

        if perturbations is None:
            perturbations = BatchedPerturbationConfig.empty(batch_size)

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None

        run_vx = video is not None and video.enabled and vx.numel() > 0
        run_ax = audio is not None and audio.enabled and ax.numel() > 0

        run_a2v = run_vx and (audio is not None and ax.numel() > 0)
        run_v2a = run_ax and (video is not None and vx.numel() > 0)

        if run_vx:
            if not perturbations.all_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx):
                norm_vx = modulated_rms_norm_from_ada(
                    vx,
                    self.scale_shift_table,
                    video.timesteps,
                    scale_index=1,
                    shift_index=0,
                    num_ada_params=6,
                    eps=self.norm_eps,
                )
                v_mask = perturbations.mask_like(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx, vx)
                vx = gated_residual_from_ada(
                    vx,
                    self.attn1(norm_vx, pe=video.positional_embeddings),
                    self.scale_shift_table,
                    video.timesteps,
                    gate_index=2,
                    num_ada_params=6,
                    mask=v_mask,
                )

            vx = vx + self.attn2(rms_norm(vx, eps=self.norm_eps), context=video.context, mask=video.context_mask)

        if run_ax:
            if not perturbations.all_in_batch(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx):
                norm_ax = modulated_rms_norm_from_ada(
                    ax,
                    self.audio_scale_shift_table,
                    audio.timesteps,
                    scale_index=1,
                    shift_index=0,
                    num_ada_params=6,
                    eps=self.norm_eps,
                )
                a_mask = perturbations.mask_like(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx, ax)
                ax = gated_residual_from_ada(
                    ax,
                    self.audio_attn1(norm_ax, pe=audio.positional_embeddings),
                    self.audio_scale_shift_table,
                    audio.timesteps,
                    gate_index=2,
                    num_ada_params=6,
                    mask=a_mask,
                )

            ax = ax + self.audio_attn2(rms_norm(ax, eps=self.norm_eps), context=audio.context, mask=audio.context_mask)

        # Audio - Video cross attention.
        if run_a2v or run_v2a:
            vx_norm3 = rms_norm(vx, eps=self.norm_eps)
            ax_norm3 = rms_norm(ax, eps=self.norm_eps)

            if run_a2v and not perturbations.all_in_batch(PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx):
                vx_scaled = modulate_from_ada(
                    vx_norm3,
                    self.scale_shift_table_a2v_ca_video,
                    video.cross_scale_shift_timestep,
                    scale_index=0,
                    shift_index=1,
                    num_ada_params=4,
                )
                ax_scaled = modulate_from_ada(
                    ax_norm3,
                    self.scale_shift_table_a2v_ca_audio,
                    audio.cross_scale_shift_timestep,
                    scale_index=0,
                    shift_index=1,
                    num_ada_params=4,
                )
                a2v_mask = perturbations.mask_like(PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx, vx)
                vx = gated_residual_from_ada(
                    vx,
                    self.audio_to_video_attn(
                        vx_scaled,
                        context=ax_scaled,
                        pe=video.cross_positional_embeddings,
                        k_pe=audio.cross_positional_embeddings,
                    ),
                    self.scale_shift_table_a2v_ca_video[4:],
                    video.cross_gate_timestep,
                    gate_index=0,
                    num_ada_params=1,
                    mask=a2v_mask,
                )

            if run_v2a and not perturbations.all_in_batch(PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx):
                ax_scaled = modulate_from_ada(
                    ax_norm3,
                    self.scale_shift_table_a2v_ca_audio,
                    audio.cross_scale_shift_timestep,
                    scale_index=2,
                    shift_index=3,
                    num_ada_params=4,
                )
                vx_scaled = modulate_from_ada(
                    vx_norm3,
                    self.scale_shift_table_a2v_ca_video,
                    video.cross_scale_shift_timestep,
                    scale_index=2,
                    shift_index=3,
                    num_ada_params=4,
                )
                v2a_mask = perturbations.mask_like(PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx, ax)
                ax = gated_residual_from_ada(
                    ax,
                    self.video_to_audio_attn(
                        ax_scaled,
                        context=vx_scaled,
                        pe=audio.cross_positional_embeddings,
                        k_pe=video.cross_positional_embeddings,
                    ),
                    self.scale_shift_table_a2v_ca_audio[4:],
                    audio.cross_gate_timestep,
                    gate_index=0,
                    num_ada_params=1,
                    mask=v2a_mask,
                )

        if run_vx:
            vx_scaled = modulated_rms_norm_from_ada(
                vx,
                self.scale_shift_table,
                video.timesteps,
                scale_index=4,
                shift_index=3,
                num_ada_params=6,
                eps=self.norm_eps,
            )
            vx = gated_residual_from_ada(
                vx,
                self.ff(vx_scaled),
                self.scale_shift_table,
                video.timesteps,
                gate_index=5,
                num_ada_params=6,
            )

        if run_ax:
            ax_scaled = modulated_rms_norm_from_ada(
                ax,
                self.audio_scale_shift_table,
                audio.timesteps,
                scale_index=4,
                shift_index=3,
                num_ada_params=6,
                eps=self.norm_eps,
            )
            ax = gated_residual_from_ada(
                ax,
                self.audio_ff(ax_scaled),
                self.audio_scale_shift_table,
                audio.timesteps,
                gate_index=5,
                num_ada_params=6,
            )

        return replace(video, x=vx) if video is not None else None, replace(audio, x=ax) if audio is not None else None

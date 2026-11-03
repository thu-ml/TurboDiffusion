from __future__ import annotations

from dataclasses import dataclass

import math
import numpy as np
import torch

from turbodiffusion.h3.dit.config import H3DiTConfig, MODALITY_AUDIO, MODALITY_TEXT, MODALITY_VIDEO, PACKED_SEQUENCE_ALIGNMENT
from turbodiffusion.h3.resolution import (  # noqa: F401
    H3_CANVAS_MAX_PIXELS,
    H3_CANVAS_SHORT_EDGE,
    model_canvas,
    resolve_canvas_size,
)

# Official H3 RoPE grid constants (ComfyUI / native packing).
_INTERP = 32.0
_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
_FRAME_RESCALE = 5.0 / 3.0
_AUDIO_CHANNELS = 2
# Official RF condition mixture: 1.0 = fully clean keyframe rows (FL2VA I2VA)
H3_IMGVID_COND_TIMESTEP = 1.0
# The keyframe layouts FL2VA packs, in packed order: the frame the video starts
# from, the frame it ends on, or both. ``-1`` is the last pixel frame, resolved
# against ``frame_count``. Nothing else is a layout the checkpoint was trained on,
# so a keyframe asked for anywhere in between is refused rather than approximated.
H3_FL2VA_KEYFRAME_ANCHORS: tuple[tuple[int, ...], ...] = ((0,), (-1,), (0, -1))
def target_latent_hw(
    short_edge: int,
    aspect_ratio: str,
    *,
    resolution: str | None = None,
    canvas_latent_hw: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Use the encoded keyframe canvas, or resolve the requested latent size."""
    if canvas_latent_hw is None:
        return compute_latent_hw(short_edge, aspect_ratio, resolution=resolution)
    lh, lw = (int(x) for x in canvas_latent_hw)
    if lh <= 0 or lw <= 0 or lh % 2 or lw % 2:
        raise ValueError(
            f"canvas_latent_hw must be positive and even (the 2x2 patch), got {(lh, lw)}"
        )
    return lh, lw


def _parse_aspect(aspect_ratio: str) -> tuple[int, int]:
    w, h = aspect_ratio.split(":")
    return int(w), int(h)


def compute_latent_hw(
    short_edge: int,
    aspect_ratio: str,
    spatial_compress: int = 16,
    max_pixels: int | None = None,
    resolution: str | None = None,
) -> tuple[int, int]:
    """Target canvas of a request → ``(latent_h, latent_w)``.

    ``resolution`` names an area tier ("720p"); unset it is H3's own short-edge rule,
    where ``max_pixels`` defaults to the area of the widest canvas that short edge
    reaches — exactly H3's ``768 * 1344`` at the released one, so a sweep that raises
    ``short_edge`` still gets a proportionally bigger canvas.
    """
    aw, ah = _parse_aspect(aspect_ratio)
    pixel_h, pixel_w = model_canvas(
        aw, ah, resolution=resolution, short_edge=short_edge, max_pixels=max_pixels
    )
    return pixel_h // spatial_compress, pixel_w // spatial_compress




def aligned_frame_count(duration_seconds: float, fps: int = 24) -> int:
    """Round duration up to H3's ``17*n + 5`` pixel-frame lattice."""
    requested = max(1, round(float(duration_seconds) * fps))
    if requested <= 5:
        return 5
    return math.ceil((requested - 5) / 17) * 17 + 5




def video_latent_time(frame_count: int) -> int:
    """Official latent temporal length from aligned frame_count."""
    frame_count = int(frame_count)
    if frame_count <= 5:
        return 2
    if (frame_count - 5) % 17:
        raise ValueError(f"frame_count={frame_count} not on 17n+5 lattice")
    return ((frame_count - 5) // 17) * 5 + 2






def audio_latent_time(frame_count: int, fps: int = 24) -> int:
    return int(round((frame_count / fps) * 40))


def _sqrt_area(height: int, width: int) -> float:
    return float(np.sqrt(height * width))


def _axis_from_sqrt_area(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    right = left + ratio
    grid = np.linspace(left, right, dim // patch, endpoint=False) * _INTERP
    return torch.from_numpy(grid).to(torch.float64)


def _video_t_spans(length: int) -> torch.Tensor:
    return torch.tensor(
        [_FRAME_RESCALE * _FRAME_PER_TOKEN[i % len(_FRAME_PER_TOKEN)] for i in range(length)],
        dtype=torch.float64,
    )


def _video_t_grid(length: int, origin: float) -> torch.Tensor:
    spans = _video_t_spans(length)
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])




def _keyframe_anchor_span(length: int) -> float:
    """Match the official last-keyframe anchor's pairwise NumPy sum."""
    spans = np.ones(int(length), dtype=np.float64) * _FRAME_RESCALE
    group = len(_FRAME_PER_TOKEN)
    for i in range(group):
        spans[i::group] *= _FRAME_PER_TOKEN[i]
    return float(spans.sum())


def _audio_width_positions(*, steps: int, channels: int, left: float, right: float) -> torch.Tensor:
    if steps == 0:
        return torch.empty(0, dtype=torch.float64)
    return torch.cat(
        (
            torch.full((steps,), left, dtype=torch.float64),
            torch.full((steps,), right, dtype=torch.float64),
        )
    )


@dataclass
class PackedBatch:
    video_latents: torch.Tensor  # [1, C, T, H, W] target noise (shape metadata)
    audio_latents: torch.Tensor  # [2, Ta, Ca]
    text_tokens: torch.Tensor  # [Tt, text_dim]
    position_ids: torch.Tensor  # [S, 3]
    token_tags: torch.Tensor  # [S]
    video_mask: torch.Tensor  # True for cond + target visual rows
    audio_mask: torch.Tensor
    text_mask: torch.Tensor
    inverse_indices: torch.Tensor
    arch: H3DiTConfig
    patch_t: int
    patch_h: int
    patch_w: int
    audio_t: int
    # FL2VA keyframe conditioning.
    update_mask: torch.Tensor | None = None  # over visual rows [n_cond+n_target]; True=denoise
    visual_cond_rows: torch.Tensor | None = None  # [n_cond, 96] clean DiT-space
    n_cond_rows: int = 0
    n_target_rows: int = 0
    frame_count: int = 0
    task: str = "i2va"

    def patchify_video(self, video: torch.Tensor) -> torch.Tensor:
        # video: [1,C,T,H,W] -> [N, C*pt*ph*pw]  (official: nctrhpwq -> nthwcrpq)
        pt, ph, pw = self.arch.patch_size
        _, c, t, h, w = video.shape
        assert t % pt == 0 and h % ph == 0 and w % pw == 0
        x = video.view(1, c, t // pt, pt, h // ph, ph, w // pw, pw)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        return x.view(-1, c * pt * ph * pw)

    def unpatchify_video(self, tokens: torch.Tensor) -> torch.Tensor:
        pt, ph, pw = self.arch.patch_size
        _, c, t, h, w = self.video_latents.shape
        nt, nh, nw = t // pt, h // ph, w // pw
        d = tokens.view(1, nt, nh, nw, c, pt, ph, pw)
        return d.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(1, c, t, h, w)

    def pack_audio_rows(self, audio: torch.Tensor) -> torch.Tensor:
        """``[2, Ta, Ca]`` → channel-major rows ``[2*Ta, Ca]``."""
        assert audio.ndim == 3 and audio.shape[0] == _AUDIO_CHANNELS
        # [C, T, D] → [C*T, D]
        return audio.reshape(-1, audio.shape[-1]).contiguous()

    def unpack_audio_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """``[2*Ta, Ca]`` → ``[2, Ta, Ca]``."""
        return rows.view(_AUDIO_CHANNELS, self.audio_t, -1).contiguous()


def _align_up(n: int, align: int) -> int:
    return ((n + align - 1) // align) * align




def build_i2va_packed(
    *,
    prompt_embeds: torch.Tensor,
    visual_cond_rows: torch.Tensor,
    duration_seconds: float,
    short_edge: int,
    aspect_ratio: str,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    arch: H3DiTConfig,
    keyframe_anchors: tuple[int, ...] = (0,),
    resolution: str | None = None,
    canvas_latent_hw: tuple[int, int] | None = None,
) -> PackedBatch:
    """Official FL2VA pack: ``[text | keyframe_cond | audio | target_video | pad]``.

    ``visual_cond_rows``: clean DiT-space patch rows ``[n_keyframes*Nh*Nw, 96]``, one
    ``Nh*Nw`` block per keyframe in the order of ``keyframe_anchors``.

    ``keyframe_anchors``: which end of the video each keyframe is, one of
    ``H3_FL2VA_KEYFRAME_ANCHORS``. A keyframe conditions by sitting on the video's
    rotary clock where the frame it depicts would: the opening frame at the video's
    own origin, the closing one a single pixel frame short of the span the video
    consumes. That offset is what makes it the last frame rather than the first
    frame of whatever comes after.
    """
    lh, lw = target_latent_hw(
        short_edge, aspect_ratio, resolution=resolution, canvas_latent_hw=canvas_latent_hw
    )
    frame_count = aligned_frame_count(duration_seconds)
    lt = video_latent_time(frame_count)
    audio_t = audio_latent_time(frame_count)
    pt, ph, pw = arch.patch_size
    lt = max(pt, (lt // pt) * pt)
    lh = max(ph, (lh // ph) * ph)
    lw = max(pw, (lw // pw) * pw)

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    video = torch.randn(1, arch.latents_dim, lt, lh, lw, generator=g, dtype=dtype).to(device)
    audio = torch.randn(_AUDIO_CHANNELS, audio_t, arch.audio_latents_dim, generator=g, dtype=dtype).to(
        device
    )

    text = prompt_embeds.to(device=device, dtype=dtype)
    if text.dim() == 3:
        text = text[0]
    tt = int(text.shape[0])

    anchors = tuple(int(a) for a in keyframe_anchors)
    if anchors not in H3_FL2VA_KEYFRAME_ANCHORS:
        raise ValueError(
            f"keyframe_anchors must be one of {H3_FL2VA_KEYFRAME_ANCHORS}, got {anchors}"
        )

    frame_rows = (lh // ph) * (lw // pw)
    n_cond = len(anchors) * frame_rows
    n_target = (lt // pt) * frame_rows
    if visual_cond_rows.shape[0] != n_cond or visual_cond_rows.shape[1] != arch.video_patch_dim:
        raise ValueError(
            f"visual_cond_rows shape {tuple(visual_cond_rows.shape)} != "
            f"({n_cond}, {arch.video_patch_dim}) for {len(anchors)} keyframe(s) at "
            f"latent_hw=({lh},{lw})"
        )
    cond_rows = visual_cond_rows.to(device=device, dtype=dtype).contiguous()

    na = audio_t * _AUDIO_CHANNELS
    used = tt + n_cond + na + n_target
    total = _align_up(used, PACKED_SEQUENCE_ALIGNMENT)

    text_slice = slice(0, tt)
    cond_slice = slice(tt, tt + n_cond)
    audio_slice = slice(tt + n_cond, tt + n_cond + na)
    video_slice = slice(tt + n_cond + na, tt + n_cond + na + n_target)

    tags = torch.full((total,), -1, device=device, dtype=torch.long)
    tags[text_slice] = MODALITY_TEXT
    tags[cond_slice] = MODALITY_VIDEO  # same modality 0 / video_patch_proj
    tags[audio_slice] = MODALITY_AUDIO
    tags[video_slice] = MODALITY_VIDEO

    pos = torch.zeros(total, 3, device=device, dtype=torch.float32)
    pos_cpu = torch.zeros(total, 3, dtype=torch.float64)
    pos_cpu[text_slice, 0] = torch.arange(tt, dtype=torch.float64)

    sqrt_area = _sqrt_area(lh, lw)
    h_grid = _axis_from_sqrt_area(lh, ph, sqrt_area)
    w_grid = _axis_from_sqrt_area(lw, pw, sqrt_area)
    hh, ww = torch.meshgrid(h_grid, w_grid, indexing="ij")
    spatial_frame = torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)

    # Each keyframe's cond block takes the rotary t of the frame it depicts.
    last_t = float(tt) + _keyframe_anchor_span(lt // pt) - _FRAME_RESCALE
    for block, anchor in enumerate(anchors):
        sl = slice(cond_slice.start + block * frame_rows,
                   cond_slice.start + (block + 1) * frame_rows)
        pos_cpu[sl, 0] = float(tt) if anchor == 0 else last_t
        pos_cpu[sl, 1:] = spatial_frame

    video_grid = torch.empty(lt // pt, frame_rows, 3, dtype=torch.float64)
    video_grid[:, :, 0] = _video_t_grid(lt // pt, float(tt))[:, None]
    video_grid[:, :, 1:] = spatial_frame[None]
    pos_cpu[video_slice] = video_grid.reshape(-1, 3)

    audio_t_grid = float(tt) + torch.arange(audio_t, dtype=torch.float64)
    pos_cpu[audio_slice, 0] = audio_t_grid.repeat(_AUDIO_CHANNELS)
    pos_cpu[audio_slice, 2] = _audio_width_positions(
        steps=audio_t,
        channels=_AUDIO_CHANNELS,
        left=float(w_grid[0]),
        right=float(w_grid[-1]),
    )
    pos.copy_(pos_cpu.to(device=device, dtype=torch.float32))

    video_mask = tags == MODALITY_VIDEO
    audio_mask = tags == MODALITY_AUDIO
    text_mask = tags == MODALITY_TEXT
    # visual update: [cond=False..., target=True...]
    update = torch.zeros(n_cond + n_target, device=device, dtype=torch.bool)
    update[n_cond:] = True

    return PackedBatch(
        video_latents=video,
        audio_latents=audio,
        text_tokens=text,
        position_ids=pos,
        token_tags=tags,
        video_mask=video_mask,
        audio_mask=audio_mask,
        text_mask=text_mask,
        inverse_indices=torch.zeros(total, device=device, dtype=torch.long),
        arch=arch,
        patch_t=lt // pt,
        patch_h=lh // ph,
        patch_w=lw // pw,
        audio_t=audio_t,
        update_mask=update,
        visual_cond_rows=cond_rows,
        n_cond_rows=n_cond,
        n_target_rows=n_target,
        frame_count=frame_count,
        task="i2va",
    )

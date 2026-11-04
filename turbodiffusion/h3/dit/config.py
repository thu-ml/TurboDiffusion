from __future__ import annotations

from dataclasses import dataclass


MODALITY_VIDEO = 0
MODALITY_TEXT = 1
MODALITY_AUDIO = 2
ADALN_MODALITY_NUM = 3
PACKED_SEQUENCE_ALIGNMENT = 64


@dataclass
class H3DiTConfig:
    num_layers: int = 50
    token_refiner_num_layers: int = 2
    hidden_size: int = 5376
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_hidden_size: int = 14336
    latents_dim: int = 24
    audio_latents_dim: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 2688
    adaln_out_features: int = 18 * 5376
    final_adaln_out_features: int = 2 * 5376
    rope_inv_freq_len: int = 16
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5

    @property
    def video_patch_dim(self) -> int:
        pt, ph, pw = self.patch_size
        return self.latents_dim * pt * ph * pw

    @classmethod
    def from_json(cls, path: str) -> "H3DiTConfig":
        import json

        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        patch = raw.get("patch_size", [1, 2, 2])
        return cls(
            num_layers=int(raw.get("num_layers", 50)),
            token_refiner_num_layers=int(raw.get("token_refiner_num_layers", raw.get("num_refiner_layers", 2))),
            hidden_size=int(raw.get("hidden_size", 5376)),
            num_attention_heads=int(raw.get("num_attention_heads", 56)),
            attention_head_dim=int(raw.get("attention_head_dim", 128)),
            ffn_hidden_size=int(raw.get("ffn_hidden_size", raw.get("ffn_dim", 14336))),
            latents_dim=int(raw.get("latents_dim", raw.get("in_channels", 24))),
            audio_latents_dim=int(raw.get("audio_latents_dim", raw.get("audio_in_channels", 32))),
            patch_size=tuple(patch),
            text_dim=int(raw.get("text_dim", 5120)),
            timestep_input_dim=int(raw.get("timestep_input_dim", 256)),
            time_embed_hidden_size=int(raw.get("time_embed_hidden_size", raw.get("time_embed_hidden_dim", 5376))),
            time_embed_dim=int(raw.get("time_embed_dim", 2688)),
            adaln_out_features=int(raw.get("adaln_out_features", 18 * 5376)),
            final_adaln_out_features=int(raw.get("final_adaln_out_features", 2 * 5376)),
            rope_inv_freq_len=int(raw.get("rope_inv_freq_len", raw.get("rope_freq_dim", 16))),
            norm_eps=float(raw.get("norm_eps", 1e-5)),
            qk_norm_eps=float(raw.get("qk_norm_eps", 1e-5)),
            final_norm_eps=float(raw.get("final_norm_eps", 1e-5)),
        )

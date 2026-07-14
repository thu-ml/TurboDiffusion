from types import SimpleNamespace

import pytest
import torch

from ltx_distillation.acceleration import (
    _linear_name_in_quant_scope,
    _parse_sla_topk_schedule,
    _scheduled_sla_topk,
    replace_ltx_attention,
)
from ltx_distillation.models.text_encoder_wrapper import GemmaTextEncoderWrapper


class _DummyTextEncoder(torch.nn.Module):
    def forward(self, text: str, padding_side: str) -> SimpleNamespace:
        del text, padding_side
        video = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
        audio = video + 100
        mask = torch.tensor([[0, 1, 1, 0]], dtype=torch.int64)
        return SimpleNamespace(video_encoding=video, audio_encoding=audio, attention_mask=mask)


def test_text_context_trimming_is_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TURBOT2AV_TRIM_TEXT_CONTEXT", raising=False)
    wrapper = GemmaTextEncoderWrapper(_DummyTextEncoder())

    result = wrapper(["prompt"])

    assert result["video_context"].shape == (1, 4, 2)
    assert result["audio_context"].shape == (1, 4, 2)
    assert result["attention_mask"] is not None


def test_text_context_trimming_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TURBOT2AV_TRIM_TEXT_CONTEXT", "1")
    wrapper = GemmaTextEncoderWrapper(_DummyTextEncoder())

    result = wrapper(["prompt"])

    expected_video = torch.tensor([[[2.0, 3.0], [4.0, 5.0]]])
    assert torch.equal(result["video_context"], expected_video)
    assert torch.equal(result["audio_context"], expected_video + 100)
    assert result["attention_mask"] is None


def test_sla_topk_schedule_selects_matching_layers() -> None:
    schedule = _parse_sla_topk_schedule("0-15:0.35,16-31:0.3,40:0.25")

    assert _scheduled_sla_topk("model.transformer_blocks.7.attn1", 0.5, schedule) == 0.35
    assert _scheduled_sla_topk("model.transformer_blocks.20.attn1", 0.5, schedule) == 0.3
    assert _scheduled_sla_topk("model.transformer_blocks.40.attn1", 0.5, schedule) == 0.25
    assert _scheduled_sla_topk("model.transformer_blocks.35.attn1", 0.5, schedule) == 0.5


@pytest.mark.parametrize(
    ("schedule", "message"),
    [
        ("3-1:0.3", "Invalid layer range"),
        ("0-2:0", "values must be in"),
        ("0-2:1.1", "values must be in"),
        ("bad", "entries must use"),
    ],
)
def test_sla_topk_schedule_rejects_invalid_values(schedule: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _parse_sla_topk_schedule(schedule)


def test_quantization_scopes_select_expected_linears() -> None:
    video_ffn = "model.transformer_blocks.0.ff.net.0.proj"
    audio_ffn = "model.transformer_blocks.0.audio_ff.net.0.proj"
    video_attn = "model.transformer_blocks.0.attn1.to_q"

    assert _linear_name_in_quant_scope(video_ffn, "video_ffn")
    assert not _linear_name_in_quant_scope(audio_ffn, "video_ffn")
    assert _linear_name_in_quant_scope(audio_ffn, "audio_ffn")
    assert _linear_name_in_quant_scope(video_attn, "video_heavy")
    assert not _linear_name_in_quant_scope(video_attn, "non_attention")


@pytest.mark.parametrize("topk", [0.0, -0.1, 1.1])
def test_sla_topk_rejects_values_outside_unit_interval(topk: float) -> None:
    with pytest.raises(ValueError, match="must be in"):
        replace_ltx_attention(torch.nn.Module(), attention_type="sagesla", sla_topk=topk)

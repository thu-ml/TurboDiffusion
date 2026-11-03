"""Small-model coverage for the bundled H3 implementation."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from turbodiffusion.h3.components import load_vae_module
from turbodiffusion.h3.dit.adaln_bake import load_adaln_bake, save_adaln_bake, schedule_timesteps
from turbodiffusion.h3.dit.config import H3DiTConfig
from turbodiffusion.h3.dit.dbcache import DBCacheConfig, DBCacheState
from turbodiffusion.h3.dit.model import MiniMaxH3Attention, MiniMaxH3DiTModel
from turbodiffusion.h3.dit.pack import build_i2va_packed
from turbodiffusion.h3.dit.processor import H3DiTProcessor, _shifted_sigma_schedule
from turbodiffusion.h3.dit.turbo_int8 import FORMAT, apply_turbo_int8, is_mlp_alias_key, load_fl2va_turbo_int8
from turbodiffusion.h3.dit.weight_loader import (
    FP32_PROJECTIONS, _diffusers_key, load_fl2va_streaming, materialize_cpu_model,
)
from turbodiffusion.h3.vae import apply_video_vae_w8a8, normalize_latents, pixels_to_uint8


def tiny_config(**overrides):
    return H3DiTConfig(**{
        "num_layers": 2, "token_refiner_num_layers": 1,
        "hidden_size": 32, "ffn_hidden_size": 64,
        "num_attention_heads": 1, "attention_head_dim": 32,
        "latents_dim": 4, "audio_latents_dim": 4, "text_dim": 16,
        "timestep_input_dim": 16, "time_embed_hidden_size": 32,
        "time_embed_dim": 16, "adaln_out_features": 18 * 32,
        "final_adaln_out_features": 2 * 32, "rope_inv_freq_len": 4,
        **overrides,
    })


def tiny_video_vae():
    model = torch.nn.Module()
    model.encoder = torch.nn.Conv3d(3, 24, 1)
    model.post_quant_conv = torch.nn.Conv3d(24, 24, 1)
    model.decoder = torch.nn.Module()
    model.decoder.x_embedder = torch.nn.Linear(24, 256)
    model.decoder.proj_out = torch.nn.Linear(256, 192)
    model.decoder.transformer_blocks = torch.nn.ModuleList()
    for _ in range(2):
        block = torch.nn.Module()
        block.attn = torch.nn.Module()
        block.attn.to_qkv = torch.nn.Linear(256, 768)
        block.attn.to_out = torch.nn.Linear(256, 256)
        block.ff = torch.nn.Module()
        block.ff.w1 = torch.nn.Linear(256, 1024)
        block.ff.w2 = torch.nn.Linear(512, 256)
        block.norm1 = torch.nn.LayerNorm(256)
        model.decoder.transformer_blocks.append(block)
    return model.bfloat16().eval()


class H3RuntimeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.arch = tiny_config()

    def model(self):
        model = MiniMaxH3DiTModel(self.arch, attn_backend="sdpa").eval()
        model.rope.inv_freq.fill_(0.01)
        return model

    def test_four_updates_and_frozen_keyframe_with_real_model(self):
        model = self.model()
        processor = H3DiTProcessor({"dit": {"model_path": ".", "attn_backend": "sdpa"}})
        processor.model = model
        processor.device = torch.device("cpu")
        processor.dtype = torch.float32
        calls = []
        handle = model.register_forward_pre_hook(
            lambda module, args, kwargs: calls.append(kwargs["video_tokens"][:1].clone()),
            with_kwargs=True,
        )
        self.addCleanup(handle.remove)
        condition = torch.randn(1, self.arch.video_patch_dim)
        video, audio = processor.sample_i2va(
            torch.randn(3, self.arch.text_dim), condition, duration_seconds=0.1,
            steps=5, canvas_latent_hw=(2, 2),
        )
        self.assertEqual(len(calls), 4)
        for rows in calls:
            torch.testing.assert_close(rows, condition)
        self.assertEqual(video.shape[:2], (1, 4))
        self.assertEqual(audio.shape[0], 2)
        self.assertTrue(torch.isfinite(video).all())
        self.assertTrue(torch.isfinite(audio).all())

    def test_packing_is_seeded_and_masks_do_not_overlap(self):
        kwargs = dict(
            prompt_embeds=torch.randn(3, 16), visual_cond_rows=torch.ones(1, 16),
            duration_seconds=0.1, short_edge=32, aspect_ratio="1:1", seed=12,
            device=torch.device("cpu"), dtype=torch.float32, arch=self.arch,
            canvas_latent_hw=(2, 2),
        )
        first, second = build_i2va_packed(**kwargs), build_i2va_packed(**kwargs)
        torch.testing.assert_close(first.video_latents, second.video_latents)
        self.assertEqual(first.token_tags.numel() % 64, 0)
        self.assertFalse((first.video_mask & first.audio_mask).any())
        self.assertFalse(first.update_mask[0])
        self.assertTrue(first.update_mask[1:].all())

    def test_sparge_sinks_reach_offload_buffers_for_each_sample(self):
        model = self.model()
        buffers = torch.nn.ModuleList([copy.deepcopy(model.blocks[0]) for _ in range(2)])
        processor = H3DiTProcessor({"dit": {
            "model_path": ".", "attn_backend": "sparge_attn",
            "sparge_attn_topk": 0.25, "sparge_attn_post_sink_length": 7,
        }})
        processor.model = model
        processor.offloader = types.SimpleNamespace(bufs=buffers)
        for text_rows in (3, 5):
            packed = build_i2va_packed(
                prompt_embeds=torch.randn(text_rows, 16), visual_cond_rows=torch.ones(1, 16),
                duration_seconds=0.1, short_edge=32, aspect_ratio="1:1", seed=12,
                device=torch.device("cpu"), dtype=torch.float32, arch=self.arch,
                canvas_latent_hw=(2, 2),
            )
            processor._apply_sparge_sinks(packed)
            expected = text_rows + int(packed.audio_mask.sum()) + packed.n_cond_rows
            for block in [*model.blocks, *buffers]:
                self.assertEqual(block.attn.sparge_attn_pre_sink_length, expected)
                self.assertEqual(block.attn.sparge_attn_post_sink_length, 7)
                self.assertEqual(block.attn.sparge_attn_topk, 0.25)
            self.assertEqual(model.token_refiner.blocks[0].attn.sparge_attn_pre_sink_length, 0)

    def test_long_prompt_refiner_is_dense_and_never_communicates(self):
        model = MiniMaxH3DiTModel(self.arch, attn_backend="sparge_attn").eval()
        model.configure_sparge(topk=0.25, pre_sink_length=129)
        refiner = model.token_refiner
        self.assertEqual(refiner.blocks[0].attn.attn_backend, "sdpa")
        self.assertFalse(refiner.blocks[0].attn.sequence_parallel)
        text = torch.randn(513, self.arch.hidden_size)
        with patch("turbodiffusion.h3.dit.model.ulysses_pre_attn", side_effect=AssertionError), patch(
            "turbodiffusion.h3.dit.model.ulysses_pre_attn_launch", side_effect=AssertionError
        ), patch("turbodiffusion.h3.dit.model.ulysses_post_attn", side_effect=AssertionError), patch(
            "turbodiffusion.h3.dit.model.get_parallel_state",
            return_value=types.SimpleNamespace(use_async_comm=True),
        ):
            output = refiner(text)
        block = refiner.blocks[0]
        normalized = block.norm1(text)
        q, k, v = block.attn.qkv_proj(normalized).chunk(3, -1)
        q, k, v = (tensor.view(513, 1, 32) for tensor in (q, k, v))
        attention = torch.nn.functional.scaled_dot_product_attention(
            block.attn.q_norm(q).transpose(0, 1),
            block.attn.k_norm(k).transpose(0, 1), v.transpose(0, 1),
        ).transpose(0, 1).reshape(513, 32)
        expected = text + block.attn.out_proj(attention)
        expected = refiner.final_norm(expected + block.mlp(block.norm2(expected)))
        torch.testing.assert_close(output, expected)

    def test_padding_does_not_change_valid_attention_outputs(self):
        attention = MiniMaxH3Attention(self.arch).eval()
        x = torch.randn(64, 32)
        expected = attention(x[:17], cos=None, sin=None)
        actual = attention(x, cos=None, sin=None, sequence_length=17)
        torch.testing.assert_close(actual[:17], expected)
        torch.testing.assert_close(actual[17:], torch.zeros_like(actual[17:]))

    def test_rope_preserves_fp32_values_across_casts_and_meta_materialization(self):
        model = self.model()
        frequencies = torch.tensor([1., 0.1234567, 0.01234567, 0.001234567])
        model.rope.inv_freq.copy_(frequencies)
        model.bfloat16()
        self.assertEqual(model.rope.inv_freq.dtype, torch.float32)
        torch.testing.assert_close(model.rope.inv_freq, frequencies, rtol=0, atol=0)
        positions = torch.tensor([[1024., 500., 400.]])
        expected_half = (positions.unsqueeze(-1) * frequencies).flatten(1)
        torch.testing.assert_close(
            model.rope(positions), torch.cat([expected_half, expected_half], -1), rtol=0, atol=0
        )
        with torch.device("meta"):
            empty = MiniMaxH3DiTModel(self.arch)
        materialize_cpu_model(empty)
        self.assertFalse(empty.rope.inv_freq.is_meta)
        self.assertEqual(empty.rope.inv_freq.dtype, torch.float32)

    def test_streaming_loader_preserves_original_mixed_precision(self):
        original = self.model().bfloat16()
        for name in ("video_patch_proj", "audio_patch_proj", "time_embedder",
                     "final_layer.video_out", "final_layer.audio_out"):
            original.get_submodule(name).float()
        tensors = {name: tensor.contiguous().clone() for name, tensor in original.state_dict().items()}
        filename = "model.safetensors"
        save_file(tensors, str(self.root / filename))
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: filename for key in tensors}})
        )
        with torch.device("meta"):
            restored = MiniMaxH3DiTModel(self.arch)
        materialize_cpu_model(restored)
        _, missing = load_fl2va_streaming(restored, self.root)
        self.assertEqual(missing, 0)
        for name, parameter in restored.named_parameters():
            dtype = torch.float32 if name.startswith(FP32_PROJECTIONS) else torch.bfloat16
            self.assertEqual(parameter.dtype, dtype, name)
            if dtype == torch.float32:
                torch.testing.assert_close(parameter, tensors[name], rtol=0, atol=0)
        torch.testing.assert_close(restored.rope.inv_freq, original.rope.inv_freq, rtol=0, atol=0)

    def test_legacy_w8a8_fp32_loss_requires_reexport(self):
        model = self.model()
        tensors = {name: tensor.contiguous().clone() for name, tensor in model.state_dict().items()}
        tensors["rope.inv_freq"] = tensors["rope.inv_freq"].bfloat16()
        path = self.root / "legacy.safetensors"
        save_file(tensors, str(path), metadata={"format": FORMAT})
        with self.assertRaisesRegex(ValueError, "re-export"):
            load_fl2va_turbo_int8(model, path)

    def test_streaming_loader_rejects_rounded_fp32_source_projections(self):
        model = self.model()
        filename = "model.safetensors"
        save_file(
            {"video_patch_proj.weight": model.video_patch_proj.weight.bfloat16().contiguous()},
            str(self.root / filename),
        )
        (self.root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"video_patch_proj.weight": filename},
        }))
        with self.assertRaisesRegex(ValueError, "original FP32"):
            load_fl2va_streaming(model, self.root)

    def test_baked_adaln_matches_uncached_sampling(self):
        model = self.model()
        prompt = torch.randn(3, self.arch.text_dim)
        condition = torch.randn(1, self.arch.video_patch_dim)
        results = []
        for baked in (False, True):
            processor = H3DiTProcessor({"dit": {
                "model_path": ".", "attn_backend": "sdpa", "bake_adaln": baked,
            }})
            processor.model = copy.deepcopy(model)
            processor.device = torch.device("cpu")
            processor.dtype = torch.float32
            results.append(processor.sample_i2va(
                prompt, condition, duration_seconds=0.1, steps=5, canvas_latent_hw=(2, 2),
            ))
        for expected, actual in zip(*results):
            torch.testing.assert_close(expected, actual)

    def test_adaln_roundtrip_and_schedule(self):
        schedule = schedule_timesteps("i2va", steps=5, video_shift=12, audio_shift=3)
        self.assertEqual(len(schedule), 4)
        sigmas = _shifted_sigma_schedule(5, 12, device=torch.device("cpu"))
        self.assertEqual(float(sigmas[0]), 1.0)
        self.assertEqual(float(sigmas[-1]), 0.0)
        self.assertTrue(torch.all(sigmas[:-1] > sigmas[1:]))
        model = self.model()
        blocks, final = model.bake_and_drop_adaln(
            schedule, device=torch.device("cpu"), dtype=torch.float32
        )
        path = self.root / "adaln.safetensors"
        save_adaln_bake(
            path, blocks, final, task="i2va", steps=5, video_shift=12, audio_shift=3,
            timesteps_per_step=schedule,
        )
        _, loaded_blocks, loaded_final, meta = load_adaln_bake(path, device="cpu")
        self.assertEqual(meta["n_blocks"], 2)
        for expected, actual in zip(blocks[0][0], loaded_blocks[0][0]):
            torch.testing.assert_close(expected, actual)
        for expected, actual in zip(final[-1], loaded_final[-1]):
            torch.testing.assert_close(expected, actual)

    def test_cache_warmup_and_consecutive_hit_limit(self):
        cfg = DBCacheConfig(enabled=True, Fn_compute_blocks=1, Bn_compute_blocks=0,
                            max_warmup_steps=1, max_continuous_cached_steps=1,
                            residual_diff_threshold=0.2)
        cache = DBCacheState(cfg, 2)
        x = torch.ones(3, 4)
        self.assertFalse(cache.decide_hit(x))
        cache.update_mid(x)
        cache.finish_step(hit=False, fn_residual=x)
        self.assertTrue(cache.decide_hit(x))
        cache.finish_step(hit=True, fn_residual=x)
        self.assertFalse(cache.decide_hit(x))
        cache.reset()
        self.assertEqual(cache.hits, 0)
        self.assertIsNone(cache.cached_mid_residual)

    def test_wrap_delegates_to_existing_int8linear_only(self):
        class FakeInt8Linear(torch.nn.Module):
            @classmethod
            def from_linear(cls, original, quantize):
                self.assertFalse(quantize)
                wrapped = cls()
                wrapped.register_buffer("int8_weight", torch.empty_like(original.weight, dtype=torch.int8))
                wrapped.register_buffer("scale", torch.empty(1))
                return wrapped

        native = types.ModuleType("turbodiffusion.ops.core")
        native.Int8Linear = FakeInt8Linear
        with patch.dict(sys.modules, {"turbodiffusion.ops.core": native}):
            model = self.model()
            self.assertEqual(apply_turbo_int8(model, quantize=False), 8)
        self.assertIsInstance(model.blocks[0].attn.qkv_proj, FakeInt8Linear)
        self.assertIs(model.blocks[0].mlp.fc1, model.blocks[0].mlp.w1)
        self.assertIsInstance(model.token_refiner.blocks[0].mlp.fc1, torch.nn.Linear)
        self.assertIsInstance(model.final_layer.video_out, torch.nn.Linear)

    def test_weight_loader_combines_diffusers_qkv(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([torch.nn.Module()])
                self.blocks[0].attn = torch.nn.Module()
                self.blocks[0].attn.qkv_proj = torch.nn.Linear(2, 6, bias=False)
                self.rope = torch.nn.Module()
                self.rope.register_buffer("inv_freq", torch.zeros(2))

        tensors = {
            f"transformer_blocks.0.attn.to_{part}.weight": torch.full((2, 2), value)
            for part, value in zip("qkv", (1., 2., 3.))
        }
        filename = "diffusion_pytorch_model.safetensors"
        save_file(tensors, str(self.root / filename))
        (self.root / "diffusion_pytorch_model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: filename for key in tensors}})
        )
        model = Tiny()
        self.assertEqual(load_fl2va_streaming(model, self.root), (1, 0))
        torch.testing.assert_close(model.blocks[0].attn.qkv_proj.weight, torch.cat(list(tensors.values())))
        self.assertEqual(_diffusers_key("norm_out.linear.weight")[0], "final_layer.adaln_proj.linear.weight")

    def test_wrong_quantization_format_rejected(self):
        path = self.root / "wrong.safetensors"
        save_file({"x": torch.ones(1)}, str(path), metadata={"format": "other"})
        with self.assertRaisesRegex(ValueError, FORMAT):
            load_fl2va_turbo_int8(None, path)

    def test_public_component_loader_requires_consent_and_preserves_imports(self):
        (self.root / "helper.py").write_text("VALUE = 17\n")
        (self.root / "minimax_h3_video_vae.py").write_text("from .helper import VALUE\n")
        with self.assertRaisesRegex(ValueError, "trust-model-code"):
            load_vae_module(self.root, "video", trust_model_code=False)
        before = list(sys.path)
        module = load_vae_module(self.root, "video", trust_model_code=True)
        self.assertEqual(module.VALUE, 17)
        self.assertEqual(sys.path, before)
        self.assertIs(module, load_vae_module(self.root, "video", trust_model_code=True))

    def test_vae_normalization_and_pixel_conversion(self):
        x = torch.randn(1, 3, 2, 4, 4)
        mean, std = torch.randn(3), torch.rand(3) + 1
        normalized = normalize_latents(x, mean, std)
        torch.testing.assert_close(normalize_latents(normalized, mean, std, inverse=True), x)
        pixels = pixels_to_uint8(torch.zeros(1, 3, 1, 2, 2), "simple")
        self.assertEqual(pixels.dtype, torch.uint8)
        self.assertTrue(torch.all(pixels == 128))

    def test_video_vae_w8a8_wraps_only_decoder_block_linears(self):
        calls = []

        class FakeInt8Linear(torch.nn.Module):
            @classmethod
            def from_linear(cls, layer, quantize):
                calls.append((layer, quantize))
                wrapped = cls()
                wrapped.register_buffer("int8_weight", torch.zeros_like(layer.weight, dtype=torch.int8))
                wrapped.register_buffer("scale", torch.ones(1, dtype=torch.float32))
                wrapped.register_buffer("bias", layer.bias.detach().clone())
                return wrapped

        native = types.ModuleType("turbodiffusion.ops.core")
        native.Int8Linear = FakeInt8Linear
        model = tiny_video_vae()
        untouched = {name: copy.deepcopy(layer.state_dict()) for name, layer in model.named_modules()
                     if name in ("encoder", "post_quant_conv", "decoder.x_embedder", "decoder.proj_out")
                     or name.endswith("norm1")}
        with patch.dict(sys.modules, {"turbodiffusion.ops.core": native}):
            self.assertEqual(apply_video_vae_w8a8(model), 8)
        self.assertEqual(len(calls), 8)
        self.assertTrue(all(quantize for _, quantize in calls))
        for name, expected in untouched.items():
            torch.testing.assert_close(model.get_submodule(name).state_dict(), expected)
        for layer in model.modules():
            if isinstance(layer, FakeInt8Linear):
                self.assertEqual(layer.scale.dtype, torch.float32)
                self.assertEqual(layer.bias.dtype, torch.bfloat16)
                self.assertFalse(layer.training)

    def test_video_vae_w8a8_validates_before_replacement(self):
        native = types.ModuleType("turbodiffusion.ops.core")
        native.Int8Linear = types.SimpleNamespace(
            from_linear=lambda *args, **kwargs: self.fail("invalid VAE must not be quantized")
        )
        with patch.dict(sys.modules, {"turbodiffusion.ops.core": native}):
            for variant, message in (
                ("empty", "no ViT decoder blocks"),
                ("dtype", "BF16"), ("shape", "unsupported native"),
            ):
                with self.subTest(variant=variant):
                    model = tiny_video_vae()
                    if variant == "empty":
                        model.decoder.transformer_blocks = torch.nn.ModuleList()
                    elif variant == "dtype":
                        model.decoder.transformer_blocks[-1].ff.w2.float()
                    else:
                        model.decoder.transformer_blocks[-1].ff.w2 = torch.nn.Linear(128, 256).bfloat16()
                    with self.assertRaisesRegex(ValueError, message):
                        apply_video_vae_w8a8(model)
                    if variant != "empty":
                        self.assertIsInstance(model.decoder.transformer_blocks[0].ff.w1, torch.nn.Linear)

    def test_nonzero_rank_does_not_copy_decoded_video_to_cpu(self):
        from turbodiffusion.h3.vae import H3VAEProcessor

        video = torch.nn.Module()
        video.model = torch.nn.Module()
        video.decode_base = lambda z: torch.zeros(1, 3, 1, 2, 2)
        processor = H3VAEProcessor.__new__(H3VAEProcessor)
        processor.video_module = types.SimpleNamespace(
            MiniMaxH3VideoVAE=types.SimpleNamespace(from_pretrained=lambda path: video)
        )
        processor.video_path = self.root
        processor.video_precision = "bf16"
        processor.device = torch.device("cpu")
        processor.rank = 1
        processor.video_mean, processor.video_std = torch.zeros(3), torch.ones(3)
        processor._configure_tiles = lambda *args, **kwargs: None
        with patch("turbodiffusion.h3.vae.pixels_to_uint8", side_effect=AssertionError):
            self.assertEqual(processor.decode(torch.zeros(1, 3, 1, 2, 2), None), (None, None))

    @unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("turbo_diffusion_ops"),
                         "requires the native TurboDiffusion CUDA extension")
    def test_video_vae_native_cuda_w8a8_accuracy_and_bias(self):
        from turbodiffusion.ops.core import Int8Linear, int8_linear

        original = tiny_video_vae().cuda()
        quantized = copy.deepcopy(original)
        self.assertEqual(apply_video_vae_w8a8(quantized), 8)
        for idx, block in enumerate(original.decoder.transformer_blocks):
            for name in ("attn.to_qkv", "attn.to_out", "ff.w1", "ff.w2"):
                with self.subTest(block=idx, projection=name):
                    reference = block.get_submodule(name)
                    layer = quantized.decoder.transformer_blocks[idx].get_submodule(name)
                    self.assertIs(type(layer), Int8Linear)
                    self.assertEqual(layer.scale.dtype, torch.float32)
                    self.assertEqual(layer.bias.dtype, torch.bfloat16)
                    self.assertEqual(layer.int8_weight.device, reference.weight.device)
                    torch.testing.assert_close(layer.bias, reference.bias, rtol=0, atol=0)
                    # Odd sequence lengths cover the VAE's register/suffix tokens.
                    x = torch.randn(1, 133, reference.in_features, device="cuda", dtype=torch.bfloat16)
                    actual = layer(x)
                    direct = int8_linear(x, layer.int8_weight, layer.scale) + layer.bias
                    torch.testing.assert_close(actual, direct, rtol=0, atol=0)
                    expected = reference(x).float()
                    self.assertTrue(torch.isfinite(actual).all())
                    error = (actual.float() - expected).norm() / expected.norm()
                    self.assertLess(float(error), 0.03)
        torch.cuda.synchronize()

    @unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("turbo_diffusion_ops"),
                         "requires the native TurboDiffusion CUDA extension")
    def test_native_cuda_w8a8_checkpoint_roundtrip(self):
        from turbodiffusion.ops.core import Int8Linear, int8_linear

        arch = tiny_config(
            hidden_size=256, ffn_hidden_size=512, num_attention_heads=4,
            attention_head_dim=64, time_embed_hidden_size=256,
            adaln_out_features=18 * 256, final_adaln_out_features=2 * 256,
        )
        model = MiniMaxH3DiTModel(arch, attn_backend="sdpa").bfloat16().eval()
        for name in ("video_patch_proj", "audio_patch_proj", "time_embedder",
                     "final_layer.video_out", "final_layer.audio_out"):
            model.get_submodule(name).float()
        model.rope.inv_freq.fill_(0.01)
        self.assertEqual(apply_turbo_int8(model, quantize=True), 8)
        layer = model.blocks[0].attn.qkv_proj.to("cuda")
        self.assertIs(type(layer), Int8Linear)
        # Exercise the native GEMM pipeline with at least two 128-wide K tiles.
        x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        expected = int8_linear(x, layer.int8_weight, layer.scale)
        torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)
        model.cpu()
        for block in model.blocks:
            block.adaln_proj = None
        model.final_layer.adaln_proj = None
        state = {k: v.contiguous().clone() for k, v in model.state_dict().items()
                 if not is_mlp_alias_key(k)}
        path = self.root / "w8a8.safetensors"
        save_file(state, str(path), metadata={"format": FORMAT})
        with torch.device("meta"):
            restored = MiniMaxH3DiTModel(arch, attn_backend="sdpa")
            apply_turbo_int8(restored, quantize=False)
        for block in restored.blocks:
            block.adaln_proj = None
        restored.final_layer.adaln_proj = None
        load_fl2va_turbo_int8(restored, path)
        actual = restored.blocks[0].attn.qkv_proj.to("cuda")(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(any(t.is_meta for t in restored.parameters()))
        self.assertFalse(any(t.is_meta for t in restored.buffers()))
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()

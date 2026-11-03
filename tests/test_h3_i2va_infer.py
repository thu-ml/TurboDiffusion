"""CPU-only coverage for the bundled H3 full-model W8A8 pipeline."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from turbodiffusion.inference import h3_i2va_infer as infer


class H3InferenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.assets = self.root / "assets"
        (self.assets / "shared").mkdir(parents=True)
        (self.assets / "transformer").mkdir()
        (self.assets / "transformer" / "config.json").touch()
        for kind in ("video", "audio"):
            directory = self.assets / "shared" / f"{kind}_vae"
            directory.mkdir()
            (directory / "config.json").touch()
            (directory / f"minimax_h3_{kind}_vae.py").touch()
        for name in ("model.safetensors", "adaln.safetensors", "adaln.json"):
            (self.assets / name).touch()
        self.image = self.root / "first.png"
        self.image.touch()

    def args(self, *extra):
        return infer.parse_args([
            "--model-root", str(self.assets / "shared"),
            "--dit-model-path", str(self.assets / "transformer"),
            "--w8a8-path", str(self.assets / "model.safetensors"),
            "--adaln-path", str(self.assets / "adaln.safetensors"),
            "--trust-model-code",
            "--output-dir", str(self.root / "output"),
            *extra,
        ])

    def test_profiles_and_full_pipeline(self):
        args = self.args("--image", str(self.image), "--prompt", "a test", "--seed", "73")
        config = infer.effective_config(args)
        infer.validate_inputs(args, config)
        self.assertEqual(config["dit"]["steps"], 5)
        self.assertEqual(config["dit"]["turbo_int8_path"], str(self.assets / "model.safetensors"))
        self.assertEqual(config["sample"]["seed"], 73)
        self.assertEqual(args.profile, "fullopt")
        self.assertEqual(config["dit"]["dbcache"], {"enabled": False})
        self.assertEqual(config["vae"]["video_decoder_precision"], "w8a8")
        commands = infer.stage_commands(args, self.root / "run.yaml", config)
        self.assertEqual(len(commands), 3)
        self.assertIn("turbodiffusion.h3.stages", commands[0])
        self.assertIn("encode", commands[0])
        self.assertIn("sample", commands[1])
        self.assertIn("decode", commands[2])
        self.assertIn("--nproc_per_node=4", commands[0])
        self.assertIn("--nproc_per_node=8", commands[1])
        self.assertIn("--master_port=14771", commands[2])

        args = self.args("--image", str(self.image), "--prompt", "a test",
                         "--profile", "fullopt-dbcache")
        config = infer.effective_config(args)
        self.assertEqual(config["dit"]["dbcache"], infer.CONSERVATIVE_CACHE)
        self.assertEqual(config["vae"]["video_decoder_precision"], "w8a8")

        args = self.args("--image", str(self.image), "--prompt", "a test",
                         "--profile", "fullopt")
        config = infer.effective_config(args)
        infer.validate_inputs(args, config)
        self.assertEqual(config["dit"]["dbcache"], {"enabled": False})
        self.assertTrue(config["dit"]["bake_adaln"])
        self.assertEqual(config["vae"]["video_decoder_precision"], "w8a8")

    def test_shipped_config_disables_cache_by_default(self):
        config = yaml.safe_load(infer.DEFAULT_CONFIG.read_text())
        self.assertEqual(config["dit"]["dbcache"], {**infer.CONSERVATIVE_CACHE, "enabled": False})
        self.assertNotIn("backend", config["text_encoder"])

    def test_prepared_inputs_and_config_origin(self):
        config_dir = self.root / "config"
        config_dir.mkdir()
        source = yaml.safe_load(infer.DEFAULT_CONFIG.read_text())
        source["model_root"] = "../assets/shared"
        source["dit"]["model_path"] = "../assets/transformer"
        config_path = config_dir / "sample.yaml"
        config_path.write_text(yaml.safe_dump(source))
        prompt = self.root / "prompt.pt"
        visual = self.root / "visual.pt"
        prompt.touch()
        visual.touch()
        args = infer.parse_args([
            "--config", str(config_path), "--trust-model-code",
            "--output-dir", str(self.root / "output"),
            "--prompt-embeds", str(prompt), "--visual-cond", str(visual),
            "--w8a8-path", str(self.assets / "model.safetensors"),
            "--adaln-path", str(self.assets / "adaln.safetensors"),
        ])
        config = infer.effective_config(args)
        self.assertEqual(config["model_root"], str(self.assets / "shared"))
        self.assertEqual(config["dit"]["model_path"], str(self.assets / "transformer"))
        infer.validate_inputs(args, config)
        self.assertEqual(len(infer.stage_commands(args, self.root / "run.yaml", config)), 2)

    def test_preflight_rejects_partial_inputs_and_wrong_schedule(self):
        prompt = self.root / "prompt.pt"
        prompt.touch()
        args = self.args("--prompt-embeds", str(prompt))
        with self.assertRaisesRegex(ValueError, "together"):
            infer.validate_inputs(args, infer.effective_config(args))

        args = self.args("--image", str(self.image), "--prompt", "test")
        # Schedule validation happens before the multi-GPU stages start.
        source = yaml.safe_load(infer.DEFAULT_CONFIG.read_text())
        source["dit"]["steps"] = 51
        path = self.root / "wrong.yaml"
        path.write_text(yaml.safe_dump(source))
        args.config = path
        with self.assertRaisesRegex(ValueError, "AdaLN bake"):
            infer.effective_config(args)

    def test_dry_run_writes_config_without_launching(self):
        args = self.args("--image", str(self.image), "--prompt", "a test", "--dry-run")
        with patch.object(infer, "parse_args", return_value=args), patch.object(
            infer.subprocess, "run"
        ) as run:
            infer.main()
        run.assert_not_called()
        cfg = yaml.safe_load((self.root / "output" / "run.yaml").read_text())
        self.assertEqual(cfg["dit"]["dbcache"], {"enabled": False})

    def test_model_code_requires_explicit_consent(self):
        args = self.args("--image", str(self.image), "--prompt", "test")
        args.trust_model_code = False
        with self.assertRaisesRegex(ValueError, "trust-model-code"):
            infer.validate_inputs(args, infer.effective_config(args))
        args.dry_run = True
        infer.validate_inputs(args, infer.effective_config(args))

    def test_video_vae_precision_override(self):
        args = self.args("--vae-precision", "bf16")
        self.assertEqual(infer.effective_config(args)["vae"]["video_decoder_precision"], "bf16")
        source = yaml.safe_load(infer.DEFAULT_CONFIG.read_text())
        source["vae"]["video_decoder_precision"] = "unsupported"
        path = self.root / "precision.yaml"
        path.write_text(yaml.safe_dump(source))
        args = self.args("--config", str(path))
        with self.assertRaisesRegex(ValueError, "video_decoder_precision"):
            infer.effective_config(args)
        args.vae_precision = "w8a8"
        self.assertEqual(infer.effective_config(args)["vae"]["video_decoder_precision"], "w8a8")

    def test_legacy_quantized_vae_config_is_rejected(self):
        args = self.args("--image", str(self.image), "--prompt", "test")
        source = yaml.safe_load(infer.DEFAULT_CONFIG.read_text())
        source["vae"]["int8_path"] = "unsupported.safetensors"
        path = self.root / "unsupported.yaml"
        path.write_text(yaml.safe_dump(source))
        args.config = path
        with self.assertRaisesRegex(ValueError, "legacy quantization"):
            infer.effective_config(args)


if __name__ == "__main__":
    unittest.main()

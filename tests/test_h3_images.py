from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch

from turbodiffusion.h3 import stages
from turbodiffusion.h3.vae import H3VAEProcessor


class H3ImageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def image(self, orientation):
        path = self.root / f"orientation_{orientation}.jpg"
        pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3) * 10
        image = Image.fromarray(pixels)
        exif = image.getexif()
        exif[274] = orientation
        image.save(path, exif=exif)
        return path

    def test_encode_canvas_uses_exif_display_dimensions(self):
        path = self.image(6)
        encoder = types.ModuleType("turbodiffusion.h3.textencoder.processor")
        encoder.encode_prompt = Mock(return_value=torch.zeros(1, 2))
        vae = Mock()
        vae.encode_image.return_value = torch.zeros(1, 4, 1, 2, 2)
        cfg = {
            "output_dir": str(self.root),
            "sample": {"image": str(path), "prompt": "test", "short_edge": 768},
        }
        with patch.dict(sys.modules, {"turbodiffusion.h3.textencoder.processor": encoder}), \
             patch("turbodiffusion.h3.vae.H3VAEProcessor", return_value=vae), \
             patch("torch.distributed.get_rank", return_value=0), patch("torch.save"):
            stages.encode(cfg)
        _, height, width = vae.encode_image.call_args.args
        self.assertGreater(height, width)

    def test_vae_pixels_follow_exif_orientation_before_resizing(self):
        processor = H3VAEProcessor.__new__(H3VAEProcessor)
        processor.device = torch.device("cuda:0")
        processor.video_mean = torch.zeros(3)
        processor.video_std = torch.ones(3)
        model = Mock()
        model.encode_images.return_value = [torch.zeros(3, 1, 2, 1)]
        processor._load_encoder = Mock(return_value=model)
        processor._configure_tiles = Mock()
        for orientation, turns in ((1, 0), (6, 3), (8, 1)):
            with self.subTest(orientation=orientation):
                path = self.image(orientation)
                with Image.open(path) as image:
                    decoded = np.asarray(image.convert("RGB"))
                expected = np.rot90(decoded, turns)
                height, width = expected.shape[:2]
                with patch("torch.random.fork_rng", return_value=nullcontext()), \
                     patch("torch.manual_seed"), patch("torch.cuda.manual_seed_all"), \
                     patch("torch.cuda.empty_cache"):
                    processor.encode_image(path, height, width)
                actual = model.encode_images.call_args.args[0][0]
                np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()

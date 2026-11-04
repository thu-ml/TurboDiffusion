"""Public SpargeAttn adapter, sink masks and partial-block pooling."""
import os
import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from turbodiffusion.h3.attention import _add_sink_blocks, attention_fn


class H3AttentionTest(unittest.TestCase):
    def test_sink_rows_and_columns_round_up_without_erasing_topk(self):
        mask = torch.zeros(1, 2, 8, 16, dtype=torch.int8)
        mask[..., 4, 7] = 1
        _add_sink_blocks(mask, 129, 65, 128, 64, query_length=1024, key_length=1024)
        expected = torch.zeros_like(mask)
        expected[..., :2, :] = 1
        expected[..., :, :3] = 1
        expected[..., -1:, :] = 1
        expected[..., :, -2:] = 1
        expected[..., 4, 7] = 1
        torch.testing.assert_close(mask, expected)

    def test_zero_and_overlapping_sinks(self):
        mask = torch.zeros(1, 1, 3, 6, dtype=torch.int8)
        _add_sink_blocks(mask, 0, 0, 128, 64, query_length=384, key_length=384)
        self.assertFalse(mask.any())
        _add_sink_blocks(mask, 10000, 1, 128, 64, query_length=384, key_length=384)
        self.assertTrue(mask.all())

    def test_partial_suffix_blocks_match_token_level_sinks(self):
        for block_q, block_k in ((128, 64), (64, 128)):
            for qlen, klen in ((193, 257), (257, 193), (384, 384)):
                for pre, post in ((0, 0), (0, 1), (0, 64), (1, 129), (65, 1000)):
                    with self.subTest(blocks=(block_q, block_k), lengths=(qlen, klen),
                                      pre=pre, post=post):
                        nq, nk = (qlen + block_q - 1) // block_q, (klen + block_k - 1) // block_k
                        mask = torch.zeros(1, 2, nq, nk, dtype=torch.int8)
                        mask[..., 0, 0] = 1
                        _add_sink_blocks(
                            mask, pre, post, block_q, block_k,
                            query_length=qlen, key_length=klen,
                        )
                        qpos, kpos = torch.arange(qlen), torch.arange(klen)
                        qsink = (qpos < pre) | (qpos >= qlen - post)
                        ksink = (kpos < pre) | (kpos >= klen - post)
                        qblocks = F.pad(qsink, (0, nq * block_q - qlen)).view(nq, block_q).any(-1)
                        kblocks = F.pad(ksink, (0, nk * block_k - klen)).view(nk, block_k).any(-1)
                        expected = (qblocks[:, None] | kblocks[None, :]).to(mask.dtype)
                        expected[0, 0] = 1
                        torch.testing.assert_close(mask, expected.expand_as(mask))

    def test_public_api_layout_scale_and_architecture(self):
        for arch, length, block_q, block_k in (
            ((12, 0), 384, 128, 64), ((9, 0), 384, 64, 128),
            ((12, 0), 3136, 128, 64),
        ):
            with self.subTest(arch=arch, length=length):
                q = torch.randn(1, length, 2, 64)
                nq, nk = (length + block_q - 1) // block_q, (length + block_k - 1) // block_k
                mask = torch.zeros(1, 2, nq, nk, dtype=torch.int8)
                pooling = types.ModuleType("turbodiffusion.h3.pooling")
                pooling.get_block_map = Mock(return_value=(mask, None, None))
                public = types.ModuleType("spas_sage_attn")
                public.block_sparse_sage2_attn_cuda = Mock(side_effect=lambda q, k, v, **kw: v)
                with patch.dict(sys.modules, {
                    "spas_sage_attn": public, "turbodiffusion.h3.pooling": pooling,
                }), patch("torch.cuda.get_device_capability", return_value=arch):
                    out = attention_fn(
                        q, q, q, backend="sparge_attn", scale=0.125,
                        sparge_attn_pre_sink_length=1, sparge_attn_topk=0.01,
                    )
                torch.testing.assert_close(out, q)
                args, kw = public.block_sparse_sage2_attn_cuda.call_args
                self.assertEqual(args[0].shape, (1, 2, length, 64))
                self.assertTrue(args[0].is_contiguous())
                self.assertIs(kw["mask_id"], mask)
                self.assertEqual(kw["scale"], 0.125)
                self.assertEqual(kw["pvthreshd"], 1e6)
                self.assertFalse(kw["attention_sink"])
                self.assertTrue(mask[..., 0, :].all())
                self.assertTrue(mask[..., :, 0].all())
                self.assertFalse(mask[..., 1:, 1:].any())
                pool_args, pool_kw = pooling.get_block_map.call_args
                self.assertEqual(pool_kw, {"BLKQ": block_q, "BLKK": block_k})
                self.assertEqual(int(pool_args[2] * nk), 1)

    def test_public_api_uses_unpadded_suffix_sink_lengths(self):
        length, post = 193, 64
        for arch, block_q, block_k in (((12, 0), 128, 64), ((9, 0), 64, 128)):
            with self.subTest(arch=arch):
                q = torch.randn(1, length, 2, 64)
                nq, nk = (length + block_q - 1) // block_q, (length + block_k - 1) // block_k
                mask = torch.zeros(1, 2, nq, nk, dtype=torch.int8)
                pooling = types.ModuleType("turbodiffusion.h3.pooling")
                pooling.get_block_map = Mock(return_value=(mask, None, None))
                public = types.ModuleType("spas_sage_attn")
                public.block_sparse_sage2_attn_cuda = Mock(side_effect=lambda q, k, v, **kw: v)
                with patch.dict(sys.modules, {
                    "spas_sage_attn": public, "turbodiffusion.h3.pooling": pooling,
                }), patch("torch.cuda.get_device_capability", return_value=arch):
                    attention_fn(q, q, q, backend="sparge_attn", sparge_attn_post_sink_length=post)
                dense = mask.bool().repeat_interleave(block_q, -2).repeat_interleave(block_k, -1)
                dense = dense[..., :length, :length]
                self.assertTrue(dense[..., -post:, :].all())
                self.assertTrue(dense[..., :, -post:].all())
                self.assertFalse(dense[..., 0, 0].any())

    def test_short_sequence_uses_sdpa(self):
        q, k, v = (torch.randn(1, 33, 2, 64) for _ in range(3))
        public = types.ModuleType("spas_sage_attn")
        public.block_sparse_sage2_attn_cuda = Mock()
        with patch.dict(sys.modules, {"spas_sage_attn": public}):
            out = attention_fn(q, k, v, backend="sparge_attn", scale=0.2)
        expected = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=0.2
        ).transpose(1, 2)
        torch.testing.assert_close(out, expected)
        public.block_sparse_sage2_attn_cuda.assert_not_called()

    def test_missing_public_dependency_is_actionable(self):
        q = torch.randn(1, 128, 1, 64)
        with patch.dict(sys.modules, {"spas_sage_attn": None}):
            with self.assertRaisesRegex(ImportError, "public thu-ml/SpargeAttn"):
                attention_fn(q, q, q, backend="sparge_attn")

    def test_invalid_sparse_settings(self):
        q = torch.randn(1, 128, 1, 64)
        for kw in (
            {"sparge_attn_topk": 0}, {"sparge_attn_topk": 1.1},
            {"sparge_attn_pre_sink_length": -1}, {"sparge_attn_post_sink_length": -1},
        ):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                attention_fn(q, q, q, backend="sparge_attn", **kw)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_partial_block_pooling_matches_reference(self):
        from turbodiffusion.h3.pooling import mean_pool

        for length in (129, 192, 257):
            for block in (64, 128):
                with self.subTest(length=length, block=block):
                    x = torch.randn(1, 2, length, 64, device="cuda", dtype=torch.bfloat16)
                    expected = torch.stack([
                        x[..., start : start + block, :].float().mean(-2)
                        for start in range(0, length, block)
                    ], dim=-2).to(x.dtype)
                    torch.testing.assert_close(mean_pool(x, block), expected)

    @unittest.skipUnless(
        torch.cuda.is_available() and os.environ.get("H3_TEST_PUBLIC_SPARGE") == "1",
        "Set H3_TEST_PUBLIC_SPARGE=1 after building the pinned public dependency",
    )
    def test_public_cuda_kernel_matches_its_block_mask(self):
        from turbodiffusion.h3.pooling import get_block_map

        torch.manual_seed(17)
        bq, bk = (64, 128) if torch.cuda.get_device_capability() == (9, 0) else (128, 64)
        for length, dim, pre, post, topk in (
            (192, 64, 0, 0, 1.0), (384, 128, 1, 1, 0.5), (512, 128, 65, 0, 0.5),
            (193, 64, 0, 64, 0.25), (257, 128, 1, 128, 0.5),
        ):
            with self.subTest(length=length, dim=dim):
                q, k, v = (torch.randn(1, 2, length, dim, device="cuda", dtype=torch.bfloat16)
                           for _ in range(3))
                mask, _, _ = get_block_map(q, k, topk, BLKQ=bq, BLKK=bk)
                _add_sink_blocks(
                    mask, pre, post, bq, bk, query_length=length, key_length=length
                )
                dense_mask = mask.bool().repeat_interleave(bq, -2).repeat_interleave(bk, -1)
                expected = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=dense_mask[..., :length, :length]
                )
                actual = attention_fn(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                    backend="sparge_attn", sparge_attn_topk=topk,
                    sparge_attn_pre_sink_length=pre, sparge_attn_post_sink_length=post,
                ).transpose(1, 2)
                self.assertEqual(actual.dtype, q.dtype)
                self.assertTrue(torch.isfinite(actual).all())
                relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
                self.assertLess(relative_l2.item(), 0.08)


if __name__ == "__main__":
    unittest.main()

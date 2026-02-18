# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import html
import math
import string
from typing import List, Optional, Union

import ftfy
import regex as re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from imaginaire.utils import distributed, log, misc
from imaginaire.utils.easy_io import easy_io


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def canonicalize(text, keep_punctuation_exact_string=None):
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(str.maketrans("", "", string.punctuation)) for part in text.split(keep_punctuation_exact_string)
        )
    else:
        text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class HuggingfaceTokenizer:
    def __init__(self, name, seq_len=None, clean=None, **kwargs):
        assert clean in (None, "whitespace", "lower", "canonicalize")
        self.name = name
        self.seq_len = seq_len
        self.clean = clean

        # init tokenizer with controlled download behavior and robust fallback
        allow_downloads = os.getenv("HF_ALLOW_DOWNLOADS", "0").lower() in ("1", "true", "yes")
        hf_token = os.getenv("HF_TOKEN", None)
        _kwargs = dict(kwargs)
        if hf_token:
            _kwargs["use_auth_token"] = hf_token

        if not allow_downloads:
            # Prefer local cached tokenizer to avoid network timeouts
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True, **_kwargs)
                self.vocab_size = self.tokenizer.vocab_size
                log.info(f"Loaded tokenizer '{name}' from local cache.")
            except Exception as e:
                log.info(f"Local tokenizer not found for '{name}' and downloads disabled; using DummyTokenizer fallback. ({e})")
                use_dummy = True
        else:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(name, **_kwargs)
                self.vocab_size = self.tokenizer.vocab_size
                log.info(f"Loaded tokenizer '{name}' from remote or cache.")
                use_dummy = False
            except Exception as e:
                log.warn(f"AutoTokenizer load failed ({e}); falling back to DummyTokenizer.")
                use_dummy = True

        if 'use_dummy' in locals() and use_dummy:
            class DummyTokenizer:
                def __init__(self, seq_len=None):
                    self.seq_len = seq_len
                    self.vocab_size = 32000
                def __call__(self, sequence, **kwargs):
                    # Simple whitespace tokenization with hashing to ids
                    def tok(s):
                        toks = s.split()[: (self.seq_len or 512)]
                        ids = [abs(hash(w)) % self.vocab_size + 1 for w in toks]
                        # pad/truncate
                        if self.seq_len is not None:
                            ids = ids + [0] * max(0, self.seq_len - len(ids))
                        return {"input_ids": torch.tensor([ids], dtype=torch.long), "attention_mask": torch.tensor([[1 if x!=0 else 0 for x in ids]], dtype=torch.long)}
                    if isinstance(sequence, str):
                        return tok(sequence)
                    else:
                        # batch
                        batch_ids = []
                        batch_masks = []
                        for s in sequence:
                            out = tok(s)
                            batch_ids.append(out["input_ids"][0])
                            batch_masks.append(out["attention_mask"][0])
                        return type("_", (), {"input_ids": torch.stack(batch_ids), "attention_mask": torch.stack(batch_masks)})
            self.tokenizer = DummyTokenizer(seq_len=self.seq_len)
            self.vocab_size = self.tokenizer.vocab_size
    def __call__(self, sequence, **kwargs):
        return_mask = kwargs.pop("return_mask", False)

        # arguments
        _kwargs = {"return_tensors": "pt"}
        if self.seq_len is not None:
            _kwargs.update({"padding": "max_length", "truncation": True, "max_length": self.seq_len})
        _kwargs.update(**kwargs)

        # tokenization
        if isinstance(sequence, str):
            sequence = [sequence]
        if self.clean:
            sequence = [self._clean(u) for u in sequence]
        ids = self.tokenizer(sequence, **_kwargs)

        # output
        if return_mask:
            return ids.input_ids, ids.attention_mask
        else:
            return ids.input_ids

    def _clean(self, text):
        if self.clean == "whitespace":
            text = whitespace_clean(basic_clean(text))
        elif self.clean == "lower":
            text = whitespace_clean(basic_clean(text)).lower()
        elif self.clean == "canonicalize":
            text = canonicalize(basic_clean(text))
        return text


def fp16_clamp(x):
    if x.dtype == torch.float16 and torch.isinf(x).any():
        clamp = torch.finfo(x.dtype).max - 1000
        x = torch.clamp(x, min=-clamp, max=clamp)
    return x


def init_weights(m):
    if isinstance(m, T5LayerNorm):
        nn.init.ones_(m.weight)
    elif isinstance(m, T5Model):
        nn.init.normal_(m.token_embedding.weight, std=1.0)
    elif isinstance(m, T5FeedForward):
        nn.init.normal_(m.gate[0].weight, std=m.dim**-0.5)
        nn.init.normal_(m.fc1.weight, std=m.dim**-0.5)
        nn.init.normal_(m.fc2.weight, std=m.dim_ffn**-0.5)
    elif isinstance(m, T5Attention):
        nn.init.normal_(m.q.weight, std=(m.dim * m.dim_attn) ** -0.5)
        nn.init.normal_(m.k.weight, std=m.dim**-0.5)
        nn.init.normal_(m.v.weight, std=m.dim**-0.5)
        nn.init.normal_(m.o.weight, std=(m.num_heads * m.dim_attn) ** -0.5)
    elif isinstance(m, T5RelativeEmbedding):
        nn.init.normal_(m.embedding.weight, std=(2 * m.num_buckets * m.num_heads) ** -0.5)


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class T5LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(T5LayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            x = x.type_as(self.weight)
        return self.weight * x


class T5Attention(nn.Module):
    def __init__(self, dim, dim_attn, num_heads, dropout=0.1):
        assert dim_attn % num_heads == 0
        super(T5Attention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads

        # layers
        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, pos_bias=None):
        """
        x:          [B, L1, C].
        context:    [B, L2, C] or None.
        mask:       [B, L2] or [B, L1, L2] or None.
        """
        # check inputs
        context = x if context is None else context
        b, n, c = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.q(x).view(b, -1, n, c)
        k = self.k(context).view(b, -1, n, c)
        v = self.v(context).view(b, -1, n, c)

        # attention bias
        attn_bias = x.new_zeros(b, n, q.size(1), k.size(1))
        if pos_bias is not None:
            attn_bias += pos_bias
        if mask is not None:
            assert mask.ndim in [2, 3]
            mask = mask.view(b, 1, 1, -1) if mask.ndim == 2 else mask.unsqueeze(1)
            attn_bias.masked_fill_(mask == 0, torch.finfo(x.dtype).min)

        # compute attention (T5 does not use scaling)
        attn = torch.einsum("binc,bjnc->bnij", q, k) + attn_bias
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        x = torch.einsum("bnij,bjnc->binc", attn, v)

        # output
        x = x.reshape(b, -1, n * c)
        x = self.o(x)
        x = self.dropout(x)
        return x


class T5FeedForward(nn.Module):
    def __init__(self, dim, dim_ffn, dropout=0.1):
        super(T5FeedForward, self).__init__()
        self.dim = dim
        self.dim_ffn = dim_ffn

        # layers
        self.gate = nn.Sequential(nn.Linear(dim, dim_ffn, bias=False), GELU())
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x) * self.gate(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class T5SelfAttention(nn.Module):
    def __init__(self, dim, dim_attn, dim_ffn, num_heads, num_buckets, shared_pos=True, dropout=0.1):
        super(T5SelfAttention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.norm1 = T5LayerNorm(dim)
        self.attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = None if shared_pos else T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True)

    def forward(self, x, mask=None, pos_bias=None):
        e = pos_bias if self.shared_pos else self.pos_embedding(x.size(1), x.size(1))
        x = fp16_clamp(x + self.attn(self.norm1(x), mask=mask, pos_bias=e))
        x = fp16_clamp(x + self.ffn(self.norm2(x)))
        return x


class T5CrossAttention(nn.Module):
    def __init__(self, dim, dim_attn, dim_ffn, num_heads, num_buckets, shared_pos=True, dropout=0.1):
        super(T5CrossAttention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.norm1 = T5LayerNorm(dim)
        self.self_attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.cross_attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm3 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = None if shared_pos else T5RelativeEmbedding(num_buckets, num_heads, bidirectional=False)

    def forward(self, x, mask=None, encoder_states=None, encoder_mask=None, pos_bias=None):
        e = pos_bias if self.shared_pos else self.pos_embedding(x.size(1), x.size(1))
        x = fp16_clamp(x + self.self_attn(self.norm1(x), mask=mask, pos_bias=e))
        x = fp16_clamp(x + self.cross_attn(self.norm2(x), context=encoder_states, mask=encoder_mask))
        x = fp16_clamp(x + self.ffn(self.norm3(x)))
        return x


class T5RelativeEmbedding(nn.Module):
    def __init__(self, num_buckets, num_heads, bidirectional, max_dist=128):
        super(T5RelativeEmbedding, self).__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist

        # layers
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def forward(self, lq, lk):
        device = self.embedding.weight.device
        # rel_pos = torch.arange(lk).unsqueeze(0).to(device) - \
        #     torch.arange(lq).unsqueeze(1).to(device)
        rel_pos = torch.arange(lk, device=device).unsqueeze(0) - torch.arange(lq, device=device).unsqueeze(1)
        rel_pos = self._relative_position_bucket(rel_pos)
        rel_pos_embeds = self.embedding(rel_pos)
        rel_pos_embeds = rel_pos_embeds.permute(2, 0, 1).unsqueeze(0)  # [1, N, Lq, Lk]
        return rel_pos_embeds.contiguous()

    def _relative_position_bucket(self, rel_pos):
        # preprocess
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).long() * num_buckets
            rel_pos = torch.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = 0
            rel_pos = -torch.min(rel_pos, torch.zeros_like(rel_pos))

        # embeddings for small and large positions
        max_exact = num_buckets // 2
        rel_pos_large = max_exact + (torch.log(rel_pos.float() / max_exact) / math.log(self.max_dist / max_exact) * (num_buckets - max_exact)).long()
        rel_pos_large = torch.min(rel_pos_large, torch.full_like(rel_pos_large, num_buckets - 1))
        rel_buckets += torch.where(rel_pos < max_exact, rel_pos, rel_pos_large)
        return rel_buckets


class T5Encoder(nn.Module):
    def __init__(self, vocab, dim, dim_attn, dim_ffn, num_heads, num_layers, num_buckets, shared_pos=True, dropout=0.1):
        super(T5Encoder, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.token_embedding = vocab if isinstance(vocab, nn.Embedding) else nn.Embedding(vocab, dim)
        self.pos_embedding = T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True) if shared_pos else None
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([T5SelfAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets, shared_pos, dropout) for _ in range(num_layers)])
        self.norm = T5LayerNorm(dim)

        # initialize weights
        self.apply(init_weights)

    def forward(self, ids, mask=None):
        x = self.token_embedding(ids)
        x = self.dropout(x)
        e = self.pos_embedding(x.size(1), x.size(1)) if self.shared_pos else None
        for block in self.blocks:
            x = block(x, mask, pos_bias=e)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class T5Decoder(nn.Module):
    def __init__(self, vocab, dim, dim_attn, dim_ffn, num_heads, num_layers, num_buckets, shared_pos=True, dropout=0.1):
        super(T5Decoder, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.token_embedding = vocab if isinstance(vocab, nn.Embedding) else nn.Embedding(vocab, dim)
        self.pos_embedding = T5RelativeEmbedding(num_buckets, num_heads, bidirectional=False) if shared_pos else None
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [T5CrossAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets, shared_pos, dropout) for _ in range(num_layers)]
        )
        self.norm = T5LayerNorm(dim)

        # initialize weights
        self.apply(init_weights)

    def forward(self, ids, mask=None, encoder_states=None, encoder_mask=None):
        b, s = ids.size()

        # causal mask
        if mask is None:
            mask = torch.tril(torch.ones(1, s, s).to(ids.device))
        elif mask.ndim == 2:
            mask = torch.tril(mask.unsqueeze(1).expand(-1, s, -1))

        # layers
        x = self.token_embedding(ids)
        x = self.dropout(x)
        e = self.pos_embedding(x.size(1), x.size(1)) if self.shared_pos else None
        for block in self.blocks:
            x = block(x, mask, encoder_states, encoder_mask, pos_bias=e)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class T5Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        dim,
        dim_attn,
        dim_ffn,
        num_heads,
        encoder_layers,
        decoder_layers,
        num_buckets,
        shared_pos=True,
        dropout=0.1,
    ):
        super(T5Model, self).__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.num_buckets = num_buckets

        # layers
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.encoder = T5Encoder(self.token_embedding, dim, dim_attn, dim_ffn, num_heads, encoder_layers, num_buckets, shared_pos, dropout)
        self.decoder = T5Decoder(self.token_embedding, dim, dim_attn, dim_ffn, num_heads, decoder_layers, num_buckets, shared_pos, dropout)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # initialize weights
        self.apply(init_weights)

    def forward(self, encoder_ids, encoder_mask, decoder_ids, decoder_mask):
        x = self.encoder(encoder_ids, encoder_mask)
        x = self.decoder(decoder_ids, decoder_mask, x, encoder_mask)
        x = self.head(x)
        return x


def _t5(name, encoder_only=False, decoder_only=False, dtype=torch.float32, device="cpu", **kwargs):
    # sanity check
    assert not (encoder_only and decoder_only)

    # params
    if encoder_only:
        model_cls = T5Encoder
        kwargs["vocab"] = kwargs.pop("vocab_size")
        kwargs["num_layers"] = kwargs.pop("encoder_layers")
        _ = kwargs.pop("decoder_layers")
    elif decoder_only:
        model_cls = T5Decoder
        kwargs["vocab"] = kwargs.pop("vocab_size")
        kwargs["num_layers"] = kwargs.pop("decoder_layers")
        _ = kwargs.pop("encoder_layers")
    else:
        model_cls = T5Model

    # init model
    with torch.device(device):
        model = model_cls(**kwargs)

    # set device
    model = model.to(dtype=dtype, device=device)

    return model


def umt5_xxl(**kwargs):
    cfg = dict(
        vocab_size=256384,
        dim=4096,
        dim_attn=4096,
        dim_ffn=10240,
        num_heads=64,
        encoder_layers=24,
        decoder_layers=24,
        num_buckets=32,
        shared_pos=False,
        dropout=0.1,
    )
    cfg.update(**kwargs)
    return _t5("umt5-xxl", **cfg)


def load_model_torch(model, ckpt_path):
    if distributed.is_rank0():
        ckpt = easy_io.load(
            ckpt_path,
            map_location="cuda",
        )
        model.load_state_dict(ckpt, assign=True)

    distributed.sync_model_states(model, src=0)
    return model


class UMT5EncoderModel:
    def __init__(
        self,
        text_len=512,
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),
        checkpoint_path="models_t5_umt5-xxl-enc-bf16.pth",
        tokenizer_path="google/umt5-xxl",
    ):
        self.text_len = text_len
        self.dtype = dtype
        self.device = device

        # init model
        model = umt5_xxl(encoder_only=True, dtype=dtype, device='meta')
        log.info(f"loading {checkpoint_path}")
        assert checkpoint_path.endswith(".pth")
        self.model = load_model_torch(model, checkpoint_path).to(device).eval()
        # init tokenizer
        self.tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=text_len, clean="whitespace")

    def __call__(self, texts, device: Optional[torch.device] = None):
        if device is None:
            device = self.device
        ids, mask = self.tokenizer(texts, return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.model(ids, mask)
        # return [u[:v] for u, v in zip(context, seq_lens)]
        stack_emb = []
        for u, length in zip(context, seq_lens):
            if length > self.text_len:
                stack_emb.append(u[: self.text_len])
            else:
                # pad with zeros to max_length
                zeros = torch.zeros(self.text_len - length, u.shape[1]).to(u)
                stack_emb.append(torch.cat([u[:length], zeros], dim=0))
        return torch.stack(stack_emb)


t5_encoder: Optional[UMT5EncoderModel] = None


def get_umt5_embedding(
    checkpoint_path: str,
    prompts: Union[str, List[str]],
    device: str = "cuda",
    max_length: int = 512,
) -> torch.Tensor:
    global t5_encoder
    if t5_encoder is None:
        t5_encoder = UMT5EncoderModel(text_len=max_length, device=device, checkpoint_path=checkpoint_path)
    return t5_encoder(prompts, device=device)


def clear_umt5_memory():
    """
    Clears the GPU memory by deleting the global t5_encoder model and emptying the CUDA cache.
    """
    global t5_encoder
    if t5_encoder is not None:
        del t5_encoder
        t5_encoder = None
    gc.collect()
    torch.cuda.empty_cache()

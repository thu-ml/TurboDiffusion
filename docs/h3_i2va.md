# H3 FL2VA with TurboDiffusion W8A8

The H3 model and inference implementation are included in this repository.
The default `fullopt` profile performs **four DMD denoising updates**
(`dit.steps: 5` means five sigma points), without DBCache.
`fullopt-dbcache` explicitly enables conservative F8/B4 caching: threshold 0.20,
one warmup, at most one consecutive hit, and TaylorSeer disabled.
Both profiles retain the precomputed AdaLN table. Cache hits depend on the input;
no cache speedup or quality-parity claim is made for the default profile.

## Implementation

- `turbodiffusion/h3/dit/model.py`: H3 Omni-DiT and token refiner.
- `turbodiffusion/h3/dit/processor.py`: four-update DMD sampling and AdaLN reuse.
- `turbodiffusion/h3/dit/turbo_int8.py`: replacement of 200 main-block projections
  with the existing `turbodiffusion.ops.core.Int8Linear`.
- `turbodiffusion/h3/dit/dbcache.py`, `parallel.py`, `offload.py`: conservative
  caching, Ulysses/SP8, quantized/asynchronous communication and block offload.
- `turbodiffusion/h3/textencoder/`: tensor-parallel Qwen3-VL layer-50 encoding.
- `turbodiffusion/h3/stages.py`, `vae.py`: image encoding, sampling, tiled
  video/audio decoding and MP4 export. The video VAE uses the same native
  `Int8Linear` for 144 attention/FFN projections in its 36 ViT decoder blocks.
- `turbodiffusion/inference/export_h3_w8a8.py`: full-model W8A8 export.
- `turbodiffusion/h3/nccl.py`: loaded-library verification and an eight-rank
  asynchronous INT8/FP32 all-to-all stress test with pinned H2D overlap.

No separate H3 inference checkout is required. W8A8 uses TurboDiffusion's
existing CUDA quantization/GEMM kernels, with 128x128 weight blocks; it does
not introduce another quantization backend. Keyframe encoding remains FP32,
video decoder block projections use W8A8, and audio VAE decoding is FP32.
Video decoder I/O projections, convolutions and other unquantized tensors
retain BF16 (normalization follows the public VAE implementation).
SpargeAttn is a separately installed attention dependency.

The DiT retains FP32 RoPE frequencies and the original FP32 video/audio input
and output projections and timestep MLP. RMSNorm weights follow the original
checkpoint (BF16); normalization is not an INT8 projection. Inputs are cast to
each projection's dtype, then back to the block-stack dtype as needed.
The replicated text refiner always uses local dense SDPA, without Ulysses,
INT8 communication or SpargeAttn. Main-block attention removes the alignment
tail before attention and restores the padded shape before reverse Ulysses,
matching the reference's separation of valid tokens from padding.

## Installation

On Linux RTX 5090 hosts, use Python 3.10 or newer, CUDA toolkit 12.8, and the official
PyTorch 2.11 cu128 wheels. Install PyTorch before building CUDA extensions:

```bash
python -m venv .venv-h3
source .venv-h3/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -e '.[h3]' --no-build-isolation
```

The tested combination is PyTorch `2.11.0+cu128`, its bundled NCCL `2.28.9`,
CUDA 12.8, Triton 3.6.0, Transformers 4.57.3 and Diffusers 0.38.0.
The H3 extra pins PyTorch 2.11 and Diffusers 0.38.0 without changing the
requirements for Wan. Rebuild both TurboDiffusion and SpargeAttn extensions
when switching PyTorch versions; binaries built for 2.7 are not reusable.

Install `ffmpeg` on the host for video/audio export. The profiles use 8 GPU
ranks for DiT/VAE and 4 for text/keyframe encoding. CPU RAM must hold each
rank's offloaded weights; these are not single-GPU profiles.
They do not use Wan's rCM schedule or SLA.

### SpargeAttn Compatibility

H3 uses the public [thu-ml/SpargeAttn](https://github.com/thu-ml/SpargeAttn)
custom block-mask API, not a private H3-specific fork:

```python
from spas_sage_attn import block_sparse_sage2_attn_cuda
```

The pinned public revision is `ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a`.
Build it with the CUDA toolkit matching PyTorch. On RTX 5090, use CUDA 12.8
or newer and expose only the intended build GPU architecture:

```bash
git clone https://github.com/thu-ml/SpargeAttn.git
git -C SpargeAttn checkout ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a
CUDA_VISIBLE_DEVICES=0 env -u TORCH_CUDA_ARCH_LIST \
  pip install ./SpargeAttn --no-build-isolation --no-deps
```

This revision's explicit `TORCH_CUDA_ARCH_LIST` allowlist predates SM120,
so the command lets its build script detect the visible GPU instead.
No SpargeAttn source patch is required for this adapter.

`turbodiffusion/h3/attention.py` uses H3-scoped pooled top-k block
selection, then makes prefix/suffix query rows and key columns dense for
text, audio and first-frame conditioning tokens. Blocks are Q128/K64
(Q64/K128 on SM90), with sinks rounded up to block boundaries. The mask is
passed as `mask_id`; `pvthreshd=1e6` disables additional PV-based skipping.
The public top-k convenience function is not called because it does not
accept H3's prefix/suffix sink lengths. `turbodiffusion/h3/pooling.py` explicitly
zero-fills masked tail elements for sequences not divisible by the block size.
The shared `turbodiffusion/SLA/utils.py` and Wan/SLA behaviour are unchanged.

This preserves the block-mask policy, not bitwise equivalence with the earlier
local backend: quantization and accumulation inside the public kernel differ.
Query sequences shorter than 128 tokens use SDPA because the public kernel
does not support them. A missing public dependency raises an actionable error,
not a silent fallback. For a fully dense reference, explicitly set
`dit.attn_backend: sdpa`; it has different performance.

### NCCL on RTX 5090

Use the NCCL 2.28.9 supplied by the official PyTorch 2.11 cu128 installation
above. Do not manually preload another NCCL version or replace files in
PyTorch's environment. The earlier source-build recommendation for NCCL
2.26.2 is withdrawn: it was not stable in the reviewer's 5090 tests.

Before inference, remove any previous NCCL overrides and check the library
actually mapped into the process:

```bash
unset LD_PRELOAD LD_LIBRARY_PATH NCCL_HOME
python -m turbodiffusion.h3.nccl
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  --module turbodiffusion.h3.nccl --stress --iterations 30
```

The checker calls `ncclGetVersion` on the library found in `/proc/self/maps`
and verifies both its runtime version and its path inside the installed
PyTorch wheel's `nvidia/nccl/lib`. `compiled_nccl` is printed separately;
`torch.cuda.nccl.version()` alone does not verify a preloaded binary.

Isolated validation on 2026-11-02 used 8x RTX 5090 32GB, driver 595.58.03,
two four-GPU PHB groups with SYS links across groups, and no NVLink.
With no NCCL preload, the official 2.11/2.28.9 combination passed six
eight-rank stress launches, H3 sampling and decoding, and separate four-rank
text/keyframe encoding. This establishes the tested combination on that
host, not the root cause of 2.26.2 failures or stability on every 5090 host.
Run the stress test on each target machine before collecting benchmarks.

If DiT all-to-all fails, restart inference with `H3_USE_NCCL_A2A=0` to select
the existing all-gather fallback. This changes communication cost and does
not eliminate NCCL or affect the VAE's own communication implementation.

## Download

The full BF16 four-update DMD checkpoint and matching AdaLN table are at
[liran12345/h3-turbo-diffusion](https://huggingface.co/liran12345/h3-turbo-diffusion).
Original small FP32 tensors are preserved. The transformer has 34 shards
and is approximately 61.73 GiB.

The text encoder, processor/tokenizer and video/audio VAEs come from
[MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3).
Download their complete directories, including VAE Python modules and `source/`
weights, but not the original many-step transformer:

```bash
hf download liran12345/h3-turbo-diffusion \
  --local-dir /models/h3-turbo-diffusion
hf download MiniMaxAI/MiniMax-H3 \
  --include 'FL2VA/text_encoder/*' \
  --include 'FL2VA/processor/*' \
  --include 'FL2VA/tokenizer/*' \
  --include 'FL2VA/video_vae/*' \
  --include 'FL2VA/audio_vae/*' \
  --local-dir /models/MiniMax-H3
```

Each download pattern has its own `--include`, as required by current `hf` CLI
versions.

The published AdaLN table was refreshed on 2026-11-03 using the original FP32
timestep MLP. Its JSON and embedded metadata record
`"timestep_mlp_dtype": "float32"`; table tensors remain BF16. For the published
schedule, use this table directly without regenerating it. If you downloaded
the older 2026-10-29 table, refresh only the two AdaLN files:

```bash
hf download liran12345/h3-turbo-diffusion \
  --include 'adaln/adaln_i2va_s5_v12_a3.safetensors' \
  --include 'adaln/adaln_i2va_s5_v12_a3.json' \
  --local-dir /models/h3-turbo-diffusion
```

The VAEs use the public Python modules supplied with those weights.
`--trust-model-code` explicitly permits loading that code; review its source
before enabling it. The code is loaded from the local model directory,
not a private checkout. Third-party attribution is in `turbodiffusion/h3/NOTICE`.
Model weights remain subject to the
[MiniMax H3 Community License Agreement](https://huggingface.co/liran12345/h3-turbo-diffusion/blob/main/LICENSE),
not this repository's code license.
The current checkpoint is hosted by its publisher's personal account;
an organization-hosted release is not yet available. This PR does not
transfer weight ownership or broaden redistribution rights.

## Export W8A8

Run from the TurboDiffusion checkout after building its CUDA extension:

```bash
CUDA_VISIBLE_DEVICES=0 python -m turbodiffusion.inference.export_h3_w8a8 \
  --model-path /models/h3-turbo-diffusion/transformer \
  --output /models/h3-turbo-diffusion/model_w8a8.safetensors
```

The exporter wraps only main-block QKV, attention output and MLP projections.
It does not modify the source checkpoint or the CUDA kernels.
To regenerate the table instead of reusing the published one, append
`--adaln-output /models/h3-turbo-diffusion/adaln/adaln_i2va_s5_v12_a3.safetensors`.
This regenerates the table for
`i2va / steps=5 / video_shift=12 / audio_shift=3`, preserving the source FP32
timestep MLP and writing its companion `.json`. Omit it to reuse an existing
matching table. W8A8 and RoPE changes alone do not require re-baking AdaLN,
but tables created by an older exporter using a BF16 timestep MLP should be
regenerated to reproduce the original mixed-precision computation.
New exports record `timestep_mlp_dtype` in both the `.json` and safetensors
metadata. The table itself remains BF16; its storage dtype does not indicate
the timestep MLP's computation dtype.

Re-export INT8 checkpoints created before the FP32-preservation fix.
The loader rejects reduced-precision RoPE/input/output/timestep tensors
with an actionable error instead of silently upcasting already-rounded values.
The full BF16/mixed-precision source checkpoint does not need to be downloaded
again or modified.

The video VAE is quantized once at decode startup from the original public
weights using the unchanged native `Int8Linear.from_linear`. No separate VAE
export or quantized VAE checkpoint is required. The source weights are not
modified. VAE quantization does not require re-baking DiT AdaLN either.

## Run

```bash
bash scripts/inference_h3_i2va.sh \
  --model-root /models/MiniMax-H3/FL2VA \
  --dit-model-path /models/h3-turbo-diffusion/transformer \
  --w8a8-path /models/h3-turbo-diffusion/model_w8a8.safetensors \
  --adaln-path /models/h3-turbo-diffusion/adaln/adaln_i2va_s5_v12_a3.safetensors \
  --trust-model-code \
  --image /path/to/first-frame.png \
  --prompt 'Video description and soundscape...' \
  --output-dir output/h3_demo
```

`fullopt` is the no-cache default.
Add `--profile fullopt-dbcache` to opt into conservative DBCache.
Both profiles default to native W8A8 video decoding; use `--vae-precision bf16`
for the unquantized video decoder reference.
Use separate output directories and identical inputs/seed/duration to compare
profiles. The output with audio is `<output-dir>/i2va.mp4`; `run.yaml` records
the configuration, and conditioning/latent artifacts remain alongside it.

To reuse conditioning, pass both `--prompt-embeds` and `--visual-cond`, plus
the original `--seed`, `--duration-seconds` and `--aspect-ratio`.
`--skip-encode` instead uses conditioning files already in the output directory.
`--dry-run` validates paths, writes the configuration and prints the bundled
stage commands without starting GPUs or loading model Python code.

Relative paths in the shipped config resolve against this repository;
relative paths in custom configs resolve against their config directory.
CLI path overrides resolve against the working directory.
Older configs referencing a separate inference checkout or a quantized video
VAE checkpoint should be replaced with `configs/h3_i2va_w8a8.yaml`.

To isolate VAE quality, reuse the exact same `latents.pt` and `visual_cond.pt`
in two separate output directories, and decode with
`vae.video_decoder_precision: bf16` and `w8a8`, respectively.
The native W8A8 VAE path differs from earlier local latency experiments.
Do not infer end-to-end speed or quality parity from the W8A8 DiT benchmark;
this repository does not claim Wan2.2's speedup for H3.

## Validation Scope

Local validation includes native W8A8 CUDA tests and actual public SpargeAttn
CUDA comparisons against the same block mask expanded for SDPA, including
partial blocks and conditioning sinks. Run these after building the pinned
public dependency:

```bash
H3_TEST_PUBLIC_SPARGE=1 python -m unittest discover -s tests -p 'test_h3*.py' -v
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  tests/h3_sequence_parallel_check.py
```

All 41 H3 tests passed with real native W8A8/public SpargeAttn CUDA coverage.
The isolated PyTorch 2.11 validation uses the official wheel NCCL and rebuilt
native W8A8/public SpargeAttn extensions. Tests cover local dense refinement of
long prompts, FP32 preservation and legacy-checkpoint rejection, padding
exclusion, rank-zero-only video conversion, and the actual NCCL version API.
On 2026-11-03, the 513-token SP8/local regression passed for both attention
backends. Fresh four-rank encoding followed by default eight-rank sampling and
decoding also passed; the optional cache profile passed sample/decode using
that same freshly encoded conditioning. Both exported 124-frame audio/video
MP4s. This is a pipeline smoke test, not a quality or latency comparison.
Validation on other GPU topologies and SM90 hardware remains outstanding.
Generated baseline/accelerated comparisons are not included in this PR;
unit tests and pipeline smoke tests do not establish quality or latency parity.

import os
import io
import math
import tarfile
import time
import torch
import argparse
from tqdm import tqdm
from imaginaire.utils import distributed
from collections import defaultdict

from einops import repeat

from imaginaire.lazy_config import LazyCall as L, LazyDict, instantiate

from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.utils.umt5 import get_umt5_embedding
from rcm.utils.model_utils import init_weights_on_device, load_state_dict
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
from rcm.networks.wan2pt1 import WanModel
from rcm.samplers.euler import FlowEulerSampler
from rcm.samplers.unipc import FlowUniPCMultistepSampler

_DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

tensor_kwargs = {"device": "cuda", "dtype": torch.bfloat16}

WAN2PT1_1PT3B_T2V: LazyDict = L(WanModel)(
    dim=1536,
    eps=1e-06,
    ffn_dim=8960,
    freq_dim=256,
    in_dim=16,
    model_type="t2v",
    num_heads=12,
    num_layers=30,
    out_dim=16,
    text_len=512,
)

WAN2PT1_14B_T2V: LazyDict = L(WanModel)(
    dim=5120,
    eps=1e-06,
    ffn_dim=13824,
    freq_dim=256,
    in_dim=16,
    model_type="t2v",
    num_heads=40,
    num_layers=40,
    out_dim=16,
    text_len=512,
)

dit_configs = {"1.3B": WAN2PT1_1PT3B_T2V, "14B": WAN2PT1_14B_T2V}


def is_shard_done(shard_path):
    return os.path.exists(shard_path) and os.path.getsize(shard_path) > 0


def write_to_tar(tar, key, data_bytes):
    ti = tarfile.TarInfo(key)
    ti.size = len(data_bytes)
    tar.addfile(ti, io.BytesIO(data_bytes))


def barrier(interval=1):
    while True:
        time.sleep(interval)


def main(args):
    distributed.init()
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    prompts = prompts * args.repeat

    total = len(prompts)

    num_shards = math.ceil(total / args.samples_per_shard)

    my_shards = [i for i in range(num_shards) if i % world_size == rank]

    print(f"[Rank {rank}] Read {total} prompts, a total of {num_shards} shards.")

    print(f"[Rank {rank}] will build {len(my_shards)} shards.")

    with init_weights_on_device():
        net = instantiate(dit_configs[args.model_size]).eval()
    state_dict = load_state_dict(args.dit_path)
    prefix_to_load = "net."
    # drop net. prefix
    state_dict_dit_compatible = dict()
    for k, v in state_dict.items():
        if k.startswith(prefix_to_load):
            state_dict_dit_compatible[k[len(prefix_to_load) :]] = v
        else:
            state_dict_dit_compatible[k] = v
    net.load_state_dict(state_dict_dit_compatible, strict=False, assign=True)
    del state_dict, state_dict_dit_compatible
    print(f"Successfully loaded DiT from {args.dit_path}")

    net = net.to(**tensor_kwargs)
    torch.cuda.empty_cache()
    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)
    neg_text_emb = get_umt5_embedding(checkpoint_path=args.text_encoder_path, prompts=args.negative_prompt).to(dtype=torch.bfloat16).cuda()

    for shard_id in my_shards:
        shard_path = os.path.join(args.output_dir, f"shard_{shard_id:06d}.tar")

        if is_shard_done(shard_path):
            print(f"[Rank {rank}] Shard {shard_id} already completed. Skip.")
            continue

        start = shard_id * args.samples_per_shard
        end = min(total, start + args.samples_per_shard)

        print(f"[Rank {rank}] Building shard {shard_id}, items {start}..{end-1}")

        tmp_path = shard_path + ".tmp"

        processed_indices = set()
        if os.path.exists(tmp_path):
            try:
                with tarfile.open(tmp_path, "r") as existing_tar:
                    files_in_tar = defaultdict(list)
                    for member in existing_tar.getmembers():
                        parts = member.name.split(".")
                        if len(parts) == 3 and parts[0].isdigit():
                            index_str, file_type, _ = parts
                            files_in_tar[int(index_str)].append(file_type)

                    for index, types in files_in_tar.items():
                        if "latent" in types and "embed" in types and "prompt" in types:
                            processed_indices.add(index)

                if processed_indices:
                    print(f"[Rank {rank}] Found {len(processed_indices)} completed items in tmp file for shard {shard_id}.")

            except (tarfile.ReadError, EOFError) as e:
                print(f"[Rank {rank}] Tmp file for shard {shard_id} is corrupted ({e}). Deleting and starting over.")
                os.remove(tmp_path)
                processed_indices.clear()

        print(f"[Rank {rank}] Building shard {shard_id}, items {start}..{end-1}")

        for i in tqdm(range(start, end, args.batch_size), disable=(rank != 0)):
            batch_start = i
            batch_end = min(i + args.batch_size, end)

            # Filter out already processed items from the current batch
            original_indices_batch = list(range(batch_start, batch_end))

            indices_to_process = [idx for idx in original_indices_batch if idx not in processed_indices]

            if not indices_to_process:
                continue

            prompts_to_process = [prompts[idx] for idx in indices_to_process]

            text_embs_batch = get_umt5_embedding(checkpoint_path=args.text_encoder_path, prompts=prompts_to_process).to(dtype=torch.bfloat16).cuda()

            latents_batch = sample_latents_from_prompts(indices_to_process, text_embs_batch, neg_text_emb, net, tokenizer, args)

            try:
                with tarfile.open(tmp_path, "a") as tar:
                    for j, original_idx in enumerate(indices_to_process):
                        prompt = prompts_to_process[j]
                        latent = latents_batch[j]
                        text_emb = text_embs_batch[j]

                        key_prefix = f"{original_idx:09d}"

                        latent_buffer = io.BytesIO()
                        torch.save(latent.cpu(), latent_buffer)
                        write_to_tar(tar, f"{key_prefix}.latent.pt", latent_buffer.getvalue())

                        embed_buffer = io.BytesIO()
                        torch.save(text_emb.cpu(), embed_buffer)
                        write_to_tar(tar, f"{key_prefix}.embed.pt", embed_buffer.getvalue())

                        write_to_tar(tar, f"{key_prefix}.prompt.txt", prompt.encode("utf-8"))
            except Exception as e:
                print(f"[Rank {rank}] Failed to write batch to {tmp_path}: {e}")
                exit(1)

            del latents_batch, text_embs_batch

        tar.close()

        os.rename(tmp_path, shard_path)
        print(f"[Rank {rank}] Finished shard {shard_id}")

    print(f"[Rank {rank}] All done.")
    barrier()


def sample_latents_from_prompts(indices, text_embs, neg_text_emb, net, tokenizer, args):
    w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]

    batch_size = len(indices)

    condition = {"crossattn_emb": text_embs.to(**tensor_kwargs)}

    uncondition = {"crossattn_emb": repeat(neg_text_emb.to(**tensor_kwargs), "b l d -> (k b) l d", k=batch_size)}

    state_shape = [
        tokenizer.latent_ch,
        tokenizer.get_latent_num_frames(args.num_frames),
        h // tokenizer.spatial_compression_factor,
        w // tokenizer.spatial_compression_factor,
    ]

    noises = []
    for idx in indices:
        generator = torch.Generator(device=tensor_kwargs["device"])
        generator.manual_seed(args.seed + idx)
        noise = torch.randn(
            1,  # Generate one noise tensor at a time
            *state_shape,
            dtype=torch.float32,
            device=tensor_kwargs["device"],
            generator=generator,
        )
        noises.append(noise)

    init_noise = torch.cat(noises, dim=0)

    x = init_noise.to(torch.float64)

    sigma_max = args.sigma_max / (args.sigma_max + 1)
    unshifted_sigma_max = sigma_max / (args.timestep_shift - (args.timestep_shift - 1) * sigma_max)

    samplers = {"Euler": FlowEulerSampler, "UniPC": FlowUniPCMultistepSampler}
    sampler = samplers[args.sampler](num_train_timesteps=1000, sigma_max=unshifted_sigma_max, sigma_min=0.0)
    sampler.set_timesteps(num_inference_steps=args.num_steps, device=tensor_kwargs["device"], shift=args.timestep_shift)

    ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
    for _, t in enumerate(tqdm(sampler.timesteps)):
        timesteps = t * ones

        with torch.no_grad():
            v_cond = net(x_B_C_T_H_W=x.to(**tensor_kwargs), timesteps_B_T=timesteps.to(**tensor_kwargs), **condition).float()
            v_uncond = net(x_B_C_T_H_W=x.to(**tensor_kwargs), timesteps_B_T=timesteps.to(**tensor_kwargs), **uncondition).float()

        v_pred = v_uncond + args.guidance_scale * (v_cond - v_uncond)

        x = sampler.step(v_pred, t, x)

    return x.to(**tensor_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Dataset building arguments
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--samples_per_shard", type=int, default=256)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for sampling. For best efficiency, choose a divisor of samples_per_shard."
    )

    # Sampling arguments
    parser.add_argument("--model_size", choices=["1.3B", "14B"], default="14B", help="Size of the model to use")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of sampling steps")
    parser.add_argument("--sigma_max", type=int, default=5000, help="Initial timestep represented by EDM sigma")
    parser.add_argument("--sampler", choices=["Euler", "UniPC"], default="Euler", help="Sampler")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="CFG scale")
    parser.add_argument("--timestep_shift", type=float, default=3.0, help="Timestep shift as in Wan")
    parser.add_argument("--dit_path", type=str, default="assets/checkpoints/Wan2.1-T2V-14B.pth", help="Path to the video diffusion model.")
    parser.add_argument("--vae_path", type=str, default="assets/checkpoints/Wan2.1_VAE.pth", help="Path to the Wan2.1 VAE.")
    parser.add_argument(
        "--text_encoder_path", type=str, default="assets/checkpoints/models_t5_umt5-xxl-enc-bf16.pth", help="Path to the umT5 text encoder."
    )
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to generate")
    parser.add_argument("--negative_prompt", type=str, default=_DEFAULT_NEGATIVE_PROMPT, help="Negative text prompt for video generation")
    parser.add_argument("--resolution", default="480p", type=str, help="Resolution of the generated output")
    parser.add_argument("--aspect_ratio", default="16:9", type=str, help="Aspect ratio of the generated output (width:height)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    main(args)

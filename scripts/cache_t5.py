#!/usr/bin/env python
"""
Pre-cache T5 text embeddings to avoid loading the 11GB model during inference.

Usage:
    # Cache a single prompt
    python scripts/cache_t5.py --prompt "slow head turn, cinematic" --output cached_embeddings.pt

    # Cache multiple prompts from file
    python scripts/cache_t5.py --prompts_file prompts.txt --output cached_embeddings.pt

Then use with inference:
    python turbodiffusion/inference/wan2.2_i2v_infer.py \
        --cached_embedding cached_embeddings.pt \
        --skip_t5 \
        ...
"""
import os
import sys
import argparse
import torch

# Add repo root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

def main():
    parser = argparse.ArgumentParser(description="Pre-cache T5 text embeddings")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt to cache")
    parser.add_argument("--prompts_file", type=str, default=None, help="File with prompts (one per line)")
    parser.add_argument("--text_encoder_path", type=str,
                        default="/media/2TB/ComfyUI/models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth",
                        help="Path to the umT5 text encoder")
    parser.add_argument("--output", type=str, default="cached_t5_embeddings.pt",
                        help="Output path for cached embeddings")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for encoding (cuda is faster, memory freed after)")
    args = parser.parse_args()

    # Collect prompts
    prompts = []
    if args.prompt:
        prompts.append(args.prompt)
    if args.prompts_file and os.path.exists(args.prompts_file):
        with open(args.prompts_file, 'r') as f:
            prompts.extend([line.strip() for line in f if line.strip()])

    if not prompts:
        print("Error: Provide --prompt or --prompts_file")
        sys.exit(1)

    print(f"Caching embeddings for {len(prompts)} prompt(s)")
    print(f"Text encoder: {args.text_encoder_path}")
    print(f"Device: {args.device}")
    print()

    # Import after path setup
    from rcm.utils.umt5 import get_umt5_embedding, clear_umt5_memory

    cache_data = {
        'prompts': prompts,
        'embeddings': [],
        'text_encoder_path': args.text_encoder_path,
    }

    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            print(f"[{i+1}/{len(prompts)}] Encoding: '{prompt[:60]}...' " if len(prompt) > 60 else f"[{i+1}/{len(prompts)}] Encoding: '{prompt}'")

            # Get embedding (loads T5 if not already loaded)
            embedding = get_umt5_embedding(
                checkpoint_path=args.text_encoder_path,
                prompts=prompt
            )

            # Move to CPU for storage
            cache_data['embeddings'].append({
                'prompt': prompt,
                'embedding': embedding.cpu(),
                'shape': list(embedding.shape),
            })

            print(f"    Shape: {embedding.shape}, dtype: {embedding.dtype}")

    # Clear T5 from memory
    print("\nClearing T5 from memory...")
    clear_umt5_memory()
    torch.cuda.empty_cache()

    # Save cache
    print(f"\nSaving to: {args.output}")
    torch.save(cache_data, args.output)

    # Summary
    file_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Done! Cache file size: {file_size:.2f} MB")
    print()
    print("Usage:")
    print(f"  python turbodiffusion/inference/wan2.2_i2v_infer.py \\")
    print(f"      --cached_embedding {args.output} \\")
    print(f"      --skip_t5 \\")
    print(f"      ... (other args)")


if __name__ == "__main__":
    main()

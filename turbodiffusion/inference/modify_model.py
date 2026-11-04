""" 
Copyright (c) 2025 by TurboDiffusion team.

Licensed under the Apache License, Version 2.0 (the "License");

Citation (please cite if you use this code):

@article{zhang2025turbodiffusion,
  title={TurboDiffusion: Accelerating Video Diffusion Models by 100-200 Times},
  author={Zhang, Jintao and Zheng, Kaiwen and Jiang, Kai and Wang, Haoxu and Stoica, Ion and Gonzalez, Joseph E and Chen, Jianfei and Zhu, Jun},
  journal={arXiv preprint arXiv:2512.16093},
  year={2025}
}
"""

import argparse

import torch
from rcm.utils.model_utils import load_state_dict
from rcm.networks.wan2pt1 import (
    WanModel as WanModel2pt1,
    WanLayerNorm as WanLayerNorm2pt1,
    WanRMSNorm as WanRMSNorm2pt1,
    WanSelfAttention as WanSelfAttention2pt1
)
from rcm.networks.wan2pt2 import (
    WanModel as WanModel2pt2,
    WanLayerNorm as WanLayerNorm2pt2,
    WanRMSNorm as WanRMSNorm2pt2,
    WanSelfAttention as WanSelfAttention2pt2
)

from ops import FastLayerNorm, FastRMSNorm, Int8Linear
from SLA import (
    SparseLinearAttention as SLA,
    SageSparseLinearAttention as SageSLA
)


def replace_attention(
    model: torch.nn.Module,
    attention_type: str,
    sla_topk: float,
) -> torch.nn.Module:
    assert attention_type in ["sla", "sagesla"], "Invalid attention type."
    
    for module in model.modules():
        if type(module) is WanSelfAttention2pt1 or type(module) is WanSelfAttention2pt2:
            if attention_type == "sla":
                module.attn_op.local_attn = SLA(head_dim=module.dim // module.num_heads, topk=sla_topk, BLKQ=128, BLKK=64)
            elif attention_type == "sagesla":
                module.attn_op.local_attn = SageSLA(head_dim=module.dim // module.num_heads, topk=sla_topk)
    return model


def replace_linear_norm(
    model: torch.nn.Module,
    replace_linear: bool = False,
    replace_norm: bool = False,
    quantize: bool = True,
    skip_layer: str = "proj_l"
) -> torch.nn.Module:
    replacements = {}
    for name, module in model.blocks.named_modules():
        if isinstance(module, torch.nn.Linear) and replace_linear:
            if skip_layer not in name:
                replacements[name] = Int8Linear.from_linear(module, quantize)
        
        if (isinstance(module, WanRMSNorm2pt1) or isinstance(module, WanRMSNorm2pt2)) and replace_norm:
            replacements[name] = FastRMSNorm.from_rmsnorm(module)
        
        if (isinstance(module, WanLayerNorm2pt1) or isinstance(module, WanLayerNorm2pt2)) and replace_norm:
            replacements[name] = FastLayerNorm.from_layernorm(module)

    for name, new_module in replacements.items():
        parent_module = model.blocks
        name_parts = name.split(".")
        for part in name_parts[:-1]:
            parent_module = getattr(parent_module, part)
        setattr(parent_module, name_parts[-1], new_module)
    return model


tensor_kwargs = {"device": "cuda", "dtype": torch.float16}

def select_model(model_name: str) -> torch.nn.Module:
    if model_name == "Wan2.1-1.3B":
        return WanModel2pt1(
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
    elif model_name == "Wan2.1-14B":
        return WanModel2pt1(
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
    elif model_name == "Wan2.2-A14B":
        return WanModel2pt2(
            dim=5120,
            eps=1e-06,
            ffn_dim=13824,
            freq_dim=256,
            in_dim=36,
            model_type="i2v",
            num_heads=40,
            num_layers=40,
            out_dim=16,
            text_len=512,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def create_model(dit_path: str, args: argparse.Namespace) -> torch.nn.Module:
    with torch.device("meta"):
        net = select_model(args.model)

    state_dict = load_state_dict(dit_path)
    if args.attention_type in ['sla', 'sagesla']:
        net = replace_attention(net, attention_type=args.attention_type, sla_topk=args.sla_topk)
    replace_linear_norm(net, replace_linear=args.quant_linear, replace_norm=not args.default_norm, quantize=False)
    net.load_state_dict(state_dict, assign=True)
    net = net.to(tensor_kwargs["device"]).eval()
    del state_dict
    return net


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TurboDiffusion replace attention module & quantize model")
    parser.add_argument("--model", choices=["Wan2.1-1.3B", "Wan2.1-14B", "Wan2.2-A14B"], default="Wan2.1-1.3B", help="Model to use")
    parser.add_argument("--input_path", type=str, default="", help="Input path to the DiT model checkpoint for Wan model after rCM-SLA finetuning")
    parser.add_argument("--output_path", type=str, default="", help="Custom path to save the modified model checkpoint")
    parser.add_argument("--attention_type", choices=["sla", "sagesla", "original"], default="original", help="Type of attention mechanism to use")
    parser.add_argument("--sla_topk", type=float, default=0.2, help="Top-k ratio for SLA/SageSLA attention")
    parser.add_argument("--quant_linear", action="store_true", help="Whether to replace Linear layers with quantized versions")
    parser.add_argument("--default_norm", action="store_true", help="Whether to replace LayerNorm/RMSNorm layers with faster versions")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    with torch.device("meta"):
        net = select_model(args.model)

    state_dict = load_state_dict(args.input_path)["state_dict"]

    # drop net. prefix
    prefix_to_load = "net."
    state_dict_dit_compatible = dict()
    for k, v in state_dict.items():
        new_key = k[len(prefix_to_load) :] if k.startswith(prefix_to_load) else k
        # reshape patch embedding if needed
        if k.endswith("patch_embedding.weight"):
            v = v.reshape(net.patch_embedding.weight.shape)
        if k.endswith("patch_embedding.bias"):
            v = v.reshape(net.patch_embedding.bias.shape)
        state_dict_dit_compatible[new_key] = v

    if args.attention_type in ['sla', 'sagesla']:
        net = replace_attention(net, attention_type=args.attention_type, sla_topk=args.sla_topk)
    net.load_state_dict(state_dict_dit_compatible, strict=False, assign=True)
    net = net.to(tensor_kwargs["device"]).eval()
    del state_dict, state_dict_dit_compatible

    net = replace_linear_norm(net, replace_linear=args.quant_linear, replace_norm=not args.default_norm)
    torch.save(net.state_dict(), args.output_path)

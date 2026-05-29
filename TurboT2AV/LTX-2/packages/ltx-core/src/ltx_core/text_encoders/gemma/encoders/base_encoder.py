import functools
from pathlib import Path

import torch
from einops import rearrange
from transformers import AutoImageProcessor, Gemma3ForConditionalGeneration, Gemma3Processor

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorProjLinear
from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
from ltx_core.utils import find_matching_file


class GemmaTextEncoderModelBase(torch.nn.Module):
    """
    Gemma Text Encoder Model.
    This base class combines the tokenizer, Gemma model and feature extractor to provide a preprocessing
    for implementation classes for multimodal pipelines. It processes input text through tokenization,
    obtains hidden states from the base language model, applies a linear feature extractor.
    Args:
        tokenizer (LTXVGemmaTokenizer): The tokenizer used for text preprocessing.
        model (Gemma3ForConditionalGeneration): The base Gemma LLM.
        feature_extractor_linear (GemmaFeaturesExtractorProjLinear): Linear projection for hidden state aggregation.
        dtype (torch.dtype, optional): The data type for model parameters (default: torch.bfloat16).
    """

    def __init__(
        self,
        feature_extractor_linear: GemmaFeaturesExtractorProjLinear,
        tokenizer: LTXVGemmaTokenizer | None = None,
        model: Gemma3ForConditionalGeneration | None = None,
        img_processor: Gemma3Processor | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.processor = img_processor
        self.feature_extractor_linear = feature_extractor_linear.to(dtype=dtype)

    def _run_feature_extractor(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, padding_side: str = "right"
    ) -> torch.Tensor:
        encoded_text_features = torch.stack(hidden_states, dim=-1)
        encoded_text_features_dtype = encoded_text_features.dtype

        sequence_lengths = attention_mask.sum(dim=-1)
        normed_concated_encoded_text_features = _norm_and_concat_padded_batch(
            encoded_text_features, sequence_lengths, padding_side=padding_side
        )

        return self.feature_extractor_linear(normed_concated_encoded_text_features.to(encoded_text_features_dtype))

    def _convert_to_additive_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return (attention_mask - 1).to(dtype).reshape(
            (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        ) * torch.finfo(dtype).max

    def _preprocess_text(self, text: str, padding_side: str = "left") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Encode a given string into feature tensors suitable for downstream tasks.
        Args:
            text (str): Input string to encode.
        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]: Encoded features and a dictionary with attention mask.
        """
        token_pairs = self.tokenizer.tokenize_with_weights(text)["gemma"]
        input_ids = torch.tensor([[t[0] for t in token_pairs]], device=self.model.device)
        attention_mask = torch.tensor([[w[1] for w in token_pairs]], device=self.model.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        projected = self._run_feature_extractor(
            hidden_states=outputs.hidden_states, attention_mask=attention_mask, padding_side=padding_side
        )
        return projected, attention_mask

    def _enhance(
        self,
        messages: list[dict[str, str]],
        image: torch.Tensor | None = None,
        max_new_tokens: int = 512,
        seed: int = 10,
    ) -> str:
        text = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        model_inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
        ).to(self.model.device)
        pad_token_id = self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else 0
        model_inputs = _pad_inputs_for_attention_alignment(model_inputs, pad_token_id=pad_token_id)

        with torch.inference_mode(), torch.random.fork_rng(devices=[self.model.device]):
            torch.manual_seed(seed)
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
            )
            generated_ids = outputs[0][len(model_inputs.input_ids[0]) :]
            enhanced_prompt = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return enhanced_prompt

    def enhance_t2v(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
        seed: int = 10,
    ) -> str:
        """Enhance a text prompt for T2V generation."""

        system_prompt = system_prompt or self.default_gemma_t2v_system_prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"user prompt: {prompt}"},
        ]

        return self._enhance(messages, max_new_tokens=max_new_tokens, seed=seed)

    def enhance_i2v(
        self,
        prompt: str,
        image: torch.Tensor,
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
        seed: int = 10,
    ) -> str:
        """Enhance a text prompt for I2V generation using a reference image."""
        system_prompt = system_prompt or self.default_gemma_i2v_system_prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"User Raw Input Prompt: {prompt}."},
                ],
            },
        ]
        return self._enhance(messages, image=image, max_new_tokens=max_new_tokens, seed=seed)

    @functools.cached_property
    def default_gemma_i2v_system_prompt(self) -> str:
        return _load_system_prompt("gemma_i2v_system_prompt.txt")

    @functools.cached_property
    def default_gemma_t2v_system_prompt(self) -> str:
        return _load_system_prompt("gemma_t2v_system_prompt.txt")

    def forward(self, text: str, padding_side: str = "left") -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("This method is not implemented for the base class")


def _norm_and_concat_padded_batch(
    encoded_text: torch.Tensor,
    sequence_lengths: torch.Tensor,
    padding_side: str = "right",
) -> torch.Tensor:
    """Normalize and flatten multi-layer hidden states, respecting padding.
    Performs per-batch, per-layer normalization using masked mean and range,
    then concatenates across the layer dimension.
    Args:
        encoded_text: Hidden states of shape [batch, seq_len, hidden_dim, num_layers].
        sequence_lengths: Number of valid (non-padded) tokens per batch item.
        padding_side: Whether padding is on "left" or "right".
    Returns:
        Normalized tensor of shape [batch, seq_len, hidden_dim * num_layers],
        with padded positions zeroed out.
    """
    b, t, d, l = encoded_text.shape  # noqa: E741
    device = encoded_text.device

    # Build mask: [B, T, 1, 1]
    token_indices = torch.arange(t, device=device)[None, :]  # [1, T]

    if padding_side == "right":
        # For right padding, valid tokens are from 0 to sequence_length-1
        mask = token_indices < sequence_lengths[:, None]  # [B, T]
    elif padding_side == "left":
        # For left padding, valid tokens are from (T - sequence_length) to T-1
        start_indices = t - sequence_lengths[:, None]  # [B, 1]
        mask = token_indices >= start_indices  # [B, T]
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")

    mask = rearrange(mask, "b t -> b t 1 1")

    eps = 1e-6

    # Compute masked mean: [B, 1, 1, L]
    masked = encoded_text.masked_fill(~mask, 0.0)
    denom = (sequence_lengths * d).view(b, 1, 1, 1)
    mean = masked.sum(dim=(1, 2), keepdim=True) / (denom + eps)

    # Compute masked min/max: [B, 1, 1, L]
    x_min = encoded_text.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
    x_max = encoded_text.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)
    range_ = x_max - x_min

    # Normalize only the valid tokens
    normed = 8 * (encoded_text - mean) / (range_ + eps)

    # concat to be [Batch, T,  D * L] - this preserves the original structure
    normed = normed.reshape(b, t, -1)  # [B, T, D * L]

    # Apply mask to preserve original padding (set padded positions to 0)
    mask_flattened = rearrange(mask, "b t 1 1 -> b t 1").expand(-1, -1, d * l)
    normed = normed.masked_fill(~mask_flattened, 0.0)

    return normed


@functools.lru_cache(maxsize=2)
def _load_system_prompt(prompt_name: str) -> str:
    with open(Path(__file__).parent / "prompts" / f"{prompt_name}", "r") as f:
        return f.read()


def module_ops_from_gemma_root(gemma_root: str) -> tuple[ModuleOps, ...]:
    tokenizer_root = str(find_matching_file(gemma_root, "tokenizer.model").parent)
    processor_root = str(find_matching_file(gemma_root, "preprocessor_config.json").parent)

    def load_tokenizer(module: GemmaTextEncoderModelBase) -> GemmaTextEncoderModelBase:
        module.tokenizer = LTXVGemmaTokenizer(tokenizer_root, 1024)
        return module

    def load_processor(module: GemmaTextEncoderModelBase) -> GemmaTextEncoderModelBase:
        image_processor = AutoImageProcessor.from_pretrained(processor_root, local_files_only=True)
        if not module.tokenizer:
            raise ValueError("Tokenizer model operation must be performed before processor model operation")
        module.processor = Gemma3Processor(image_processor=image_processor, tokenizer=module.tokenizer.tokenizer)
        return module

    tokenizer_load_ops = ModuleOps(
        "TokenizerLoad",
        matcher=lambda module: isinstance(module, GemmaTextEncoderModelBase) and module.tokenizer is None,
        mutator=load_tokenizer,
    )
    processor_load_ops = ModuleOps(
        "ProcessorLoad",
        matcher=lambda module: isinstance(module, GemmaTextEncoderModelBase) and module.processor is None,
        mutator=load_processor,
    )
    return (tokenizer_load_ops, processor_load_ops)


def encode_text(text_encoder: GemmaTextEncoderModelBase, prompts: list[str]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Encode a list of prompts using the provided Gemma text encoder.
    Args:
        text_encoder: The Gemma text encoder instance.
        prompts: List of prompt strings to encode.
    Returns:
        List of tuples, each containing (v_context, a_context) tensors for each prompt.
    """
    result = []
    for prompt in prompts:
        v_context, a_context, _ = text_encoder(prompt)
        result.append((v_context, a_context))
    return result


def _cat_with_padding(
    tensor: torch.Tensor,
    padding_length: int,
    value: int | float,
) -> torch.Tensor:
    """Concatenate a tensor with a padding tensor of the given value."""
    return torch.cat(
        [
            tensor,
            torch.full(
                (1, padding_length),
                value,
                dtype=tensor.dtype,
                device=tensor.device,
            ),
        ],
        dim=1,
    )


def _pad_inputs_for_attention_alignment(
    model_inputs: dict[str, torch.Tensor],
    pad_token_id: int = 0,
    alignment: int = 8,
) -> dict[str, torch.Tensor]:
    """Pad sequence length to multiple of alignment for Flash Attention compatibility.
    Flash Attention within SDPA requires sequence lengths aligned to 8 bytes.
    This pads input_ids, attention_mask, and token_type_ids (if present) to prevent
    'p.attn_bias_ptr is not correctly aligned' errors.
    """
    seq_len = model_inputs.input_ids.shape[1]
    padded_len = ((seq_len + alignment - 1) // alignment) * alignment
    padding_length = padded_len - seq_len

    if padding_length > 0:
        model_inputs["input_ids"] = _cat_with_padding(model_inputs.input_ids, padding_length, pad_token_id)

        model_inputs["attention_mask"] = _cat_with_padding(model_inputs.attention_mask, padding_length, 0)

        if "token_type_ids" in model_inputs and model_inputs["token_type_ids"] is not None:
            model_inputs["token_type_ids"] = _cat_with_padding(model_inputs["token_type_ids"], padding_length, 0)

    return model_inputs

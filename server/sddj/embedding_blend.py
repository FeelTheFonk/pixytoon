"""Prompt embedding utilities — SLERP/LERP for smooth prompt transitions.

Provides functions to interpolate between CLIP text embeddings, enabling
smooth visual transitions between prompts instead of hard cuts or flickering
frame alternation.
"""

from __future__ import annotations

import logging
import math

import torch

log = logging.getLogger("sddj.embedding_blend")


def slerp(
    embed_a: torch.Tensor,
    embed_b: torch.Tensor,
    t: float,
) -> torch.Tensor:
    """Spherical linear interpolation between two embedding tensors.

    More geometrically correct than LERP for unit-sphere-normalized embeddings
    (CLIP text embeddings lie approximately on a hypersphere).

    Args:
        embed_a: First embedding tensor [batch, seq_len, dim]
        embed_b: Second embedding tensor [batch, seq_len, dim]
        t: Interpolation weight [0.0=A, 1.0=B]

    Returns:
        Interpolated embedding tensor.
    """
    # Edge cases
    if t <= 0.0:
        return embed_a
    if t >= 1.0:
        return embed_b

    # Flatten for dot product computation, then restore shape
    orig_shape = embed_a.shape
    flat_a = embed_a.reshape(-1).float()
    flat_b = embed_b.reshape(-1).float()

    # Normalize
    norm_a = torch.linalg.norm(flat_a)
    norm_b = torch.linalg.norm(flat_b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        # Degenerate: fall back to LERP
        return lerp(embed_a, embed_b, t)

    unit_a = flat_a / norm_a
    unit_b = flat_b / norm_b

    # Compute angle
    cos_omega = torch.clamp(torch.dot(unit_a, unit_b), -1.0, 1.0).item()

    # If embeddings are nearly identical, use LERP to avoid division by zero
    if abs(cos_omega) > 0.9995:
        return lerp(embed_a, embed_b, t)

    omega = math.acos(cos_omega)
    sin_omega = math.sin(omega)

    # SLERP formula
    coeff_a = math.sin((1.0 - t) * omega) / sin_omega
    coeff_b = math.sin(t * omega) / sin_omega

    # Interpolate with magnitude preservation
    avg_norm = norm_a * (1.0 - t) + norm_b * t
    result_flat = coeff_a * unit_a + coeff_b * unit_b
    result_flat = result_flat * avg_norm

    return result_flat.reshape(orig_shape).to(embed_a.dtype)


def lerp(
    embed_a: torch.Tensor,
    embed_b: torch.Tensor,
    t: float,
) -> torch.Tensor:
    """Linear interpolation between two embedding tensors.

    Simpler than SLERP but may produce slightly shorter intermediate vectors.
    Suitable when embeddings are near-parallel.
    """
    if t <= 0.0:
        return embed_a
    if t >= 1.0:
        return embed_b
    return embed_a * (1.0 - t) + embed_b * t


def blend_prompt_embeds(
    pipe,
    prompt_a: str,
    prompt_b: str,
    blend_weight: float,
    negative_prompt: str = "",
    negative_prompt_b: str = "",
    clip_skip: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode two prompts and blend their embeddings via SLERP.

    Args:
        pipe: A diffusers pipeline with a text encoder and tokenizer.
        prompt_a: Outgoing prompt (weight = 1 - blend_weight).
        prompt_b: Incoming prompt (weight = blend_weight).
        blend_weight: 0.0 = fully A, 1.0 = fully B.
        negative_prompt: Outgoing negative prompt.
        negative_prompt_b: Incoming negative prompt (blended with outgoing).
        clip_skip: Number of CLIP layers to skip.

    Returns:
        (prompt_embeds, negative_embeds) ready for pipeline __call__.
    """
    # No blending needed
    if blend_weight <= 0.0:
        return _encode_prompt(pipe, prompt_a, negative_prompt, clip_skip)
    if blend_weight >= 1.0:
        neg = negative_prompt_b or negative_prompt
        return _encode_prompt(pipe, prompt_b, neg, clip_skip)

    # Encode both prompts
    embed_a, neg_embeds_a = _encode_prompt(pipe, prompt_a, negative_prompt, clip_skip)
    neg_b = negative_prompt_b or negative_prompt
    embed_b, neg_embeds_b = _encode_prompt(pipe, prompt_b, neg_b, clip_skip)

    # SLERP the positive embeddings
    blended = slerp(embed_a, embed_b, blend_weight)

    # SLERP the negative embeddings when both negatives differ
    if negative_prompt_b and negative_prompt_b != negative_prompt:
        neg_blended = slerp(neg_embeds_a, neg_embeds_b, blend_weight)
    else:
        neg_blended = neg_embeds_a

    return blended, neg_blended


def _encode_prompt(
    pipe,
    prompt: str,
    negative_prompt: str,
    clip_skip: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a single prompt + negative via the pipeline's text encoder.

    Handles clip_skip by extracting from the appropriate hidden layer.
    """
    # Use the pipeline's built-in encoding method
    if hasattr(pipe, "encode_prompt"):
        # Modern diffusers API
        result = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            device=pipe.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )
        if len(result) >= 2:
            return result[0], result[1]  # prompt_embeds, negative_embeds
        return result[0], result[0]  # fallback

    # Manual encoding fallback
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    # Positive
    pos_tokens = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(text_encoder.device)

    with torch.no_grad():
        if clip_skip > 0 and hasattr(text_encoder.config, "num_hidden_layers"):
            outputs = text_encoder(pos_tokens, output_hidden_states=True)
            layer_idx = -(clip_skip + 1)
            pos_embeds = outputs.hidden_states[layer_idx]
            pos_embeds = text_encoder.text_model.final_layer_norm(pos_embeds)
        else:
            pos_embeds = text_encoder(pos_tokens)[0]

    # Negative
    neg_tokens = tokenizer(
        negative_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(text_encoder.device)

    with torch.no_grad():
        if clip_skip > 0 and hasattr(text_encoder.config, "num_hidden_layers"):
            outputs = text_encoder(neg_tokens, output_hidden_states=True)
            layer_idx = -(clip_skip + 1)
            neg_embeds = outputs.hidden_states[layer_idx]
            neg_embeds = text_encoder.text_model.final_layer_norm(neg_embeds)
        else:
            neg_embeds = text_encoder(neg_tokens)[0]

    return pos_embeds, neg_embeds

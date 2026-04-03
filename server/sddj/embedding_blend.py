"""Prompt embedding utilities — SLERP/LERP for smooth prompt transitions.

Provides functions to interpolate between CLIP text embeddings, enabling
smooth visual transitions between prompts instead of hard cuts or flickering
frame alternation.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict

import torch

log = logging.getLogger("sddj.embedding_blend")


class _EmbeddingCache:
    """LRU cache for CLIP prompt embeddings. Eliminates redundant tokenization+encoding."""

    def __init__(self, maxsize: int = 1024):
        self._cache: OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: tuple) -> tuple[torch.Tensor, torch.Tensor] | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: tuple, embeds: tuple[torch.Tensor, torch.Tensor]):
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
            self._cache[key] = embeds

    def clear(self):
        self._cache.clear()


# Thread-safety contract: _embed_cache and _model_generation are accessed only
# from the engine thread under _generate_lock (via run_in_executor). Only one
# executor task runs at a time due to the asyncio.Lock serialization in
# server.py — this lock, not the GIL, provides the actual safety guarantee.
# Defense in depth: _embed_cache_lock protects cache get/put even though the
# contract guarantees serialized access. Cost: ~25ns per uncontended acquire.
_embed_cache = _EmbeddingCache()
_embed_cache_lock = threading.Lock()

_NORM_EPSILON = 1e-8
_COLLINEAR_THRESHOLD = 0.9995

_model_generation: int = 0


def bump_model_generation() -> None:
    """Increment the model generation counter. Call when pipeline/model is reloaded."""
    global _model_generation
    _model_generation += 1


def slerp(
    embed_a: torch.Tensor,
    embed_b: torch.Tensor,
    t: float,
) -> torch.Tensor:
    """Spherical linear interpolation between two embedding tensors.

    Per-token SLERP: normalizes and interpolates along the embedding dimension
    (last axis) independently for each token position. This preserves the
    sequential structure of CLIP embeddings [batch, seq_len, dim] instead of
    collapsing the entire tensor into a single vector.

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

    orig_dtype = embed_a.dtype
    a = embed_a.float()
    b = embed_b.float()

    # Per-token normalization along embedding dim (last axis)
    norm_a = torch.linalg.norm(a, dim=-1, keepdim=True)
    norm_b = torch.linalg.norm(b, dim=-1, keepdim=True)

    # Degenerate check: if any token has near-zero norm, fall back to LERP
    if (norm_a < _NORM_EPSILON).any() or (norm_b < _NORM_EPSILON).any():
        return lerp(embed_a, embed_b, t)

    unit_a = a / norm_a
    unit_b = b / norm_b

    # Per-token cosine similarity (dot product along last dim)
    cos_omega = torch.sum(unit_a * unit_b, dim=-1, keepdim=True).clamp(-1.0, 1.0)

    # Where tokens are nearly collinear, use LERP; elsewhere use SLERP
    is_collinear = torch.abs(cos_omega) > _COLLINEAR_THRESHOLD

    omega = torch.acos(cos_omega)
    sin_omega = torch.sin(omega)
    # Avoid division by zero for collinear tokens
    safe_sin = torch.where(is_collinear, torch.ones_like(sin_omega), sin_omega)

    coeff_a = torch.sin((1.0 - t) * omega) / safe_sin
    coeff_b = torch.sin(t * omega) / safe_sin

    # Per-token magnitude preservation
    avg_norm = norm_a * (1.0 - t) + norm_b * t
    slerp_result = (coeff_a * unit_a + coeff_b * unit_b) * avg_norm

    # Blend: LERP for collinear tokens, SLERP for the rest
    lerp_result = a * (1.0 - t) + b * t
    result = torch.where(is_collinear, lerp_result, slerp_result)

    return result.to(orig_dtype)


def slerp_batch(
    embed_a: torch.Tensor,
    embed_b: torch.Tensor,
    t_values: torch.Tensor,
) -> torch.Tensor:
    """Batch SLERP: compute all interpolation steps in one GPU kernel.

    Instead of calling slerp() N times with scalar t (N separate GPU kernel
    launches), this computes all N interpolations in a single batched operation.

    Args:
        embed_a: First embedding tensor [seq_len, dim] or [batch, seq_len, dim]
        embed_b: Second embedding tensor (same shape as embed_a)
        t_values: 1-D tensor of interpolation weights, shape (N,)

    Returns:
        Interpolated embeddings, shape (N, seq_len, dim).
    """
    N = t_values.shape[0]

    # Ensure 3-D: (seq_len, dim) -> (1, seq_len, dim)
    if embed_a.dim() == 2:
        embed_a = embed_a.unsqueeze(0)
    if embed_b.dim() == 2:
        embed_b = embed_b.unsqueeze(0)

    # t_values shape: (N,) -> (N, 1, 1) for broadcasting with (1, seq_len, dim)
    t = t_values.view(N, 1, 1)

    a = embed_a.expand(N, -1, -1).float()   # (N, seq_len, dim)
    b = embed_b.expand(N, -1, -1).float()

    norm_a = torch.linalg.norm(a, dim=-1, keepdim=True)
    norm_b = torch.linalg.norm(b, dim=-1, keepdim=True)

    unit_a = a / norm_a.clamp(min=_NORM_EPSILON)
    unit_b = b / norm_b.clamp(min=_NORM_EPSILON)

    cos_omega = (unit_a * unit_b).sum(dim=-1, keepdim=True).clamp(-1, 1)
    omega = torch.acos(cos_omega)
    sin_omega = torch.sin(omega).clamp(min=_NORM_EPSILON)

    coeff_a = torch.sin((1 - t) * omega) / sin_omega
    coeff_b = torch.sin(t * omega) / sin_omega

    avg_norm = norm_a * (1 - t) + norm_b * t
    result = (coeff_a * unit_a + coeff_b * unit_b) * avg_norm

    # Collinear fallback: use LERP where embeddings are nearly parallel
    is_collinear = cos_omega.abs() > _COLLINEAR_THRESHOLD
    lerp_result = a * (1 - t) + b * t
    result = torch.where(is_collinear, lerp_result, result)

    return result.to(embed_a.dtype)


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


def clear_embedding_cache():
    """Call when pipeline/model changes to invalidate cached embeddings."""
    with _embed_cache_lock:
        _embed_cache.clear()


def _encode_prompt(
    pipe,
    prompt: str,
    negative_prompt: str,
    clip_skip: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a single prompt + negative via the pipeline's text encoder.

    Handles clip_skip by extracting from the appropriate hidden layer.
    Uses LRU cache to eliminate redundant tokenization+encoding across frames.
    """
    cache_key = (prompt, negative_prompt or "", clip_skip, _model_generation)
    with _embed_cache_lock:
        cached = _embed_cache.get(cache_key)
    if cached is not None:
        return cached

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
            embeds = (result[0], result[1])  # prompt_embeds, negative_embeds
        else:
            embeds = (result[0], result[0])  # fallback
        with _embed_cache_lock:
            _embed_cache.put(cache_key, embeds)
        return embeds

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

    with torch.inference_mode():
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

    with torch.inference_mode():
        if clip_skip > 0 and hasattr(text_encoder.config, "num_hidden_layers"):
            outputs = text_encoder(neg_tokens, output_hidden_states=True)
            layer_idx = -(clip_skip + 1)
            neg_embeds = outputs.hidden_states[layer_idx]
            neg_embeds = text_encoder.text_model.final_layer_norm(neg_embeds)
        else:
            neg_embeds = text_encoder(neg_tokens)[0]

    result = (pos_embeds, neg_embeds)
    with _embed_cache_lock:
        _embed_cache.put(cache_key, result)
    return result

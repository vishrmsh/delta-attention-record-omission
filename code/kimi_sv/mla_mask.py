"""Mask selected key positions out of Kimi Linear's global MLA layers.

The paper's "attention masking" control evicts a record's rows from the MLA
key/value cache while leaving the KDA recurrent state untouched. The original
control ran through the SV-gated attention graft with the gate disabled, which
replaces the fused attention kernel by a blocked softmax and therefore changes
the arithmetic slightly for every condition, masked or not.

This module masks the same rows without leaving the native code path. Kimi's
MLA layer hands the fused kernel an additive score matrix (the rotary part of
the logits) as its mask, so setting those scores to the dtype minimum at the
dropped positions gives those keys exactly zero softmax weight while every
other operation stays the one the released implementation performs. With no
positions dropped the wrapper reproduces the base layer bit for bit, which the
tests check.
"""
from __future__ import annotations

from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import scaled_dot_product_attention

from .state import classify_layers


class MaskedMLAAttention(nn.Module):
    """Drop-in for ``KimiMLAAttention`` with an optional set of evicted keys."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.is_linear = False
        # Absolute context positions whose keys receive zero attention weight.
        self.drop_pos: Optional[List[int]] = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        b = self.base
        B, L, _ = x.shape

        q = b.q_proj(x).reshape(B, L, b.num_heads, b.q_head_dim)
        q = q.transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [b.qk_nope_head_dim], axis=-1)

        compressed_kv = b.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [b.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, b.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = b.kv_a_layernorm(compressed_kv)
        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache is not None:
            kv_latent, k_pe = cache.update_and_fetch(kv_latent, k_pe)

        pe_scores = (q_pe * b.scale) @ k_pe.swapaxes(-1, -2)
        floor = mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype)
        if mask is not None:
            pe_scores = mx.where(mask, pe_scores, floor)

        if self.drop_pos:
            total = int(pe_scores.shape[-1])
            dropped = [p for p in self.drop_pos if 0 <= p < total]
            if dropped:
                positions = mx.arange(total)
                targets = mx.array(dropped, dtype=positions.dtype)
                hit = mx.any(positions[None, :] == targets[:, None], axis=0)
                pe_scores = mx.where(hit[None, None, None, :], floor, pe_scores)

        if L == 1:
            q_nope = b.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = b.embed_q(kv_latent, transpose=False)
            v = b.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q_nope, k, v, cache=cache, scale=b.scale, mask=pe_scores
        )

        if L == 1:
            output = b.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return b.o_proj(output)


def install_mla_mask(model: Any) -> List[int]:
    """Wrap every global MLA layer in place; returns the wrapped indices.

    Call after the weights are loaded: wrapping nests the base parameters under
    ``self_attn.base.*``, which no longer matches checkpoint key names.
    """
    _, mla = classify_layers(model)
    for i in mla:
        layer = model.layers[i]
        if not isinstance(layer.self_attn, MaskedMLAAttention):
            layer.self_attn = MaskedMLAAttention(layer.self_attn)
    return list(mla)


def set_mla_drop_positions(model: Any, positions: Optional[List[int]]) -> None:
    """Set (or clear, with ``None``) the evicted key positions on every wrapped layer."""
    for layer in model.layers:
        attn = getattr(layer, "self_attn", None)
        if isinstance(attn, MaskedMLAAttention):
            attn.drop_pos = None if positions is None else list(positions)

"""Experimental model-level reference weighting for MiniMax H3.

The public helpers are intentionally independent of ComfyUI's node schema so
their packed-layout mapping can be unit tested. Runtime patch installation is
performed lazily to keep this module importable in lightweight environments.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable

import torch

from .spans import WeightedSpan, reference_spans, target_start, weights_are_native


WEIGHT_EPSILON = math.exp(-20.0)
PATCH_KEY = "moc_minimax_h3_reference_weights"


def make_key_log_bias(
    sequence_length: int,
    spans: Iterable[WeightedSpan],
    *,
    strength: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a broadcast SDPA mask where log(weight) is a key prior."""
    strength = float(strength)
    if not math.isfinite(strength) or strength < 0.0:
        raise ValueError("weight strength must be finite and non-negative")
    bias = torch.zeros((1, 1, 1, int(sequence_length)), device=device, dtype=dtype)
    for span in spans:
        log_prior = math.log(max(span.weight, WEIGHT_EPSILON)) * strength
        bias[..., span.start:span.stop] = log_prior
    return bias


def apply_value_gates(
    values: torch.Tensor,
    spans: Iterable[WeightedSpan],
    strength: float,
    *,
    clone: bool = True,
) -> torch.Tensor:
    """Gate reference V slices, leaving every other token untouched.

    ``clone=True`` keeps this helper side-effect free for standalone callers.
    MiniMax H3 already gives the override a private V clone, so its hot path
    safely uses ``clone=False`` after the unmodified prefix pass has completed.
    """
    strength = float(strength)
    if not math.isfinite(strength) or strength < 0.0:
        raise ValueError("weight strength must be finite and non-negative")
    weighted = values.clone() if clone else values
    # Exponential interpolation stays non-negative: strength=0 is native,
    # strength=1 applies the requested gate, and >1 exaggerates it.
    for span in spans:
        effective = math.pow(span.weight, strength)
        weighted[..., span.start:span.stop, :].mul_(effective)
    return weighted


def _slice_mask_for_queries(mask: torch.Tensor | None, start: int, stop: int, full_q: int):
    if mask is None:
        return None
    if mask.ndim >= 2 and mask.shape[-2] == full_q:
        return mask[..., start:stop, :]
    # Key-only broadcast masks (query dimension 1) need no slicing.
    if mask.ndim >= 2 and mask.shape[-2] == 1:
        return mask
    raise ValueError("Experimental H3 weighting cannot safely slice this attention mask shape")


def _call_attention(
    previous_override: Callable | None,
    original_func: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: torch.Tensor | None,
    kwargs: dict[str, Any],
):
    if previous_override is not None:
        return previous_override(original_func, q, k, v, heads, mask=mask, **kwargs)
    return original_func(q, k, v, heads, mask=mask, **kwargs)


def make_value_gate_override(
    spans: list[WeightedSpan],
    first_target: int,
    strength: float,
    previous_override: Callable | None,
) -> Callable:
    """Make a backend-compatible override that only gates ref V for target queries."""

    def override(original_func, q, k, v, heads, mask=None, **kwargs):
        full_q = int(q.shape[-2])
        if not kwargs.get("skip_reshape", False) or kwargs.get("skip_output_reshape", False):
            return _call_attention(previous_override, original_func, q, k, v, heads, mask, kwargs)
        if first_target <= 0 or first_target >= full_q:
            return _call_attention(previous_override, original_func, q, k, v, heads, mask, kwargs)
        prefix_mask = _slice_mask_for_queries(mask, 0, first_target, full_q)
        target_mask = _slice_mask_for_queries(mask, first_target, full_q, full_q)
        prefix = _call_attention(
            previous_override,
            original_func,
            q[..., :first_target, :],
            k,
            v,
            heads,
            prefix_mask,
            kwargs,
        )
        weighted_v = apply_value_gates(v, spans, strength, clone=False)
        target = _call_attention(
            previous_override,
            original_func,
            q[..., first_target:, :],
            k,
            weighted_v,
            heads,
            target_mask,
            kwargs,
        )
        # H3 uses skip_reshape=True and receives [batch, query, hidden].
        concat_dim = 1 if prefix.ndim == 3 else -2
        return torch.cat((prefix, target), dim=concat_dim)

    return override


def make_attention_prior_override(
    spans: list[WeightedSpan],
    first_target: int,
    strength: float,
    previous_override: Callable | None,
    attention_impl: Callable | None = None,
) -> Callable:
    """Make a PyTorch-SDPA override with an exact multiplicative key prior."""
    if attention_impl is None:
        from comfy.ldm.modules.attention import attention_pytorch as attention_impl

    def override(original_func, q, k, v, heads, mask=None, **kwargs):
        full_q = int(q.shape[-2])
        if not kwargs.get("skip_reshape", False) or kwargs.get("skip_output_reshape", False):
            return _call_attention(previous_override, original_func, q, k, v, heads, mask, kwargs)
        if first_target <= 0 or first_target >= full_q:
            return _call_attention(previous_override, original_func, q, k, v, heads, mask, kwargs)
        prefix_mask = _slice_mask_for_queries(mask, 0, first_target, full_q)
        prefix = _call_attention(
            previous_override,
            original_func,
            q[..., :first_target, :],
            k,
            v,
            heads,
            prefix_mask,
            kwargs,
        )
        key_bias = make_key_log_bias(k.shape[-2], spans, strength=strength, device=q.device, dtype=q.dtype)
        target_mask = _slice_mask_for_queries(mask, first_target, full_q, full_q)
        if target_mask is not None:
            if target_mask.dtype == torch.bool:
                floor = -torch.finfo(q.dtype).max
                target_mask = torch.where(
                    target_mask,
                    torch.zeros((), device=q.device, dtype=q.dtype),
                    torch.full((), floor, device=q.device, dtype=q.dtype),
                )
            key_bias = key_bias + target_mask.to(device=q.device, dtype=q.dtype)
        # Preserve the wrapper guard so calling the decorated SDPA function does
        # not re-enter this override. Runtime installation rejects pre-existing
        # optimized-attention overrides before this callable is installed.
        target_kwargs = dict(kwargs)
        target_kwargs["_inside_attn_wrapper"] = True
        target = _call_attention(
            previous_override,
            attention_impl,
            q[..., first_target:, :],
            k,
            v,
            heads,
            key_bias,
            target_kwargs,
        )
        concat_dim = 1 if prefix.ndim == 3 else -2
        return torch.cat((prefix, target), dim=concat_dim)

    return override


def install_reference_weight_patch(model, *, method: str, strength: float):
    """Clone a ModelPatcher and install an H3-only diffusion wrapper."""
    if method not in ("value_gate", "attention_prior"):
        raise ValueError(f"Unsupported reference weight method: {method!r}")
    strength = float(strength)
    if not math.isfinite(strength) or strength < 0.0:
        raise ValueError("weight strength must be finite and non-negative")

    diffusion_model = model.get_model_object("diffusion_model")
    cls = diffusion_model.__class__
    if cls.__name__ != "MiniMaxH3Model" or "minimax" not in cls.__module__.lower():
        raise ValueError(
            "MOC H3 reference weighting requires a native MiniMax H3 diffusion model. "
            f"Received {cls.__module__}.{cls.__name__}."
        )

    import comfy.patcher_extension

    patched = model.clone()

    def diffusion_wrapper(executor, x, timestep, context, transformer_options=None, minimax_payload=None, **kwargs):
        options = dict(transformer_options or {})
        payload = minimax_payload or {}
        refs = payload.get("refs") or []
        layout = payload.get("layout")
        if not refs or layout is None:
            return executor(x, timestep, context, options, minimax_payload=minimax_payload, **kwargs)
        spans = reference_spans(layout.segments, refs)
        if not spans or weights_are_native(spans) or strength == 0.0:
            return executor(x, timestep, context, options, minimax_payload=minimax_payload, **kwargs)
        first_target = target_start(layout.segments)
        previous = options.get("optimized_attention_override")
        # Weighted H3 deliberately evaluates prefix and target queries in two
        # passes. Arbitrary overrides may consume containers, keep call state,
        # assume full self-attention, or expand the attention-prior mask.
        if previous is not None:
            raise RuntimeError(
                "MOC H3 weighting cannot safely compose with an existing optimized-attention override because weighted "
                "H3 splits prefix and target queries. Remove the other attention patch or set all reference weights to 1.0."
            )
        if method == "value_gate":
            options["optimized_attention_override"] = make_value_gate_override(spans, first_target, strength, previous)
        else:
            options["optimized_attention_override"] = make_attention_prior_override(spans, first_target, strength, previous)
        return executor(x, timestep, context, options, minimax_payload=minimax_payload, **kwargs)

    # Re-applying the node replaces its previous configuration rather than
    # stacking multiple gates on the same model clone.
    patched.remove_wrappers_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        PATCH_KEY,
    )
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        PATCH_KEY,
        diffusion_wrapper,
    )
    return patched

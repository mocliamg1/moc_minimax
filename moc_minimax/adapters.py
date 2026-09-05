"""Validated linear LoRA, LoKr and full-difference weight updates.

LoKr scaling follows ComfyUI's additive calculate_weight path: the last
reconstructed factor supplies alpha's rank; direct/direct factors use scale 1.
No model weights are needed. Architectural target names are never guessed.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import re

import torch

SUFFIXES = {
    '.lora_down.weight': 'a', '.lora_up.weight': 'b',
    '.lora_A.weight': 'a', '.lora_B.weight': 'b',
    '.lora_A.default.weight': 'a', '.lora_B.default.weight': 'b',
    '.lora_A': 'a', '.lora_B': 'b', '.alpha': 'alpha', '.diff': 'diff',
    **{'.' + key: key for key in ('lokr_w1', 'lokr_w2', 'lokr_w1_a',
                                 'lokr_w1_b', 'lokr_w2_a', 'lokr_w2_b')},
}
PREFIXES = ('base_model.model.', 'model.diffusion_model.', 'diffusion_model.')


def normalize_target(name):
    while True:
        prefix = next((p for p in PREFIXES if name.startswith(p)), None)
        if prefix is None:
            return name
        name = name[len(prefix):]


def block_name(name):
    match = re.search(r'(?:^|\.)(?:blocks|transformer_blocks|refiner_blocks)\.\d+(?=\.|$)', name)
    if match:
        return name[:match.end()]
    # Preserve Kohya target spelling, including its full family path.
    match = re.search(r'(?:^|_)(?:blocks|transformer_blocks|refiner_blocks)_\d+(?=_|$)', name)
    return name[:match.end()] if match else None


@dataclass
class Adapter:
    kind: str
    tensors: dict
    shape: tuple[int, int]
    scale: float
    alpha: float | None
    alpha_defaulted: bool
    alpha_ignored: bool
    export_target: str

    def signature(self):
        return self.kind, tuple((k, tuple(v.shape)) for k, v in sorted(self.tensors.items()))

    def describe(self):
        return dict(kind=self.kind, weight_shape=list(self.shape), scale=self.scale,
                    factor_shapes={k: list(v.shape) for k, v in sorted(self.tensors.items())},
                    alpha=self.alpha, alpha_defaulted=self.alpha_defaulted, alpha_ignored=self.alpha_ignored)

    def factors(self):
        t = {k: v.to(dtype=torch.float64) for k, v in self.tensors.items()}
        if self.kind == 'lora':
            return t['a'], t['b']
        if self.kind == 'lokr':
            return tuple(t[key] if key in t else t[key + '_a'] @ t[key + '_b']
                         for key in ('lokr_w1', 'lokr_w2'))
        return (t['diff'],)

    def rows(self, start, end, factors):
        if self.kind == 'lora':
            return (factors[1][start:end] @ factors[0]) * self.scale
        if self.kind == 'diff':
            return factors[0][start:end]
        w1, w2 = factors
        indices = torch.arange(start, end)
        # kron row ordering is [w1 output, w2 output, w1 input, w2 input].
        return (w1[indices // w2.shape[0], :, None] *
                w2[indices % w2.shape[0], None, :]).reshape(end - start, -1) * self.scale


def _build(name, raw_name, entry):
    alpha_tensor = entry.get('alpha')
    if alpha_tensor is not None and alpha_tensor.numel() != 1:
        raise ValueError('alpha must be scalar')
    alpha = float(alpha_tensor.item()) if alpha_tensor is not None else None
    t = {k: v for k, v in entry.items() if k != 'alpha'}
    if any(v.ndim != 2 or min(v.shape) <= 0 for v in t.values()):
        raise ValueError('only nonempty two-dimensional linear adapters are supported')
    export_target = raw_name.removeprefix('base_model.model.')
    if export_target.startswith('model.diffusion_model.'):
        export_target = export_target.removeprefix('model.')
    # Native bare block paths need the generic ComfyUI diffusion-model prefix.
    if export_target == name and re.match(r'^(blocks\.\d+\.|token_refiner\.|refiner_blocks\.)', name):
        export_target = 'diffusion_model.' + name
    if set(t) == {'a', 'b'}:
        a, b = t['a'], t['b']
        if a.shape[0] != b.shape[1]:
            raise ValueError('inconsistent A/B ranks')
        rank = a.shape[0]
        return Adapter('lora', t, (b.shape[0], a.shape[1]), (alpha / rank) if alpha is not None else 1.,
                       alpha if alpha is not None else float(rank), alpha is None, False, export_target)
    if set(t) == {'diff'} and alpha is None:
        return Adapter('diff', t, tuple(t['diff'].shape), 1., None, False, False, export_target)
    if t and all(k.startswith('lokr_') for k in t):
        shapes, rank = [], None
        for key in ('lokr_w1', 'lokr_w2'):
            direct, pa, pb = key in t, key + '_a' in t, key + '_b' in t
            if direct and (pa or pb):
                raise ValueError(f'ambiguous direct and decomposed {key}')
            if direct:
                shapes.append(t[key].shape)
            elif pa and pb:
                a, b = t[key + '_a'], t[key + '_b']
                if a.shape[1] != b.shape[0]:
                    raise ValueError(f'inconsistent {key} factor ranks')
                shapes.append((a.shape[0], b.shape[1]))
                rank = b.shape[0]
            else:
                raise ValueError(f'incomplete {key} pair')
        shape = tuple(shapes[0][i] * shapes[1][i] for i in (0, 1))
        scale = alpha / rank if alpha is not None and rank is not None else 1.
        return Adapter('lokr', t, shape, scale, alpha, alpha is None,
                       rank is None and alpha is not None, export_target)
    raise ValueError('incomplete, mixed, or unsupported adapter entries')


def parse_adapters(payload, side, diagnostics):
    state = payload.get('tensors') if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping) or not state:
        diagnostics.append(f'{side}: expected a nonempty loaded adapter tensor dictionary.')
        return {}
    pairs, raw_names = {}, {}
    for key, value in sorted(state.items(), key=lambda item: str(item[0])):
        suffix = next((s for s in SUFFIXES if isinstance(key, str) and key.endswith(s)), None)
        if suffix is None:
            diagnostics.append(f'{side}: unsupported tensor or adapter entry: {key} (DoRA/Tucker/convolutional variants are not supported).')
            continue
        raw_name, role = key[:-len(suffix)], SUFFIXES[suffix]
        name = normalize_target(raw_name)
        entry = pairs.setdefault(name, {})
        if not name or role in entry or (name in raw_names and raw_names[name] != raw_name):
            diagnostics.append(f'{side}: ambiguous target mapping: {key}.')
            continue
        raw_names[name] = raw_name
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided or
                value.is_complex() or value.is_quantized or value.device.type == 'meta'):
            diagnostics.append(f'{side}: {key} must be a real, dense, unquantized tensor.')
            continue
        if not torch.isfinite(value).all().item():
            diagnostics.append(f'{side}: nonfinite values in {key}.')
            continue
        entry[role] = value.detach().cpu()
    valid = {}
    for name, entry in sorted(pairs.items()):
        try:
            valid[name] = _build(name, raw_names.get(name, name), entry)
        except ValueError as error:
            diagnostics.append(f'{side}: {name}: {error}.')
    return valid


def row_chunks(shape, max_elements=1024 * 1024):
    step = max(1, max_elements // shape[1])
    for start in range(0, shape[0], step):
        yield start, min(start + step, shape[0])


def adapter_stats(left, right):
    lf, rf = left.factors(), right.factors()
    if left.kind == right.kind == 'lora':
        a, b = lf
        c, d = rf
        xx = float(((b.T @ b) * (a @ a.T)).sum()) * left.scale ** 2
        yy = float(((d.T @ d) * (c @ c.T)).sum()) * right.scale ** 2
        xy = float(((b.T @ d) * (a @ c.T)).sum()) * left.scale * right.scale
    elif left.kind == right.kind == 'lokr' and all(x.shape == y.shape for x, y in zip(lf, rf)):
        a, b = lf
        c, d = rf
        xx = float((a*a).sum()) * float((b*b).sum()) * left.scale ** 2
        yy = float((c*c).sum()) * float((d*d).sum()) * right.scale ** 2
        xy = float((a*c).sum()) * float((b*d).sum()) * left.scale * right.scale
    else:
        stats = []
        for start, end in row_chunks(left.shape):
            x, y = left.rows(start, end, lf), right.rows(start, end, rf)
            stats.append((float((x*x).sum()), float((y*y).sum()), float((x*y).sum())))
        xx, yy, xy = (math.fsum(s[i] for s in stats) for i in range(3))
    if not all(math.isfinite(v) for v in (xx, yy, xy)):
        raise ValueError('Effective-weight calculation overflowed float64.')
    return max(0., xx), max(0., yy), xy

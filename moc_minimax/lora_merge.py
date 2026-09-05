"""Layer-at-a-time weighted adapter merging and streaming safetensors export."""
from __future__ import annotations

from collections import defaultdict
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile

import torch

from .adapters import block_name, parse_adapters, row_chunks

FORMATS = ('full_diff', 'lokr_shared_or_diff', 'lora_svd')
DTYPES = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}


def _checked_cast(tensor, dtype):
    tensor = tensor.to(dtype=dtype).contiguous()
    if not torch.isfinite(tensor).all().item():
        raise ValueError('Merged weights overflow the selected storage dtype; choose float32 or lower the strengths.')
    return tensor


def _shared_lokr(contributors):
    if not all(adapter.kind == 'lokr' for adapter, _ in contributors):
        return None
    factors = [adapter.factors() for adapter, _ in contributors]
    for shared in (0, 1):
        if not all(pair[shared].shape == factors[0][shared].shape and
                   torch.equal(pair[shared], factors[0][shared]) and
                   pair[1-shared].shape == factors[0][1-shared].shape for pair in factors):
            continue
        combined = torch.zeros_like(factors[0][1-shared])
        for (adapter, strength), pair in zip(contributors, factors):
            combined.add_(pair[1-shared], alpha=adapter.scale * strength)
        output = [None, None]
        output[shared], output[1-shared] = factors[0][shared], combined
        return output
    return None


def _error_summary(norm2, error2):
    norm, error = math.sqrt(norm2), math.sqrt(error2)
    return dict(update_norm=norm, error_norm=error,
                relative_error=error / norm if norm else (0. if error == 0 else None))


def merge_adapters(payloads, strengths, output_format='full_diff', rank=64,
                   storage_dtype='float32', tensor_sink=None):
    """Return (payload, report); optional sink streams tensors instead of retaining them.

    Strengths are never normalized. Missing targets contribute zero. Validation
    finishes before any tensor is emitted; failures must discard the sink's file.
    """
    if output_format not in FORMATS or storage_dtype not in DTYPES:
        raise ValueError('Unknown merge format or storage dtype.')
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
        raise ValueError('SVD rank must be a positive integer.')
    if not payloads or len(payloads) != len(strengths):
        raise ValueError('Each input adapter needs one strength.')
    strengths = [float(s) for s in strengths]
    if not all(math.isfinite(s) for s in strengths):
        raise ValueError('Strengths must be finite.')
    if not any(strengths):
        raise ValueError('At least one adapter strength must be nonzero.')
    diagnostics, targets = [], defaultdict(list)
    for i, (payload, strength) in enumerate(zip(payloads, strengths)):
        if strength == 0:
            continue
        for name, adapter in parse_adapters(payload, f'Input {i+1}', diagnostics).items():
            targets[name].append((adapter, strength))
    export_names, flattened_names = {}, {}
    for name, contributors in targets.items():
        if len({a.shape for a, _ in contributors}) > 1:
            diagnostics.append(f'Weight dimension mismatch for {name}: {[a.shape for a, _ in contributors]}.')
        export = contributors[0][0].export_target
        if export in export_names and export_names[export] != name:
            diagnostics.append(f'Ambiguous exported target {export}.')
        export_names[export] = name
        flat = name.removeprefix('lora_unet_').removeprefix('lycoris_').replace('.', '_')
        if flat in flattened_names and flattened_names[flat] != name:
            diagnostics.append(f'Potential duplicate layer aliases: {flattened_names[flat]} and {name}. Use consistent target naming across inputs.')
        flattened_names[flat] = name
    if diagnostics:
        raise ValueError('Cannot merge adapters:\n' + '\n'.join(diagnostics))
    if not targets:
        raise ValueError('No supported target layers to merge.')

    tensors = {}
    emit = tensor_sink if tensor_sink is not None else tensors.__setitem__
    report = dict(schema_version=1, output_format=output_format, storage_dtype=storage_dtype,
                  requested_rank=rank if output_format == 'lora_svd' else None,
                  sources=[dict(source=p.get('source', '<unknown>'), strength=s)
                           for p, s in zip(payloads, strengths)],
                  normalized_strengths=False, apply_strength=1.0,
                  note='Weighted sum of additive updates; storage/compression error does not predict visual quality.',
                  modules=[], blocks=[], overall=None)
    block_totals = defaultdict(lambda: [0., 0.])
    overall = [0., 0.]
    dtype = DTYPES[storage_dtype]
    for name, contributors in sorted(targets.items()):
        adapter = contributors[0][0]
        prefix, shape = adapter.export_target, adapter.shape
        shared = _shared_lokr(contributors) if output_format == 'lokr_shared_or_diff' else None
        # Reconstructed factors live only for the current layer.
        prepared = [(a, strength, a.factors()) for a, strength in contributors]
        if shared is not None:
            stored = {prefix + '.lokr_w1': _checked_cast(shared[0], dtype),
                      prefix + '.lokr_w2': _checked_cast(shared[1], dtype)}
            method = 'shared_lokr'
            target = None
        else:
            target = torch.empty(shape, dtype=torch.float64)
            for start, end in row_chunks(shape):
                target[start:end].zero_()
                for a, strength, factors in prepared:
                    target[start:end].add_(a.rows(start, end, factors), alpha=strength)
            if not torch.isfinite(target).all().item():
                raise ValueError(f'{name}: merged update overflowed float64.')
            if output_format == 'lora_svd':
                u, singular, vh = torch.linalg.svd(target, full_matrices=False)
                kept = min(rank, min(shape))
                stored = {prefix + '.lora_up.weight': _checked_cast(u[:, :kept] * singular[:kept], dtype),
                          prefix + '.lora_down.weight': _checked_cast(vh[:kept], dtype)}
                # Omit alpha: the native default is scale 1, with singular values in up.
                del u, singular, vh
                method = 'lora_svd'
            else:
                stored = {prefix + '.diff': _checked_cast(target, dtype)}
                method = 'full_diff'
        parsed = parse_adapters({'tensors': stored}, 'Output', [])
        reconstructed = next(iter(parsed.values()))
        out_factors = reconstructed.factors()
        norms, errors = [], []
        for start, end in row_chunks(shape):
            if target is not None:
                expected = target[start:end]
            else:
                expected = torch.zeros((end-start, shape[1]), dtype=torch.float64)
                for a, strength, factors in prepared:
                    expected.add_(a.rows(start, end, factors), alpha=strength)
            actual = reconstructed.rows(start, end, out_factors)
            norms.append(float((expected * expected).sum()))
            errors.append(float(((expected - actual) ** 2).sum()))
        norm2, error2 = math.fsum(norms), math.fsum(errors)
        if not math.isfinite(norm2 + error2):
            raise ValueError(f'{name}: error measurement overflowed float64.')
        block = block_name(name)
        module = dict(target=name, export_target=prefix, block=block, method=method,
                      weight_shape=list(shape), contributors=len(contributors),
                      output_rank=min(rank, min(shape)) if method == 'lora_svd' else None,
                      inputs=[dict(strength=s, **a.describe()) for a, s in contributors],
                      **_error_summary(norm2, error2))
        report['modules'].append(module)
        group = block_totals[block]
        group[0] += norm2
        group[1] += error2
        overall[0] += norm2
        overall[1] += error2
        for key, tensor in stored.items():
            emit(key, tensor)
        # Release the full layer before constructing the next one.
        del target, prepared, out_factors, parsed, reconstructed, stored, expected, actual, tensor, factors
    if not all(math.isfinite(v) for v in overall):
        raise ValueError('Aggregate error measurement overflowed float64.')
    report['blocks'] = [dict(block=block, **_error_summary(*values))
                        for block, values in sorted(block_totals.items(), key=lambda kv: kv[0] or '')]
    report['overall'] = _error_summary(*overall)
    return dict(tensors=tensors, source='merged'), report


def render_merge(report):
    def error_text(value):
        return f'{value:.8g}' if value is not None else 'undefined (zero reference norm)'
    lines = ['MiniMax H3 adapter merge', report['note'],
             f"Format: {report['output_format']}; storage: {report['storage_dtype']}; apply at strength 1.0",
             'Input strengths are summed without normalization.']
    lines.extend(f"  {s['source']} × {s['strength']}" for s in report['sources'])
    lines.append(f"Overall relative error: {error_text(report['overall']['relative_error'])}")
    for block in report['blocks']:
        lines.append(f"{block['block'] or 'Non-block modules'}: relative error={error_text(block['relative_error'])}")
    for module in report['modules']:
        lines.append(f"  {module['target']}: {module['method']}; rank={module['output_rank']}; relative error={error_text(module['relative_error'])}")
    return '\n'.join(lines)


class SafetensorsStream:
    """Spool tensor bytes to disk, then publish one file without overwriting.

    Uses the documented safetensors header + byte buffer format. Keeping tensor
    data on disk bounds export memory to the current layer, even for full deltas.
    """
    def __init__(self, directory):
        self.directory = Path(directory)
        self.data = tempfile.TemporaryFile(dir=directory)
        self.header = {}
        self.offset = 0

    def add(self, key, tensor):
        if key in self.header:
            raise ValueError(f'Duplicate output key: {key}')
        dtype = {torch.float32: 'F32', torch.float16: 'F16', torch.bfloat16: 'BF16'}[tensor.dtype]
        raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
        size = tensor.numel() * tensor.element_size()
        self.header[key] = dict(dtype=dtype, shape=list(tensor.shape), data_offsets=[self.offset, self.offset+size])
        self.data.write(memoryview(raw).cast('B'))
        self.offset += size

    def finish(self, path, report):
        header = dict(self.header)
        header['__metadata__'] = {'moc_merge': json.dumps(report, allow_nan=False)}
        encoded = json.dumps(header, separators=(',', ':'), allow_nan=False).encode('utf-8')
        encoded += b' ' * (-len(encoded) % 8)
        staging = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, delete=False) as output:
                staging = output.name
                output.write(struct.pack('<Q', len(encoded)))
                output.write(encoded)
                self.data.seek(0)
                shutil.copyfileobj(self.data, output, length=8 * 1024 * 1024)
            # Atomic publication, with a hard failure if another file exists.
            os.link(staging, path)
        finally:
            if staging is not None:
                os.unlink(staging)

    def close(self):
        self.data.close()

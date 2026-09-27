"""Layer-at-a-time weighted adapter merging and streaming safetensors export."""
from __future__ import annotations

from collections import defaultdict
import errno
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile

import torch

from .adapters import block_name, parse_adapters, row_chunks

FORMATS = ('auto', 'full_diff', 'lokr_shared_or_diff', 'lora_svd')
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


def _truncated_svd(target, rank, oversample=16, power_iterations=4):
    """Return (u, s, vh) for the leading singular triplets.

    Full CPU SVD of a large layer takes seconds; a seeded randomized range
    finder with power iterations matches its truncation error far faster.
    """
    kept = min(rank, min(target.shape))
    width = kept + oversample
    if 2 * width >= min(target.shape):
        u, singular, vh = torch.linalg.svd(target, full_matrices=False)
        return u[:, :kept], singular[:kept], vh[:kept]
    generator = torch.Generator().manual_seed(0)
    probe = torch.randn(target.shape[1], width, dtype=target.dtype, generator=generator)
    basis = torch.linalg.qr(target @ probe).Q
    for _ in range(power_iterations):
        basis = torch.linalg.qr(target.T @ basis).Q
        basis = torch.linalg.qr(target @ basis).Q
    u, singular, vh = torch.linalg.svd(basis.T @ target, full_matrices=False)
    return (basis @ u[:, :kept]), singular[:kept], vh[:kept]


def _error_summary(norm2, error2):
    norm, error = math.sqrt(norm2), math.sqrt(error2)
    return dict(update_norm=norm, error_norm=error,
                relative_error=error / norm if norm else (0. if error == 0 else None))


def _output_plan(contributors, output_format, rank):
    """Choose an exact representation and count its elements before writing."""
    shape = contributors[0][0].shape
    dense_size = math.prod(shape)
    if output_format == 'lora_svd':
        return 'lora_svd', min(rank, min(shape)) * sum(shape)
    if output_format == 'auto' and all(a.kind == 'lora' for a, _ in contributors):
        size = sum(a.tensors['a'].shape[0] for a, _ in contributors) * sum(shape)
        if size < dense_size:
            return 'lora_concat', size
    if output_format in ('auto', 'lokr_shared_or_diff'):
        shared = _shared_lokr(contributors)
        if shared is not None:
            size = sum(t.numel() for t in shared)
            if output_format != 'auto' or size < dense_size:
                return 'shared_lokr', size
    return 'full_diff', dense_size


def merge_adapters(payloads, strengths, output_format='auto', rank=64,
                   storage_dtype='float32', tensor_sink=None, tensor_preflight=None):
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

    plans = {name: _output_plan(contributors, output_format, rank)
             for name, contributors in targets.items()}
    data_bytes = sum(size for _, size in plans.values()) * torch.empty((), dtype=DTYPES[storage_dtype]).element_size()
    if tensor_preflight is not None:
        # Generous metadata estimate; outgrowing the reserve would shift all tensor bytes.
        header_bytes = 4096 * len(payloads) + sum(2048 + 1024 * len(c) for c in targets.values())
        tensor_preflight(data_bytes, header_bytes)

    tensors = {}
    emit = tensor_sink if tensor_sink is not None else tensors.__setitem__
    report = dict(schema_version=1, output_format=output_format, storage_dtype=storage_dtype,
                  requested_rank=rank if output_format == 'lora_svd' else None,
                  sources=[dict(source=p.get('source', '<unknown>'), strength=s)
                           for p, s in zip(payloads, strengths)],
                  normalized_strengths=False, apply_strength=1.0,
                  tensor_data_bytes=data_bytes,
                  note='Weighted sum of additive updates; storage/compression error does not predict visual quality.',
                  modules=[], blocks=[], overall=None)
    block_totals = defaultdict(lambda: [0., 0.])
    overall = [0., 0.]
    dtype = DTYPES[storage_dtype]
    for name, contributors in sorted(targets.items()):
        adapter = contributors[0][0]
        prefix, shape = adapter.export_target, adapter.shape
        method = plans[name][0]
        # Reconstructed factors live only for the current layer.
        prepared = [(a, strength, a.factors()) for a, strength in contributors]
        if method == 'shared_lokr':
            shared = _shared_lokr(contributors)
            stored = {prefix + '.lokr_w1': _checked_cast(shared[0], dtype),
                      prefix + '.lokr_w2': _checked_cast(shared[1], dtype)}
            del shared
            target = None
        elif method == 'lora_concat':
            # [s1*B1, s2*B2, ...] @ [A1; A2; ...] is the exact sum.
            stored = {
                prefix + '.lora_down.weight': _checked_cast(torch.cat([f[0] for _, _, f in prepared], dim=0), dtype),
                prefix + '.lora_up.weight': _checked_cast(torch.cat([f[1] * (a.scale * s) for a, s, f in prepared], dim=1), dtype),
            }
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
                u, singular, vh = _truncated_svd(target, rank)
                stored = {prefix + '.lora_up.weight': _checked_cast(u * singular, dtype),
                          prefix + '.lora_down.weight': _checked_cast(vh, dtype)}
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
                      output_rank=stored[prefix + '.lora_down.weight'].shape[0] if method in ('lora_svd', 'lora_concat') else None,
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
    """Write a single staging file, then publish it without copying tensor data.

    Reserve header space ahead of the byte buffer. Safetensors permits trailing
    whitespace in its header. Oversized metadata grows the same file in place.
    """
    HEADER_RESERVE = 1024 * 1024

    def __init__(self, directory):
        self.directory = Path(directory)
        self.data = tempfile.NamedTemporaryFile(prefix='.moc-merge-', suffix='.tmp', dir=directory, delete=False)
        self.header_size = self.HEADER_RESERVE
        self.data.seek(8 + self.header_size)
        self.header = {}
        self.offset = 0
        self.published = False

    def preflight(self, data_bytes, header_bytes=0):
        if self.header or self.published:
            raise RuntimeError('Preflight must run before tensors are written.')
        self.header_size = max(self.HEADER_RESERVE, (header_bytes + 7) // 8 * 8)
        self.data.seek(8 + self.header_size)
        self._require_space(8 + self.header_size + data_bytes)

    def _require_space(self, required):
        free = shutil.disk_usage(self.directory).free
        if required > free:
            raise OSError(errno.ENOSPC,
                          f'Merge needs {required / 2**30:.2f} GiB of additional disk space; '
                          f'only {free / 2**30:.2f} GiB is free. '
                          'Use auto to preserve compact factors, lora_svd for a smaller approximate merge, '
                          'float16/bfloat16 storage, or free space in the output directory.')

    def add(self, key, tensor):
        if self.published:
            raise RuntimeError('This merge has already been published.')
        if key in self.header:
            raise ValueError(f'Duplicate output key: {key}')
        dtype = {torch.float32: 'F32', torch.float16: 'F16', torch.bfloat16: 'BF16'}[tensor.dtype]
        raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
        size = tensor.numel() * tensor.element_size()
        self.header[key] = dict(dtype=dtype, shape=list(tensor.shape), data_offsets=[self.offset, self.offset+size])
        self.data.write(memoryview(raw).cast('B'))
        self.offset += size

    def finish(self, path, report):
        if Path(path).exists():
            raise FileExistsError(errno.EEXIST, 'Output file already exists', str(path))
        if self.published:
            raise RuntimeError('This merge has already been published.')
        header = dict(self.header)
        header['__metadata__'] = {'moc_merge': json.dumps(report, allow_nan=False)}
        encoded = json.dumps(header, separators=(',', ':'), allow_nan=False).encode('utf-8')
        needed = (len(encoded) + 7) // 8 * 8
        if needed > self.header_size:
            self._require_space(needed - self.header_size)
            # Move backwards to safely handle overlapping source/destination.
            end = self.offset
            while end:
                start = max(0, end - 8 * 1024 * 1024)
                self.data.seek(8 + self.header_size + start)
                chunk = self.data.read(end - start)
                self.data.seek(8 + needed + start)
                self.data.write(chunk)
                end = start
            self.header_size = needed
        self.data.seek(0)
        self.data.write(struct.pack('<Q', self.header_size))
        self.data.write(encoded)
        self.data.write(b' ' * (self.header_size - len(encoded)))
        self.data.flush()
        # Both names point to the same bytes; no second full-size file is made.
        os.link(self.data.name, path)
        self.published = True

    def close(self):
        try:
            self.data.close()
        finally:
            Path(self.data.name).unlink(missing_ok=True)

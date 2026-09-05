"""CPU-only comparison of linear LoRA, LoKr and full-difference updates."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping

from .adapters import adapter_stats as _stats, block_name as _block, parse_adapters as _parse

ATOL = 1e-8
RTOL = 1e-5

def _metrics(stats):
    xx, yy, xy = stats
    nx, ny = math.sqrt(xx), math.sqrt(yy)
    difference = math.sqrt(max(0.0, xx + yy - 2 * xy))
    score = 100.0 if nx + ny == 0 else 100 * (1 - (difference / (nx + ny)) ** 2)
    cosine = max(-1.0, min(1.0, xy / nx / ny)) if nx and ny else None
    return dict(score=max(0.0, min(100.0, score)), cosine_similarity=cosine,
                norm_a=nx, norm_b=ny, difference_norm=difference)


def compare_loras(lora_a, lora_b, matching="strict"):
    if matching not in ("strict", "effective_dimensions"):
        raise ValueError(f"Unknown comparison matching mode: {matching}")
    diagnostics = []
    result = dict(schema_version=1,
                  sources={s: p.get('source', '<unknown>') if isinstance(p, Mapping) else '<unknown>'
                           for s, p in [('a', lora_a), ('b', lora_b)]},
                  compatible=False, score=None, overall=None, matching=matching,
                  metric='100 * (1 - ||X-Y||^2 / (||X||+||Y||)^2)',
                  interpretation='Weight-update similarity at strength 1; does not predict visual similarity.',
                  tolerances=dict(atol=ATOL, rtol=RTOL), blocks=[], non_block_modules=[],
                  diagnostics=diagnostics)
    left, right = _parse(lora_a, 'A', diagnostics), _parse(lora_b, 'B', diagnostics)
    for name in sorted(left.keys() - right.keys()):
        diagnostics.append(f'B: missing target {name}.')
    for name in sorted(right.keys() - left.keys()):
        diagnostics.append(f'A: missing target {name}.')
    for name in sorted(left.keys() & right.keys()):
        if left[name].shape != right[name].shape:
            diagnostics.append(f'Weight dimension mismatch for {name}: A={left[name].shape}, B={right[name].shape}.')
        elif matching == 'strict' and left[name].signature() != right[name].signature():
            diagnostics.append(f'Shape/rank/adapter format mismatch for {name}; use effective_dimensions to compare different representations.')
    if diagnostics:
        return result
    groups, totals = {}, []
    try:
        for name in sorted(left):
            stats = _stats(left[name], right[name])
            metrics = _metrics(stats)
            entry = dict(target=name, **metrics,
                         differs=metrics['difference_norm'] > ATOL + RTOL * max(metrics['norm_a'], metrics['norm_b']),
                         factor_shapes=list(left[name].describe()['factor_shapes'].values()),
                         alpha_a=left[name].alpha, alpha_b=right[name].alpha,
                         alpha_defaulted_a=left[name].alpha_defaulted, alpha_defaulted_b=right[name].alpha_defaulted,
                         adapter_a=left[name].describe(), adapter_b=right[name].describe())
            totals.append(stats)
            block = _block(name)
            if block is None:
                result['non_block_modules'].append(entry)
            else:
                group = groups.setdefault(block, ([], []))
                group[0].append(entry)
                group[1].append(stats)
        aggregate = lambda values: tuple(math.fsum(v[i] for v in values) for i in range(3))
        result['overall'] = _metrics(aggregate(totals))
        for name, (modules, values) in groups.items():
            result['blocks'].append(dict(block=name, **_metrics(aggregate(values)),
                                         differs=any(m['differs'] for m in modules), modules=modules))
    except (ValueError, OverflowError) as error:
        diagnostics.append(str(error))
        result.update(overall=None, blocks=[], non_block_modules=[])
        return result
    result['blocks'].sort(key=lambda b: (not b['differs'], b['score'], b['block']))
    result['non_block_modules'].sort(key=lambda m: (not m['differs'], m['score'], m['target']))
    result.update(compatible=True, score=result['overall']['score'])
    return result


def render_comparison(result):
    lines = ['MiniMax H3 LoRA comparison', f"A: {result['sources']['a']}", f"B: {result['sources']['b']}", result['interpretation']]
    if not result['compatible']:
        return '\n'.join([*lines, 'Incompatible — no similarity score.', *result['diagnostics']])
    overall = result['overall']
    lines.extend([f"Matching: {result['matching']}", f"Similarity: {overall['score']:.6f}/100", f"Cosine similarity: {overall['cosine_similarity']}",
                  f"Norms A/B: {overall['norm_a']:.8g} / {overall['norm_b']:.8g}; difference norm: {overall['difference_norm']:.8g}",
                  f"Difference tolerance: {ATOL} + {RTOL} × max(norm A, norm B)",
                  f"Metric: {result['metric']}",
                  f"Differing blocks: {sum(b['differs'] for b in result['blocks'])}/{len(result['blocks'])}"])
    def module_line(m):
        defaults = ', '.join(s for s in ('a', 'b') if m[f'alpha_defaulted_{s}']) or 'none'
        return (f"  {'DIFF' if m['differs'] else 'same'} {m['target']}: {m['score']:.6f}/100; "
                f"cosine={m['cosine_similarity']}; difference={m['difference_norm']:.8g}; "
                f"norms={m['norm_a']:.8g}/{m['norm_b']:.8g}; shapes={m['factor_shapes']}; "
                f"alpha A/B={m['alpha_a']}/{m['alpha_b']} (defaulted: {defaults}); A={m['adapter_a']}; B={m['adapter_b']}")
    for block in result['blocks']:
        lines.append(f"{'DIFF' if block['differs'] else 'same'} {block['block']}: {block['score']:.6f}/100; cosine={block['cosine_similarity']}; difference={block['difference_norm']:.8g}")
        lines.extend(module_line(m) for m in block['modules'])
    if result['non_block_modules']:
        lines.append('Non-block modules:')
        lines.extend(module_line(m) for m in result['non_block_modules'])
    return '\n'.join(lines)


def comparison_json(result):
    return json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)

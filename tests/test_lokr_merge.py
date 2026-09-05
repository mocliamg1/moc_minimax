from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

AVAILABLE = importlib.util.find_spec('torch') is not None
if AVAILABLE:
    import torch
    from moc_minimax.adapters import parse_adapters, adapter_stats
    from moc_minimax.lora_compare import compare_loras
    from moc_minimax.lora_merge import merge_adapters, render_merge, SafetensorsStream


def lokr(w1, w2, name='diffusion_model.blocks.0.attn.q', alpha=None):
    state = {name + '.lokr_w1': w1, name + '.lokr_w2': w2}
    if alpha is not None:
        state[name + '.alpha'] = torch.tensor(alpha, dtype=torch.float64)
    return dict(tensors=state, source='synthetic LoKr')


def parsed(payload):
    errors = []
    result = parse_adapters(payload, 'Test', errors)
    if errors:
        raise AssertionError(errors)
    return result


def dense(adapter):
    return adapter.rows(0, adapter.shape[0], adapter.factors())


@unittest.skipUnless(AVAILABLE, 'torch is required')
class LoKrTests(unittest.TestCase):
    def setUp(self):
        self.w1 = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
        self.w2 = torch.tensor([[2., -1., 3.], [0., 1., 2.]], dtype=torch.float64)
        self.payload = lokr(self.w1, self.w2)

    def test_direct_kron_and_ignored_alpha(self):
        for alpha in (None, 0., 100.):
            value = next(iter(parsed(lokr(self.w1, self.w2, alpha=alpha)).values()))
            torch.testing.assert_close(dense(value), torch.kron(self.w1, self.w2))
            self.assertEqual(value.alpha_ignored, alpha is not None)

    def test_decomposed_factors_native_scaling(self):
        prefix = 'diffusion_model.blocks.0.attn.q.'
        eye = torch.eye(2, dtype=torch.float64)
        for decompose_first, decompose_second in ((True, False), (False, True), (True, True)):
            state = {prefix+'alpha': torch.tensor(3.)}
            for index, (value, decomposed) in enumerate([(self.w1, decompose_first), (self.w2, decompose_second)], 1):
                if decomposed:
                    state[prefix+f'lokr_w{index}_a'] = eye
                    state[prefix+f'lokr_w{index}_b'] = value
                else:
                    state[prefix+f'lokr_w{index}'] = value
            actual = next(iter(parsed({'tensors': state}).values()))
            torch.testing.assert_close(dense(actual), torch.kron(self.w1, self.w2) * 1.5)
        # If both are decomposed with distinct ranks, native calculate_weight uses w2's rank.
        state = {prefix+'alpha': torch.tensor(6.), prefix+'lokr_w1_a': self.w1[:, :1],
                 prefix+'lokr_w1_b': self.w1[:1], prefix+'lokr_w2_a': eye, prefix+'lokr_w2_b': self.w2}
        actual = next(iter(parsed({'tensors': state}).values()))
        expected = torch.kron(self.w1[:, :1] @ self.w1[:1], self.w2) * 3
        torch.testing.assert_close(dense(actual), expected)
        del state[prefix+'alpha']
        torch.testing.assert_close(dense(next(iter(parsed({'tensors': state}).values()))), expected/3)

    def test_comparison_reparameterization_and_strength(self):
        equivalent = lokr(self.w1 * 2, self.w2 / 2)
        result = compare_loras(self.payload, equivalent)
        self.assertTrue(result['compatible'])
        self.assertAlmostEqual(result['score'], 100)
        scaled = compare_loras(self.payload, lokr(self.w1 * 2, self.w2))
        self.assertAlmostEqual(scaled['score'], 100 * 8 / 9)
        self.assertTrue(scaled['blocks'][0]['differs'])
        a, b = next(iter(parsed(self.payload).values())), next(iter(parsed(equivalent).values()))
        x, y = dense(a), dense(b)
        for actual, expected in zip(adapter_stats(a, b), [(x*x).sum(), (y*y).sum(), (x*y).sum()]):
            self.assertAlmostEqual(actual, float(expected))

    def test_cross_format_and_partition_comparison(self):
        matrix = torch.kron(self.w1, self.w2)
        full = {'tensors': {'diffusion_model.blocks.0.attn.q.diff': matrix}}
        other_partition = lokr(torch.ones(1, 1), matrix)
        for candidate in (full, other_partition):
            self.assertFalse(compare_loras(self.payload, candidate)['compatible'])
            result = compare_loras(self.payload, candidate, 'effective_dimensions')
            self.assertTrue(result['compatible'])
            self.assertAlmostEqual(result['score'], 100)

    def test_reject_malformed_or_nonlinear(self):
        cases = []
        for key, value in [('lokr_w1_a', self.w1), ('lokr_t2', torch.ones(1,1,1,1)),
                           ('dora_scale', self.w1), ('alpha', torch.ones(2))]:
            p = lokr(self.w1, self.w2)
            p['tensors']['diffusion_model.blocks.0.attn.q.' + key] = value
            cases.append(p)
        cases += [lokr(self.w1, self.w2 * float('inf')), lokr(self.w1, self.w2.unsqueeze(0))]
        cases.append({'tensors': {'blocks.0.q.lokr_w1_a': self.w1}})
        for candidate in cases:
            self.assertFalse(compare_loras(candidate, candidate)['compatible'])
            with self.assertRaises(ValueError):
                merge_adapters([candidate], [1.])

    def test_kohya_and_refiner_block_attribution(self):
        p = lokr(self.w1, self.w2, 'lora_unet_token_refiner_refiner_blocks_0_attn_q')
        report = compare_loras(p, p)
        self.assertEqual(report['blocks'][0]['block'], 'lora_unet_token_refiner_refiner_blocks_0')


@unittest.skipUnless(AVAILABLE, 'torch is required')
class MergeTests(unittest.TestCase):
    def setUp(self):
        self.w1 = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
        self.w2 = torch.tensor([[2., -1., 3.], [0., 1., 2.]], dtype=torch.float64)
        self.p = lokr(self.w1, self.w2)
        self.q = lokr(self.w1 + 1, self.w2 - 1)

    def test_exact_weighted_sum_and_unique_layers(self):
        self.q['tensors'].update(lokr(self.w1, self.w2, 'diffusion_model.blocks.1.attn.q')['tensors'])
        output, report = merge_adapters([self.p, self.q], [.7, -.25])
        layers = parsed(output)
        expected = .7 * torch.kron(self.w1, self.w2) - .25 * torch.kron(self.w1+1, self.w2-1)
        torch.testing.assert_close(dense(layers['blocks.0.attn.q']), expected, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(dense(layers['blocks.1.attn.q']), -.25 * torch.kron(self.w1, self.w2))
        self.assertFalse(report['normalized_strengths'])
        self.assertEqual(report['sources'][0]['strength'], .7)
        self.assertLess(report['overall']['relative_error'], 1e-6)
        self.assertIn('without normalization', render_merge(report))
        json.dumps(report, allow_nan=False)

    def test_shared_factor_shortcut_both_sides_and_fallback(self):
        for q in (lokr(self.w1, self.w2 + 1), lokr(self.w1 + 1, self.w2)):
            output, report = merge_adapters([self.p, q], [.5, -.25], 'lokr_shared_or_diff')
            self.assertEqual(report['modules'][0]['method'], 'shared_lokr')
            merged = next(iter(parsed(output).values()))
            expected = .5 * dense(next(iter(parsed(self.p).values()))) - .25 * dense(next(iter(parsed(q).values())))
            torch.testing.assert_close(dense(merged), expected)
        output, report = merge_adapters([self.p, self.q], [1., 1.], 'lokr_shared_or_diff')
        self.assertEqual(report['modules'][0]['method'], 'full_diff')
        self.assertTrue(next(iter(output['tensors'])).endswith('.diff'))

    def test_svd_rank_and_measured_error(self):
        exact, _ = merge_adapters([self.p, self.q], [1., .5])
        reference = dense(next(iter(parsed(exact).values())))
        previous_error = float('inf')
        for rank in (1, 2, 4):
            output, report = merge_adapters([self.p, self.q], [1., .5], 'lora_svd', rank)
            actual = dense(next(iter(parsed(output).values())))
            relative_error = float(torch.linalg.vector_norm(actual-reference) / torch.linalg.vector_norm(reference))
            self.assertAlmostEqual(report['overall']['relative_error'], relative_error, places=7)
            self.assertLessEqual(relative_error, previous_error)
            previous_error = relative_error
            self.assertEqual(report['modules'][0]['output_rank'], rank)
        self.assertLess(previous_error, 1e-6)
        self.assertAlmostEqual(compare_loras(exact, output, 'effective_dimensions')['score'], 100., places=5)

    def test_lora_lokr_mixed_merge_and_different_ranks(self):
        lora = {'tensors': {'diffusion_model.blocks.0.attn.q.lora_A.weight': torch.ones(1,6),
                            'diffusion_model.blocks.0.attn.q.lora_B.weight': torch.ones(4,1)}}
        output, _ = merge_adapters([self.p, lora], [.5, 2.])
        actual = dense(next(iter(parsed(output).values())))
        torch.testing.assert_close(actual, .5*torch.kron(self.w1, self.w2) + 2)
        lora2 = {'tensors': {'diffusion_model.blocks.0.attn.q.lora_A.weight': torch.ones(2,6),
                             'diffusion_model.blocks.0.attn.q.lora_B.weight': torch.ones(4,2)}}
        output, _ = merge_adapters([lora, lora2], [1., 1.])
        torch.testing.assert_close(dense(next(iter(parsed(output).values()))), torch.full((4,6), 3., dtype=torch.float64))

    def test_zero_strength_cancellation_shape_and_overflow(self):
        output, report = merge_adapters([self.p, self.p], [1., -1.])
        self.assertEqual(report['overall']['relative_error'], 0.)
        self.assertEqual(float(next(iter(output['tensors'].values())).abs().max()), 0.)
        # A disabled checkpoint contributes no targets and cannot block dimensions.
        mismatch = lokr(torch.ones(3,2), self.w2)
        merge_adapters([self.p, mismatch], [1., 0.])
        for args in [([self.p, mismatch], [1., 1.]), ([self.p], [float('nan')]), ([self.p], [0.])]:
            with self.assertRaises(ValueError):
                merge_adapters(*args)
        with self.assertRaisesRegex(ValueError, 'storage dtype'):
            merge_adapters([self.p], [1e8], storage_dtype='float16')

    def test_duplicate_target_aliases_are_rejected(self):
        alias = lokr(self.w1, self.w2, 'lora_unet_blocks_0_attn_q')
        with self.assertRaisesRegex(ValueError, 'duplicate layer aliases'):
            merge_adapters([self.p, alias], [1., 1.])

    def test_streamed_safetensors_roundtrip_and_no_overwrite(self):
        from safetensors import safe_open
        from safetensors.torch import load_file
        for dtype in ('float32', 'float16', 'bfloat16'):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'merged.safetensors'
                stream = SafetensorsStream(folder)
                try:
                    payload, report = merge_adapters([self.p, self.q], [.7, .25], storage_dtype=dtype, tensor_sink=stream.add)
                    self.assertFalse(payload['tensors'])
                    stream.finish(path, report)
                    before = path.read_bytes()
                    with self.assertRaises(FileExistsError):
                        stream.finish(path, report)
                    self.assertEqual(path.read_bytes(), before)
                    state = load_file(str(path))
                    self.assertTrue(state)
                    expected, _ = merge_adapters([self.p, self.q], [.7, .25], storage_dtype=dtype)
                    for key, value in state.items():
                        torch.testing.assert_close(value, expected['tensors'][key])
                    with safe_open(str(path), framework='pt') as file:
                        self.assertEqual(json.loads(file.metadata()['moc_merge']), report)
                finally:
                    stream.close()

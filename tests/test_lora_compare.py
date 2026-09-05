from __future__ import annotations

import json
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from moc_minimax.lora_compare import compare_loras, comparison_json, render_comparison, _stats, _parse


def payload(a, b, name='blocks.0.attn.q', suffixes=('lora_A.weight', 'lora_B.weight'), alpha=None):
    tensors = {f'{name}.{suffixes[0]}': a, f'{name}.{suffixes[1]}': b}
    if alpha is not None:
        tensors[f'{name}.alpha'] = torch.tensor(alpha)
    return dict(source='test', tensors=tensors)


@unittest.skipIf(torch is None, 'torch is not installed')
class LoraComparisonTests(unittest.TestCase):
    def setUp(self):
        self.a = torch.tensor([[1., 2., 3.], [3., 1., 2.]], dtype=torch.float64)
        self.b = torch.tensor([[1., 2.], [0., 1.]], dtype=torch.float64)
        self.p = payload(self.a, self.b)

    def test_identical_and_equivalent_factorizations(self):
        for q in [self.p, payload(self.a * 2, self.b / 2),
                  payload(self.a.flip(0), self.b.flip(1))]:
            result = compare_loras(self.p, q)
            self.assertTrue(result['compatible'])
            self.assertAlmostEqual(result['score'], 100)
            self.assertFalse(result['blocks'][0]['differs'])

    def test_magnitude_and_direction(self):
        scaled = compare_loras(self.p, payload(self.a, self.b * 2))
        self.assertAlmostEqual(scaled['score'], 100 * 8 / 9)
        self.assertAlmostEqual(scaled['overall']['cosine_similarity'], 1)
        opposite = compare_loras(self.p, payload(self.a, -self.b))
        self.assertAlmostEqual(opposite['score'], 0)
        self.assertAlmostEqual(opposite['overall']['cosine_similarity'], -1)

    def test_zeros(self):
        zero = payload(self.a, self.b * 0)
        self.assertEqual(compare_loras(zero, zero)['score'], 100)
        result = compare_loras(zero, self.p)
        self.assertEqual(result['score'], 0)
        self.assertIsNone(result['overall']['cosine_similarity'])

    def test_alpha_and_names(self):
        for suffixes in [('lora_down.weight', 'lora_up.weight'),
                         ('lora_A.default.weight', 'lora_B.default.weight'), ('lora_A', 'lora_B')]:
            q = payload(self.a, self.b * 2, 'base_model.model.blocks.0.attn.q', suffixes, alpha=1.)
            result = compare_loras(self.p, q)
            self.assertEqual(result['score'], 100)
            module = result['blocks'][0]['modules'][0]
            self.assertTrue(module['alpha_defaulted_a'])
            self.assertFalse(module['alpha_defaulted_b'])

    def test_incompatible_and_unsupported(self):
        cases = [payload(self.a[:1], self.b[:, :1]),
                 payload(self.a[:, :2], self.b), payload(self.a, self.b, 'blocks.1.attn.q'),
                 payload(self.a.unsqueeze(0), self.b), payload(self.a * float('nan'), self.b),
                 payload(self.a, self.b, alpha=[1., 2.]), {'tensors': {}}, None]
        for key in ['blocks.0.attn.q.dora_scale', 'other.weight', 'blocks.0.attn.q.lora_mid.weight',
                    'base_model.model.blocks.0.attn.q.lora_A.weight']:
            q = payload(self.a, self.b)
            q['tensors'][key] = self.a
            cases.append(q)
        q = payload(self.a, self.b)
        del q['tensors']['blocks.0.attn.q.lora_B.weight']
        cases.append(q)
        for q in cases:
            with self.subTest(q=str(q)[:60]):
                result = compare_loras(self.p, q)
                self.assertFalse(result['compatible'])
                self.assertIsNone(result['score'])
                self.assertTrue(result['diagnostics'])
                json.loads(comparison_json(result))

    def test_difference_tolerance(self):
        for multiplier, differs in [(1 + 1e-6, False), (1 + 1e-4, True)]:
            result = compare_loras(self.p, payload(self.a, self.b * multiplier))
            self.assertEqual(result['blocks'][0]['differs'], differs)

    def test_gram_matches_dense(self):
        generator = torch.Generator().manual_seed(23)
        for _ in range(8):
            a, c = [torch.randn(3, 7, generator=generator, dtype=torch.float64) for _ in range(2)]
            b, d = [torch.randn(5, 3, generator=generator, dtype=torch.float64) for _ in range(2)]
            x, y = b @ a * (2 / 3), d @ c * (-4 / 3)
            expected = ((x*x).sum(), (y*y).sum(), (x*y).sum())
            left = _parse(payload(a, b, alpha=2), 'A', [])
            right = _parse(payload(c, d, alpha=-4), 'B', [])
            actual = _stats(left['blocks.0.attn.q'], right['blocks.0.attn.q'])
            for measured, reference in zip(actual, expected):
                self.assertAlmostEqual(measured, float(reference), places=9)

    def test_blocks_aggregation_and_serialization(self):
        p, q = {'tensors': {}}, {'tensors': {}}
        names = ['transformer_blocks.0.q', 'token_refiner.refiner_blocks.0.q', 'blocks.0.q', 'proj_out']
        for i, name in enumerate(names):
            p['tensors'].update(payload(self.a, self.b, name)['tensors'])
            q['tensors'].update(payload(self.a, self.b * (2 if i == 1 else 1), name)['tensors'])
        result = compare_loras(p, q)
        self.assertEqual(result['blocks'][0]['block'], 'token_refiner.refiner_blocks.0')
        self.assertEqual(len(result['blocks']), 3)
        self.assertEqual(len(result['non_block_modules']), 1)
        self.assertAlmostEqual(result['score'], 100 * (1 - 1 / (2 + 7**.5)**2))
        self.assertEqual(json.loads(comparison_json(result)), result)
        self.assertIn('Differing blocks: 1/3', render_comparison(result))
        q['tensors'] = dict(reversed(list(q['tensors'].items())))
        self.assertEqual(compare_loras(p, q), result)


if __name__ == '__main__':
    unittest.main()

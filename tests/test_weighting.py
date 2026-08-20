from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - lightweight CI can still test pure mapping logic with a stub
    torch = None

from moc_minimax.spans import WeightedSpan, reference_spans, target_start, weights_are_native

if torch is not None:
    from moc_minimax.weighting import (
        apply_value_gates,
        make_attention_prior_override,
        make_key_log_bias,
        make_value_gate_override,
    )


class PackedSpanTests(unittest.TestCase):
    def test_hybrid_keyframe_rows_are_skipped_before_references(self):
        segments = [
            (0, 2, "text"),
            (2, 6, "cond"),
            (6, 10, "cond_audio"),
            (10, 14, "ref_img"),
            (14, 18, "audio"),
            (18, 22, "video"),
        ]
        spans = reference_spans(segments, [{"kind": "image", "name": "look"}])
        self.assertEqual([(span.start, span.stop, span.alias) for span in spans], [(10, 14, "look")])

    def test_mixed_reference_span_mapping(self):
        segments = [
            (0, 10, "text"),
            (10, 14, "ref_img"),
            (14, 20, "ref_audio"),
            (20, 28, "ref_audio"),
            (28, 40, "ref_img"),
            (40, 56, "ref_img"),
            (56, 66, "audio"),
            (66, 80, "video"),
        ]
        refs = [
            {"kind": "image", "moc_name": "hero", "moc_visual_weight": 1.5},
            {"kind": "audio", "moc_name": "voice", "ref_audio_t": 3, "moc_audio_weight": 0.5},
            {
                "kind": "video_audio",
                "moc_name": "walk",
                "ref_audio_t": 4,
                "moc_visual_weight": 0.8,
                "moc_audio_weight": 1.2,
            },
            {"kind": "video", "moc_name": "camera", "ref_audio_t": 0, "moc_visual_weight": 0.25},
        ]
        spans = reference_spans(segments, refs)
        self.assertEqual(
            spans,
            [
                WeightedSpan(10, 14, "ref_img", "hero", 1.5),
                WeightedSpan(14, 20, "ref_audio", "voice", 0.5),
                WeightedSpan(20, 28, "ref_audio", "walk", 1.2),
                WeightedSpan(28, 40, "ref_img", "walk", 0.8),
                WeightedSpan(40, 56, "ref_img", "camera", 0.25),
            ],
        )
        self.assertEqual(target_start(segments), 56)
        self.assertTrue(all(span.stop <= target_start(segments) for span in spans))

    def test_native_weights(self):
        self.assertTrue(weights_are_native([WeightedSpan(1, 2, "ref_img", "a", 1.0)]))
        self.assertFalse(weights_are_native([WeightedSpan(1, 2, "ref_img", "a", 0.999)]))

    def test_layout_mismatch_raises(self):
        with self.assertRaises(ValueError):
            reference_spans([(0, 2, "text"), (2, 4, "audio"), (4, 6, "video")], [{"kind": "image"}])

    def test_unclaimed_reference_segment_raises(self):
        segments = [(0, 2, "text"), (2, 4, "ref_img"), (4, 6, "ref_img"), (6, 8, "audio"), (8, 10, "video")]
        with self.assertRaisesRegex(ValueError, "unclaimed prefix"):
            reference_spans(segments, [{"kind": "image"}])


@unittest.skipIf(torch is None, "torch is not installed in this test environment")
class TensorWeightingTests(unittest.TestCase):
    def test_strength_above_one_never_inverts_value_gate(self):
        values = torch.ones((1, 1, 3, 1))
        spans = [WeightedSpan(1, 2, "ref_img", "hero", 0.25)]
        weighted = apply_value_gates(values, spans, 2.0)
        self.assertAlmostEqual(float(weighted[..., 1, :].item()), 0.0625)
        self.assertTrue(torch.all(weighted >= 0))

    def test_log_bias_is_key_only_and_finite_for_zero(self):
        spans = [WeightedSpan(2, 4, "ref_img", "a", 0.0), WeightedSpan(5, 6, "ref_audio", "b", 2.0)]
        bias = make_key_log_bias(8, spans, strength=1.0, device=torch.device("cpu"), dtype=torch.float32)
        self.assertEqual(tuple(bias.shape), (1, 1, 1, 8))
        self.assertTrue(torch.isfinite(bias).all())
        self.assertLess(float(bias[..., 2].item()), -19.0)
        self.assertAlmostEqual(float(bias[..., 5].item()), 0.693147, places=5)

    @staticmethod
    def _attention(q, k, v, _heads, mask=None, **_kwargs):
        scores = q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5)
        if mask is not None:
            scores = scores + mask
        output = scores.softmax(dim=-1) @ v
        return output.transpose(1, 2).reshape(output.shape[0], output.shape[2], -1)

    def test_target_only_value_gate_matches_manual_attention(self):
        torch.manual_seed(7)
        q = torch.randn((1, 1, 6, 3))
        k = torch.randn((1, 1, 6, 3))
        v = torch.randn((1, 1, 6, 3))
        original_v = v.clone()
        spans = [WeightedSpan(1, 3, "ref_img", "hero", 0.25)]
        override = make_value_gate_override(spans, first_target=4, strength=1.0, previous_override=None)
        actual = override(self._attention, q, k, v, 1, skip_reshape=True, skip_output_reshape=False)

        expected_prefix = self._attention(q[..., :4, :], k, original_v, 1)
        weighted_v = original_v.clone()
        weighted_v[..., 1:3, :] *= 0.25
        expected_target = self._attention(q[..., 4:, :], k, weighted_v, 1)
        expected = torch.cat((expected_prefix, expected_target), dim=1)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

        baseline_prefix = self._attention(q, k, original_v, 1)[:, :4]
        self.assertTrue(torch.equal(actual[:, :4], baseline_prefix))

    def test_attention_prior_matches_log_weighted_attention(self):
        torch.manual_seed(11)
        q = torch.randn((1, 1, 6, 3))
        k = torch.randn((1, 1, 6, 3))
        v = torch.randn((1, 1, 6, 3))
        spans = [WeightedSpan(1, 3, "ref_img", "hero", 0.25)]
        override = make_attention_prior_override(
            spans,
            first_target=4,
            strength=1.0,
            previous_override=None,
            attention_impl=self._attention,
        )
        actual = override(self._attention, q, k, v, 1, skip_reshape=True, skip_output_reshape=False)

        expected_prefix = self._attention(q[..., :4, :], k, v, 1)
        bias = make_key_log_bias(6, spans, strength=1.0, device=q.device, dtype=q.dtype)
        expected_target = self._attention(q[..., 4:, :], k, v, 1, mask=bias)
        expected = torch.cat((expected_prefix, expected_target), dim=1)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()

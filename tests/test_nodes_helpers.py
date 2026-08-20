from __future__ import annotations

import unittest
from unittest import mock

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    import comfy_api  # noqa: F401
except ImportError:  # pragma: no cover
    comfy_api = None

if torch is not None and comfy_api is not None:
    from moc_minimax.core import make_reference, plan_references
    from moc_minimax.nodes import (
        MocH3ImageReferenceNode,
        _attach_temporal_keyframes,
        _limit_diagnostics,
        _native_inputs,
        _native_video_frame_count,
        _prepare_temporal_keyframes,
        _resolved_target_frames,
    )


@unittest.skipIf(torch is None or comfy_api is None, "torch/ComfyUI is not installed in this test environment")
class NativePreparationTests(unittest.TestCase):
    def test_temporal_keyframes_coexist_with_non_temporal_refs(self):
        class FakeVae:
            @staticmethod
            def encode(image):
                return image.movedim(-1, 1)

        image = torch.ones((1, 8, 8, 3))
        crops = []

        def fake_upscale(samples, width, height, method, crop):
            self.assertEqual((width, height, method), (8, 8, "lanczos"))
            crops.append(crop)
            return samples

        with mock.patch("comfy.utils.common_upscale", side_effect=fake_upscale):
            keyframes, summary = _prepare_temporal_keyframes(
                FakeVae(), 8, 8, 22, first_frame=image, last_frame=image
            )

        self.assertEqual(crops, ["disabled", "center"])
        self.assertEqual([item["resolved_frame_index"] for item in keyframes], [0, 21])
        conditioning = [[object(), {"minimax_refs": [{"kind": "image"}]}]]
        combined = _attach_temporal_keyframes(conditioning, keyframes, summary, 22)
        metadata = combined[0][1]
        self.assertEqual(metadata["minimax_refs"], [{"kind": "image"}])
        self.assertEqual([item["resolved_frame_index"] for item in metadata["minimax_keyframes"]], [0, 21])
        self.assertEqual(metadata["minimax_frame_count"], 22)
        self.assertEqual(metadata["moc_h3_temporal_guides"], summary)

    def test_masked_builder_tensor_is_the_native_reference_image(self):
        image = torch.ones((1, 8, 8, 3))
        mask = torch.zeros((1, 8, 8))
        mask[:, 2:6, 2:6] = 1.0
        output = MocH3ImageReferenceNode.execute(
            image,
            "hero",
            "identity",
            "primary",
            1.0,
            "match_output",
            "",
            "",
            mask=mask,
            mask_feather_px=0,
            neutral_level=0.25,
        )
        reference = output.result[0]
        self.assertTrue(reference["metadata"]["mask_applied"])
        self.assertTrue(torch.all(reference["media"][:, 0, 0] == 0.25))
        self.assertTrue(torch.all(reference["media"][:, 2:6, 2:6] == 1.0))
        plan = plan_references([reference])
        native_inputs = _native_inputs(plan, 1344, 768, 124, "match_output")
        self.assertTrue(torch.equal(native_inputs[1]["ref_image_0"], reference["media"]))

    def test_trim_to_video_soundtrack_uses_exact_native_grid_duration(self):
        frames = torch.zeros((360, 64, 96, 3))
        waveform = torch.zeros((1, 2, 15 * 1000))
        ref = make_reference(
            kind="video",
            name="walk",
            media=frames,
            soundtrack={"waveform": waveform, "sample_rate": 1000},
            role="motion",
            priority="supporting",
            metadata={
                "processed_frames": 360,
                "duration_s": 15.0,
                "soundtrack_policy": "trim_to_video",
                "soundtrack_role": "timing",
                "soundtrack_relationship": "reference",
                "soundtrack": {
                    "duration_s": 15.0,
                    "processed_samples": 15000,
                    "sample_rate": 1000,
                },
                "warnings": [],
            },
        )
        plan = plan_references([ref])
        expected_frames = _native_video_frame_count(plan.native_order[0], _resolved_target_frames(124))
        result = _native_inputs(plan, 1344, 768, 124, "match_output")
        paired_audio = result[3]["ref_video_audio_0"]
        self.assertEqual(expected_frames, 124)
        self.assertEqual(paired_audio["waveform"].shape[-1], round(124 / 24 * 1000))
        self.assertAlmostEqual(
            plan.native_order[0]["metadata"]["native_soundtrack_duration_s"],
            round(124 / 24 * 1000) / 1000,
        )

    def test_keep_soundtrack_reports_duration_mismatch(self):
        ref = make_reference(
            kind="video",
            name="walk",
            media=torch.zeros((240, 64, 96, 3)),
            soundtrack={"waveform": torch.zeros((1, 2, 10000)), "sample_rate": 1000},
            role="motion",
            priority="supporting",
            metadata={
                "processed_frames": 240,
                "duration_s": 10.0,
                "soundtrack_policy": "keep",
                "soundtrack": {"duration_s": 10.0},
                "warnings": [],
            },
        )
        plan = plan_references([ref])
        codes = [item.code for item in _limit_diagnostics(plan, 124)]
        self.assertIn("soundtrack_duration_mismatch", codes)


if __name__ == "__main__":
    unittest.main()

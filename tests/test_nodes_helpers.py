from __future__ import annotations

import asyncio
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
        MocH3ImageToVideoSimpleNode,
        MocH3Extension,
        _attach_temporal_keyframes,
        _limit_diagnostics,
        _native_inputs,
        _native_video_frame_count,
        _prepare_temporal_keyframes,
        _resolved_target_frames,
    )


@unittest.skipIf(torch is None or comfy_api is None, "torch/ComfyUI is not installed in this test environment")
class NativePreparationTests(unittest.TestCase):
    def test_simple_node_schema_and_registration(self):
        self.assertIn(MocH3ImageToVideoSimpleNode, asyncio.run(MocH3Extension().get_node_list()))
        schema = MocH3ImageToVideoSimpleNode.define_schema()
        schema.finalize()
        schema.validate()
        self.assertEqual([item.id for item in schema.inputs], [
            "clip", "vae", "prompt", "width", "height", "length", "first_frame", "last_frame",
            "reference_images",
        ])
        self.assertEqual([item.io_type for item in schema.outputs], ["CONDITIONING", "LATENT"])
        self.assertTrue(all(item.optional for item in schema.inputs[6:]))
        template = schema.inputs[-1].template
        self.assertEqual((template.min, template.max), (1, 9))
        self.assertTrue(template.input.optional)

    def test_simple_node_without_references_delegates_unchanged_to_stock(self):
        for first, last in ((None, None), (object(), None), (None, object()), (object(), object())):
            with self.subTest(first=first is not None, last=last is not None):
                inputs = dict(clip=object(), vae=object(), prompt="A quiet scene", width=640,
                              height=480, length=23, first_frame=first, last_frame=last)
                for references in (None, {}, {"reference_image_0": None}):
                    with mock.patch("moc_minimax.nodes._native_node_class") as factory:
                        result = MocH3ImageToVideoSimpleNode.execute(**inputs, reference_images=references)
                    factory.assert_called_once_with("MiniMaxH3ImageToVideo")
                    factory.return_value.execute.assert_called_once_with(**inputs)
                    self.assertIs(result, factory.return_value.execute.return_value)

    def test_simple_node_sparse_references_and_temporal_frames(self):
        from comfy_api.latest import io

        class FakeVae:
            @staticmethod
            def encode(image):
                return image.movedim(-1, 1)

        first, last = torch.ones((1, 32, 32, 3)), torch.zeros((1, 32, 32, 3))
        ref2, ref4 = torch.full_like(first, 0.2), torch.full_like(first, 0.4)
        refs = [{"kind": "image", "latent": ref2}, {"kind": "image", "latent": ref4}]
        original_metadata = {"minimax_refs": refs}
        latent = {"samples": object()}
        clip, vae = object(), FakeVae()

        # Keyword-only arguments catch accidental positional delegation when native
        # ComfyUI moves optional VAE arguments behind the required parameters.
        def native_execute(*, clip, vae, audio_vae, prompt, width, height, length,
                           ref_image_size, ref_images, ref_videos, ref_video_audios, ref_audios):
            self.assertEqual(prompt, "Use <Picture 1> and <Picture 2>")
            self.assertEqual(ref_image_size, "match")
            self.assertEqual(list(ref_images), ["ref_image_0", "ref_image_1"])
            self.assertIs(ref_images["ref_image_0"], ref2)
            self.assertIs(ref_images["ref_image_1"], ref4)
            self.assertEqual((ref_videos, ref_video_audios, ref_audios), ({}, {}, {}))
            return io.NodeOutput([[object(), original_metadata]], latent)

        with mock.patch("moc_minimax.nodes._native_node_class") as factory, \
             mock.patch("comfy.utils.common_upscale", side_effect=lambda samples, *args: samples):
            factory.return_value.execute.side_effect = native_execute
            result = MocH3ImageToVideoSimpleNode.execute(
                clip, vae, "Use <Picture 1> and <Picture 2>", 32, 32, 23,
                first_frame=first, last_frame=last,
                reference_images={"reference_image_0": None, "reference_image_1": ref2,
                                  "reference_image_2": None, "reference_image_3": ref4},
            )
        metadata = result.result[0][0][1]
        self.assertEqual([kf["resolved_frame_index"] for kf in metadata["minimax_keyframes"]], [0, 38])
        self.assertIs(metadata["minimax_refs"], refs)
        self.assertNotIn("minimax_keyframes", original_metadata)
        self.assertIs(result.result[1], latent)

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

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - lightweight CI still exercises pure planning logic
    torch = None

if torch is not None:
    from moc_minimax.media import prepare_audio, prepare_image, prepare_video


@unittest.skipIf(torch is None, "torch is not installed in this test environment")
class MediaPreparationTests(unittest.TestCase):
    def test_image_uses_first_batch_item_without_mutating_source(self):
        image = torch.zeros((2, 4, 6, 4))
        image[0, ..., 0] = 0.25
        image[1, ..., 0] = 0.75
        selected, metadata = prepare_image(image)
        self.assertEqual(tuple(selected.shape), (1, 4, 6, 3))
        self.assertTrue(torch.all(selected[..., 0] == 0.25))
        self.assertEqual(metadata["source_batch"], 2)
        self.assertTrue(metadata["warnings"])

    def test_image_without_mask_preserves_original_tensor_view(self):
        image = torch.rand((1, 5, 7, 3))
        selected, metadata = prepare_image(image)
        self.assertEqual(selected.data_ptr(), image.data_ptr())
        self.assertTrue(torch.equal(selected, image))
        self.assertFalse(metadata["mask_applied"])

    def test_neutral_mask_and_inverse_polarity(self):
        image = torch.ones((1, 4, 4, 3))
        mask = torch.zeros((1, 4, 4))
        mask[:, 1:3, 1:3] = 1.0
        kept, metadata = prepare_image(
            image,
            mask=mask,
            mask_feather_px=0,
            neutral_level=0.25,
        )
        self.assertTrue(torch.all(kept[:, 1:3, 1:3] == 1.0))
        self.assertTrue(torch.all(kept[:, 0, 0] == 0.25))
        self.assertAlmostEqual(metadata["mask_coverage"], 0.25)
        self.assertEqual(metadata["mask_bounds_xyxy"], [1, 1, 3, 3])

        removed, inverse_metadata = prepare_image(
            image,
            mask=mask,
            mask_polarity="white_removes",
            mask_feather_px=0,
            neutral_level=0.0,
        )
        self.assertTrue(torch.all(removed[:, 1:3, 1:3] == 0.0))
        self.assertTrue(torch.all(removed[:, 0, 0] == 1.0))
        self.assertAlmostEqual(inverse_metadata["mask_coverage"], 0.75)

    def test_mask_resize_batch_and_nonfinite_validation(self):
        image = torch.ones((1, 8, 6, 3))
        masks = torch.ones((2, 4, 3))
        _, metadata = prepare_image(image, mask=masks, mask_feather_px=0)
        self.assertTrue(metadata["mask_resized"])
        self.assertEqual((metadata["mask_source_width"], metadata["mask_source_height"]), (3, 4))
        self.assertEqual((metadata["mask_resolved_width"], metadata["mask_resolved_height"]), (6, 8))
        self.assertTrue(any("Mask batch" in warning for warning in metadata["warnings"]))
        self.assertTrue(any("Resized mask" in warning for warning in metadata["warnings"]))

        invalid = torch.ones((1, 8, 6))
        invalid[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            prepare_image(image, mask=invalid)

    def test_mask_expand_erode_and_feather(self):
        image = torch.ones((1, 7, 7, 3))
        point = torch.zeros((1, 7, 7))
        point[:, 3, 3] = 1.0
        expanded, expanded_meta = prepare_image(
            image,
            mask=point,
            mask_expand_px=1,
            mask_feather_px=0,
            neutral_level=0.0,
        )
        self.assertEqual(int((expanded[..., 0] > 0).sum()), 9)
        self.assertAlmostEqual(expanded_meta["mask_coverage"], 9 / 49)

        block = torch.zeros((1, 7, 7))
        block[:, 2:5, 2:5] = 1.0
        eroded, _ = prepare_image(
            image,
            mask=block,
            mask_expand_px=-1,
            mask_feather_px=0,
            neutral_level=0.0,
        )
        self.assertEqual(int((eroded[..., 0] > 0).sum()), 1)

        feathered, _ = prepare_image(
            image,
            mask=point,
            mask_feather_px=2,
            neutral_level=0.0,
        )
        self.assertTrue(((feathered > 0.0) & (feathered < 1.0)).any().item())

    def test_crop_padding_and_blurred_background(self):
        image = torch.zeros((1, 10, 12, 3))
        image[:, :, :, 0] = torch.linspace(0.0, 1.0, 12).reshape(1, 1, 12)
        mask = torch.zeros((1, 10, 12))
        mask[:, 2:6, 3:7] = 1.0
        cropped, metadata = prepare_image(
            image,
            mask=mask,
            mask_presentation="crop_to_mask",
            mask_feather_px=0,
            crop_padding_pct=50.0,
        )
        self.assertEqual(tuple(cropped.shape), (1, 8, 8, 3))
        self.assertEqual(metadata["mask_crop_xyxy"], [1, 0, 9, 8])

        blurred, blur_metadata = prepare_image(
            image,
            mask=mask,
            mask_presentation="blur_background",
            mask_feather_px=0,
            background_blur_px=2,
        )
        self.assertEqual(tuple(blurred.shape), tuple(image.shape))
        self.assertTrue(blur_metadata["mask_applied"])
        self.assertTrue(torch.equal(blurred[:, 2:6, 3:7], image[:, 2:6, 3:7]))
        self.assertFalse(torch.equal(blurred[:, 0, 0], image[:, 0, 0]))

    def test_empty_and_extreme_coverage_diagnostics(self):
        image = torch.ones((1, 20, 20, 3))
        with self.assertRaisesRegex(ValueError, "retains no pixels"):
            prepare_image(image, mask=torch.zeros((1, 20, 20)))

        tiny = torch.zeros((1, 20, 20))
        tiny[:, 10, 10] = 1.0
        _, tiny_meta = prepare_image(image, mask=tiny, mask_feather_px=0)
        self.assertTrue(any("only" in warning for warning in tiny_meta["warnings"]))
        _, full_meta = prepare_image(image, mask=torch.ones((1, 20, 20)), mask_feather_px=0)
        self.assertTrue(any("no-op" in warning for warning in full_meta["warnings"]))

    def test_video_trim_and_resample_to_24_fps(self):
        frames = torch.arange(60, dtype=torch.float32).reshape(60, 1, 1, 1).expand(-1, 2, 3, 3)
        prepared, metadata = prepare_video(
            frames,
            input_fps=30.0,
            trim_start_s=0.5,
            max_duration_s=1.0,
        )
        self.assertEqual(tuple(prepared.shape), (24, 2, 3, 3))
        self.assertEqual(metadata["processed_fps"], 24.0)
        self.assertAlmostEqual(metadata["duration_s"], 1.0)
        self.assertEqual(float(prepared[0, 0, 0, 0]), 15.0)
        self.assertEqual(float(prepared[-1, 0, 0, 0]), 44.0)

    def test_audio_trim_keeps_first_batch_and_channels(self):
        waveform = torch.arange(400, dtype=torch.float32).reshape(2, 2, 100)
        prepared, metadata = prepare_audio(
            {"waveform": waveform, "sample_rate": 20},
            trim_start_s=1.0,
            max_duration_s=2.0,
        )
        self.assertEqual(tuple(prepared["waveform"].shape), (1, 2, 40))
        self.assertEqual(metadata["processed_samples"], 40)
        self.assertEqual(float(prepared["waveform"][0, 0, 0]), 20.0)


if __name__ == "__main__":
    unittest.main()

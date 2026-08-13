"""Deterministic media preparation used by the ComfyUI reference builders."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


TARGET_VIDEO_FPS = 24.0


def _finite_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


MASK_POLARITIES = ("white_keeps", "white_removes")
MASK_PRESENTATIONS = ("neutral_fill", "blur_background", "crop_to_mask")


def _bounded_float(value: float, name: str, minimum: float, maximum: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be finite and between {minimum:g} and {maximum:g}")
    return value


def _bounded_int(value: int, name: str, minimum: int, maximum: int) -> int:
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _gaussian_blur(tensor: torch.Tensor, radius: int) -> torch.Tensor:
    """Apply a deterministic separable Gaussian blur to BCHW data."""
    radius = int(radius)
    if radius <= 0:
        return tensor
    sigma = max(radius / 3.0, 0.5)
    positions = torch.arange(-radius, radius + 1, device=tensor.device, dtype=tensor.dtype)
    kernel = torch.exp(-(positions * positions) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    channels = int(tensor.shape[1])
    horizontal = kernel.reshape(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.reshape(1, 1, -1, 1).expand(channels, 1, -1, 1)
    # Reflect padding fails when the radius is at least the corresponding image
    # dimension. Replication remains deterministic and avoids artificial zeros.
    horizontal_mode = "reflect" if radius < tensor.shape[-1] else "replicate"
    vertical_mode = "reflect" if radius < tensor.shape[-2] else "replicate"
    blurred = F.conv2d(F.pad(tensor, (radius, radius, 0, 0), mode=horizontal_mode), horizontal, groups=channels)
    return F.conv2d(F.pad(blurred, (0, 0, radius, radius), mode=vertical_mode), vertical, groups=channels)


def _expand_or_erode(mask: torch.Tensor, pixels: int) -> torch.Tensor:
    if pixels == 0:
        return mask
    radius = abs(int(pixels))
    kernel = radius * 2 + 1
    padded = F.pad(mask, (radius, radius, radius, radius), mode="constant", value=0.0)
    if pixels > 0:
        return F.max_pool2d(padded, kernel_size=kernel, stride=1)
    return -F.max_pool2d(-padded, kernel_size=kernel, stride=1)


def _mask_bounds(mask: torch.Tensor) -> tuple[int, int, int, int] | None:
    active = mask[0, 0] > 1e-6
    locations = active.nonzero(as_tuple=False)
    if locations.numel() == 0:
        return None
    y0 = int(locations[:, 0].min().item())
    y1 = int(locations[:, 0].max().item()) + 1
    x0 = int(locations[:, 1].min().item())
    x1 = int(locations[:, 1].max().item()) + 1
    return x0, y0, x1, y1


def _padded_bounds(
    bounds: tuple[int, int, int, int],
    width: int,
    height: int,
    padding_pct: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bounds
    pad_x = int(math.ceil((x1 - x0) * padding_pct / 100.0))
    pad_y = int(math.ceil((y1 - y0) * padding_pct / 100.0))
    return max(0, x0 - pad_x), max(0, y0 - pad_y), min(width, x1 + pad_x), min(height, y1 + pad_y)


def prepare_image(
    image: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    mask_polarity: str = "white_keeps",
    mask_presentation: str = "neutral_fill",
    mask_expand_px: int = 0,
    mask_feather_px: int = 4,
    neutral_level: float = 0.5,
    background_blur_px: int = 32,
    crop_padding_pct: float = 10.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise ValueError("Image references must be IMAGE tensors shaped [batch, height, width, channels].")
    if image.shape[0] < 1 or image.shape[1] < 1 or image.shape[2] < 1 or image.shape[3] < 3:
        raise ValueError("Image reference tensor is empty or has fewer than three channels.")
    warnings: list[str] = []
    if image.shape[0] > 1:
        warnings.append(f"Image batch contained {image.shape[0]} images; only the first image is used.")
    selected = image[:1, :, :, :3]
    metadata: dict[str, Any] = {
        "source_batch": int(image.shape[0]),
        "processed_width": int(selected.shape[2]),
        "processed_height": int(selected.shape[1]),
        "mask_applied": False,
        "warnings": warnings,
    }
    # This exact return path intentionally preserves 0.1.x behaviour and tensor
    # identity when the optional mask is not connected.
    if mask is None:
        return selected, metadata

    if mask_polarity not in MASK_POLARITIES:
        raise ValueError(f"Unsupported mask_polarity: {mask_polarity!r}")
    if mask_presentation not in MASK_PRESENTATIONS:
        raise ValueError(f"Unsupported mask_presentation: {mask_presentation!r}")
    mask_expand_px = _bounded_int(mask_expand_px, "mask_expand_px", -128, 128)
    mask_feather_px = _bounded_int(mask_feather_px, "mask_feather_px", 0, 128)
    neutral_level = _bounded_float(neutral_level, "neutral_level", 0.0, 1.0)
    background_blur_px = _bounded_int(background_blur_px, "background_blur_px", 0, 128)
    crop_padding_pct = _bounded_float(crop_padding_pct, "crop_padding_pct", 0.0, 100.0)

    if not isinstance(mask, torch.Tensor) or mask.ndim not in (2, 3):
        raise ValueError("Image reference masks must be shaped [height, width] or [batch, height, width].")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.shape[0] < 1 or mask.shape[1] < 1 or mask.shape[2] < 1:
        raise ValueError("Image reference mask tensor is empty.")
    if not torch.isfinite(mask).all().item():
        raise ValueError("Image reference masks must contain only finite values.")
    if mask.shape[0] > 1:
        warnings.append(f"Mask batch contained {mask.shape[0]} masks; only the first mask is used.")

    source_mask_height = int(mask.shape[1])
    source_mask_width = int(mask.shape[2])
    work_mask = mask[:1].to(device=selected.device, dtype=torch.float32).unsqueeze(1).clamp_(0.0, 1.0)
    image_height, image_width = int(selected.shape[1]), int(selected.shape[2])
    mask_resized = (source_mask_height, source_mask_width) != (image_height, image_width)
    if mask_resized:
        work_mask = F.interpolate(work_mask, size=(image_height, image_width), mode="bilinear", align_corners=False)
        warnings.append(
            f"Resized mask from {source_mask_width}x{source_mask_height} to {image_width}x{image_height} using bilinear interpolation."
        )
    if mask_polarity == "white_removes":
        work_mask = 1.0 - work_mask
    work_mask = _expand_or_erode(work_mask, mask_expand_px).clamp_(0.0, 1.0)
    bounds = _mask_bounds(work_mask)
    if bounds is None:
        raise ValueError("Image reference mask retains no pixels after polarity and grow/shrink processing.")
    if mask_feather_px:
        work_mask = _gaussian_blur(work_mask, mask_feather_px).clamp_(0.0, 1.0)

    coverage = float(work_mask.mean().item())
    if coverage < 0.01:
        warnings.append(f"Mask retains only {coverage * 100.0:.2f}% of the source image; reference evidence may be too small.")
    elif coverage > 0.99:
        warnings.append(f"Mask retains {coverage * 100.0:.2f}% of the source image and is effectively a no-op.")

    original_dtype = selected.dtype
    work_image = selected.permute(0, 3, 1, 2).to(dtype=torch.float32)
    crop_bounds: tuple[int, int, int, int] | None = None
    if mask_presentation == "crop_to_mask":
        crop_bounds = _padded_bounds(bounds, image_width, image_height, crop_padding_pct)
        x0, y0, x1, y1 = crop_bounds
        work_image = work_image[:, :, y0:y1, x0:x1]
        work_mask = work_mask[:, :, y0:y1, x0:x1]
        background = torch.full_like(work_image, neutral_level)
    elif mask_presentation == "blur_background":
        background = _gaussian_blur(work_image, background_blur_px)
    else:
        background = torch.full_like(work_image, neutral_level)
    processed = (work_image * work_mask + background * (1.0 - work_mask)).permute(0, 2, 3, 1).to(dtype=original_dtype)

    metadata.update({
        "processed_width": int(processed.shape[2]),
        "processed_height": int(processed.shape[1]),
        "mask_applied": True,
        "mask_polarity": mask_polarity,
        "mask_presentation": mask_presentation,
        "mask_expand_px": mask_expand_px,
        "mask_feather_px": mask_feather_px,
        "mask_coverage": coverage,
        "mask_source_width": source_mask_width,
        "mask_source_height": source_mask_height,
        "mask_resolved_width": image_width,
        "mask_resolved_height": image_height,
        "mask_resized": mask_resized,
        "mask_bounds_xyxy": list(bounds),
        "mask_crop_xyxy": list(crop_bounds) if crop_bounds is not None else None,
        "neutral_level": neutral_level,
        "background_blur_px": background_blur_px,
        "crop_padding_pct": crop_padding_pct,
    })
    return processed, metadata


def prepare_video(
    frames: torch.Tensor,
    *,
    input_fps: float,
    trim_start_s: float = 0.0,
    max_duration_s: float = 0.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Trim and nearest-neighbour resample an IMAGE batch to H3's 24 fps."""
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
        raise ValueError("Video references must be IMAGE batches shaped [frames, height, width, channels].")
    if frames.shape[0] < 1 or frames.shape[1] < 1 or frames.shape[2] < 1 or frames.shape[3] < 3:
        raise ValueError("Video reference tensor is empty or has fewer than three channels.")
    input_fps = float(input_fps)
    if not math.isfinite(input_fps) or input_fps <= 0:
        raise ValueError("input_fps must be finite and greater than zero")
    trim_start_s = _finite_nonnegative(trim_start_s, "trim_start_s")
    max_duration_s = _finite_nonnegative(max_duration_s, "max_duration_s")

    source_count = int(frames.shape[0])
    start_index = min(source_count, int(math.floor(trim_start_s * input_fps + 1e-8)))
    end_index = source_count
    if max_duration_s > 0:
        end_index = min(end_index, start_index + max(1, int(round(max_duration_s * input_fps))))
    clipped = frames[start_index:end_index, :, :, :3]
    if clipped.shape[0] == 0:
        raise ValueError("Video trim produced zero frames. Reduce trim_start_s or increase max_duration_s.")

    source_duration = clipped.shape[0] / input_fps
    target_count = max(1, int(round(source_duration * TARGET_VIDEO_FPS)))
    if target_count == clipped.shape[0] and abs(input_fps - TARGET_VIDEO_FPS) < 1e-6:
        processed = clipped
    else:
        # Use timestamps centred on output frames, mapped to nearest source frame.
        positions = (torch.arange(target_count, device=clipped.device, dtype=torch.float64) + 0.5)
        positions *= input_fps / TARGET_VIDEO_FPS
        indices = positions.floor().long().clamp_(0, clipped.shape[0] - 1)
        processed = clipped.index_select(0, indices)

    warnings: list[str] = []
    if abs(input_fps - TARGET_VIDEO_FPS) > 1e-6:
        warnings.append(f"Resampled reference video from {input_fps:g} fps to 24 fps using nearest frames.")
    duration_s = processed.shape[0] / TARGET_VIDEO_FPS
    if duration_s < 2.0:
        warnings.append("Reference video is shorter than MiniMax's recommended 2 second minimum.")
    if duration_s > 15.0:
        warnings.append("Reference video exceeds MiniMax's recommended 15 second per-clip maximum.")
    grid_frames = int(processed.shape[0])
    while grid_frames >= 5 and grid_frames % 17 != 5:
        grid_frames -= 1
    if grid_frames >= 5 and grid_frames != processed.shape[0]:
        warnings.append(
            "Native H3 will trim trailing frames to its 17k+5 temporal grid after applying the target-length cap; "
            "the exact encoded count is reported downstream."
        )
    if processed.shape[0] < 5:
        warnings.append("Reference video has fewer than 5 processed frames and cannot be encoded by the native H3 node.")

    return processed, {
        "source_frames": source_count,
        "source_fps": input_fps,
        "trim_start_s": trim_start_s,
        "processed_frames": int(processed.shape[0]),
        "processed_fps": TARGET_VIDEO_FPS,
        "processed_width": int(processed.shape[2]),
        "processed_height": int(processed.shape[1]),
        "duration_s": duration_s,
        "native_grid_frames": grid_frames if grid_frames >= 5 else 0,
        "warnings": warnings,
    }


def prepare_audio(
    audio: dict[str, Any],
    *,
    trim_start_s: float = 0.0,
    max_duration_s: float = 0.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("Audio references must contain 'waveform' and 'sample_rate'.")
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3:
        raise ValueError("Audio waveform must be shaped [batch, channels, samples].")
    if sample_rate <= 0 or waveform.shape[-1] < 1:
        raise ValueError("Audio sample rate and sample count must be positive.")
    trim_start_s = _finite_nonnegative(trim_start_s, "trim_start_s")
    max_duration_s = _finite_nonnegative(max_duration_s, "max_duration_s")
    start = min(waveform.shape[-1], int(math.floor(trim_start_s * sample_rate + 1e-8)))
    end = waveform.shape[-1]
    if max_duration_s > 0:
        end = min(end, start + max(1, int(round(max_duration_s * sample_rate))))
    selected = waveform[:1, :, start:end]
    if selected.shape[-1] == 0:
        raise ValueError("Audio trim produced zero samples. Reduce trim_start_s or increase max_duration_s.")
    duration_s = selected.shape[-1] / sample_rate
    warnings: list[str] = []
    if waveform.shape[0] > 1:
        warnings.append(f"Audio batch contained {waveform.shape[0]} items; only the first item is used.")
    if duration_s < 2.0:
        warnings.append("Reference audio is shorter than MiniMax's recommended 2 second minimum.")
    if duration_s > 15.0:
        warnings.append("Reference audio exceeds MiniMax's recommended 15 second per-clip maximum.")
    prepared = {"waveform": selected, "sample_rate": sample_rate}
    return prepared, {
        "source_samples": int(waveform.shape[-1]),
        "sample_rate": sample_rate,
        "channels": int(selected.shape[1]),
        "trim_start_s": trim_start_s,
        "processed_samples": int(selected.shape[-1]),
        "duration_s": duration_s,
        "warnings": warnings,
    }


def trim_audio_to_duration(
    audio: dict[str, Any],
    *,
    trim_start_s: float,
    duration_s: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return prepare_audio(audio, trim_start_s=trim_start_s, max_duration_s=duration_s)

"""Pure packed-layout mapping for MiniMax H3 reference controls."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class WeightedSpan:
    start: int
    stop: int
    kind: str
    alias: str
    weight: float


def _validate_weight(value: Any, alias: str, modality: str) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"@{alias} has a non-numeric {modality} weight") from exc
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError(f"@{alias} {modality} weight must be finite and non-negative")
    return weight


def reference_spans(layout_segments: Iterable[tuple[int, int, str]], refs: Iterable[dict[str, Any]]) -> list[WeightedSpan]:
    """Map H3 reference records to their packed visual/audio token spans."""
    segments = list(layout_segments)
    cursor = 1 if segments and segments[0][2] == "text" else 0
    # Hybrid H3 layouts place temporal keyframe rows before the non-temporal
    # reference bank. They are anchors, not weighted reference records.
    while cursor < len(segments) and segments[cursor][2] in ("cond", "cond_audio"):
        cursor += 1
    spans: list[WeightedSpan] = []

    def take(expected: str, alias: str) -> tuple[int, int, str]:
        nonlocal cursor
        if cursor >= len(segments):
            raise ValueError(f"Packed MiniMax layout ended before reference @{alias}")
        segment = segments[cursor]
        if segment[2] != expected:
            raise ValueError(
                f"Packed MiniMax layout mismatch for @{alias}: expected {expected!r}, found {segment[2]!r}"
            )
        cursor += 1
        return segment

    for index, ref in enumerate(refs, 1):
        kind = ref.get("kind")
        alias = str(ref.get("moc_name") or ref.get("name") or f"reference_{index}")
        visual_weight = _validate_weight(ref.get("moc_visual_weight", ref.get("visual_weight", 1.0)), alias, "visual")
        audio_weight = _validate_weight(ref.get("moc_audio_weight", ref.get("audio_weight", 1.0)), alias, "audio")
        if kind == "image":
            start, stop, segment_kind = take("ref_img", alias)
            spans.append(WeightedSpan(start, stop, segment_kind, alias, visual_weight))
        elif kind == "audio":
            if int(ref.get("ref_audio_t", 0)) > 0:
                start, stop, segment_kind = take("ref_audio", alias)
                spans.append(WeightedSpan(start, stop, segment_kind, alias, audio_weight))
        elif kind in ("video", "video_audio"):
            if int(ref.get("ref_audio_t", 0)) > 0:
                start, stop, segment_kind = take("ref_audio", alias)
                spans.append(WeightedSpan(start, stop, segment_kind, alias, audio_weight))
            start, stop, segment_kind = take("ref_img", alias)
            spans.append(WeightedSpan(start, stop, segment_kind, alias, visual_weight))
        else:
            raise ValueError(f"Unsupported MiniMax reference block kind {kind!r} for @{alias}")
    try:
        first_target_index = next(i for i, segment in enumerate(segments) if segment[2] in ("audio", "video"))
    except StopIteration as exc:
        raise ValueError("Packed MiniMax layout has no target audio/video segments") from exc
    if cursor != first_target_index:
        leftovers = ", ".join(segment[2] for segment in segments[cursor:first_target_index]) or "unknown"
        raise ValueError(
            "Packed MiniMax layout/reference metadata mismatch: unclaimed prefix segment(s) "
            f"before the target: {leftovers}"
        )
    return spans


def target_start(layout_segments: Iterable[tuple[int, int, str]]) -> int:
    for start, _stop, kind in layout_segments:
        if kind in ("audio", "video"):
            return int(start)
    raise ValueError("Packed MiniMax layout has no target audio/video segments")


def weights_are_native(spans: Iterable[WeightedSpan]) -> bool:
    return all(span.weight == 1.0 for span in spans)

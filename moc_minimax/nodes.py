"""ComfyUI V3 nodes for richer MiniMax H3 reference conditioning."""

from __future__ import annotations

import importlib
import inspect
import math
from typing import Any

import torch

from comfy_api.latest import ComfyExtension, io, ui

from .core import (
    AUDIO_RELATIONSHIPS,
    AUDIO_ROLES,
    IMAGE_ROLES,
    PROMPT_MODES,
    PRIORITIES,
    VALIDATION_MODES,
    VIDEO_ROLES,
    Diagnostic,
    ReferencePlan,
    ReferencePlanError,
    compile_prompt,
    enforce,
    make_reference,
    manifest_json,
    plan_references,
    render_report,
    serializable_manifest,
    tag_mapping_text,
)
from .media import (
    MASK_POLARITIES,
    MASK_PRESENTATIONS,
    prepare_audio,
    prepare_image,
    prepare_video,
    trim_audio_to_duration,
)
from .weighting import install_reference_weight_patch
from .lora_nodes import MocH3LoadLoraNode, MocH3CompareLorasNode, MocH3MergeLorasNode


CATEGORY = "MiniMax H3/MOC References"
MocH3Reference = io.Custom("MOC_H3_REFERENCE")
MocH3ReferenceSet = io.Custom("MOC_H3_REFERENCE_SET")


def _round_canvas_axis(value: float) -> int:
    return max(32, round(value / 32) * 32)


def _resolved_image_canvas(
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
    detail: str,
) -> tuple[int, int]:
    if detail == "high_2048":
        scale = min(1.0, 2048.0 / min(source_width, source_height))
    else:
        scale = min(1.0, math.sqrt((output_width * output_height) / (source_width * source_height)))
    return _round_canvas_axis(source_width * scale), _round_canvas_axis(source_height * scale)


def _resolved_video_canvas(source_width: int, source_height: int) -> tuple[int, int]:
    ratio = source_width / source_height
    if ratio >= 1.0:
        nominal_width, nominal_height = 768.0 * ratio, 768.0
    else:
        nominal_width, nominal_height = 768.0, 768.0 / ratio
    if nominal_width * nominal_height > 768 * 1344:
        scale = math.sqrt((768 * 1344) / (nominal_width * nominal_height))
        nominal_width *= scale
        nominal_height *= scale
    canvas_width, canvas_height = _round_canvas_axis(nominal_width), _round_canvas_axis(nominal_height)
    if source_width * source_height < canvas_width * canvas_height:
        canvas_width, canvas_height = _round_canvas_axis(source_width), _round_canvas_axis(source_height)
    return canvas_width, canvas_height


def _priority_input() -> io.Combo.Input:
    return io.Combo.Input(
        "priority",
        options=list(PRIORITIES),
        default="supporting",
        tooltip=(
            "Semantic conflict priority. Primary wins conflicts; Supporting contributes within its scope; "
            "Weak is inspiration only; Disabled is excluded from encoding and numbering."
        ),
    )


def _scope_inputs() -> list[io.Input]:
    return [
        io.String.Input(
            "use_for",
            default="",
            multiline=True,
            tooltip="Optional exact scope. Overrides the role's default 'use for' instruction.",
        ),
        io.String.Input(
            "do_not_copy",
            default="",
            multiline=True,
            tooltip="Optional exclusion boundary. Overrides the role's default 'do not copy' instruction.",
        ),
    ]


def _weight_input(name: str = "signal_weight", default: float = 1.0, tooltip: str | None = None) -> io.Float.Input:
    return io.Float.Input(
        name,
        default=default,
        min=0.0,
        max=2.0,
        step=0.05,
        display_mode=io.NumberDisplay.slider,
        tooltip=tooltip or (
            "Experimental direct-signal weight. It is metadata until 'Apply Reference Weights [Experimental]' "
            "is connected to MODEL. 1.0 is native; semantic priority remains separate prompt guidance."
        ),
    )


def _summary(ref: dict[str, Any]) -> str:
    meta = ref.get("metadata", {})
    parts = [
        f"@{ref['name']}",
        ref["kind"],
        ref["role"],
        ref["priority"],
        f"visual={ref['visual_weight']:.2f}",
        f"audio={ref['audio_weight']:.2f}",
    ]
    if meta.get("processed_width") and meta.get("processed_height"):
        parts.append(f"{meta['processed_width']}x{meta['processed_height']}")
    if meta.get("processed_frames") is not None:
        parts.append(f"{meta['processed_frames']} frames")
    if meta.get("duration_s") is not None:
        parts.append(f"{meta['duration_s']:.2f}s")
    if meta.get("mask_applied"):
        parts.append(
            f"mask={meta.get('mask_coverage', 0.0) * 100.0:.1f}% "
            f"{meta.get('mask_presentation', 'neutral_fill')}"
        )
    warnings = meta.get("warnings", [])
    if warnings:
        parts.append(f"{len(warnings)} warning(s)")
    return " | ".join(parts)


class MocH3ImageReferenceNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MocH3ImageReference",
            display_name="MOC • H3 Image Reference",
            category=CATEGORY,
            description=(
                "Create a named, scoped MiniMax H3 image reference with semantic priority, optional source-isolation "
                "masking, and optional signal weight. Masks isolate source evidence; they do not control output placement."
            ),
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip=(
                        "Optional source-isolation mask. White keeps source pixels by default. The processed RGB image is "
                        "sent to both H3 reference encoders; this is not an output-region or inpainting mask."
                    ),
                ),
                io.String.Input("name", default="hero", tooltip="Stable prompt alias, used as @name. Do not include spaces."),
                io.Combo.Input(
                    "role",
                    options=list(IMAGE_ROLES),
                    default="identity",
                    tooltip="Semantic job compiled into guided prompt instructions; it is not a trained model switch.",
                ),
                _priority_input(),
                _weight_input(),
                io.Combo.Input(
                    "detail",
                    options=["inherit", "match_output", "high_2048"],
                    default="inherit",
                    tooltip=(
                        "match_output downsizes to the output pixel area while preserving aspect ratio; high_2048 keeps "
                        "up to a 2048px short edge and can be several times slower. Neither option upscales the source."
                    ),
                ),
                *_scope_inputs(),
                io.Combo.Input(
                    "mask_polarity",
                    options=list(MASK_POLARITIES),
                    default="white_keeps",
                    advanced=True,
                    optional=True,
                    tooltip="white_keeps retains white mask pixels; white_removes inverts that meaning.",
                ),
                io.Combo.Input(
                    "mask_presentation",
                    options=list(MASK_PRESENTATIONS),
                    default="neutral_fill",
                    advanced=True,
                    optional=True,
                    tooltip=(
                        "How excluded source pixels are shown to H3: neutral gray, a blurred source background, or a "
                        "padded crop around the retained region with neutral fill."
                    ),
                ),
                io.Int.Input(
                    "mask_expand_px",
                    default=0,
                    min=-128,
                    max=128,
                    step=1,
                    advanced=True,
                    optional=True,
                    tooltip="Grow positive or shrink negative the retained mask region before feathering.",
                ),
                io.Int.Input(
                    "mask_feather_px",
                    default=4,
                    min=0,
                    max=128,
                    step=1,
                    advanced=True,
                    optional=True,
                    tooltip="Approximate Gaussian feather radius applied after grow/shrink.",
                ),
                io.Float.Input(
                    "neutral_level",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                    optional=True,
                    tooltip="Grayscale value used outside the retained region for neutral-fill and crop modes.",
                ),
                io.Int.Input(
                    "background_blur_px",
                    default=32,
                    min=0,
                    max=128,
                    step=1,
                    advanced=True,
                    optional=True,
                    tooltip="Gaussian background blur radius used only by blur_background.",
                ),
                io.Float.Input(
                    "crop_padding_pct",
                    default=10.0,
                    min=0.0,
                    max=100.0,
                    step=1.0,
                    advanced=True,
                    optional=True,
                    tooltip="Padding around the retained mask bounds, as a percentage of the retained width and height.",
                ),
            ],
            outputs=[MocH3Reference.Output("reference"), io.String.Output("summary")],
        )

    @classmethod
    def execute(
        cls,
        image,
        name,
        role,
        priority,
        signal_weight,
        detail,
        use_for,
        do_not_copy,
        mask=None,
        mask_polarity="white_keeps",
        mask_presentation="neutral_fill",
        mask_expand_px=0,
        mask_feather_px=4,
        neutral_level=0.5,
        background_blur_px=32,
        crop_padding_pct=10.0,
    ):
        selected, metadata = prepare_image(
            image,
            mask=mask,
            mask_polarity=mask_polarity,
            mask_presentation=mask_presentation,
            mask_expand_px=mask_expand_px,
            mask_feather_px=mask_feather_px,
            neutral_level=neutral_level,
            background_blur_px=background_blur_px,
            crop_padding_pct=crop_padding_pct,
        )
        metadata["detail"] = detail
        if detail == "high_2048":
            metadata["warnings"].append("high_2048 retains substantially more reference tokens and increases runtime/VRAM pressure.")
        ref = make_reference(
            kind="image",
            name=name,
            media=selected,
            role=role,
            priority=priority,
            use_for=use_for,
            do_not_copy=do_not_copy,
            visual_weight=signal_weight,
            metadata=metadata,
        )
        return io.NodeOutput(ref, _summary(ref))


class MocH3VideoReferenceNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MocH3VideoReference",
            display_name="MOC • H3 Video Reference",
            category=CATEGORY,
            description="Create a named motion/camera/performance reference, with deterministic trim and 24 fps preparation.",
            inputs=[
                io.Image.Input("frames", tooltip="Reference video frames as an IMAGE batch."),
                io.String.Input("name", default="motion", tooltip="Stable prompt alias, used as @name."),
                io.Combo.Input(
                    "role",
                    options=list(VIDEO_ROLES),
                    default="motion",
                    tooltip="Semantic job compiled into guided prompt instructions; it is not a trained model switch.",
                ),
                _priority_input(),
                _weight_input("visual_weight"),
                _weight_input(
                    "audio_weight",
                    tooltip="Experimental weight for the paired soundtrack. Ignored when no soundtrack is connected.",
                ),
                io.Float.Input(
                    "input_fps",
                    default=24.0,
                    min=0.01,
                    max=240.0,
                    step=0.01,
                    tooltip="FPS represented by the incoming IMAGE batch after any loader sampling. A wrong value changes motion speed.",
                ),
                io.Float.Input(
                    "trim_start_s",
                    default=0.0,
                    min=0.0,
                    max=3600.0,
                    step=0.05,
                    advanced=True,
                    tooltip="Start time in the original video timebase. trim_to_video applies the same offset to the soundtrack.",
                ),
                io.Float.Input(
                    "max_duration_s",
                    default=15.0,
                    min=0.0,
                    max=3600.0,
                    step=0.05,
                    tooltip="0 keeps everything after trim_start_s. H3 recommends no more than 15 seconds.",
                    advanced=True,
                ),
                io.Audio.Input("soundtrack", optional=True, tooltip="Optional soundtrack paired with this reference video."),
                io.Combo.Input(
                    "soundtrack_policy",
                    options=["trim_to_video", "keep", "ignore"],
                    default="trim_to_video",
                    advanced=True,
                    tooltip=(
                        "trim_to_video applies the video offset and exact native encoded duration; keep retains the whole "
                        "audio file; ignore excludes it. A connected soundtrack gets the generated alias @name_audio."
                    ),
                ),
                io.Combo.Input(
                    "soundtrack_role",
                    options=list(AUDIO_ROLES),
                    default="timing",
                    advanced=True,
                    tooltip="Semantic job assigned to a connected soundtrack in guided prompting.",
                ),
                io.Combo.Input(
                    "soundtrack_relationship",
                    options=list(AUDIO_RELATIONSHIPS),
                    default="reference",
                    advanced=True,
                    tooltip=(
                        "How guided prompting asks H3 to use the audio signal. Copy modes request authorized signal reuse; "
                        "they are generative instructions, not a byte-exact audio editor."
                    ),
                ),
                *_scope_inputs(),
            ],
            outputs=[MocH3Reference.Output("reference"), io.String.Output("summary")],
        )

    @classmethod
    def execute(
        cls,
        frames,
        name,
        role,
        priority,
        visual_weight,
        audio_weight,
        input_fps,
        trim_start_s,
        max_duration_s,
        soundtrack_policy,
        soundtrack_role,
        soundtrack_relationship,
        use_for,
        do_not_copy,
        soundtrack=None,
    ):
        prepared_frames, metadata = prepare_video(
            frames,
            input_fps=input_fps,
            trim_start_s=trim_start_s,
            max_duration_s=max_duration_s,
        )
        prepared_soundtrack = None
        if soundtrack is not None and soundtrack_policy != "ignore":
            if soundtrack_policy == "trim_to_video":
                prepared_soundtrack, audio_meta = trim_audio_to_duration(
                    soundtrack,
                    trim_start_s=trim_start_s,
                    duration_s=metadata["duration_s"],
                )
            else:
                prepared_soundtrack, audio_meta = prepare_audio(soundtrack)
            metadata["soundtrack"] = audio_meta
            metadata["warnings"].extend(f"Soundtrack: {warning}" for warning in audio_meta.get("warnings", []))
        elif soundtrack is not None:
            metadata["warnings"].append("A soundtrack was connected but soundtrack_policy=ignore, so it is excluded.")
        metadata["soundtrack_policy"] = soundtrack_policy
        metadata["soundtrack_role"] = soundtrack_role
        metadata["soundtrack_relationship"] = soundtrack_relationship
        ref = make_reference(
            kind="video",
            name=name,
            media=prepared_frames,
            soundtrack=prepared_soundtrack,
            role=role,
            priority=priority,
            use_for=use_for,
            do_not_copy=do_not_copy,
            visual_weight=visual_weight,
            audio_weight=audio_weight,
            metadata=metadata,
        )
        return io.NodeOutput(ref, _summary(ref))


class MocH3AudioReferenceNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MocH3AudioReference",
            display_name="MOC • H3 Audio Reference",
            category=CATEGORY,
            description="Create a named voice, music, ambience, sound-effect, or timing reference.",
            inputs=[
                io.Audio.Input("audio"),
                io.String.Input("name", default="voice", tooltip="Stable prompt alias, used as @name."),
                io.Combo.Input(
                    "role",
                    options=list(AUDIO_ROLES),
                    default="voice",
                    tooltip="Semantic job compiled into guided prompt instructions; it is not a trained model switch.",
                ),
                _priority_input(),
                io.Combo.Input(
                    "relationship",
                    options=list(AUDIO_RELATIONSHIPS),
                    default="reference",
                    tooltip=(
                        "reference transfers audible attributes without copying the signal. Copy modes request authorized "
                        "signal reuse; they are generative instructions, not a byte-exact audio editor."
                    ),
                ),
                _weight_input(),
                io.Float.Input("trim_start_s", default=0.0, min=0.0, max=3600.0, step=0.05, advanced=True),
                io.Float.Input(
                    "max_duration_s",
                    default=15.0,
                    min=0.0,
                    max=3600.0,
                    step=0.05,
                    tooltip="0 keeps everything after trim_start_s. H3 recommends no more than 15 seconds.",
                    advanced=True,
                ),
                *_scope_inputs(),
            ],
            outputs=[MocH3Reference.Output("reference"), io.String.Output("summary")],
        )

    @classmethod
    def execute(
        cls,
        audio,
        name,
        role,
        priority,
        relationship,
        signal_weight,
        trim_start_s,
        max_duration_s,
        use_for,
        do_not_copy,
    ):
        prepared, metadata = prepare_audio(audio, trim_start_s=trim_start_s, max_duration_s=max_duration_s)
        ref = make_reference(
            kind="audio",
            name=name,
            media=prepared,
            role=role,
            priority=priority,
            use_for=use_for,
            do_not_copy=do_not_copy,
            audio_weight=signal_weight,
            audio_relationship=relationship,
            metadata=metadata,
        )
        return io.NodeOutput(ref, _summary(ref))


def _reference_values(references: dict[str, Any] | None) -> list[dict[str, Any]]:
    return [value for value in (references or {}).values() if value is not None]


def _set_record(references: list[dict[str, Any]]) -> dict[str, Any]:
    return {"_type": "moc_h3_reference_set", "schema_version": 1, "references": references}


def _references_from_set(reference_set: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(reference_set, dict) or reference_set.get("_type") != "moc_h3_reference_set":
        raise ReferencePlanError("Expected an MOC H3 Reference Set. Connect the output of 'MOC • H3 Reference Set'.")
    references = reference_set.get("references")
    if not isinstance(references, list):
        raise ReferencePlanError("Reference Set is malformed: 'references' must be a list.")
    return references


class MocH3ReferenceSetNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MocH3ReferenceSet",
            display_name="MOC • H3 Reference Set",
            category=CATEGORY,
            description="Collect named references, validate them, and resolve stable aliases to MiniMax's native tags.",
            inputs=[
                io.Autogrow.Input(
                    "references",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=MocH3Reference.Input("reference"),
                        prefix="reference_",
                        min=0,
                        max=15,
                    ),
                ),
            ],
            outputs=[
                MocH3ReferenceSet.Output("reference_set"),
                io.String.Output("mapping"),
                io.String.Output("report"),
                io.String.Output("manifest_json"),
                io.Boolean.Output("valid"),
            ],
        )

    @classmethod
    def execute(cls, references=None):
        values = _reference_values(references)
        plan = plan_references(values)
        report = render_report(plan)
        return io.NodeOutput(
            _set_record(values),
            tag_mapping_text(plan),
            report,
            manifest_json(plan),
            plan.valid,
            ui=ui.PreviewText(report),
        )


def _resolved_target_frames(length: int) -> int:
    target_frames = max(5, int(length))
    while target_frames % 17 != 5:
        target_frames += 1
    return target_frames


def _native_video_frame_count(ref: dict[str, Any], target_frames: int) -> int:
    prepared = int(ref.get("metadata", {}).get("processed_frames", 0))
    encoded = min(prepared, target_frames)
    while encoded >= 5 and encoded % 17 != 5:
        encoded -= 1
    return encoded


def _native_video_latent_frames(frame_count: int) -> int:
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def _estimate_native_metadata(
    plan: ReferencePlan,
    width: int,
    height: int,
    length: int,
    default_image_detail: str,
) -> tuple[dict[str, str], str]:
    """Populate read-only native sizing/cost estimates used by reports."""
    images = [ref for ref in plan.native_order if ref["kind"] == "image"]
    desired_detail = {
        ref["name"]: (
            default_image_detail
            if ref.get("metadata", {}).get("detail", "inherit") == "inherit"
            else ref["metadata"]["detail"]
        )
        for ref in images
    }
    native_image_size = "max" if any(value == "high_2048" for value in desired_detail.values()) else "match"
    for ref in images:
        image = ref["media"]
        native_width, native_height = _resolved_image_canvas(
            int(image.shape[2]),
            int(image.shape[1]),
            width,
            height,
            desired_detail[ref["name"]],
        )
        ref["metadata"].update({
            "resolved_detail": desired_detail[ref["name"]],
            "native_width": native_width,
            "native_height": native_height,
            "estimated_visual_tokens": (native_width // 32) * (native_height // 32),
        })

    target_frames = _resolved_target_frames(length)
    for ref in plan.native_order:
        if ref["kind"] != "video":
            continue
        native_frames = _native_video_frame_count(ref, target_frames)
        native_width, native_height = _resolved_video_canvas(
            int(ref["media"].shape[2]),
            int(ref["media"].shape[1]),
        )
        ref["metadata"].update({
            "native_width": native_width,
            "native_height": native_height,
            "native_encoded_frames": native_frames,
            "native_duration_s": native_frames / 24.0,
            "estimated_visual_tokens": (
                (native_width // 32)
                * (native_height // 32)
                * (_native_video_latent_frames(native_frames) if native_frames >= 5 else 0)
            ),
        })
        soundtrack = ref.get("metadata", {}).get("soundtrack")
        if isinstance(soundtrack, dict) and soundtrack.get("duration_s") is not None:
            prepared_duration = float(soundtrack["duration_s"])
            if ref["metadata"].get("soundtrack_policy", "trim_to_video") == "trim_to_video":
                native_soundtrack_duration = min(prepared_duration, native_frames / 24.0)
            else:
                native_soundtrack_duration = prepared_duration
            ref["metadata"]["native_soundtrack_duration_s"] = native_soundtrack_duration
            ref["metadata"]["estimated_audio_tokens"] = round(native_soundtrack_duration * 40.0) * 2
    for ref in plan.native_order:
        if ref["kind"] != "audio":
            continue
        duration = float(ref.get("metadata", {}).get("duration_s", 0.0))
        ref["metadata"]["estimated_audio_tokens"] = round(duration * 40.0) * 2
    return desired_detail, native_image_size


def _limit_diagnostics(plan: ReferencePlan, length: int) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    target_frames = _resolved_target_frames(length)
    if target_frames > 362:
        diagnostics.append(Diagnostic(
            "warning",
            "duration_range",
            "Resolved output exceeds the documented ~15 second trained range.",
        ))
    video_duration = sum(
        max(0, _native_video_frame_count(ref, target_frames)) / 24.0
        for ref in plan.active
        if ref["kind"] == "video"
    )
    standalone_audio_duration = sum(
        float(ref.get("metadata", {}).get("duration_s", 0.0))
        for ref in plan.active
        if ref["kind"] == "audio"
    )
    soundtrack_duration = 0.0
    for ref in plan.active:
        if ref["kind"] != "video" or ref.get("soundtrack") is None:
            continue
        metadata = ref.get("metadata", {})
        source_audio_duration = float(metadata.get("soundtrack", {}).get("duration_s", 0.0))
        encoded_frames = _native_video_frame_count(ref, target_frames)
        encoded_duration = max(0, encoded_frames) / 24.0
        policy = metadata.get("soundtrack_policy", "trim_to_video")
        effective_audio_duration = (
            min(source_audio_duration, encoded_duration)
            if policy == "trim_to_video" and encoded_duration > 0
            else source_audio_duration
        )
        soundtrack_duration += effective_audio_duration
        if source_audio_duration + (1.0 / 24.0) < encoded_duration:
            diagnostics.append(Diagnostic(
                "warning",
                "soundtrack_shorter_than_video",
                f"Paired soundtrack is {source_audio_duration:.2f}s but the native reference video is "
                f"{encoded_duration:.2f}s; the end will have no paired reference audio.",
                ref["name"],
            ))
        elif policy == "keep" and abs(source_audio_duration - encoded_duration) > (1.0 / 24.0):
            diagnostics.append(Diagnostic(
                "warning",
                "soundtrack_duration_mismatch",
                f"soundtrack_policy=keep retains {source_audio_duration:.2f}s beside a "
                f"{encoded_duration:.2f}s native reference video.",
                ref["name"],
            ))
    audio_duration = standalone_audio_duration + soundtrack_duration
    if video_duration > 15.0 + 1e-6:
        diagnostics.append(Diagnostic(
            "warning",
            "effective_total_video_duration",
            f"Native-grid reference videos total {video_duration:.2f}s; H3 recommends at most 15s.",
        ))
    if audio_duration > 15.0 + 1e-6:
        diagnostics.append(Diagnostic(
            "warning",
            "effective_total_audio_duration",
            f"Native-duration reference audio including paired soundtracks totals {audio_duration:.2f}s; "
            "H3 recommends at most 15s.",
        ))
    for ref in plan.active:
        if ref["kind"] != "video":
            continue
        prepared = int(ref.get("metadata", {}).get("processed_frames", 0))
        if prepared < 5:
            diagnostics.append(Diagnostic("error", "video_too_short", "Native H3 requires at least 5 processed video frames.", ref["name"]))
            continue
        encoded = _native_video_frame_count(ref, target_frames)
        if encoded < 5:
            diagnostics.append(Diagnostic("error", "video_grid_empty", "Target-length capping leaves no valid H3 reference-video grid.", ref["name"]))
        elif encoded != prepared:
            diagnostics.append(Diagnostic(
                "warning",
                "video_frame_resolution",
                f"Native H3 will encode {encoded} of {prepared} prepared frames after target-length capping and 17k+5 grid alignment.",
                ref["name"],
            ))
    return diagnostics


def _compile(
    reference_set: dict[str, Any],
    prompt: str,
    prompt_mode: str,
    validation_mode: str,
    width: int,
    height: int,
    length: int,
    default_image_detail: str = "match_output",
    *,
    enforce_mode: bool = True,
) -> tuple[ReferencePlan, str, list[Diagnostic], str]:
    plan = plan_references(_references_from_set(reference_set))
    _estimate_native_metadata(plan, width, height, length, default_image_detail)
    if plan.errors:
        compiled, prompt_diagnostics = str(prompt or "").strip(), []
    else:
        compiled, prompt_diagnostics = compile_prompt(prompt, plan, prompt_mode)
    detail_diagnostics = [
        Diagnostic(
            "warning",
            "high_image_detail",
            "Inherited high_2048 detail retains substantially more reference tokens and increases runtime/VRAM pressure.",
            ref["name"],
        )
        for ref in plan.active
        if ref.get("kind") == "image"
        and ref.get("metadata", {}).get("detail", "inherit") == "inherit"
        and ref.get("metadata", {}).get("resolved_detail") == "high_2048"
    ]
    extra = [*prompt_diagnostics, *detail_diagnostics, *_limit_diagnostics(plan, length)]
    if enforce_mode:
        enforce(plan, extra, validation_mode)
    report = render_report(
        plan,
        compiled_prompt=compiled,
        width=width,
        height=height,
        length=length,
        extra_diagnostics=extra,
    )
    return plan, compiled, extra, report


class MocH3CompilePromptNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MocH3CompileReferencePrompt",
            display_name="MOC • H3 Compile Reference Prompt",
            category=CATEGORY,
            description="Resolve @aliases, generate scoped reference instructions, and inspect diagnostics without encoding media.",
            inputs=[
                MocH3ReferenceSet.Input("reference_set"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Combo.Input("prompt_mode", options=list(PROMPT_MODES), default="guided"),
                io.Combo.Input("validation_mode", options=list(VALIDATION_MODES), default="warn"),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17),
                io.Combo.Input(
                    "default_image_detail",
                    options=["match_output", "high_2048"],
                    default="match_output",
                    tooltip="Must match Reference to Video+ for representative native canvas/token estimates.",
                ),
            ],
            outputs=[
                io.String.Output("compiled_prompt"),
                io.String.Output("report"),
                io.String.Output("mapping"),
                io.String.Output("manifest_json"),
                io.Boolean.Output("valid"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        reference_set,
        prompt,
        prompt_mode,
        validation_mode,
        width,
        height,
        length,
        default_image_detail,
    ):
        plan, compiled, extra, report = _compile(
            reference_set,
            prompt,
            prompt_mode,
            validation_mode,
            width,
            height,
            length,
            default_image_detail,
            enforce_mode=False,
        )
        diagnostics = [*plan.diagnostics, *extra]
        valid = not any(item.level == "error" for item in diagnostics)
        if validation_mode == "strict" and any(item.level == "warning" for item in diagnostics):
            valid = False
        return io.NodeOutput(
            compiled,
            report,
            tag_mapping_text(plan),
            manifest_json(plan, extra, validation_mode),
            valid,
            ui=ui.PreviewText(report),
        )


def _resize_down_to_area(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    import comfy.utils

    h, w = int(image.shape[1]), int(image.shape[2])
    scale = min(1.0, math.sqrt((width * height) / (w * h)))
    target_w = max(32, round(w * scale / 32) * 32)
    target_h = max(32, round(h * scale / 32) * 32)
    if target_w == w and target_h == h:
        return image
    samples = image[..., :3].movedim(-1, 1)
    return comfy.utils.common_upscale(samples, target_w, target_h, "lanczos", "disabled").movedim(1, -1)


def _native_inputs(plan: ReferencePlan, width: int, height: int, length: int, default_image_detail: str):
    images = [ref for ref in plan.native_order if ref["kind"] == "image"]
    videos = [ref for ref in plan.native_order if ref["kind"] == "video"]
    audios = [ref for ref in plan.native_order if ref["kind"] == "audio"]
    desired_detail, native_image_size = _estimate_native_metadata(
        plan,
        width,
        height,
        length,
        default_image_detail,
    )
    image_inputs: dict[str, torch.Tensor] = {}
    for index, ref in enumerate(images):
        image = ref["media"]
        if native_image_size == "max" and desired_detail[ref["name"]] == "match_output":
            image = _resize_down_to_area(image, width, height)
        image_inputs[f"ref_image_{index}"] = image
    video_inputs = {}
    video_audio_inputs: dict[str, dict[str, Any]] = {}
    target_frames = _resolved_target_frames(length)
    for index, ref in enumerate(videos):
        video_inputs[f"ref_video_{index}"] = ref["media"]
        soundtrack = ref.get("soundtrack")
        if soundtrack is None:
            continue
        if ref.get("metadata", {}).get("soundtrack_policy", "trim_to_video") == "trim_to_video":
            encoded_frames = _native_video_frame_count(ref, target_frames)
            if encoded_frames >= 5:
                soundtrack, native_soundtrack_metadata = trim_audio_to_duration(
                    soundtrack,
                    trim_start_s=0.0,
                    duration_s=encoded_frames / 24.0,
                )
                ref["metadata"].update({
                    "native_soundtrack_duration_s": native_soundtrack_metadata["duration_s"],
                    "native_soundtrack_samples": native_soundtrack_metadata["processed_samples"],
                    "estimated_audio_tokens": round(native_soundtrack_metadata["duration_s"] * 40.0) * 2,
                })
        video_audio_inputs[f"ref_video_audio_{index}"] = soundtrack
    audio_inputs = {f"ref_audio_{index}": ref["media"] for index, ref in enumerate(audios)}
    controls = [*images, *videos, *audios]
    return native_image_size, image_inputs, video_inputs, video_audio_inputs, audio_inputs, controls


def _resize_temporal_frame(image: torch.Tensor, width: int, height: int, crop: str) -> torch.Tensor:
    """Match the native FL2VA first/last-frame resize policy."""
    import comfy.utils

    samples = image[:1, ..., :3].movedim(-1, 1)
    return comfy.utils.common_upscale(samples, width, height, "lanczos", crop).movedim(1, -1)


def _prepare_temporal_keyframes(
    vae,
    width: int,
    height: int,
    length: int,
    first_frame: torch.Tensor | None = None,
    last_frame: torch.Tensor | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Encode temporal anchors separately from the non-temporal reference bank."""
    frame_count = _resolved_target_frames(length)
    keyframes: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    if first_frame is not None:
        image = _resize_temporal_frame(first_frame, width, height, "disabled")
        keyframes.append({"resolved_frame_index": 0, "latent": vae.encode(image)})
        summary.append({"position": "first", "resolved_frame_index": 0})
    if last_frame is not None:
        image = _resize_temporal_frame(last_frame, width, height, "center")
        keyframes.append({"resolved_frame_index": frame_count - 1, "latent": vae.encode(image)})
        summary.append({"position": "last", "resolved_frame_index": frame_count - 1})
    return keyframes, summary


def _attach_temporal_keyframes(conditioning, keyframes, summary, frame_count: int):
    if not keyframes:
        return conditioning
    output = []
    for entry in conditioning:
        embedding, metadata = entry[0], dict(entry[1])
        metadata["minimax_keyframes"] = [*metadata.get("minimax_keyframes", []), *keyframes]
        # Retained for compatibility with ComfyUI builds whose H3 payload uses
        # the explicit pixel-frame count when resolving the final anchor.
        metadata["minimax_frame_count"] = int(frame_count)
        metadata["moc_h3_temporal_guides"] = [dict(item) for item in summary]
        copied_entry = list(entry)
        copied_entry[0] = embedding
        copied_entry[1] = metadata
        output.append(copied_entry)
    return output


def _annotate_conditioning(
    conditioning,
    controls: list[dict[str, Any]],
    plan: ReferencePlan,
    visual_fidelity: float,
    audio_fidelity: float,
    extra_diagnostics: list[Diagnostic] | tuple[Diagnostic, ...] = (),
    validation_mode: str | None = None,
):
    output = []
    manifest = serializable_manifest(plan, extra_diagnostics, validation_mode)
    for entry in conditioning:
        embedding, metadata = entry[0], dict(entry[1])
        blocks = metadata.get("minimax_refs")
        if blocks is not None:
            if len(blocks) != len(controls):
                raise RuntimeError(
                    f"Native MiniMax encoded {len(blocks)} reference blocks, but MOC prepared {len(controls)} controls. "
                    "Update ComfyUI and report this compatibility mismatch."
                )
            annotated = []
            for block, control in zip(blocks, controls):
                copied = dict(block)
                copied.update({
                    "moc_name": control["name"],
                    "moc_kind": control["kind"],
                    "moc_role": control["role"],
                    "moc_priority": control["priority"],
                    "moc_visual_weight": float(control.get("visual_weight", 1.0)),
                    "moc_audio_weight": float(control.get("audio_weight", 1.0)),
                })
                annotated.append(copied)
            metadata["minimax_refs"] = annotated
        metadata["moc_h3_reference_manifest"] = manifest
        metadata["minimax_visual_cond_noise_aug"] = float(visual_fidelity)
        metadata["minimax_audio_cond_noise_aug"] = float(audio_fidelity)
        copied_entry = list(entry)
        copied_entry[0] = embedding
        copied_entry[1] = metadata
        output.append(copied_entry)
    return output


def _native_node_class():
    try:
        module = importlib.import_module("comfy_extras.nodes_minimax_h3")
    except ModuleNotFoundError as exc:
        if exc.name != "comfy_extras.nodes_minimax_h3":
            raise RuntimeError(
                f"Native MiniMax H3 could not load its dependency {exc.name!r}. Repair the ComfyUI environment."
            ) from exc
        raise RuntimeError(
            "MOC MiniMax requires native MiniMax H3 support from ComfyUI 0.32.0 or newer. Update ComfyUI first."
        ) from exc
    native = getattr(module, "MiniMaxH3ReferenceToVideo", None)
    if native is None:
        raise RuntimeError(
            "This ComfyUI build does not expose MiniMaxH3ReferenceToVideo. Update ComfyUI to a compatible release."
        )
    required = {
        "clip", "vae", "audio_vae", "prompt", "width", "height", "length",
        "ref_image_size", "ref_images", "ref_videos", "ref_video_audios", "ref_audios",
    }
    parameters = set(inspect.signature(native.execute).parameters)
    missing = sorted(required - parameters)
    if missing:
        raise RuntimeError(
            "Native MiniMax H3 has an incompatible execute signature; missing "
            + ", ".join(missing)
            + ". Update MOC MiniMax and ComfyUI together."
        )
    return native


def _node_output_values(result) -> tuple[Any, ...]:
    """Normalize V3 NodeOutput (and legacy tuple-like results) from delegation."""
    values = result.result if hasattr(result, "result") else result
    if not isinstance(values, (tuple, list)) or len(values) < 2:
        raise RuntimeError(
            "Native MiniMax H3 returned an unexpected result. Update this extension and ComfyUI together."
        )
    return tuple(values)


class MocH3ReferenceToVideoPlusNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MocH3ReferenceToVideoPlus",
            display_name="MOC • H3 Reference to Video+",
            category=CATEGORY,
            description=(
                "Feature-rich replacement for native MiniMax H3 R2V: stable aliases, roles, semantic priorities, "
                "per-reference detail, validation, diagnostics, and optional experimental numeric weights."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                MocH3ReferenceSet.Input("reference_set"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17),
                io.Combo.Input("prompt_mode", options=list(PROMPT_MODES), default="guided"),
                io.Combo.Input("validation_mode", options=list(VALIDATION_MODES), default="warn"),
                io.Combo.Input(
                    "default_image_detail",
                    options=["match_output", "high_2048"],
                    default="match_output",
                    tooltip="Default for image references whose detail setting is inherit.",
                ),
                io.Float.Input(
                    "visual_reference_fidelity",
                    default=0.999,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    advanced=True,
                    tooltip="Global visual condition fidelity. Lower values add noise to all direct visual reference latents.",
                ),
                io.Float.Input(
                    "audio_reference_fidelity",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    advanced=True,
                    tooltip="Global audio condition fidelity. Lower values add noise to all direct audio reference latents.",
                ),
            ],
            outputs=[
                io.Conditioning.Output("positive"),
                io.Latent.Output("latent"),
                io.String.Output("compiled_prompt"),
                io.String.Output("report"),
                io.String.Output("manifest_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        vae,
        audio_vae,
        reference_set,
        prompt,
        width,
        height,
        length,
        prompt_mode,
        validation_mode,
        default_image_detail,
        visual_reference_fidelity,
        audio_reference_fidelity,
    ):
        plan, compiled, _extra, report = _compile(
            reference_set,
            prompt,
            prompt_mode,
            validation_mode,
            width,
            height,
            length,
            default_image_detail,
        )
        native_size, images, videos, video_audios, audios, controls = _native_inputs(
            plan, width, height, length, default_image_detail
        )
        report = render_report(
            plan,
            compiled_prompt=compiled,
            width=width,
            height=height,
            length=length,
            extra_diagnostics=_extra,
        )
        native = _native_node_class()
        result = native.execute(
            clip,
            vae,
            audio_vae,
            compiled,
            width,
            height,
            length,
            ref_image_size=native_size,
            ref_images=images,
            ref_videos=videos,
            ref_video_audios=video_audios,
            ref_audios=audios,
        )
        conditioning, latent = _node_output_values(result)[:2]
        conditioning = _annotate_conditioning(
            conditioning,
            controls,
            plan,
            visual_reference_fidelity,
            audio_reference_fidelity,
            _extra,
            validation_mode,
        )
        return io.NodeOutput(
            conditioning,
            latent,
            compiled,
            report,
            manifest_json(plan, _extra, validation_mode),
            ui=ui.PreviewText(report),
        )


class MocH3HybridImageReferencesToVideoNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MocH3HybridImageReferencesToVideo",
            display_name="MOC • H3 Image to Video + References",
            category=CATEGORY,
            description=(
                "The native MiniMax H3 Image to Video interface plus ordinary IMAGE inputs for independent, "
                "non-temporal references. First/last images remain temporal keyframes."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17),
                io.Image.Input(
                    "first_frame",
                    optional=True,
                    tooltip=(
                        "Native Image to Video first-frame input. The first image in the batch is stretched to the "
                        "output canvas and fixed to frame 0; it is not added to the reference bank."
                    ),
                ),
                io.Image.Input(
                    "last_frame",
                    optional=True,
                    tooltip=(
                        "Native Image to Video last-frame input. The first image in the batch is cover-cropped and "
                        "fixed to the final frame; it is not added to the reference bank."
                    ),
                ),
                io.Combo.Input(
                    "ref_image_size",
                    options=["match", "max"],
                    default="match",
                    tooltip=(
                        "Reference sizing passed to native H3. 'match' limits each reference to the output pixel area; "
                        "'max' retains up to the native 2048px short-edge reference canvas."
                    ),
                ),
                io.Autogrow.Input(
                    "reference_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "reference_image",
                            tooltip="Ordinary IMAGE input used as a non-temporal H3 reference, not as an output frame.",
                        ),
                        prefix="reference_image_",
                        min=0,
                        max=9,
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output("positive"),
                io.Latent.Output("latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        vae,
        prompt,
        width,
        height,
        length,
        ref_image_size,
        first_frame=None,
        last_frame=None,
        reference_images=None,
    ):
        refs = {
            f"ref_image_{index}": image
            for index, image in enumerate(
                value for value in (reference_images or {}).values() if value is not None
            )
        }
        native = _native_node_class()
        native_result = native.execute(
            clip,
            vae,
            vae,  # The native signature requires audio_vae; image-only references never use it.
            prompt,
            width,
            height,
            length,
            ref_image_size=ref_image_size,
            ref_images=refs,
            ref_videos={},
            ref_video_audios={},
            ref_audios={},
        )
        conditioning, latent = _node_output_values(native_result)[:2]
        keyframes, temporal_summary = _prepare_temporal_keyframes(
            vae,
            width,
            height,
            length,
            first_frame=first_frame,
            last_frame=last_frame,
        )
        conditioning = _attach_temporal_keyframes(
            conditioning,
            keyframes,
            temporal_summary,
            _resolved_target_frames(length),
        )
        return io.NodeOutput(conditioning, latent)


class MocH3ApplyReferenceWeightsNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MocH3ApplyReferenceWeights",
            display_name="MOC • H3 Apply Reference Weights [Experimental]",
            category=f"{CATEGORY}/Experimental",
            description=(
                "Opt-in model patch for numeric reference weights. It affects direct packed VAE reference tokens, "
                "not Qwen's semantic vision tokens. 1.0 is native and semantic priority remains active."
            ),
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "method",
                    options=["value_gate", "attention_prior"],
                    default="value_gate",
                    tooltip=(
                        "value_gate is backend-friendly and scales direct value contribution. attention_prior applies "
                        "log(weight) to reference keys for target queries, but forces PyTorch SDPA and may be slower."
                    ),
                ),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="0 is a native no-op; 1 applies requested weights; values above 1 exaggerate them.",
                ),
            ],
            outputs=[io.Model.Output("model"), io.String.Output("status")],
        )

    @classmethod
    def execute(cls, model, method, strength):
        patched = install_reference_weight_patch(model, method=method, strength=strength)
        status = (
            f"Experimental MiniMax H3 reference weighting enabled: method={method}, strength={strength:.2f}. "
            "The patch reads weights stored by MOC H3 Reference to Video+ at sampling time. "
            "Qwen semantic interpretation is influenced separately by the compiled priority/role prompt."
        )
        return io.NodeOutput(patched, status)


class MocH3Extension(ComfyExtension):
    async def get_node_list(self):
        return [
            MocH3LoadLoraNode,
            MocH3CompareLorasNode,
            MocH3MergeLorasNode,
            MocH3ImageReferenceNode,
            MocH3VideoReferenceNode,
            MocH3AudioReferenceNode,
            MocH3ReferenceSetNode,
            MocH3CompilePromptNode,
            MocH3ReferenceToVideoPlusNode,
            MocH3HybridImageReferencesToVideoNode,
            MocH3ApplyReferenceWeightsNode,
        ]

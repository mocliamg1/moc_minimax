"""Pure-Python reference planning, validation, prompting, and diagnostics.

This module deliberately has no ComfyUI or torch imports so its most important
behaviour can be tested without loading a model runtime.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable


SCHEMA_VERSION = 1
ALIAS_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")
# Parse the whole alias-like token so overlong typos become diagnostics rather
# than partial replacements. Explicit aliases are still validated at 32 chars;
# generated soundtrack aliases may append ``_audio``.
ALIAS_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_-])@([A-Za-z][A-Za-z0-9_-]*)")
NATIVE_TAG_RE = re.compile(r"<(Picture|Video|Audio)\s+([1-9][0-9]*)>", re.IGNORECASE)

KINDS = ("image", "video", "audio")
PRIORITIES = ("primary", "supporting", "weak", "disabled")
PROMPT_MODES = ("guided", "aliases_only", "manual")
VALIDATION_MODES = ("strict", "warn")

IMAGE_ROLES = (
    "identity",
    "object",
    "style",
    "composition",
    "environment",
    "lighting_color",
    "general",
)
VIDEO_ROLES = (
    "motion",
    "camera",
    "performance",
    "identity",
    "style",
    "environment",
    "full_reference",
    "general",
)
AUDIO_ROLES = (
    "voice",
    "prosody_dialogue",
    "music",
    "ambience",
    "sfx",
    "timing",
    "general",
)
AUDIO_RELATIONSHIPS = ("reference", "fully_copy", "partially_copy", "weak_reference")


class ReferencePlanError(ValueError):
    """Raised when a reference plan cannot be compiled safely."""


@dataclass(slots=True)
class Diagnostic:
    level: str
    code: str
    message: str
    alias: str | None = None

    def render(self) -> str:
        scope = f" @{self.alias}" if self.alias else ""
        return f"[{self.level.upper()}] {self.code}{scope}: {self.message}"


@dataclass(slots=True)
class ReferencePlan:
    references: list[dict[str, Any]]
    active: list[dict[str, Any]]
    native_order: list[dict[str, Any]]
    alias_to_tag: dict[str, str]
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def errors(self) -> list[Diagnostic]:
        return [item for item in self.diagnostics if item.level == "error"]

    @property
    def warnings(self) -> list[Diagnostic]:
        return [item for item in self.diagnostics if item.level == "warning"]

    @property
    def valid(self) -> bool:
        return not self.errors


ROLE_GUIDANCE: dict[tuple[str, str], tuple[str, str]] = {
    ("image", "identity"): (
        "identity-defining facial features, hair, body proportions, and recurring wardrobe details",
        "pose, camera angle, background, and lighting unless the scene explicitly requests them",
    ),
    ("image", "object"): (
        "the object's identity, silhouette, materials, markings, and defining construction details",
        "the source pose, background, camera angle, and lighting unless explicitly requested",
    ),
    ("image", "style"): (
        "rendering style, palette, texture language, and lighting character",
        "the source person, object identity, pose, composition, and scene content",
    ),
    ("image", "composition"): (
        "spatial layout, framing, scale relationships, and visual balance",
        "the source identities, wardrobe, detailed textures, and narrative action",
    ),
    ("image", "environment"): (
        "environment design, architecture, props, weather, and location character",
        "the source people, poses, and camera framing unless explicitly requested",
    ),
    ("image", "lighting_color"): (
        "lighting direction, contrast, exposure character, and color palette",
        "the source identity, pose, composition, and scene contents",
    ),
    ("video", "motion"): (
        "action, gait, timing, physical rhythm, and motion trajectory",
        "the source actor, face, wardrobe, environment, visual style, and camera unless explicitly requested",
    ),
    ("video", "camera"): (
        "camera path, framing changes, lens behaviour, pacing, and shot rhythm",
        "the source actor, action, wardrobe, setting, and visual style",
    ),
    ("video", "performance"): (
        "gesture, facial performance, body language, timing, and emotional delivery",
        "the source performer's identity, wardrobe, environment, and camera treatment",
    ),
    ("video", "identity"): (
        "the recurring subject identity and appearance across motion and changing viewpoints",
        "the source action, camera path, setting, and soundtrack unless explicitly requested",
    ),
    ("video", "style"): (
        "moving-image style, grading, texture, lighting, transition language, and pacing",
        "the source actor, identity, wardrobe, exact action, and setting",
    ),
    ("video", "environment"): (
        "the moving environment, atmosphere, spatial relationships, weather, and environmental dynamics",
        "the source actors, wardrobe, exact action, and camera path",
    ),
    ("video", "full_reference"): (
        "the source video's subjects, motion, camera, environment, pacing, and audiovisual structure",
        "details explicitly replaced by the target description",
    ),
    ("audio", "voice"): (
        "voice timbre, vocal identity, accent character, and natural delivery",
        "the source words, background sounds, and music unless explicitly requested",
    ),
    ("audio", "prosody_dialogue"): (
        "prosody, speaking rhythm, emotional delivery, pauses, and conversational timing",
        "the source voice identity, exact words, music, and background sounds unless explicitly requested",
    ),
    ("audio", "music"): (
        "musical style, instrumentation, tempo, harmony, energy, and production character",
        "the exact copyrighted recording or melody unless the prompt explicitly and lawfully requests reuse",
    ),
    ("audio", "ambience"): (
        "ambient texture, acoustic space, environmental bed, and atmosphere",
        "speech, foreground sound effects, and music",
    ),
    ("audio", "sfx"): (
        "sound-effect texture, impact, timing, material character, and spatial impression",
        "speech, music, and unrelated background sound",
    ),
    ("audio", "timing"): (
        "beat positions, event timing, rhythmic structure, and synchronization cues",
        "the exact timbre, words, or melody unless explicitly requested",
    ),
}


def normalize_alias(value: str) -> str:
    alias = str(value or "").strip()
    if alias.startswith("@"):
        alias = alias[1:]
    return alias


def make_reference(
    *,
    kind: str,
    name: str,
    media: Any,
    role: str,
    priority: str,
    use_for: str = "",
    do_not_copy: str = "",
    visual_weight: float = 1.0,
    audio_weight: float = 1.0,
    audio_relationship: str = "reference",
    soundtrack: Any = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the stable runtime record emitted by reference builder nodes."""
    return {
        "_type": "moc_h3_reference",
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "name": normalize_alias(name),
        "media": media,
        "soundtrack": soundtrack,
        "role": role,
        "priority": priority,
        "use_for": str(use_for or "").strip(),
        "do_not_copy": str(do_not_copy or "").strip(),
        "visual_weight": float(visual_weight),
        "audio_weight": float(audio_weight),
        "audio_relationship": str(audio_relationship),
        "metadata": dict(metadata or {}),
    }


def _is_finite_weight(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and 0.0 <= float(value) <= 2.0
    except (TypeError, ValueError):
        return False


def _duration_value(value: Any) -> float:
    """Return a finite non-negative duration, or zero for stale diagnostics metadata."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) and number >= 0.0 else 0.0


def _expected_roles(kind: str) -> tuple[str, ...]:
    return {
        "image": IMAGE_ROLES,
        "video": VIDEO_ROLES,
        "audio": AUDIO_ROLES,
    }.get(kind, ())


def _assign_tags(native_order: Iterable[dict[str, Any]]) -> dict[str, str]:
    counters = Counter()
    mapping: dict[str, str] = {}
    for ref in native_order:
        kind = ref["kind"]
        alias = ref["name"]
        if kind == "image":
            counters["picture"] += 1
            mapping[alias] = f"<Picture {counters['picture']}>"
        elif kind == "video":
            if ref.get("soundtrack") is not None:
                counters["audio"] += 1
                mapping[f"{alias}_audio"] = f"<Audio {counters['audio']}>"
            counters["video"] += 1
            mapping[alias] = f"<Video {counters['video']}>"
        elif kind == "audio":
            counters["audio"] += 1
            mapping[alias] = f"<Audio {counters['audio']}>"
    return mapping


def plan_references(references: Iterable[dict[str, Any]] | None) -> ReferencePlan:
    # Copy records so validation/normalisation never mutates cached builder
    # outputs or another consumer's view of the same reference set.
    refs = []
    diagnostics: list[Diagnostic] = []
    for ref in references or []:
        if not isinstance(ref, dict):
            refs.append(ref)
            continue
        copied = dict(ref)
        raw_metadata = ref.get("metadata")
        if raw_metadata is None:
            copied["metadata"] = {}
        elif isinstance(raw_metadata, dict):
            copied["metadata"] = dict(raw_metadata)
        else:
            copied["metadata"] = {}
            diagnostics.append(Diagnostic(
                "error",
                "invalid_metadata",
                "Reference metadata must be a dictionary; the malformed metadata was ignored.",
                normalize_alias(ref.get("name", "")) or None,
            ))
        soundtrack_metadata = copied["metadata"].get("soundtrack")
        if soundtrack_metadata is not None and not isinstance(soundtrack_metadata, dict):
            copied["metadata"]["soundtrack"] = {}
            diagnostics.append(Diagnostic(
                "error",
                "invalid_metadata",
                "Soundtrack metadata must be a dictionary; the malformed metadata was ignored.",
                normalize_alias(ref.get("name", "")) or None,
            ))
        refs.append(copied)

    for index, ref in enumerate(refs, 1):
        if not isinstance(ref, dict) or ref.get("_type") != "moc_h3_reference":
            diagnostics.append(Diagnostic("error", "invalid_reference", f"Input {index} is not an MOC H3 reference record."))
            continue
        alias = normalize_alias(ref.get("name", ""))
        ref["name"] = alias
        kind = ref.get("kind")
        priority = ref.get("priority")
        role = ref.get("role")
        if not ALIAS_RE.fullmatch(alias):
            diagnostics.append(Diagnostic(
                "error",
                "invalid_alias",
                "Use 1-32 characters: start with a letter, then letters, numbers, '_' or '-'.",
                alias or None,
            ))
        if kind not in KINDS:
            diagnostics.append(Diagnostic("error", "invalid_kind", f"Unsupported reference kind: {kind!r}.", alias or None))
        if priority not in PRIORITIES:
            diagnostics.append(Diagnostic("error", "invalid_priority", f"Unsupported priority: {priority!r}.", alias or None))
        if role not in _expected_roles(kind):
            diagnostics.append(Diagnostic("error", "invalid_role", f"Role {role!r} is invalid for {kind!r} references.", alias or None))
        if priority != "disabled" and ref.get("media") is None:
            diagnostics.append(Diagnostic("error", "missing_media", "The enabled reference has no media input.", alias or None))
        for field_name in ("visual_weight", "audio_weight"):
            if not _is_finite_weight(ref.get(field_name, 1.0)):
                diagnostics.append(Diagnostic("error", "invalid_weight", f"{field_name} must be finite and between 0 and 2.", alias or None))
        if kind == "audio" and ref.get("audio_relationship", "reference") not in AUDIO_RELATIONSHIPS:
            diagnostics.append(Diagnostic(
                "error",
                "invalid_audio_relationship",
                f"Unsupported audio relationship: {ref.get('audio_relationship')!r}.",
                alias or None,
            ))
        metadata = ref.get("metadata", {})
        duration_fields = [("duration_s", metadata.get("duration_s"))]
        if isinstance(metadata.get("soundtrack"), dict):
            duration_fields.append(("soundtrack.duration_s", metadata["soundtrack"].get("duration_s")))
        for field_name, value in duration_fields:
            if value is not None and _duration_value(value) == 0.0 and str(value) not in ("0", "0.0"):
                diagnostics.append(Diagnostic(
                    "error",
                    "invalid_metadata",
                    f"{field_name} must be a finite non-negative number.",
                    alias or None,
                ))

    active = [
        ref for ref in refs
        if isinstance(ref, dict)
        and ref.get("_type") == "moc_h3_reference"
        and ref.get("priority") != "disabled"
        and ref.get("kind") in KINDS
        and ref.get("priority") in PRIORITIES
        and ref.get("role") in _expected_roles(ref.get("kind"))
        and ALIAS_RE.fullmatch(normalize_alias(ref.get("name", "")))
        and ref.get("media") is not None
        and _is_finite_weight(ref.get("visual_weight", 1.0))
        and _is_finite_weight(ref.get("audio_weight", 1.0))
        and (ref.get("kind") != "audio" or ref.get("audio_relationship", "reference") in AUDIO_RELATIONSHIPS)
    ]
    aliases = [normalize_alias(ref.get("name", "")).lower() for ref in active]
    for alias, count in Counter(aliases).items():
        if alias and count > 1:
            diagnostics.append(Diagnostic("error", "duplicate_alias", "Active aliases are case-insensitively unique.", alias))
    active_aliases = {normalize_alias(ref.get("name", "")).lower() for ref in active}
    for ref in active:
        if ref.get("kind") != "video" or ref.get("soundtrack") is None:
            continue
        derived_alias = f"{ref['name']}_audio"
        if derived_alias.lower() in active_aliases:
            diagnostics.append(Diagnostic(
                "error",
                "reserved_alias_collision",
                f"The paired soundtrack reserves @{derived_alias}; rename the real reference that uses this alias.",
                ref["name"],
            ))
    if not active:
        diagnostics.append(Diagnostic("error", "no_references", "Connect at least one enabled reference."))

    counts = Counter(ref.get("kind") for ref in active)
    for kind, maximum in (("image", 9), ("video", 3), ("audio", 3)):
        if counts[kind] > maximum:
            diagnostics.append(Diagnostic(
                "error",
                "too_many_references",
                f"MiniMax H3 accepts at most {maximum} {kind} reference(s); received {counts[kind]}.",
            ))
    mixed_file_count = len(active) + sum(1 for ref in active if ref.get("soundtrack") is not None)
    if mixed_file_count > 12:
        diagnostics.append(Diagnostic(
            "warning",
            "mixed_file_limit",
            f"The active set represents {mixed_file_count} media files including paired soundtracks. "
            "The official H3 API caps mixed reference inputs at 12 files; local ComfyUI may accept more, "
            "but behaviour is unvalidated.",
        ))

    source_video_duration = sum(
        _duration_value(ref.get("metadata", {}).get("duration_s"))
        for ref in active
        if ref.get("kind") == "video"
    )
    source_audio_duration = sum(
        _duration_value(ref.get("metadata", {}).get("duration_s"))
        for ref in active
        if ref.get("kind") == "audio"
    ) + sum(
        _duration_value(ref.get("metadata", {}).get("soundtrack", {}).get("duration_s"))
        for ref in active
        if ref.get("kind") == "video"
        and ref.get("soundtrack") is not None
        and isinstance(ref.get("metadata", {}).get("soundtrack"), dict)
    )
    if source_video_duration > 15.0 + 1e-6:
        diagnostics.append(Diagnostic(
            "warning",
            "source_total_video_duration",
            f"Prepared reference videos total {source_video_duration:.2f}s; H3 recommends at most 15s per modality.",
        ))
    if source_audio_duration > 15.0 + 1e-6:
        diagnostics.append(Diagnostic(
            "warning",
            "source_total_audio_duration",
            f"Prepared reference audio including soundtracks totals {source_audio_duration:.2f}s; "
            "H3 recommends at most 15s per modality.",
        ))

    primary_refs = [ref for ref in active if ref.get("priority") == "primary"]
    overlapping_primary = Counter(
        ("audio" if ref.get("kind") == "audio" else "visual", ref.get("role"))
        for ref in primary_refs
    )
    for (modality, role), count in overlapping_primary.items():
        if count > 1:
            diagnostics.append(Diagnostic(
                "warning",
                "primary_conflict",
                f"{count} primary {modality} references share role '{role}'. Give each a non-overlapping scope to resolve conflicts.",
            ))
    visual_primary = [ref for ref in primary_refs if ref.get("kind") in ("image", "video")]
    if len(visual_primary) > 1 and any(ref.get("role") == "full_reference" for ref in visual_primary):
        diagnostics.append(Diagnostic(
            "warning",
            "primary_full_reference_overlap",
            "A primary full-reference video overlaps other primary visual references. Add precise scopes or lower one priority.",
        ))

    for ref in active:
        alias = ref.get("name")
        metadata = ref.get("metadata", {})
        for warning in metadata.get("warnings", []):
            diagnostics.append(Diagnostic("warning", "media_processing", str(warning), alias))
        if (
            ref.get("kind") in ("image", "video")
            and _is_finite_weight(ref.get("visual_weight", 1.0))
            and float(ref.get("visual_weight", 1.0)) != 1.0
        ):
            diagnostics.append(Diagnostic(
                "warning",
                "experimental_weight",
                "Its numeric visual weight only becomes active when the experimental MODEL patch node is connected.",
                alias,
            ))
        if (
            (ref.get("kind") == "audio" or ref.get("soundtrack") is not None)
            and _is_finite_weight(ref.get("audio_weight", 1.0))
            and float(ref.get("audio_weight", 1.0)) != 1.0
        ):
            diagnostics.append(Diagnostic(
                "warning",
                "experimental_weight",
                "Its numeric audio weight only becomes active when the experimental MODEL patch node is connected.",
                alias,
            ))

    native_order = (
        [ref for ref in active if ref.get("kind") == "image"]
        + [ref for ref in active if ref.get("kind") == "video"]
        + [ref for ref in active if ref.get("kind") == "audio"]
    )
    mapping = _assign_tags(native_order)
    return ReferencePlan(refs, active, native_order, mapping, diagnostics)


def substitute_aliases(prompt: str, alias_to_tag: dict[str, str]) -> tuple[str, set[str]]:
    """Resolve @aliases without accidentally replacing substrings."""
    lower_mapping = {key.lower(): value for key, value in alias_to_tag.items()}
    unresolved: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        alias = match.group(1)
        tag = lower_mapping.get(alias.lower())
        if tag is None:
            unresolved.add(alias)
            return match.group(0)
        return tag

    return ALIAS_TOKEN_RE.sub(replace, str(prompt or "")), unresolved


def validate_native_tags(prompt: str, plan: ReferencePlan) -> list[Diagnostic]:
    available = {tag.lower() for tag in plan.alias_to_tag.values()}
    diagnostics: list[Diagnostic] = []
    seen: set[str] = set()
    for match in NATIVE_TAG_RE.finditer(str(prompt or "")):
        canonical = f"<{match.group(1).title()} {int(match.group(2))}>"
        if canonical.lower() not in available and canonical.lower() not in seen:
            diagnostics.append(Diagnostic(
                "error",
                "missing_native_tag",
                f"The prompt references {canonical}, but the active set does not produce that tag.",
            ))
        seen.add(canonical.lower())
    return diagnostics


def _priority_language(priority: str, kind: str, audio_relationship: str = "reference") -> tuple[str, str]:
    if kind == "audio":
        return priority, audio_relationship
    return {
        "primary": ("primary", "fully_preserved"),
        "supporting": ("supporting", "partially_preserved"),
        "weak": ("weak", "weak_reference"),
    }[priority]


def _guidance_for(ref: dict[str, Any], tag: str) -> str:
    kind = ref["kind"]
    role = ref["role"]
    priority, marker = _priority_language(
        ref["priority"],
        kind,
        ref.get("audio_relationship", "reference"),
    )
    preserve, exclude = ROLE_GUIDANCE.get(
        (kind, role),
        ("only the attributes explicitly assigned to it", "unassigned identities, composition, motion, style, and audio"),
    )
    if ref.get("use_for"):
        preserve = ref["use_for"]
    if ref.get("do_not_copy"):
        exclude = ref["do_not_copy"]
    if kind == "audio" and marker == "fully_copy":
        return (
            f"{tag} is the {priority} {role.replace('_', ' ')} audio reference ({marker}). "
            "Reuse the complete authorized source signal as the target audio, preserving its timing and audible layers."
        )
    if kind == "audio" and marker == "partially_copy":
        return (
            f"{tag} is the {priority} {role.replace('_', ' ')} audio reference ({marker}). "
            f"Reuse only the authorized portions or layers assigned for {preserve}. Do not copy {exclude}."
        )
    guidance = (
        f"{tag} is the {priority} {role.replace('_', ' ')} reference "
        f"({marker}). Use it for {preserve}. Do not copy {exclude}."
    )
    if kind == "image" and ref.get("metadata", {}).get("mask_applied"):
        guidance += (
            " Only the source pixels retained by its isolation mask are valid reference evidence; "
            "ignore the synthetic filled, blurred, or cropped exterior. The mask does not specify target-frame placement."
        )
    return guidance


def compile_prompt(prompt: str, plan: ReferencePlan, mode: str = "guided") -> tuple[str, list[Diagnostic]]:
    if mode not in PROMPT_MODES:
        raise ReferencePlanError(f"Unsupported prompt mode: {mode!r}")
    if mode == "manual":
        manual = str(prompt or "").strip()
        return manual, validate_native_tags(manual, plan)

    resolved, unresolved = substitute_aliases(prompt, plan.alias_to_tag)
    diagnostics = [
        Diagnostic("error", "unknown_alias", "The prompt references an alias that is not enabled in this set.", alias)
        for alias in sorted(unresolved, key=str.lower)
    ]
    diagnostics.extend(validate_native_tags(resolved, plan))
    if mode == "aliases_only":
        return resolved.strip(), diagnostics

    guidance: list[str] = []
    for ref in plan.native_order:
        tag = plan.alias_to_tag[ref["name"]]
        guidance.append(_guidance_for(ref, tag))
        if ref["kind"] == "video" and ref.get("soundtrack") is not None:
            audio_tag = plan.alias_to_tag[f"{ref['name']}_audio"]
            soundtrack_relationship = ref.get("metadata", {}).get("soundtrack_relationship", "reference")
            audio_priority, marker = _priority_language(
                ref["priority"],
                "audio",
                soundtrack_relationship,
            )
            audio_role = ref.get("metadata", {}).get("soundtrack_role", "timing")
            preserve, exclude = ROLE_GUIDANCE.get(
                ("audio", audio_role),
                ("only the audible attributes explicitly assigned to it", "unassigned dialogue, music, and effects"),
            )
            if marker == "fully_copy":
                audio_instruction = "Reuse the complete authorized source soundtrack signal."
            elif marker == "partially_copy":
                audio_instruction = f"Reuse only the authorized portions or layers assigned for {preserve}."
            else:
                audio_instruction = f"Use it for {preserve}. Do not copy {exclude}."
            guidance.append(
                f"{audio_tag} is the {audio_priority} {str(audio_role).replace('_', ' ')} soundtrack reference "
                f"({marker}) paired with {tag}. {audio_instruction} Keep it synchronized with the referenced motion."
            )

    prefix = "Reference plan:\n" + "\n".join(guidance)
    conflict_rule = (
        "Conflict rule: Primary references override Supporting references; Supporting references override Weak references. "
        "Respect every 'Do not copy' boundary above."
    )
    scene = f"Target scene:\n{resolved.strip()}" if resolved.strip() else "Target scene:\nCreate the requested target shot using the scoped references above."
    return f"{prefix}\n\n{conflict_rule}\n\n{scene}", diagnostics


def tag_mapping_text(plan: ReferencePlan) -> str:
    if not plan.alias_to_tag:
        return "No active reference mappings."
    return "\n".join(f"@{alias} -> {tag}" for alias, tag in plan.alias_to_tag.items())


def _metadata_summary(ref: dict[str, Any]) -> str:
    metadata = ref.get("metadata", {})
    pieces: list[str] = []
    if metadata.get("processed_width") and metadata.get("processed_height"):
        pieces.append(f"{metadata['processed_width']}x{metadata['processed_height']}")
    if metadata.get("processed_frames") is not None:
        pieces.append(f"{metadata['processed_frames']} frames")
    if metadata.get("duration_s") is not None:
        pieces.append(f"{_duration_value(metadata['duration_s']):.2f}s")
    if metadata.get("native_width") and metadata.get("native_height"):
        pieces.append(f"native {metadata['native_width']}x{metadata['native_height']}")
    if metadata.get("native_encoded_frames") is not None:
        pieces.append(f"native {metadata['native_encoded_frames']} frames")
    if metadata.get("estimated_visual_tokens") is not None:
        pieces.append(f"~{int(metadata['estimated_visual_tokens']):,} direct visual tokens")
    if metadata.get("estimated_audio_tokens") is not None:
        pieces.append(f"~{int(metadata['estimated_audio_tokens']):,} direct audio tokens")
    if metadata.get("mask_applied"):
        coverage = _duration_value(metadata.get("mask_coverage")) * 100.0
        mask_text = f"mask {coverage:.1f}% {metadata.get('mask_presentation', 'neutral_fill')}"
        bounds = metadata.get("mask_bounds_xyxy")
        if isinstance(bounds, list) and len(bounds) == 4:
            mask_text += f" bounds {bounds}"
        source_width = metadata.get("mask_source_width")
        source_height = metadata.get("mask_source_height")
        resolved_width = metadata.get("mask_resolved_width")
        resolved_height = metadata.get("mask_resolved_height")
        if metadata.get("mask_resized") and all(
            isinstance(value, int) for value in (source_width, source_height, resolved_width, resolved_height)
        ):
            mask_text += f" resized {source_width}x{source_height}->{resolved_width}x{resolved_height}"
        pieces.append(mask_text)
    soundtrack = metadata.get("soundtrack")
    if isinstance(soundtrack, dict) and soundtrack.get("duration_s") is not None:
        soundtrack_text = f"soundtrack {_duration_value(soundtrack['duration_s']):.2f}s prepared"
        if metadata.get("native_soundtrack_duration_s") is not None:
            soundtrack_text += f" -> {_duration_value(metadata['native_soundtrack_duration_s']):.2f}s native"
        pieces.append(
            f"{soundtrack_text} ({metadata.get('soundtrack_policy', 'trim_to_video')}, "
            f"{metadata.get('soundtrack_role', 'timing')})"
        )
    return ", ".join(pieces) if pieces else "runtime media"


def render_report(
    plan: ReferencePlan,
    *,
    compiled_prompt: str | None = None,
    width: int | None = None,
    height: int | None = None,
    length: int | None = None,
    extra_diagnostics: Iterable[Diagnostic] = (),
) -> str:
    diagnostics = [*plan.diagnostics, *extra_diagnostics]
    lines = ["MOC MiniMax H3 reference report", "=" * 36]
    if width is not None and height is not None and length is not None:
        aligned = max(5, int(length))
        while aligned % 17 != 5:
            aligned += 1
        lines.append(
            f"Target: {int(width)}x{int(height)}, requested {int(length)} frames, resolved {aligned} frames "
            f"({aligned / 24.0:.2f}s at 24 fps)"
        )
    lines.append(f"References: {len(plan.active)} active / {len(plan.references)} connected")
    lines.append("")
    lines.append("Resolved mappings:")
    lines.extend(f"  @{alias:<20} {tag}" for alias, tag in plan.alias_to_tag.items())
    lines.append("")
    lines.append("Reference controls:")
    for ref in plan.native_order:
        tag = plan.alias_to_tag.get(ref["name"], "<unresolved>")
        visual_weight = float(ref.get("visual_weight", 1.0))
        audio_weight = float(ref.get("audio_weight", 1.0))
        weight_text = f"visual={visual_weight:.2f}"
        if ref["kind"] == "audio" or ref.get("soundtrack") is not None:
            weight_text += f", audio={audio_weight:.2f}"
        lines.append(
            f"  {tag} @{ref['name']}: {ref['kind']} / {ref['role']} / {ref['priority']} / "
            f"{weight_text} / {_metadata_summary(ref)}"
        )
    disabled = [ref for ref in plan.references if isinstance(ref, dict) and ref.get("priority") == "disabled"]
    for ref in disabled:
        lines.append(f"  (disabled) @{ref.get('name', '?')}: excluded from encoding and numbering")
    lines.append("")
    if diagnostics:
        lines.append("Diagnostics:")
        lines.extend(f"  {item.render()}" for item in diagnostics)
    else:
        lines.append("Diagnostics: no issues detected")
    if compiled_prompt is not None:
        lines.append("")
        lines.append(f"Compiled prompt: {len(compiled_prompt)} characters")
    return "\n".join(lines)


def serializable_manifest(
    plan: ReferencePlan,
    extra_diagnostics: Iterable[Diagnostic] = (),
    validation_mode: str | None = None,
) -> dict[str, Any]:
    diagnostics = [*plan.diagnostics, *extra_diagnostics]
    has_errors = any(item.level == "error" for item in diagnostics)
    has_warnings = any(item.level == "warning" for item in diagnostics)
    ready = not has_errors and not (validation_mode == "strict" and has_warnings)
    refs: list[dict[str, Any]] = []
    for ref in plan.references:
        if not isinstance(ref, dict):
            continue
        refs.append({
            "schema_version": ref.get("schema_version", SCHEMA_VERSION),
            "name": ref.get("name"),
            "kind": ref.get("kind"),
            "role": ref.get("role"),
            "priority": ref.get("priority"),
            "tag": plan.alias_to_tag.get(ref.get("name")),
            "audio_tag": plan.alias_to_tag.get(f"{ref.get('name')}_audio"),
            "soundtrack_role": ref.get("metadata", {}).get("soundtrack_role") if ref.get("soundtrack") is not None else None,
            "soundtrack_relationship": (
                ref.get("metadata", {}).get("soundtrack_relationship")
                if ref.get("soundtrack") is not None
                else None
            ),
            "visual_weight": ref.get("visual_weight", 1.0),
            "audio_weight": ref.get("audio_weight", 1.0),
            "audio_relationship": ref.get("audio_relationship", "reference"),
            "use_for": ref.get("use_for", ""),
            "do_not_copy": ref.get("do_not_copy", ""),
            "enabled": ref.get("priority") != "disabled",
            "metadata": {
                key: value
                for key, value in ref.get("metadata", {}).items()
                if isinstance(value, (str, int, float, bool, type(None), list, dict))
            },
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": ready,
        "validation_mode": validation_mode,
        "mapping": plan.alias_to_tag,
        "references": refs,
        "diagnostics": [
            {"level": item.level, "code": item.code, "message": item.message, "alias": item.alias}
            for item in diagnostics
        ],
    }


def manifest_json(
    plan: ReferencePlan,
    extra_diagnostics: Iterable[Diagnostic] = (),
    validation_mode: str | None = None,
) -> str:
    return json.dumps(
        serializable_manifest(plan, extra_diagnostics, validation_mode),
        indent=2,
        sort_keys=True,
    )


def enforce(plan: ReferencePlan, extra: Iterable[Diagnostic] = (), mode: str = "strict") -> None:
    if mode not in VALIDATION_MODES:
        raise ReferencePlanError(f"Unsupported validation mode: {mode!r}")
    errors = [*plan.errors, *(item for item in extra if item.level == "error")]
    # Structural errors cannot be delegated safely to native H3: several make
    # alias compilation or packed-reference alignment ambiguous. ``warn`` only
    # controls advisory diagnostics; it never authorizes malformed execution.
    if errors:
        raise ReferencePlanError("Reference plan validation failed:\n" + "\n".join(item.render() for item in errors))
    if mode == "strict":
        warnings = [*plan.warnings, *(item for item in extra if item.level == "warning")]
        if warnings:
            raise ReferencePlanError(
                "Reference plan strict validation stopped on warning(s):\n"
                + "\n".join(item.render() for item in warnings)
            )

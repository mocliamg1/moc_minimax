from __future__ import annotations

import unittest

from moc_minimax.core import (
    Diagnostic,
    ReferencePlanError,
    compile_prompt,
    enforce,
    make_reference,
    render_report,
    serializable_manifest,
    plan_references,
    substitute_aliases,
)


def ref(kind, name, role, priority="supporting", soundtrack=None, **kwargs):
    return make_reference(
        kind=kind,
        name=name,
        media=object(),
        soundtrack=soundtrack,
        role=role,
        priority=priority,
        **kwargs,
    )


class ReferencePlanTests(unittest.TestCase):
    def test_native_order_and_independent_numbering(self):
        refs = [
            ref("video", "walk", "motion", soundtrack=object()),
            ref("audio", "voice", "voice"),
            ref("image", "hero", "identity", "primary"),
            ref("image", "look", "style", "weak"),
        ]
        plan = plan_references(refs)
        self.assertTrue(plan.valid)
        self.assertEqual([item["name"] for item in plan.native_order], ["hero", "look", "walk", "voice"])
        self.assertEqual(
            plan.alias_to_tag,
            {
                "hero": "<Picture 1>",
                "look": "<Picture 2>",
                "walk_audio": "<Audio 1>",
                "walk": "<Video 1>",
                "voice": "<Audio 2>",
            },
        )

    def test_disabled_reference_does_not_consume_tag(self):
        plan = plan_references([
            ref("image", "disabled", "identity", "disabled"),
            ref("image", "hero", "identity", "primary"),
        ])
        self.assertEqual(plan.alias_to_tag, {"hero": "<Picture 1>"})

    def test_disabled_duplicate_alias_does_not_block_active_reference(self):
        plan = plan_references([
            ref("image", "hero", "identity", "disabled"),
            ref("image", "hero", "identity", "primary"),
        ])
        self.assertTrue(plan.valid)
        self.assertEqual(plan.alias_to_tag, {"hero": "<Picture 1>"})

    def test_duplicate_alias_is_case_insensitive(self):
        plan = plan_references([
            ref("image", "Hero", "identity"),
            ref("video", "hero", "motion"),
        ])
        self.assertFalse(plan.valid)
        self.assertIn("duplicate_alias", [item.code for item in plan.errors])

    def test_reference_count_limit(self):
        plan = plan_references([ref("video", f"motion{i}", "motion") for i in range(4)])
        self.assertFalse(plan.valid)
        self.assertIn("too_many_references", [item.code for item in plan.errors])

    def test_primary_conflicts_are_detected_across_visual_media(self):
        plan = plan_references([
            ref("image", "portrait", "identity", "primary"),
            ref("video", "actor", "identity", "primary"),
        ])
        self.assertIn("primary_conflict", [item.code for item in plan.warnings])

    def test_set_reports_source_modality_duration_totals(self):
        first = ref("video", "walk", "motion")
        second = ref("video", "run", "motion")
        first["metadata"]["duration_s"] = 10.0
        second["metadata"]["duration_s"] = 10.0
        plan = plan_references([first, second])
        self.assertIn("source_total_video_duration", [item.code for item in plan.warnings])

    def test_paired_soundtrack_alias_is_reserved(self):
        plan = plan_references([
            ref("video", "walk", "motion", soundtrack=object()),
            ref("audio", "walk_audio", "voice"),
        ])
        self.assertFalse(plan.valid)
        self.assertIn("reserved_alias_collision", [item.code for item in plan.errors])

    def test_invalid_alias_and_weight(self):
        plan = plan_references([
            ref("image", "bad alias", "identity", visual_weight=-1),
        ])
        codes = [item.code for item in plan.errors]
        self.assertIn("invalid_alias", codes)
        self.assertIn("invalid_weight", codes)

    def test_malformed_stale_record_returns_diagnostics_instead_of_crashing(self):
        malformed = ref("image", "hero", "identity")
        malformed["priority"] = "bogus"
        malformed["visual_weight"] = "not-a-number"
        plan = plan_references([malformed])
        self.assertFalse(plan.valid)
        self.assertIn("invalid_priority", [item.code for item in plan.errors])
        self.assertIn("invalid_weight", [item.code for item in plan.errors])
        self.assertEqual(plan.active, [])

    def test_planning_does_not_mutate_input_record(self):
        original = ref("image", "@Hero", "identity")
        plan = plan_references([original])
        self.assertEqual(original["name"], "Hero")
        self.assertIsNot(original, plan.references[0])

    def test_enforce_strict_and_warn(self):
        plan = plan_references([])
        with self.assertRaises(ReferencePlanError):
            enforce(plan, mode="strict")
        with self.assertRaises(ReferencePlanError):
            enforce(plan, mode="warn")

        valid = plan_references([ref("image", "hero", "identity", "primary")])
        enforce(valid, [Diagnostic("warning", "advisory", "Advisory only")], mode="warn")
        with self.assertRaises(ReferencePlanError):
            enforce(valid, [Diagnostic("warning", "advisory", "Advisory only")], mode="strict")

    def test_manifest_includes_compilation_diagnostics(self):
        plan = plan_references([ref("image", "hero", "identity", "primary")])
        manifest = serializable_manifest(
            plan,
            [Diagnostic("error", "unknown_alias", "Unknown alias @missing", "missing")],
        )
        self.assertFalse(manifest["valid"])
        self.assertEqual(manifest["diagnostics"][-1]["code"], "unknown_alias")


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.plan = plan_references([
            ref("image", "hero", "identity", "primary"),
            ref("image", "look", "style", "weak"),
            ref("video", "walk", "motion", "supporting"),
        ])

    def test_alias_substitution_has_token_boundaries(self):
        text, unknown = substitute_aliases("@hero @heroine foo@hero", self.plan.alias_to_tag)
        self.assertEqual(text, "<Picture 1> @heroine foo@hero")
        self.assertEqual(unknown, {"heroine"})

    def test_guided_prompt_contains_scope_and_conflict_order(self):
        compiled, diagnostics = compile_prompt(
            "A tracking shot of @hero. Use @walk for gait and @look for palette.",
            self.plan,
            "guided",
        )
        self.assertFalse(diagnostics)
        self.assertIn("<Picture 1> is the primary identity reference", compiled)
        self.assertIn("<Video 1> is the supporting motion reference", compiled)
        self.assertIn("<Picture 2> is the weak style reference", compiled)
        self.assertIn("Primary references override Supporting references", compiled)

    def test_mask_guidance_and_metadata_are_visible_only_where_intended(self):
        masked_plan = plan_references([
            ref(
                "image",
                "hero",
                "identity",
                "primary",
                metadata={
                    "mask_applied": True,
                    "mask_coverage": 0.25,
                    "mask_presentation": "neutral_fill",
                    "mask_bounds_xyxy": [1, 2, 5, 8],
                    "mask_source_width": 6,
                    "mask_source_height": 10,
                    "mask_resolved_width": 12,
                    "mask_resolved_height": 20,
                    "mask_resized": True,
                    "warnings": [],
                },
            )
        ])
        guided, _ = compile_prompt("Use @hero.", masked_plan, "guided")
        aliases_only, _ = compile_prompt("Use @hero.", masked_plan, "aliases_only")
        manual, _ = compile_prompt("Use <Picture 1>.", masked_plan, "manual")
        self.assertIn("Only the source pixels retained by its isolation mask", guided)
        self.assertIn("does not specify target-frame placement", guided)
        self.assertNotIn("isolation mask", aliases_only)
        self.assertNotIn("isolation mask", manual)

        report = render_report(masked_plan)
        self.assertIn("mask 25.0% neutral_fill", report)
        self.assertIn("resized 6x10->12x20", report)
        manifest = serializable_manifest(masked_plan)
        self.assertEqual(manifest["references"][0]["metadata"]["mask_coverage"], 0.25)

    def test_aliases_only_does_not_add_plan(self):
        compiled, diagnostics = compile_prompt("Use @hero.", self.plan, "aliases_only")
        self.assertEqual(compiled, "Use <Picture 1>.")
        self.assertFalse(diagnostics)

    def test_manual_is_unchanged(self):
        compiled, diagnostics = compile_prompt("Use <Picture 1>: exactly.", self.plan, "manual")
        self.assertEqual(compiled, "Use <Picture 1>: exactly.")
        self.assertFalse(diagnostics)

    def test_manual_nonexistent_native_tag_is_error(self):
        _, diagnostics = compile_prompt("Use <Picture 9>.", self.plan, "manual")
        self.assertEqual([item.code for item in diagnostics], ["missing_native_tag"])

    def test_unknown_alias_is_error(self):
        _, diagnostics = compile_prompt("Use @missing.", self.plan, "guided")
        self.assertEqual([item.code for item in diagnostics], ["unknown_alias"])

    def test_audio_relationship_is_independent_of_priority(self):
        audio = ref("audio", "score", "music", "primary", audio_relationship="partially_copy")
        plan = plan_references([audio])
        compiled, diagnostics = compile_prompt("Use @score.", plan, "guided")
        self.assertFalse(diagnostics)
        self.assertIn("primary music audio reference (partially_copy)", compiled)
        self.assertIn("Reuse only the authorized portions", compiled)

    def test_long_generated_soundtrack_alias_is_resolved(self):
        name = "v" * 32
        plan = plan_references([ref("video", name, "motion", soundtrack=object())])
        compiled, diagnostics = compile_prompt(f"Use @{name}_audio for timing.", plan, "aliases_only")
        self.assertFalse(diagnostics)
        self.assertEqual(compiled, "Use <Audio 1> for timing.")


if __name__ == "__main__":
    unittest.main()

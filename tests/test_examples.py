from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ExampleArtifactTests(unittest.TestCase):
    def test_api_example_is_a_clean_prompt_graph(self):
        graph = json.loads((ROOT / "examples" / "minimax_h3_reference_plus_api.json").read_text())
        self.assertTrue(graph)
        for node_id, node in graph.items():
            self.assertTrue(str(node_id).isdigit())
            self.assertIsInstance(node.get("class_type"), str)
            self.assertIsInstance(node.get("inputs"), dict)
        weights = [
            node["inputs"].get("signal_weight")
            for node in graph.values()
            if node["class_type"] == "MocH3ImageReference"
        ]
        self.assertTrue(weights)
        self.assertTrue(all(weight == 1.0 for weight in weights))
        compile_nodes = [node for node in graph.values() if node["class_type"] == "MocH3CompileReferencePrompt"]
        self.assertEqual(len(compile_nodes), 1)
        self.assertEqual(compile_nodes[0]["inputs"]["validation_mode"], "warn")

    def test_ui_example_links_reference_set_to_output_preflight(self):
        workflow = json.loads(
            (ROOT / "example_workflows" / "minimax_h3_moc_authoring_preflight.json").read_text()
        )
        nodes = {node["id"]: node for node in workflow["nodes"]}
        compile_nodes = [node for node in nodes.values() if node["type"] == "MocH3CompileReferencePrompt"]
        self.assertEqual(len(compile_nodes), 1)
        compile_node = compile_nodes[0]
        self.assertIn("match_output", compile_node["widgets_values"])
        image_nodes = [node for node in nodes.values() if node["type"] == "MocH3ImageReference"]
        self.assertTrue(image_nodes)
        for image_node in image_nodes:
            self.assertIn("mask", {item["name"] for item in image_node["inputs"]})
            self.assertEqual(image_node["widgets_values"][-7:], [
                "white_keeps", "neutral_fill", 0, 4, 0.5, 32, 10,
            ])
        incoming_link_ids = {item["link"] for item in compile_node["inputs"] if item.get("link") is not None}
        declared_link_ids = {link[0] for link in workflow["links"]}
        self.assertTrue(incoming_link_ids)
        self.assertTrue(incoming_link_ids <= declared_link_ids)


if __name__ == "__main__":
    unittest.main()

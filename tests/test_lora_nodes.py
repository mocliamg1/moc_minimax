from __future__ import annotations

import asyncio
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

AVAILABLE = importlib.util.find_spec('torch') is not None and importlib.util.find_spec('comfy_api') is not None
if AVAILABLE:
    import torch
    from moc_minimax.lora_nodes import MocH3LoadLoraNode, MocH3CompareLorasNode, MocH3MergeLorasNode


@unittest.skipUnless(AVAILABLE, 'torch/ComfyUI are required')
class LoraNodeTests(unittest.TestCase):
    def test_schema_registration(self):
        from moc_minimax.nodes import MocH3Extension
        nodes = asyncio.run(MocH3Extension().get_node_list())
        for node in (MocH3LoadLoraNode, MocH3CompareLorasNode, MocH3MergeLorasNode):
            self.assertIn(node, nodes)
            schema = node.define_schema()
            schema.finalize()
            schema.validate()
        self.assertEqual([p.io_type for p in MocH3CompareLorasNode.define_schema().outputs], ['STRING', 'STRING'])

    def test_safe_load_compare_and_cache_invalidation(self):
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.safetensors'
            state = {'blocks.0.q.lora_A.weight': torch.ones(1, 2),
                     'blocks.0.q.lora_B.weight': torch.ones(2, 1)}
            save_file(state, str(path))
            with mock.patch('moc_minimax.lora_nodes._path', return_value=str(path)):
                first = MocH3LoadLoraNode.fingerprint_inputs('test.safetensors')
                loaded = MocH3LoadLoraNode.execute('test.safetensors').result[0]
                self.assertEqual(first, MocH3LoadLoraNode.fingerprint_inputs('test.safetensors'))
                self.assertTrue(all(t.device.type == 'cpu' for t in loaded['tensors'].values()))
                output = MocH3CompareLorasNode.execute(loaded, loaded)
                self.assertIn('100.000000/100', output.result[0])
                self.assertEqual(json.loads(output.result[1])['score'], 100)
                self.assertIsNotNone(output.ui)
                # A same-sized replacement with preserved mtime must still invalidate.
                old_stat = path.stat()
                replacement = Path(directory) / 'replacement.safetensors'
                save_file({k: v * 2 for k, v in state.items()}, str(replacement))
                os.utime(replacement, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
                os.replace(replacement, path)
                self.assertNotEqual(first, MocH3LoadLoraNode.fingerprint_inputs('test.safetensors'))
                path.unlink()
                missing = MocH3LoadLoraNode.fingerprint_inputs('test.safetensors')
                self.assertNotEqual(missing, missing)

    def test_torch_loader_is_weights_only(self):
        with mock.patch('moc_minimax.lora_nodes._path', return_value='/tmp/test.pt'), \
             mock.patch('torch.load', return_value={'state_dict': {'x': torch.ones(1)}}) as load:
            output = MocH3LoadLoraNode.execute('test.pt')
            load.assert_called_once_with('/tmp/test.pt', map_location='cpu', weights_only=True)
            self.assertEqual(list(output.result[0]['tensors']), ['x'])

    def test_merge_node_export_roundtrip_and_path_guards(self):
        from safetensors.torch import load_file
        p = dict(source='test', tensors={
            'diffusion_model.blocks.0.q.lokr_w1': torch.ones(2, 2),
            'diffusion_model.blocks.0.q.lokr_w2': torch.ones(2, 2),
        })
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch('folder_paths.get_output_directory', return_value=directory):
                output = MocH3MergeLorasNode.execute(p, .7, p, .5, lora_c=p, strength_c=.3)
                merged, report, report_json, saved_path = output.result
                self.assertTrue(Path(saved_path).is_file())
                self.assertIn('apply at strength 1.0', report)
                self.assertEqual(len(json.loads(report_json)['sources']), 3)
                self.assertIsNotNone(output.ui)
                self.assertTrue(all(torch.allclose(t, torch.full_like(t, 1.5)) for t in load_file(saved_path).values()))
                comparison = MocH3CompareLorasNode.execute(merged, merged, 'effective_dimensions')
                self.assertEqual(json.loads(comparison.result[1])['score'], 100)
                second = MocH3MergeLorasNode.execute(p, .7, p, .5)
                self.assertNotEqual(saved_path, second.result[3])
                self.assertTrue(Path(saved_path).is_file())
                with self.assertRaisesRegex(ValueError, 'within'):
                    MocH3MergeLorasNode.execute(p, 1., p, 1., filename_prefix='../escape')

    def test_failed_merge_publishes_no_files(self):
        p = dict(tensors={'blocks.0.q.dora_scale': torch.ones(2, 2)})
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch('folder_paths.get_output_directory', return_value=directory):
                with self.assertRaisesRegex(ValueError, 'Cannot merge'):
                    MocH3MergeLorasNode.execute(p, 1., p, 1.)
            self.assertEqual(list(Path(directory).rglob('*.safetensors')), [])

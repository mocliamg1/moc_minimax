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
        loader_inputs = MocH3LoadLoraNode.define_schema().inputs
        self.assertEqual([i.id for i in loader_inputs], ['lora_name', 'strength'])
        self.assertEqual(loader_inputs[1].default, 1.0)
        merge_inputs = MocH3MergeLorasNode.define_schema().inputs
        self.assertEqual([i.id for i in merge_inputs], ['loras', 'output_format', 'rank', 'storage_dtype', 'filename_prefix'])
        template = merge_inputs[0].template
        self.assertEqual((template.prefix, template.min, template.max), ('lora_', 2, 20))
        self.assertFalse(template.input.optional)

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
                self.assertEqual(loaded['strength'], 1.0)
                self.assertEqual(first, MocH3LoadLoraNode.fingerprint_inputs('test.safetensors'))
                self.assertTrue(all(t.device.type == 'cpu' for t in loaded['tensors'].values()))
                output = MocH3CompareLorasNode.execute(loaded, loaded)
                self.assertIn('100.000000/100', output.result[0])
                self.assertEqual(json.loads(output.result[1])['score'], 100)
                self.assertIsNotNone(output.ui)
                # Compare applies loader strengths: half-strength vs a half-scaled copy matches.
                half = MocH3LoadLoraNode.execute('test.safetensors', .5).result[0]
                scaled = dict(loaded, tensors={k: v * (.5 if 'lora_B' in k else 1.) for k, v in state.items()})
                self.assertEqual(json.loads(MocH3CompareLorasNode.execute(half, scaled).result[1])['score'], 100)
                self.assertLess(json.loads(MocH3CompareLorasNode.execute(half, loaded).result[1])['score'], 100)
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
        at = lambda strength: dict(p, strength=strength)
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch('folder_paths.get_output_directory', return_value=directory):
                # Unconnected autogrow sockets arrive as None and are skipped.
                output = MocH3MergeLorasNode.execute(
                    {'lora_0': at(.7), 'lora_1': at(.5), 'lora_2': None, 'lora_3': at(.3)},
                    output_format='full_diff')
                merged, report, report_json, saved_path = output.result
                self.assertTrue(Path(saved_path).is_file())
                self.assertIn('apply at strength 1.0', report)
                self.assertEqual([src['strength'] for src in json.loads(report_json)['sources']], [.7, .5, .3])
                self.assertIsNotNone(output.ui)
                self.assertTrue(all(torch.allclose(t, torch.full_like(t, 1.5)) for t in load_file(saved_path).values()))
                comparison = MocH3CompareLorasNode.execute(merged, merged, 'effective_dimensions')
                self.assertEqual(json.loads(comparison.result[1])['score'], 100)
                second = MocH3MergeLorasNode.execute({'lora_0': at(.7), 'lora_1': at(.5)})
                self.assertNotEqual(saved_path, second.result[3])
                self.assertTrue(Path(saved_path).is_file())
                with self.assertRaisesRegex(ValueError, 'within'):
                    MocH3MergeLorasNode.execute({'lora_0': p, 'lora_1': p}, filename_prefix='../escape')

    def test_merge_accepts_twenty_inputs_at_loader_strengths(self):
        from safetensors.torch import load_file
        loras = {f'lora_{i}': dict(source=f'test_{i}', strength=.05, tensors={
            'blocks.0.q.lora_A.weight': torch.full((1, 2), float(i + 1)),
            'blocks.0.q.lora_B.weight': torch.ones(2, 1),
        }) for i in range(20)}
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch('folder_paths.get_output_directory', return_value=directory):
                output = MocH3MergeLorasNode.execute(loras, output_format='full_diff', storage_dtype='float32')
            (diff,) = load_file(output.result[3]).values()
            # 0.05 × (1 + 2 + … + 20) = 10.5
            self.assertTrue(torch.allclose(diff, torch.full((2, 2), 10.5)))
            self.assertEqual(len(json.loads(output.result[2])['sources']), 20)

    def test_failed_merge_publishes_no_files(self):
        p = dict(tensors={'blocks.0.q.dora_scale': torch.ones(2, 2)})
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch('folder_paths.get_output_directory', return_value=directory):
                with self.assertRaisesRegex(ValueError, 'Cannot merge'):
                    MocH3MergeLorasNode.execute({'lora_0': p, 'lora_1': p})
            self.assertEqual(list(Path(directory).rglob('*.safetensors')), [])

    def test_low_disk_space_fails_before_writing_and_cleans_staging_file(self):
        p = dict(tensors={'blocks.0.q.lora_A.weight': torch.ones(1, 16),
                          'blocks.0.q.lora_B.weight': torch.ones(16, 1)})
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch('folder_paths.get_output_directory', return_value=directory), \
                 mock.patch('moc_minimax.lora_merge.shutil.disk_usage', return_value=mock.Mock(free=1)):
                with self.assertRaisesRegex(OSError, 'Merge needs'):
                    MocH3MergeLorasNode.execute({'lora_0': p, 'lora_1': p})
            self.assertEqual([p for p in Path(directory).rglob('*') if p.is_file()], [])

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

AVAILABLE = importlib.util.find_spec('torch') is not None and importlib.util.find_spec('comfy_api') is not None
if AVAILABLE:
    from moc_minimax import lora_stack
    from moc_minimax.lora_stack import (MocApplyLoraStackByNameNode, MocLoraStackNode, apply_stack,
                                        list_stacks, parse_stack, read_stack, validate_name)


def stack(*entries):
    return json.dumps(dict(schema_version=1, entries=[dict(lora=l, strength=s, enabled=e) for l, s, e in entries]))


@unittest.skipUnless(AVAILABLE, 'torch/ComfyUI are required')
class LoraStackTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.loras = self.root / 'loras'
        self.loras.mkdir()
        for name in ('a.safetensors', 'b.safetensors', 'c.safetensors'):
            (self.loras / name).write_bytes(name.encode())
        full_path = lambda folder, name: str(self.loras / name) if (self.loras / name).is_file() else None
        user_dir = lambda user: str(self.root / 'user' / user)
        patches = [mock.patch('folder_paths.get_full_path', side_effect=full_path),
                   mock.patch('folder_paths.get_public_user_directory', side_effect=user_dir),
                   mock.patch('comfy.utils.load_torch_file',
                              side_effect=lambda path, **kw: ({'file': Path(path).name}, {'meta': Path(path).name})),
                   mock.patch('comfy.sd.load_lora_for_models',
                              side_effect=lambda model, clip, lora, s_model, s_clip, lora_metadata=None:
                              (model + [(lora['file'], s_model)], clip))]
        for patch in patches:
            self.addCleanup(patch.stop)
        _, _, self.load_file, self.apply_lora = [p.start() for p in patches]
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(lora_stack._cache.clear)

    def save(self, name, text):
        path = self.root / 'user' / 'default' / 'lora_stacks' / f'{name}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def test_schema_registration(self):
        from moc_minimax.nodes import MocH3Extension
        nodes = asyncio.run(MocH3Extension().get_node_list())
        for node in (MocLoraStackNode, MocApplyLoraStackByNameNode):
            self.assertIn(node, nodes)
            schema = node.define_schema()
            schema.finalize()
            schema.validate()
            self.assertEqual([o.io_type for o in schema.outputs], ['MODEL'])
        self.assertEqual([i.id for i in MocLoraStackNode.define_schema().inputs], ['model', 'stack_data'])

    def test_applies_in_order_like_chained_model_only_loaders(self):
        import nodes
        data = stack(('a.safetensors', .2, True), ('b.safetensors', .5, False),
                     ('c.safetensors', 0, True), ('b.safetensors', -.3, True), ('a.safetensors', 1.5, True))
        with mock.patch.object(MocLoraStackNode, 'hidden', mock.Mock(unique_id='7'), create=True):
            result = MocLoraStackNode.execute([], data).result[0]
        self.assertEqual(result, [('a.safetensors', .2), ('b.safetensors', -.3), ('a.safetensors', 1.5)])
        native_calls = self.apply_lora.call_args_list
        self.apply_lora.reset_mock()
        chained = []
        for name, strength in (('a.safetensors', .2), ('b.safetensors', -.3), ('a.safetensors', 1.5)):
            chained = nodes.LoraLoaderModelOnly().load_lora_model_only(chained, name, strength)[0]
        self.assertEqual(chained, result)
        self.assertEqual(native_calls, self.apply_lora.call_args_list)
        # A file used twice is read once.
        self.assertEqual(self.load_file.call_count, 2 + 3)

    def test_cache_reuses_unchanged_files_and_reloads_replaced_ones(self):
        data = stack(('a.safetensors', 1., True), ('b.safetensors', 1., True))
        apply_stack([], parse_stack(data), cache_key='1')
        apply_stack([], parse_stack(data), cache_key='1')
        self.assertEqual(self.load_file.call_count, 2)
        (self.loras / 'b.safetensors').write_bytes(b'replaced and longer')
        apply_stack([], parse_stack(data), cache_key='1')
        self.assertEqual(self.load_file.call_count, 3)
        apply_stack([], parse_stack(stack(('a.safetensors', 1., True))), cache_key='1')
        self.assertEqual([Path(p).name for p in lora_stack._cache['1']], ['a.safetensors'])

    def test_empty_disabled_and_zero_stacks_pass_model_through(self):
        model = object()
        for data in (lora_stack.EMPTY_STACK, stack(('a.safetensors', 1., False)), stack(('', 0, True)),
                     stack(('missing.safetensors', 1., False))):
            self.assertIs(apply_stack(model, parse_stack(data)), model)
        self.apply_lora.assert_not_called()

    def test_missing_files_fail_before_applying_anything(self):
        data = stack(('a.safetensors', 1., True), ('gone.safetensors', 1., True), ('gone2.safetensors', .5, True))
        with self.assertRaisesRegex(FileNotFoundError, 'gone.safetensors, gone2.safetensors'):
            apply_stack([], parse_stack(data))
        self.apply_lora.assert_not_called()

    def test_validation(self):
        for data, message in (
                ('{', 'not valid JSON'),
                ('{"entries": []}', 'schema_version'),
                ('{"schema_version": 1, "entries": {}}', 'must be a list'),
                (stack(('a.safetensors', 101, True)), 'from -100 to 100'),
                (stack(('a.safetensors', float('nan'), True)), 'strength'),
                (stack(('', 1., True)), 'choose a LoRA file'),
                (json.dumps(dict(schema_version=1, entries=[dict(lora='a', strength=True)])), 'strength'),
                (json.dumps(dict(schema_version=1, entries=[dict(lora='a', enabled='yes')])), 'enabled'),
                (json.dumps(dict(schema_version=1, entries=[dict(lora='a')] * 101)), 'limit is 100')):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                parse_stack(data)
        self.assertEqual(parse_stack(json.dumps(dict(schema_version=1, entries=[dict(lora='a.safetensors')]))),
                         [dict(lora='a.safetensors', strength=1., enabled=True)])
        for name in ('mix', 'portraits/mix 2', 'v1.5_style-x'):
            self.assertEqual(validate_name(name), name)
        for name in ('', '../x', 'a/../b', '.hidden', 'a/', 'trailing.', 'x\\y', 'a:b', None):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Invalid'):
                validate_name(name)

    def test_saved_stacks_by_name(self):
        self.assertEqual(list_stacks(), [])
        self.save('portrait', stack(('a.safetensors', .4, True)))
        self.save('Styles/ink', stack(('b.safetensors', .6, True)))
        self.save('broken name!', stack())
        self.assertEqual(list_stacks(), ['portrait', 'Styles/ink'])
        self.assertEqual(MocApplyLoraStackByNameNode.define_schema().inputs[1].options, ['portrait', 'Styles/ink'])
        self.assertEqual(read_stack('Styles/ink'), [dict(lora='b.safetensors', strength=.6, enabled=True)])
        with mock.patch.object(MocApplyLoraStackByNameNode, 'hidden', mock.Mock(unique_id='3'), create=True):
            self.assertEqual(MocApplyLoraStackByNameNode.execute([], 'portrait').result[0], [('a.safetensors', .4)])
        with self.assertRaisesRegex(FileNotFoundError, "'nope' not found"):
            read_stack('nope')
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            read_stack('../../escape')
        self.save('bad', '{"schema_version": 2}')
        with self.assertRaisesRegex(ValueError, "Saved LoRA stack 'bad'"):
            read_stack('bad')

    def test_by_name_fingerprint_tracks_file_content(self):
        self.save('mix', stack(('a.safetensors', .4, True)))
        first = MocApplyLoraStackByNameNode.fingerprint_inputs(stack_name='mix', model=object())
        self.assertEqual(first, MocApplyLoraStackByNameNode.fingerprint_inputs(stack_name='mix'))
        self.save('mix', stack(('a.safetensors', .5, True)))
        self.assertNotEqual(first, MocApplyLoraStackByNameNode.fingerprint_inputs(stack_name='mix'))
        missing = MocApplyLoraStackByNameNode.fingerprint_inputs(stack_name='nope')
        self.assertNotEqual(missing, missing)


if __name__ == '__main__':
    unittest.main()

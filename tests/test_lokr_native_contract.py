"""Check actual ComfyUI patch math without importing its GPU/runtime dependencies."""
import ast
import importlib.util
import logging
from pathlib import Path
import types
import unittest

TORCH = importlib.util.find_spec('torch')
COMFY = importlib.util.find_spec('comfy')
AVAILABLE = TORCH is not None and COMFY is not None
if AVAILABLE:
    import torch
    from moc_minimax.adapters import parse_adapters
    from moc_minimax.lora_merge import merge_adapters
    COMFY_ROOT = Path(next(iter(COMFY.submodule_search_locations)))


def load_function(path, function_name, namespace, class_name=None):
    source = ast.parse(path.read_text())
    statements = source.body
    if class_name:
        statements = next(n for n in statements if isinstance(n, ast.ClassDef) and n.name == class_name).body
    function = next(n for n in statements if isinstance(n, ast.FunctionDef) and n.name == function_name)
    function.decorator_list = []
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[function_name]


@unittest.skipUnless(AVAILABLE, 'torch and a ComfyUI checkout are required')
class NativeLoKrContractTests(unittest.TestCase):
    def test_native_additive_patch_matches_reconstruction(self):
        path = COMFY_ROOT / 'weight_adapter/lokr.py'
        if not path.exists():
            self.skipTest('ComfyUI checkout has no LoKr adapter')
        cast = lambda t, device, dtype: t.to(device=device, dtype=dtype)
        calculate = load_function(path, 'calculate_weight', {
            'torch': torch, 'logging': logging,
            'comfy': types.SimpleNamespace(model_management=types.SimpleNamespace(cast_to_device=cast)),
        }, class_name='LoKrAdapter')
        w1 = torch.tensor([[1.,2.],[3.,4.]], dtype=torch.float64)
        w2 = torch.tensor([[1.,2.,3.],[4.,5.,6.]], dtype=torch.float64)
        for first, second in ((False, False), (True, False), (False, True), (True, True)):
            state = {'blocks.0.q.alpha': torch.tensor(6.)}
            for index, (weight, decomposed) in enumerate([(w1,first),(w2,second)], 1):
                if decomposed:
                    # Distinct ranks exercise native rank precedence.
                    a = weight[:, :1] if index == 1 else torch.eye(2, dtype=torch.float64)
                    b = weight[:1] if index == 1 else weight
                    state[f'blocks.0.q.lokr_w{index}_a'] = a
                    state[f'blocks.0.q.lokr_w{index}_b'] = b
                else:
                    state[f'blocks.0.q.lokr_w{index}'] = weight
            v = lambda key: state.get('blocks.0.q.' + key)
            holder = types.SimpleNamespace(weights=(v('lokr_w1'),v('lokr_w2'),6.,v('lokr_w1_a'),
                v('lokr_w1_b'),v('lokr_w2_a'),v('lokr_w2_b'),None,None))
            errors = []
            adapter = parse_adapters({'tensors':state}, 'Test', errors)['blocks.0.q']
            self.assertFalse(errors)
            expected = adapter.rows(0, adapter.shape[0], adapter.factors()) * .7
            actual = calculate(holder, torch.zeros(adapter.shape,dtype=torch.float64), 'blocks.0.q',
                               .7, 1., None, lambda x:x, intermediate_dtype=torch.float64)
            torch.testing.assert_close(actual, expected)

    def test_native_loader_consumes_exported_diff(self):
        native_load = load_function(COMFY_ROOT / 'lora.py', 'load_lora', {
            'logging': logging, 'torch': torch, 'weight_adapter': types.SimpleNamespace(adapters=[]),
        })
        payload = {'tensors': {'diffusion_model.blocks.0.q.lokr_w1':torch.ones(2,2),
                               'diffusion_model.blocks.0.q.lokr_w2':torch.ones(2,2)}}
        merged, _ = merge_adapters([payload, payload], [.7,.5])
        patches = native_load(merged['tensors'], {'diffusion_model.blocks.0.q':'diffusion_model.blocks.0.q.weight'})
        self.assertEqual(set(patches), {'diffusion_model.blocks.0.q.weight'})
        kind, (delta,) = patches['diffusion_model.blocks.0.q.weight']
        self.assertEqual(kind, 'diff')
        torch.testing.assert_close(delta, torch.full((4,4), 1.2))

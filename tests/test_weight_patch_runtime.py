from __future__ import annotations

import copy
import types
import unittest

try:
    import torch  # noqa: F401
    import comfy.patcher_extension
except ImportError:  # pragma: no cover
    comfy = None
else:
    from moc_minimax.weighting import PATCH_KEY, install_reference_weight_patch


class _FakeModelPatcher:
    def __init__(self, diffusion_model, wrappers=None):
        self.diffusion_model = diffusion_model
        self.wrappers = copy.deepcopy(wrappers or {})

    def get_model_object(self, name):
        if name != "diffusion_model":
            raise KeyError(name)
        return self.diffusion_model

    def clone(self):
        return _FakeModelPatcher(self.diffusion_model, self.wrappers)

    def remove_wrappers_with_key(self, wrapper_type, key):
        self.wrappers.get(wrapper_type, {}).pop(key, None)

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)


@unittest.skipIf(comfy is None, "ComfyUI runtime is not installed in this test environment")
class ModelPatchRuntimeTests(unittest.TestCase):
    def setUp(self):
        model_type = type("MiniMaxH3Model", (), {"__module__": "comfy.ldm.minimax.model"})
        self.model = _FakeModelPatcher(model_type())
        self.wrapper_type = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL

    def test_reapplying_patch_replaces_keyed_wrapper(self):
        first = install_reference_weight_patch(self.model, method="value_gate", strength=1.0)
        second = install_reference_weight_patch(first, method="attention_prior", strength=0.5)
        wrappers = second.wrappers[self.wrapper_type][PATCH_KEY]
        self.assertEqual(len(wrappers), 1)

    def test_nonunit_weights_reject_existing_attention_override(self):
        patched = install_reference_weight_patch(self.model, method="value_gate", strength=1.0)
        wrapper = patched.wrappers[self.wrapper_type][PATCH_KEY][0]
        payload = {
            "refs": [{"kind": "image", "moc_visual_weight": 0.5}],
            "layout": types.SimpleNamespace(
                segments=[(0, 2, "text"), (2, 4, "ref_img"), (4, 6, "audio"), (6, 8, "video")]
            ),
        }
        with self.assertRaisesRegex(RuntimeError, "cannot safely compose"):
            wrapper(
                lambda *_args, **_kwargs: None,
                object(),
                object(),
                object(),
                transformer_options={"optimized_attention_override": lambda *_args, **_kwargs: None},
                minimax_payload=payload,
            )

    def test_unit_weights_preserve_native_override_path(self):
        patched = install_reference_weight_patch(self.model, method="value_gate", strength=1.0)
        wrapper = patched.wrappers[self.wrapper_type][PATCH_KEY][0]
        marker = object()
        payload = {
            "refs": [{"kind": "image", "moc_visual_weight": 1.0}],
            "layout": types.SimpleNamespace(
                segments=[(0, 2, "text"), (2, 4, "ref_img"), (4, 6, "audio"), (6, 8, "video")]
            ),
        }
        result = wrapper(
            lambda *_args, **_kwargs: marker,
            object(),
            object(),
            object(),
            transformer_options={"optimized_attention_override": lambda *_args, **_kwargs: None},
            minimax_payload=payload,
        )
        self.assertIs(result, marker)


if __name__ == "__main__":
    unittest.main()

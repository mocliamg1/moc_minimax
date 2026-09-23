#!/usr/bin/env python3
"""Import this extension and validate every V3 schema in a ComfyUI runtime."""

from __future__ import annotations

import asyncio
import ast
import contextlib
import importlib.util
import inspect
import json
import re
import sys
import types
from pathlib import Path


MIN_COMFYUI_VERSION = (0, 32, 0)
NATIVE_NODE_ID = "MiniMaxH3ReferenceToVideo"
NATIVE_EXECUTE_INPUTS = (
    "clip",
    "vae",
    "audio_vae",
    "prompt",
    "width",
    "height",
    "length",
    "ref_image_size",
    "ref_images",
    "ref_videos",
    "ref_video_audios",
    "ref_audios",
)


def _result(output):
    return output.result if hasattr(output, "result") else output


def _read_comfyui_version(comfy_root: Path) -> str:
    """Read the generated version without importing an unrelated checkout."""
    version_file = comfy_root / "comfyui_version.py"
    if not version_file.is_file():
        raise RuntimeError(f"ComfyUI checkout has no comfyui_version.py: {comfy_root}")
    tree = ast.parse(version_file.read_text(encoding="utf-8"), filename=str(version_file))
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if not any(isinstance(target, ast.Name) and target.id == "__version__" for target in targets):
            continue
        value = ast.literal_eval(statement.value)
        if isinstance(value, str):
            return value
    raise RuntimeError(f"Could not read __version__ from {version_file}")


def _release_tuple(version: str) -> tuple[int, int, int]:
    match = re.match(r"^\s*(\d+)\.(\d+)\.(\d+)", version)
    if match is None:
        raise RuntimeError(f"Unsupported ComfyUI version format: {version!r}")
    return tuple(int(part) for part in match.groups())


@contextlib.contextmanager
def _native_contract_import_stubs():
    """Avoid booting all of ComfyUI when only the native class schema is needed.

    A complete ComfyUI environment takes the normal import path. This fallback
    is for lightweight CI environments that intentionally provide Torch and the
    V3 API but omit runtime-only packages such as torchaudio and comfy-aimdo.
    The real native module is still executed; only dependencies referenced from
    method bodies (which this checkpoint-free validator never calls) are stubs.
    """
    import comfy
    from comfy_api.latest import io as _io  # noqa: F401 - ensure the real V3 API is loaded first

    module_stubs = {
        "nodes": types.ModuleType("nodes"),
        "node_helpers": types.ModuleType("node_helpers"),
        "torchaudio": types.ModuleType("torchaudio"),
        "comfy.model_management": types.ModuleType("comfy.model_management"),
        "comfy.model_sampling": types.ModuleType("comfy.model_sampling"),
        "comfy.nested_tensor": types.ModuleType("comfy.nested_tensor"),
        "comfy.utils": types.ModuleType("comfy.utils"),
    }
    module_stubs["nodes"].MAX_RESOLUTION = 16384
    module_stubs["torchaudio"].functional = types.SimpleNamespace()

    missing = object()
    saved_modules = {name: sys.modules.get(name, missing) for name in module_stubs}
    saved_comfy_attrs = {
        name.rsplit(".", 1)[-1]: getattr(comfy, name.rsplit(".", 1)[-1], missing)
        for name in module_stubs
        if name.startswith("comfy.")
    }
    try:
        for name, stub in module_stubs.items():
            sys.modules[name] = stub
            if name.startswith("comfy."):
                setattr(comfy, name.rsplit(".", 1)[-1], stub)
        yield
    finally:
        for name, previous in saved_modules.items():
            if previous is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        for attr, previous in saved_comfy_attrs.items():
            if previous is missing:
                delattr(comfy, attr)
            else:
                setattr(comfy, attr, previous)


def _isolated_native_import(comfy_root: Path):
    module_path = comfy_root / "comfy_extras" / "nodes_minimax_h3.py"
    if not module_path.is_file():
        raise RuntimeError(f"This ComfyUI checkout has no native MiniMax H3 node: {module_path}")
    module_name = "_moc_minimax_native_h3_contract"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not construct native H3 import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    with _native_contract_import_stubs():
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
    return module.MiniMaxH3ReferenceToVideo, module_path.resolve()


def _validate_native_contract(comfy_root: Path):
    """Import native H3 and prove our named delegation call still binds."""
    import_mode = "runtime import"
    expected_source = None
    try:
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
    except ModuleNotFoundError:
        MiniMaxH3ReferenceToVideo, expected_source = _isolated_native_import(comfy_root)
        import_mode = "isolated contract import"
    except Exception as exc:
        raise RuntimeError(
            "Could not import native MiniMaxH3ReferenceToVideo from this ComfyUI runtime"
        ) from exc

    if expected_source is None:
        source = inspect.getsourcefile(MiniMaxH3ReferenceToVideo)
        expected_source = Path(source).resolve() if source is not None else None
    if expected_source is None or comfy_root not in expected_source.parents:
        raise RuntimeError(
            "Native MiniMaxH3ReferenceToVideo resolved outside the requested ComfyUI checkout: "
            f"{expected_source or '<unknown>'}"
        )

    schema = MiniMaxH3ReferenceToVideo.define_schema()
    schema.finalize()
    schema.validate()
    if schema.node_id != NATIVE_NODE_ID:
        raise RuntimeError(f"Unexpected native H3 node id: {schema.node_id!r}")
    output_types = tuple(output.io_type for output in schema.outputs)
    if output_types != ("CONDITIONING", "LATENT"):
        raise RuntimeError(f"Unexpected native H3 output contract: {output_types!r}")

    signature = inspect.signature(MiniMaxH3ReferenceToVideo.execute)
    probe = object()
    try:
        bound = signature.bind(
            clip=probe,
            vae=probe,
            audio_vae=probe,
            prompt=probe,
            width=1344,
            height=768,
            length=124,
            ref_image_size="match",
            ref_images={},
            ref_videos={},
            ref_video_audios={},
            ref_audios={},
        )
    except TypeError as exc:
        raise RuntimeError(
            "Native MiniMaxH3ReferenceToVideo.execute is incompatible with MOC delegation: "
            f"{signature}"
        ) from exc
    missing = [name for name in NATIVE_EXECUTE_INPUTS if name not in bound.arguments]
    if missing:
        raise RuntimeError(f"Native H3 execute contract is missing input(s): {', '.join(missing)}")
    return MiniMaxH3ReferenceToVideo, signature, import_mode


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python scripts/validate_comfy.py /path/to/ComfyUI", file=sys.stderr)
        return 2
    comfy_root = Path(sys.argv[1]).expanduser().resolve()
    if not (comfy_root / "comfy_api" / "latest").exists():
        print(f"not a current ComfyUI checkout: {comfy_root}", file=sys.stderr)
        return 2
    version = _read_comfyui_version(comfy_root)
    if _release_tuple(version) < MIN_COMFYUI_VERSION:
        required = ".".join(str(part) for part in MIN_COMFYUI_VERSION)
        print(f"ComfyUI {version} is unsupported; MOC requires >= {required}", file=sys.stderr)
        return 2
    print(f"ok  ComfyUI version: {version}")

    plugin_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(comfy_root))

    _native_node, native_signature, native_import_mode = _validate_native_contract(comfy_root)
    print(
        f"ok  native H3 contract ({native_import_mode}): "
        f"{NATIVE_NODE_ID}.execute{native_signature}"
    )

    package_name = "moc_minimax_custom_node_validation"
    spec = importlib.util.spec_from_file_location(
        package_name,
        plugin_root / "__init__.py",
        submodule_search_locations=[str(plugin_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not construct plugin import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    extension = asyncio.run(module.comfy_entrypoint())
    node_list = asyncio.run(extension.get_node_list())
    by_id = {}
    for node in node_list:
        schema = node.define_schema()
        schema.validate()
        by_id[schema.node_id] = node
        print(f"ok  {schema.node_id}: {schema.display_name}")
    print(f"validated {len(node_list)} nodes")

    # Exercise builder -> set -> compiler and the main node's V3 delegation
    # boundary without loading any H3 checkpoints.
    import torch

    smoke_lora = {"source": "synthetic", "tensors": {
        "blocks.0.attn.q.lora_A.weight": torch.ones((2, 4)),
        "blocks.0.attn.q.lora_B.weight": torch.ones((4, 2)),
    }}
    comparison = _result(by_id["MocH3CompareLoras"].execute(smoke_lora, smoke_lora))
    comparison_data = json.loads(comparison[1])
    if not comparison_data["compatible"] or comparison_data["score"] != 100.0:
        raise RuntimeError("LoRA comparison smoke test failed")
    print("ok  execution smoke: identical LoRA comparison")
    lora_module = sys.modules[f"{package_name}.moc_minimax.lora_nodes"]
    smoke_lokr = {"source": "synthetic LoKr", "tensors": {
        "diffusion_model.blocks.0.attn.q.lokr_w1": torch.ones((2, 2)),
        "diffusion_model.blocks.0.attn.q.lokr_w2": torch.ones((2, 2)),
    }}
    merged_lokr, merge_report = lora_module.merge_adapters([smoke_lokr, smoke_lokr], [.7, .3])
    lokr_comparison = _result(by_id["MocH3CompareLoras"].execute(
        smoke_lokr, merged_lokr, matching="effective_dimensions"))
    if json.loads(lokr_comparison[1])["score"] != 100.0 or merge_report["overall"]["relative_error"] != 0.0:
        raise RuntimeError("LoKr merge/comparison smoke test failed")
    print("ok  execution smoke: LoKr merge and cross-format comparison")

    smoke_image = torch.ones((1, 32, 32, 3))
    smoke_mask = torch.zeros((1, 32, 32))
    smoke_mask[:, 8:24, 8:24] = 1.0
    image_output = by_id["MocH3ImageReference"].execute(
        smoke_image,
        "hero",
        "identity",
        "primary",
        1.0,
        "match_output",
        "",
        "",
        mask=smoke_mask,
        mask_feather_px=0,
        neutral_level=0.25,
    )
    reference = _result(image_output)[0]
    if not reference["metadata"].get("mask_applied"):
        raise RuntimeError("image builder did not record source-isolation mask metadata")
    if not torch.all(reference["media"][:, 0, 0] == 0.25):
        raise RuntimeError("image builder did not composite the masked source exterior")
    set_output = by_id["MocH3ReferenceSet"].execute({"reference_0": reference})
    reference_set, mapping = _result(set_output)[:2]
    if "@hero" not in mapping or "<Picture 1>" not in mapping:
        raise RuntimeError(f"unexpected alias mapping: {mapping}")
    compile_output = by_id["MocH3CompileReferencePrompt"].execute(
        reference_set,
        "A portrait of @hero.",
        "guided",
        "strict",
        1344,
        768,
        124,
        "match_output",
    )
    if "<Picture 1>" not in _result(compile_output)[0]:
        raise RuntimeError("compiled prompt did not resolve @hero")

    nodes_module = sys.modules[f"{package_name}.moc_minimax.nodes"]
    original_native_factory = nodes_module._native_node_class

    native_call = {}

    class FakeNative:
        @classmethod
        def execute(cls, *args, **kwargs):
            native_call["args"] = args
            native_call["kwargs"] = kwargs
            conditioning = [[object(), {"minimax_refs": [{"kind": "image", "latent_h": 2, "latent_w": 2}]}]]
            return nodes_module.io.NodeOutput(conditioning, {"samples": "smoke"})

    try:
        nodes_module._native_node_class = lambda: FakeNative
        main_output = by_id["MocH3ReferenceToVideoPlus"].execute(
            object(),
            object(),
            object(),
            reference_set,
            "A portrait of @hero.",
            1344,
            768,
            124,
            "guided",
            "strict",
            "match_output",
            0.999,
            1.0,
        )
    finally:
        nodes_module._native_node_class = original_native_factory
    main_values = _result(main_output)
    if not isinstance(main_values, tuple) or len(main_values) != 5:
        raise RuntimeError(f"main node returned an unexpected V3 NodeOutput boundary: {type(main_values)!r}")
    if native_call.get("args", ()):
        raise RuntimeError(f"main node passed an unexpected native positional boundary: {native_call!r}")
    expected_native_kwargs = {
        "clip", "vae", "audio_vae", "prompt", "width", "height", "length",
        "ref_image_size",
        "ref_images",
        "ref_videos",
        "ref_video_audios",
        "ref_audios",
    }
    if set(native_call.get("kwargs", {})) != expected_native_kwargs:
        raise RuntimeError(f"main node passed unexpected native named inputs: {native_call!r}")
    if list(native_call["kwargs"]["ref_images"]) != ["ref_image_0"]:
        raise RuntimeError(f"main node did not preserve native image numbering: {native_call!r}")
    if not torch.equal(native_call["kwargs"]["ref_images"]["ref_image_0"], reference["media"]):
        raise RuntimeError("main node did not delegate the exact masked RGB reference tensor")

    conditioning_metadata = main_values[0][0][1]
    annotated = conditioning_metadata["minimax_refs"][0]
    expected_annotation = {
        "moc_name": "hero",
        "moc_kind": "image",
        "moc_role": "identity",
        "moc_priority": "primary",
        "moc_visual_weight": 1.0,
        "moc_audio_weight": 1.0,
    }
    if any(annotated.get(key) != value for key, value in expected_annotation.items()):
        raise RuntimeError(f"main node emitted incomplete reference metadata: {annotated!r}")
    if conditioning_metadata.get("minimax_visual_cond_noise_aug") != 0.999:
        raise RuntimeError("main node did not attach visual-reference fidelity metadata")
    if conditioning_metadata.get("minimax_audio_cond_noise_aug") != 1.0:
        raise RuntimeError("main node did not attach audio-reference fidelity metadata")
    embedded_manifest = conditioning_metadata.get("moc_h3_reference_manifest")
    returned_manifest = json.loads(main_values[4])
    if embedded_manifest != returned_manifest:
        raise RuntimeError("conditioning metadata and manifest output disagree")
    if main_values[1].get("samples") != "smoke":
        raise RuntimeError("main-node native delegation smoke test failed")

    native_call.clear()
    try:
        nodes_module._native_node_class = lambda: FakeNative
        hybrid_output = by_id["MocH3HybridImageReferencesToVideo"].execute(
            object(),
            object(),
            "A portrait using <Picture 1> as a non-temporal reference.",
            1344,
            768,
            124,
            "match",
            reference_images={"reference_image_0": smoke_image},
        )
    finally:
        nodes_module._native_node_class = original_native_factory
    hybrid_values = _result(hybrid_output)
    if not isinstance(hybrid_values, tuple) or len(hybrid_values) != 2:
        raise RuntimeError("hybrid Image to Video node did not preserve the native two-output boundary")
    if list(native_call.get("kwargs", {}).get("ref_images", {})) != ["ref_image_0"]:
        raise RuntimeError(f"hybrid node did not delegate its ordinary IMAGE reference input: {native_call!r}")
    if not torch.equal(native_call["kwargs"]["ref_images"]["ref_image_0"], smoke_image):
        raise RuntimeError("hybrid node changed the direct IMAGE reference before native H3 delegation")
    print("ok  execution smoke: MOC reference and direct-IMAGE hybrid native delegation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

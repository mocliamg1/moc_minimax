"""ComfyUI V3 loader and inspection nodes for MiniMax LoRAs."""
from pathlib import Path

from comfy_api.latest import io, ui

from .adapters import payload_strength
from .lora_compare import compare_loras, comparison_json, render_comparison
from .lora_merge import FORMATS, DTYPES, SafetensorsStream, merge_adapters, render_merge

MocH3Lora = io.Custom('MOC_H3_LORA')
CATEGORY = 'MiniMax H3/MOC LoRA'
MAX_MERGE_INPUTS = 20


def _path(name):
    import folder_paths
    return folder_paths.get_full_path_or_raise('loras', name)


class MocH3LoadLoraNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        import folder_paths
        return io.Schema(
            node_id='MocH3LoadLora', display_name='MOC • H3 Load LoRA', category=CATEGORY,
            description='Load LoRA, LoKr, or full-difference tensors on CPU for comparison or merging. No base model required.',
            inputs=[io.Combo.Input('lora_name', options=folder_paths.get_filename_list('loras')),
                    io.Float.Input('strength', default=1.0, min=-20.0, max=20.0, step=0.05,
                                   tooltip='Use the same strength as your working stack. Merge and Compare apply it; merge strengths are never normalized and 0 excludes this adapter.')],
            outputs=[MocH3Lora.Output('lora')])

    @classmethod
    def fingerprint_inputs(cls, lora_name, strength=1.0):
        try:
            path = Path(_path(lora_name))
            stat = path.stat()
            return (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
        except (FileNotFoundError, OSError):
            return float('nan')

    @classmethod
    def execute(cls, lora_name, strength=1.0):
        # Use safe tensor loading without initializing ComfyUI's model/GPU runtime.
        import torch
        path = _path(lora_name)
        if Path(path).suffix.lower() in ('.safetensors', '.sft'):
            from safetensors.torch import load_file
            tensors = load_file(path, device='cpu')
        else:
            tensors = torch.load(path, map_location='cpu', weights_only=True)
            if isinstance(tensors, dict) and 'state_dict' in tensors:
                tensors = tensors['state_dict']
        if not isinstance(tensors, dict) or not tensors:
            raise ValueError(f'{lora_name}: expected a nonempty LoRA state dictionary.')
        return io.NodeOutput(dict(tensors=tensors, source=lora_name, strength=float(strength)))


class MocH3CompareLorasNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='MocH3CompareLoras', display_name='MOC • H3 Compare LoRAs', category=CATEGORY,
            description='Compare LoRA, LoKr, or full-difference updates with strict or effective-dimension matching. Scores do not predict visual similarity.',
            is_output_node=True,
            inputs=[MocH3Lora.Input('lora_a'), MocH3Lora.Input('lora_b'),
                    io.Combo.Input('matching', options=['strict', 'effective_dimensions'], default='strict', optional=True,
                                   tooltip='Strict requires the same adapter format and factor shapes. Effective dimensions allows different formats/ranks with identical target layers and weight dimensions.')],
            outputs=[io.String.Output('report'), io.String.Output('report_json')])

    @classmethod
    def execute(cls, lora_a, lora_b, matching="strict"):
        result = compare_loras(lora_a, lora_b, matching=matching)
        report = render_comparison(result)
        return io.NodeOutput(report, comparison_json(result), ui=ui.PreviewText(report))


class MocH3MergeLorasNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        inputs = [
            io.Autogrow.Input('loras', template=io.Autogrow.TemplatePrefix(
                input=MocH3Lora.Input('lora', tooltip='Adapter merged at its Load LoRA strength. Connecting the last socket reveals another.'),
                prefix='lora_', min=2, max=MAX_MERGE_INPUTS)),
            io.Combo.Input('output_format', options=list(FORMATS), default='auto',
                           tooltip='auto preserves compact LoRA/shared LoKr factors when smaller, otherwise uses full differences. full_diff can be model-sized. lokr_shared_or_diff keeps shared LoKr factors when possible. lora_svd is an approximate rank-limited LoRA.'),
            io.Int.Input('rank', default=64, min=1, max=4096, step=1, advanced=True,
                         tooltip='Used only for lora_svd. Large layers use a seeded randomized SVD; the report measures the actual truncation error.'),
            io.Combo.Input('storage_dtype', options=list(DTYPES), default='bfloat16', advanced=True,
                           tooltip='bfloat16 halves float32 file size; its rounding is far below the rounding of applying an update to bfloat16 model weights. Use float32 for bit-for-bit reference merges.'),
            io.String.Input('filename_prefix', default='loras/moc_minimax_merge',
                            tooltip='Save under ComfyUI output. A numbered suffix prevents overwriting files.'),
        ]
        return io.Schema(
            node_id='MocH3MergeLoras', display_name='MOC • H3 Merge LoRA / LoKr', category=CATEGORY,
            description=f'Merge 2–{MAX_MERGE_INPUTS} additive adapters at their Load LoRA strengths and save safetensors. Missing layers contribute zero; shared layers must have the same effective weight dimensions. No base model needed.',
            is_output_node=True, inputs=inputs,
            outputs=[MocH3Lora.Output('lora'), io.String.Output('report'),
                     io.String.Output('report_json'), io.String.Output('saved_path')])

    @classmethod
    def execute(cls, loras, output_format='auto', rank=64, storage_dtype='bfloat16',
                filename_prefix='loras/moc_minimax_merge'):
        import folder_paths
        from safetensors.torch import load_file
        payloads = [payload for payload in (loras or {}).values() if payload is not None]
        strengths = [payload_strength(payload) for payload in payloads]
        root = Path(folder_paths.get_output_directory()).resolve()
        requested = (root / filename_prefix).resolve()
        if not filename_prefix.strip() or not requested.is_relative_to(root) or requested == root:
            raise ValueError('filename_prefix must name a file within the ComfyUI output directory.')
        folder, filename, counter, _, _ = folder_paths.get_save_image_path(filename_prefix, str(root))
        if not Path(folder).resolve().is_relative_to(root):
            raise ValueError('Export folder must be within the ComfyUI output directory.')
        stream = SafetensorsStream(folder)
        try:
            _, data = merge_adapters(payloads, strengths, output_format=output_format, rank=rank,
                                     storage_dtype=storage_dtype, tensor_sink=stream.add,
                                     tensor_preflight=stream.preflight)
            while True:
                path = Path(folder) / f'{filename}_{counter:05}_.safetensors'
                data['saved_path'] = str(path)
                try:
                    stream.finish(path, data)
                    break
                except FileExistsError:
                    counter += 1
        finally:
            stream.close()
        report = render_merge(data) + f'\nSaved: {path}'
        # Safetensors maps the output file; it need not be copied into RAM.
        merged = dict(tensors=load_file(str(path), device='cpu'), source=path.name)
        return io.NodeOutput(merged, report, comparison_json(data), str(path), ui=ui.PreviewText(report))

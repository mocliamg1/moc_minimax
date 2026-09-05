"""ComfyUI V3 loader and inspection nodes for MiniMax LoRAs."""
from pathlib import Path

from comfy_api.latest import io, ui

from .lora_compare import compare_loras, comparison_json, render_comparison
from .lora_merge import FORMATS, DTYPES, SafetensorsStream, merge_adapters, render_merge

MocH3Lora = io.Custom('MOC_H3_LORA')
CATEGORY = 'MiniMax H3/MOC LoRA'


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
            inputs=[io.Combo.Input('lora_name', options=folder_paths.get_filename_list('loras'))],
            outputs=[MocH3Lora.Output('lora')])

    @classmethod
    def fingerprint_inputs(cls, lora_name):
        try:
            path = Path(_path(lora_name))
            stat = path.stat()
            return (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
        except (FileNotFoundError, OSError):
            return float('nan')

    @classmethod
    def execute(cls, lora_name):
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
        return io.NodeOutput(dict(tensors=tensors, source=lora_name))


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
        inputs = []
        for letter in 'abcdefgh':
            inputs.extend([
                MocH3Lora.Input('lora_' + letter, optional=letter not in 'ab'),
                io.Float.Input('strength_' + letter, default=1.0, min=-20.0, max=20.0,
                               step=0.05, optional=letter not in 'ab',
                               tooltip='Use the same strength as your working stack. Strengths are never normalized; 0 excludes this adapter.'),
            ])
        inputs.extend([
            io.Combo.Input('output_format', options=list(FORMATS), default='full_diff',
                           tooltip='full_diff preserves the sum within storage precision. lokr_shared_or_diff keeps identical shared LoKr factors when possible. lora_svd is an approximate rank-limited LoRA.'),
            io.Int.Input('rank', default=64, min=1, max=4096, step=1, advanced=True,
                         tooltip='Used only for lora_svd. CPU SVD can be expensive on large layers.'),
            io.Combo.Input('storage_dtype', options=list(DTYPES), default='float32', advanced=True),
            io.String.Input('filename_prefix', default='loras/moc_minimax_merge',
                            tooltip='Save under ComfyUI output. A numbered suffix prevents overwriting files.'),
        ])
        return io.Schema(
            node_id='MocH3MergeLoras', display_name='MOC • H3 Merge LoRA / LoKr', category=CATEGORY,
            description='Merge 2–8 additive adapters with individual strengths and save safetensors. Missing layers contribute zero; shared layers must have the same effective weight dimensions. No base model needed.',
            is_output_node=True, inputs=inputs,
            outputs=[MocH3Lora.Output('lora'), io.String.Output('report'),
                     io.String.Output('report_json'), io.String.Output('saved_path')])

    @classmethod
    def execute(cls, lora_a, strength_a, lora_b, strength_b, output_format='full_diff',
                rank=64, storage_dtype='float32', filename_prefix='loras/moc_minimax_merge', **kwargs):
        import folder_paths
        from safetensors.torch import load_file
        payloads, strengths = [lora_a, lora_b], [strength_a, strength_b]
        for letter in 'cdefgh':
            payload = kwargs.get('lora_' + letter)
            if payload is not None:
                payloads.append(payload)
                strengths.append(kwargs.get('strength_' + letter, 1.0))
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
                                     storage_dtype=storage_dtype, tensor_sink=stream.add)
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

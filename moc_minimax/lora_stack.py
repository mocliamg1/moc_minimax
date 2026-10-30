"""Ordered model-only LoRA stacks with named presets in the ComfyUI user directory.

A stack applies its enabled entries top to bottom, each at strength × the node's
strength multiplier, with the same call as chained LoraLoaderModelOnly nodes.
Presets are JSON files under user/<user>/lora_stacks; the stack node's interface
(web/lora_stack.js) saves them through ComfyUI's user-data API and keeps a copy
of the stack in the workflow. The multiplier is a node setting, not part of presets.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re

from comfy_api.latest import io

SCHEMA_VERSION = 1
STACK_DIR = 'lora_stacks'
MAX_ENTRIES = 100
STRENGTH_LIMIT = 100.0
MULTIPLIER_LIMIT = 10.0
EMPTY_STACK = json.dumps(dict(schema_version=SCHEMA_VERSION, entries=[]))
CATEGORY = 'MOC/LoRA'
_NAME_PART = re.compile(r'[A-Za-z0-9_\-][A-Za-z0-9 _.\-]{0,99}')

# Loaded LoRA files per node id, limited to the files that node's last run used.
_cache: dict[str, dict[str, tuple]] = {}


def validate_name(name):
    """Stack names are relative paths without '.json'; '/' makes subfolders."""
    parts = name.split('/') if isinstance(name, str) else []
    if not parts or not all(_NAME_PART.fullmatch(p) and not p.endswith((' ', '.')) for p in parts):
        raise ValueError(f'Invalid LoRA stack name {name!r}: use letters, digits, spaces, "_", "-", "." '
                         'and "/" for subfolders.')
    return name


def parse_stack(data):
    """Validate a stack (JSON text or dict) and return its entries."""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as error:
            raise ValueError(f'LoRA stack is not valid JSON: {error}.') from None
    if not isinstance(data, dict) or data.get('schema_version') != SCHEMA_VERSION:
        raise ValueError(f'LoRA stack must be an object with schema_version {SCHEMA_VERSION}.')
    entries = data.get('entries')
    if not isinstance(entries, list):
        raise ValueError('LoRA stack entries must be a list.')
    if len(entries) > MAX_ENTRIES:
        raise ValueError(f'LoRA stack has {len(entries)} entries; the limit is {MAX_ENTRIES}.')
    parsed = []
    for index, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            raise ValueError(f'LoRA stack entry {index} must be an object.')
        lora, strength, enabled = entry.get('lora', ''), entry.get('strength', 1.0), entry.get('enabled', True)
        if not isinstance(lora, str):
            raise ValueError(f'LoRA stack entry {index}: lora must be a file name.')
        if (isinstance(strength, bool) or not isinstance(strength, (int, float)) or
                not math.isfinite(strength) or abs(strength) > STRENGTH_LIMIT):
            raise ValueError(f'LoRA stack entry {index}: strength must be a number from '
                             f'-{STRENGTH_LIMIT:g} to {STRENGTH_LIMIT:g}.')
        if not isinstance(enabled, bool):
            raise ValueError(f'LoRA stack entry {index}: enabled must be true or false.')
        if enabled and strength != 0 and not lora:
            raise ValueError(f'LoRA stack entry {index}: choose a LoRA file or disable the entry.')
        parsed.append(dict(lora=lora, strength=float(strength), enabled=enabled))
    return parsed


def stack_directory():
    import folder_paths
    return Path(folder_paths.get_public_user_directory('default')) / STACK_DIR


def stack_path(name):
    root = stack_directory().resolve()
    path = (root / f'{validate_name(name)}.json').resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Invalid LoRA stack name {name!r}.')
    return path


def list_stacks():
    root = stack_directory()
    if not root.is_dir():
        return []
    names = []
    for path in root.rglob('*.json'):
        name = path.relative_to(root).with_suffix('').as_posix()
        try:
            names.append(validate_name(name))
        except ValueError:
            continue
    return sorted(names, key=str.casefold)


def read_stack(name):
    path = stack_path(name)
    try:
        text = path.read_text(encoding='utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(f'Saved LoRA stack {name!r} not found in {path.parent}.') from None
    try:
        return parse_stack(text)
    except ValueError as error:
        raise ValueError(f'Saved LoRA stack {name!r}: {error}') from None


def _load(path, previous):
    import comfy.utils
    stat = os.stat(path)
    signature = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
    cached = previous.get(path)
    if cached is not None and cached[0] == signature:
        return cached
    lora, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    return signature, lora, metadata


def apply_stack(model, entries, multiplier=1.0, cache_key=None):
    """Apply enabled entries in order at strength × multiplier, like chained model-only LoRA loaders."""
    import comfy.sd
    import folder_paths
    if isinstance(multiplier, bool) or not isinstance(multiplier, (int, float)) or not math.isfinite(multiplier):
        raise ValueError('Strength multiplier must be a finite number.')
    active = [dict(e, strength=e['strength'] * multiplier) for e in entries
              if e['enabled'] and e['strength'] * multiplier != 0]
    paths = {e['lora']: folder_paths.get_full_path('loras', e['lora']) for e in active}
    missing = [name for name, path in paths.items() if path is None]
    if missing:
        raise FileNotFoundError('LoRA files not found in the loras folders: ' + ', '.join(missing))
    previous = _cache.get(cache_key, {}) if cache_key is not None else {}
    loaded = {}
    for entry in active:
        path = paths[entry['lora']]
        loaded[path] = loaded.get(path) or _load(path, previous)
        _, lora, metadata = loaded[path]
        model = comfy.sd.load_lora_for_models(model, None, lora, entry['strength'], 0,
                                              lora_metadata=metadata)[0]
    if cache_key is not None:
        _cache[cache_key] = loaded
    return model


class MocLoraStackNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='MocLoraStack', display_name='MOC • LoRA Stack', category=CATEGORY,
            description='Apply an ordered list of LoRAs to the model, exactly like chaining model-only LoRA '
                        'loaders. Save and load stacks by name; the workflow keeps its own copy.',
            search_aliases=['lora stack', 'lora chain', 'multiple loras', 'lora preset'],
            inputs=[
                io.Model.Input('model'),
                io.String.Input('stack_data', default=EMPTY_STACK,
                                tooltip='Managed by the stack editor. Holds the rows applied when the workflow runs.'),
                io.Float.Input('strength_multiplier', default=1.0, min=-MULTIPLIER_LIMIT, max=MULTIPLIER_LIMIT, step=0.05,
                               tooltip='Multiplies every strength in the stack. Not saved with presets; 0 disables the stack.'),
            ],
            outputs=[io.Model.Output('model')],
            hidden=[io.Hidden.unique_id])

    @classmethod
    def execute(cls, model, stack_data=EMPTY_STACK, strength_multiplier=1.0):
        return io.NodeOutput(apply_stack(model, parse_stack(stack_data), strength_multiplier,
                                         cache_key=str(cls.hidden.unique_id)))


class MocApplyLoraStackByNameNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='MocApplyLoraStackByName', display_name='MOC • Apply LoRA Stack by Name', category=CATEGORY,
            description='Apply a saved LoRA stack, reading the file on every run, so edits to the saved stack '
                        'apply to every workflow that uses it.',
            inputs=[
                io.Model.Input('model'),
                io.Combo.Input('stack_name', options=list_stacks(),
                               tooltip=f'Stacks saved by MOC • LoRA Stack in user/default/{STACK_DIR}.'),
                io.Float.Input('strength_multiplier', default=1.0, min=-MULTIPLIER_LIMIT, max=MULTIPLIER_LIMIT, step=0.05,
                               tooltip='Multiplies every strength in the stack. Not saved with presets; 0 disables the stack.'),
            ],
            outputs=[io.Model.Output('model')],
            hidden=[io.Hidden.unique_id])

    @classmethod
    def fingerprint_inputs(cls, stack_name, **kwargs):
        # Re-run when the saved file changes.
        try:
            return hashlib.sha256(stack_path(stack_name).read_bytes()).hexdigest()
        except (OSError, ValueError):
            return float('nan')

    @classmethod
    def execute(cls, model, stack_name, strength_multiplier=1.0):
        return io.NodeOutput(apply_stack(model, read_stack(stack_name), strength_multiplier,
                                         cache_key=str(cls.hidden.unique_id)))

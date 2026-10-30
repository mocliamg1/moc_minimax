"""ComfyUI entrypoint for MOC MiniMax H3 References."""

from .moc_minimax.nodes import MocH3Extension


async def comfy_entrypoint() -> MocH3Extension:
    return MocH3Extension()


# Frontend for the LoRA Stack editor.
WEB_DIRECTORY = "./web"

__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]


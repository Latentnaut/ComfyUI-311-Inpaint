"""
ComfyUI-311-Inpaint — Inpaint helper nodes (311 series).

V3: Add Margins 311 (comfy_entrypoint).
V1 drop-in: mask/compositing nodes registered in on_load with original
workflow type ids so existing graphs keep loading.
"""

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from .add_margins_311 import AddMargins311
from . import mask_ops

_PACK_MODULE = "custom_nodes.ComfyUI-311-Inpaint"


def _register_mask_ops() -> None:
    """Register V1 mask ops onto ComfyUI global mappings (drop-in type ids)."""
    import nodes as comfy_nodes

    for name, node_cls in mask_ops.NODE_CLASS_MAPPINGS.items():
        comfy_nodes.NODE_CLASS_MAPPINGS[name] = node_cls
        node_cls.RELATIVE_PYTHON_MODULE = _PACK_MODULE
    comfy_nodes.NODE_DISPLAY_NAME_MAPPINGS.update(mask_ops.NODE_DISPLAY_NAME_MAPPINGS)


class Inpaint311Extension(ComfyExtension):
    @override
    async def on_load(self) -> None:
        _register_mask_ops()

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            AddMargins311,
        ]


async def comfy_entrypoint() -> Inpaint311Extension:
    return Inpaint311Extension()

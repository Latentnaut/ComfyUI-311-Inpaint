"""
ComfyUI-311-Inpaint — Inpaint helper nodes (311 series).

Nodes:
  - Add Margins 311: Pad image + mask with extra width/height (centered).
"""

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from .add_margins_311 import AddMargins311


class Inpaint311Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            AddMargins311,
        ]


async def comfy_entrypoint() -> Inpaint311Extension:
    return Inpaint311Extension()

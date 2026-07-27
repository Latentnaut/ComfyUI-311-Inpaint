"""
Add Margins 311 — pad an image and its mask with configurable margins.

Size modes:
  - square_longest (default): black 1:1 canvas = longest_side * scale_by (1.25),
    content centered — matches ImageSize(Longest) → ScaleByAspectRatio(1:1) →
    Upscale By(1.25) → Image Align(center).
  - add_margins: expand by symmetric extras or custom per-side (px/%).
  - target_size: fit (contain) onto an absolute canvas.
"""

from __future__ import annotations

import re

import torch
import comfy.utils
from comfy_api.latest import io


def _parse_fill_color(fill_color: str) -> tuple[float, float, float]:
    """Parse #RGB / #RRGGBB (or without #) into float RGB in [0, 1]."""
    raw = (fill_color or "#000000").strip()
    if raw.startswith("#"):
        raw = raw[1:]
    if not re.fullmatch(r"[0-9A-Fa-f]{3}|[0-9A-Fa-f]{6}", raw):
        raise ValueError(
            f"AddMargins311: invalid fill_color '{fill_color}'. "
            "Use #RGB or #RRGGBB (e.g. #000000)."
        )
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    r = int(raw[0:2], 16) / 255.0
    g = int(raw[2:4], 16) / 255.0
    b = int(raw[4:6], 16) / 255.0
    return r, g, b


def _normalize_mask(mask: torch.Tensor, batch: int, height: int, width: int) -> torch.Tensor:
    """Normalize MASK tensor to [B, H, W] matching the image spatial size."""
    if mask.ndim == 4:
        if mask.shape[-1] in (1, 3, 4) and mask.shape[1] == height and mask.shape[2] == width:
            mask = mask.mean(dim=-1)
        elif mask.shape[1] == 1:
            mask = mask.squeeze(1)
        else:
            mask = mask[:, 0]
    elif mask.ndim == 2:
        mask = mask.unsqueeze(0)
    elif mask.ndim != 3:
        raise ValueError(f"AddMargins311: unsupported mask shape {tuple(mask.shape)}")

    if mask.shape[1] != height or mask.shape[2] != width:
        raise ValueError(
            f"AddMargins311: mask size {mask.shape[1]}x{mask.shape[2]} "
            f"does not match image size {height}x{width}."
        )

    if mask.shape[0] == 1 and batch > 1:
        mask = mask.expand(batch, -1, -1).contiguous()
    elif mask.shape[0] != batch:
        raise ValueError(
            f"AddMargins311: mask batch {mask.shape[0]} does not match image batch {batch}."
        )
    return mask


def _align_offset(content: int, canvas: int, align: str) -> int:
    align = (align or "center").lower()
    if content >= canvas:
        return 0
    if align in ("left", "top"):
        return 0
    if align in ("right", "bottom"):
        return canvas - content
    return (canvas - content) // 2


def _fill_vector(channels: int, fill_color: str) -> list[float]:
    r, g, b = _parse_fill_color(fill_color)
    fill = [r, g, b]
    if channels > 3:
        fill = fill + [1.0] * (channels - 3)
    elif channels < 3:
        fill = fill[:channels]
    return fill


def _make_image_canvas(
    batch: int,
    height: int,
    width: int,
    channels: int,
    fill: list[float],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    canvas = torch.zeros((batch, height, width, channels), dtype=dtype, device=device)
    for c, value in enumerate(fill):
        canvas[..., c] = value
    return canvas


def _resize_image(image: torch.Tensor, new_h: int, new_w: int) -> torch.Tensor:
    """Resize IMAGE [B,H,W,C] with lanczos (via common_upscale)."""
    if image.shape[1] == new_h and image.shape[2] == new_w:
        return image
    return comfy.utils.common_upscale(
        image.movedim(-1, 1), new_w, new_h, "lanczos", "disabled"
    ).movedim(1, -1)


def _resize_mask(mask: torch.Tensor, new_h: int, new_w: int) -> torch.Tensor:
    """Resize MASK [B,H,W] with bilinear."""
    if mask.shape[1] == new_h and mask.shape[2] == new_w:
        return mask
    return torch.nn.functional.interpolate(
        mask.unsqueeze(1),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


def _side_to_pixels(value: int | float, unit: str, base: int) -> int:
    """Convert a side margin to pixels. percent is relative to base (W or H)."""
    v = max(0.0, float(value))
    if (unit or "pixels").strip().lower() == "percent":
        return max(0, int(round(base * v / 100.0)))
    return max(0, int(round(v)))


def _paste(
    image: torch.Tensor,
    mask: torch.Tensor | None,
    canvas_h: int,
    canvas_w: int,
    top: int,
    left: int,
    fill_color: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Paste image/mask onto a filled canvas at (top, left)."""
    batch, height, width, channels = image.shape
    fill = _fill_vector(channels, fill_color)
    canvas = _make_image_canvas(
        batch, canvas_h, canvas_w, channels, fill, image.dtype, image.device
    )
    canvas[:, top : top + height, left : left + width, :] = image

    mask_canvas = torch.zeros(
        (batch, canvas_h, canvas_w), dtype=image.dtype, device=image.device
    )
    if mask is not None:
        mask_canvas[:, top : top + height, left : left + width] = mask.to(
            dtype=image.dtype, device=image.device
        )
    return canvas, mask_canvas


def _place_on_canvas(
    image: torch.Tensor,
    mask: torch.Tensor | None,
    canvas_h: int,
    canvas_w: int,
    fill_color: str,
    halign: str,
    valign: str,
    fit: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Place image (+ optional mask) onto a canvas of canvas_h x canvas_w.
    If fit=True, scale content to contain within the canvas (may up/downscale).
    """
    batch, height, width, channels = image.shape
    content_h, content_w = height, width
    placed = image
    placed_mask = mask

    if fit and (height != canvas_h or width != canvas_w):
        scale = min(canvas_w / max(width, 1), canvas_h / max(height, 1))
        content_w = max(1, int(round(width * scale)))
        content_h = max(1, int(round(height * scale)))
        content_w = min(content_w, canvas_w)
        content_h = min(content_h, canvas_h)
        placed = _resize_image(image, content_h, content_w)
        if placed_mask is not None:
            placed_mask = _resize_mask(placed_mask, content_h, content_w)

    if content_h > canvas_h or content_w > canvas_w:
        scale = min(canvas_w / content_w, canvas_h / content_h)
        content_w = max(1, int(round(content_w * scale)))
        content_h = max(1, int(round(content_h * scale)))
        content_w = min(content_w, canvas_w)
        content_h = min(content_h, canvas_h)
        placed = _resize_image(placed, content_h, content_w)
        if placed_mask is not None:
            placed_mask = _resize_mask(placed_mask, content_h, content_w)

    left = _align_offset(content_w, canvas_w, halign)
    top = _align_offset(content_h, canvas_h, valign)
    return _paste(placed, placed_mask, canvas_h, canvas_w, top, left, fill_color)


def _pad_sides(
    image: torch.Tensor,
    mask: torch.Tensor | None,
    pad_left: int,
    pad_right: int,
    pad_top: int,
    pad_bottom: int,
    fill_color: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad image/mask with explicit per-side pixel margins."""
    _batch, height, width, _channels = image.shape
    canvas_w = width + pad_left + pad_right
    canvas_h = height + pad_top + pad_bottom
    return _paste(image, mask, canvas_h, canvas_w, pad_top, pad_left, fill_color)


class AddMargins311(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AddMargins311",
            display_name="Add Margins 311",
            category="311/Inpaint",
            description=(
                "Add margins to an image and its mask. "
                "Default (square_longest): black 1:1 canvas from the longest side, "
                "then scale_by 1.25, content centered — matches ImageSize→ScaleByAspectRatio→Upscale By→Align. "
                "Also supports add_margins (px/%) and target_size."
            ),
            search_aliases=["pad", "margin", "padding", "inpaint", "expand", "border", "canvas"],
            inputs=[
                io.Image.Input(id="image", display_name="image"),
                io.Mask.Input(id="mask", optional=True, display_name="mask"),
                io.Combo.Input(
                    id="size_mode",
                    options=["square_longest", "add_margins", "target_size"],
                    default="square_longest",
                    display_name="size_mode",
                    tooltip=(
                        "square_longest: square canvas = longest_side * scale_by, center content. "
                        "add_margins: expand with symmetric or custom side margins. "
                        "target_size: fit onto target_width x target_height."
                    ),
                ),
                io.Float.Input(
                    id="scale_by",
                    default=1.25,
                    min=0.01,
                    max=8.0,
                    step=0.01,
                    display_name="scale_by",
                    tooltip="Multiplier for longest side when size_mode=square_longest (default 1.25).",
                ),
                io.Combo.Input(
                    id="margin_layout",
                    options=["symmetric", "custom"],
                    default="symmetric",
                    display_name="margin_layout",
                    tooltip=(
                        "symmetric: extra_width/height split equally L/R and T/B. "
                        "custom: independent left/right/top/bottom (each with own unit)."
                    ),
                ),
                io.Combo.Input(
                    id="unit",
                    options=["pixels", "percent"],
                    default="pixels",
                    display_name="unit",
                    tooltip="Unit for extra_width / extra_height when margin_layout=symmetric.",
                ),
                io.Int.Input(
                    id="extra_width",
                    default=0,
                    min=0,
                    max=8192,
                    step=1,
                    display_name="extra_width",
                    tooltip="Total added width (split L/R). Used when size_mode=add_margins + symmetric.",
                ),
                io.Int.Input(
                    id="extra_height",
                    default=0,
                    min=0,
                    max=8192,
                    step=1,
                    display_name="extra_height",
                    tooltip="Total added height (split T/B). Used when size_mode=add_margins + symmetric.",
                ),
                io.Float.Input(
                    id="left",
                    default=0.0,
                    min=0.0,
                    max=8192.0,
                    step=1.0,
                    display_name="left",
                    tooltip="Left margin (custom layout).",
                ),
                io.Combo.Input(
                    id="left_unit",
                    options=["pixels", "percent"],
                    default="pixels",
                    display_name="left_unit",
                ),
                io.Float.Input(
                    id="right",
                    default=0.0,
                    min=0.0,
                    max=8192.0,
                    step=1.0,
                    display_name="right",
                    tooltip="Right margin (custom layout).",
                ),
                io.Combo.Input(
                    id="right_unit",
                    options=["pixels", "percent"],
                    default="pixels",
                    display_name="right_unit",
                ),
                io.Float.Input(
                    id="top",
                    default=0.0,
                    min=0.0,
                    max=8192.0,
                    step=1.0,
                    display_name="top",
                    tooltip="Top margin (custom layout).",
                ),
                io.Combo.Input(
                    id="top_unit",
                    options=["pixels", "percent"],
                    default="pixels",
                    display_name="top_unit",
                ),
                io.Float.Input(
                    id="bottom",
                    default=0.0,
                    min=0.0,
                    max=8192.0,
                    step=1.0,
                    display_name="bottom",
                    tooltip="Bottom margin (custom layout).",
                ),
                io.Combo.Input(
                    id="bottom_unit",
                    options=["pixels", "percent"],
                    default="pixels",
                    display_name="bottom_unit",
                ),
                io.Int.Input(
                    id="target_width",
                    default=512,
                    min=1,
                    max=8192,
                    step=1,
                    display_name="target_width",
                    tooltip="Canvas width when size_mode=target_size.",
                ),
                io.Int.Input(
                    id="target_height",
                    default=512,
                    min=1,
                    max=8192,
                    step=1,
                    display_name="target_height",
                    tooltip="Canvas height when size_mode=target_size.",
                ),
                io.String.Input(
                    id="fill_color",
                    default="#000000",
                    multiline=False,
                    display_name="fill_color",
                    tooltip="Canvas fill color as #RGB or #RRGGBB.",
                ),
                io.Combo.Input(
                    id="halign",
                    options=["center", "left", "right"],
                    default="center",
                    display_name="halign",
                ),
                io.Combo.Input(
                    id="valign",
                    options=["center", "top", "bottom"],
                    default="center",
                    display_name="valign",
                ),
            ],
            outputs=[
                io.Image.Output(id="image", display_name="IMAGE"),
                io.Mask.Output(id="mask", display_name="MASK"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        mask: torch.Tensor | None = None,
        size_mode: str = "square_longest",
        scale_by: float = 1.25,
        margin_layout: str = "symmetric",
        unit: str = "pixels",
        extra_width: int = 0,
        extra_height: int = 0,
        left: float = 0.0,
        left_unit: str = "pixels",
        right: float = 0.0,
        right_unit: str = "pixels",
        top: float = 0.0,
        top_unit: str = "pixels",
        bottom: float = 0.0,
        bottom_unit: str = "pixels",
        target_width: int = 512,
        target_height: int = 512,
        fill_color: str = "#000000",
        halign: str = "center",
        valign: str = "center",
    ) -> io.NodeOutput:
        if image is None or not isinstance(image, torch.Tensor):
            raise ValueError("AddMargins311: image is required.")
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(f"AddMargins311: expected IMAGE [B,H,W,C], got {tuple(image.shape)}")

        batch, height, width, _channels = image.shape
        norm_mask = None
        if mask is not None:
            norm_mask = _normalize_mask(mask, batch, height, width)

        mode = (size_mode or "square_longest").strip().lower()
        layout = (margin_layout or "symmetric").strip().lower()

        # Default workflow: square from longest side, then * scale_by, center content
        if mode == "square_longest":
            factor = max(0.01, float(scale_by))
            side = max(width, height)
            canvas = max(1, int(round(side * factor)))
            out_image, out_mask = _place_on_canvas(
                image,
                norm_mask,
                canvas,
                canvas,
                fill_color,
                halign,
                valign,
                fit=False,  # keep original resolution; only pad (like Image Align)
            )
            return io.NodeOutput(out_image, out_mask)

        if mode == "add_margins":
            if layout == "custom":
                pad_l = _side_to_pixels(left, left_unit, width)
                pad_r = _side_to_pixels(right, right_unit, width)
                pad_t = _side_to_pixels(top, top_unit, height)
                pad_b = _side_to_pixels(bottom, bottom_unit, height)
                out_image, out_mask = _pad_sides(
                    image, norm_mask, pad_l, pad_r, pad_t, pad_b, fill_color
                )
                return io.NodeOutput(out_image, out_mask)

            ew = max(0, int(extra_width))
            eh = max(0, int(extra_height))
            if (unit or "pixels").strip().lower() == "percent":
                ew = int(round(width * ew / 100.0))
                eh = int(round(height * eh / 100.0))
            pad_l = ew // 2
            pad_r = ew - pad_l
            pad_t = eh // 2
            pad_b = eh - pad_t
            out_image, out_mask = _pad_sides(
                image, norm_mask, pad_l, pad_r, pad_t, pad_b, fill_color
            )
            return io.NodeOutput(out_image, out_mask)

        # target_size
        canvas_w = max(1, int(target_width))
        canvas_h = max(1, int(target_height))
        out_image, out_mask = _place_on_canvas(
            image,
            norm_mask,
            canvas_h,
            canvas_w,
            fill_color,
            halign,
            valign,
            fit=True,
        )
        return io.NodeOutput(out_image, out_mask)

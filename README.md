# ComfyUI-311-Inpaint

Inpaint helper nodes for ComfyUI (311 series).

## Add Margins 311

### Default (`square_longest`)

Matches the working workflow:

1. Take the **longest side** of the input image
2. Build a **1:1** canvas of that size
3. Multiply by **`scale_by` = 1.25**
4. **Center** the original image and mask on that canvas (no content stretch)

Example: 500×712 → longest 712 → canvas **890×890** (`712 × 1.25`).

| Setting | Default |
|---------|---------|
| `size_mode` | `square_longest` |
| `scale_by` | `1.25` |
| `fill_mode` | `color` |
| `fill_color` | `#000000` |
| `halign` / `valign` | `center` |

### `fill_mode`

| Mode | Effect |
|------|--------|
| `color` | Solid `fill_color` in the margin (default) |
| `tile` | Repeat the image around the perimeter (`F.pad` circular) |
| `mirror` | Reflect edges for visual continuity (`F.pad` reflect) |

Mask margins stay **zeros** (inpaint semantics). Only the IMAGE is tiled/mirrored.

Tile/mirror use a single `torch.nn.functional.pad` pass (chunked reflect if pad ≥ dim) — no full-image clones.

### Other size modes

- **`add_margins`**: symmetric extras or custom per-side (`pixels` / `percent`)
- **`target_size`**: fit (contain) onto absolute `target_width` × `target_height`

## Mask / compositing (drop-in)

These nodes keep the original workflow type ids, so existing graphs load without rewiring. Search titles use the ` 311` suffix. Category: `311/Inpaint`.

Do not run a second pack that registers the same type ids.

`Paste By Mask` / `Cut By Mask` resample with **lanczos** (same path as Add Margins 311). Same-size cut→paste is a 1:1 copy. Default `resize` still fits the paste into the mask bounding box; it no longer uses raw bicubic (that aliased on downscale).

| Search title | Workflow type id |
|--------------|------------------|
| Paste By Mask 311 | `Paste By Mask` |
| Cut By Mask 311 | `Cut By Mask` |
| Mix Images By Mask 311 | `Mix Images By Mask` |
| Mix Color By Mask 311 | `Mix Color By Mask` |
| Mask By Text 311 | `Mask By Text` |
| Mask Morphology 311 | `Mask Morphology` |
| Combine Masks 311 | `Combine Masks` |
| Unary Mask Op 311 | `Unary Mask Op` |
| Unary Image Op 311 | `Unary Image Op` |
| Blur 311 | `Blur` |
| Image To Mask 311 | `Image To Mask` |
| Mask To Region 311 | `Mask To Region` |
| Get Image Size 311 | `Get Image Size` |
| Change Channel Count 311 | `Change Channel Count` |
| Constant Mask 311 | `Constant Mask` |
| Prune By Mask 311 | `Prune By Mask` |
| Separate Mask Components 311 | `Separate Mask Components` |
| Create Rect Mask 311 | `Create Rect Mask` |
| Make Image Batch 311 | `Make Image Batch` |
| Create QR Code 311 | `Create QR Code` |
| Convert Color Space 311 | `Convert Color Space` |
| Incrementer 311 | `MasqueradeIncrementer` |

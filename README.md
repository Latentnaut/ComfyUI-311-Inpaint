# ComfyUI-311-Inpaint

Inpaint helper nodes for ComfyUI (311 series).

## Add Margins 311

### Default (`square_longest`)

Matches the working workflow:

1. Take the **longest side** of the input image
2. Build a black **1:1** canvas of that size
3. Multiply by **`scale_by` = 1.25**
4. **Center** the original image and mask on that canvas (no content stretch)

Example: 500×712 → longest 712 → canvas **890×890** (`712 × 1.25`).

| Setting | Default |
|---------|---------|
| `size_mode` | `square_longest` |
| `scale_by` | `1.25` |
| `fill_color` | `#000000` |
| `halign` / `valign` | `center` |

### Other modes

- **`add_margins`**: symmetric extras or custom per-side (`pixels` / `percent`)
- **`target_size`**: fit (contain) onto absolute `target_width` × `target_height`

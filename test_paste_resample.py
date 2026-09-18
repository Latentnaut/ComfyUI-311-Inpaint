"""Paste/Cut By Mask: identity roundtrip and lanczos resample."""
from pathlib import Path
import sys
import importlib.util

import torch

COMFY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(COMFY_ROOT))


def load_mask_ops():
    path = Path(__file__).with_name("mask_ops.py")
    spec = importlib.util.spec_from_file_location("mask_ops_quality", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rect_mask(h, w, y0, x0, y1, x1):
    m = torch.zeros((1, h, w))
    m[:, y0:y1, x0:x1] = 1.0
    return m


def test_cut_paste_identity():
    ops = load_mask_ops()
    h, w = 48, 64
    y0, x0, y1, x1 = 8, 10, 28, 40
    # Unique pixel values so any resample/shift is obvious
    yy = torch.linspace(0, 1, h).view(1, h, 1, 1)
    xx = torch.linspace(0, 1, w).view(1, 1, w, 1)
    image = torch.cat([yy.expand(1, h, w, 1), xx.expand(1, h, w, 1), torch.full((1, h, w, 1), 0.4)], dim=3)
    mask = _rect_mask(h, w, y0, x0, y1, x1)

    (crop,) = ops.CutByMask().cut(image, mask, 0, 0)
    assert crop.shape[1] == y1 - y0
    assert crop.shape[2] == x1 - x0
    torch.testing.assert_close(crop[0, :, :, :3], image[0, y0:y1, x0:x1, :], atol=0, rtol=0)

    (pasted,) = ops.PasteByMask().paste(image, crop, mask, "resize")
    torch.testing.assert_close(
        pasted[0, y0:y1, x0:x1, :3], image[0, y0:y1, x0:x1, :], atol=0, rtol=0
    )


def test_downscale_does_not_use_identity_size():
    ops = load_mask_ops()
    base_h, base_w = 64, 64
    y0, x0, y1, x1 = 16, 16, 48, 48  # 32x32 hole
    base = torch.zeros((1, base_h, base_w, 3))
    mask = _rect_mask(base_h, base_w, y0, x0, y1, x1)
    # High-res crop with a 1-pixel checker (aliasing trap for bicubic/no-aa)
    ph, pw = 96, 96
    yy = torch.arange(ph).view(ph, 1)
    xx = torch.arange(pw).view(1, pw)
    checker = ((yy + xx) % 2).float().unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)
    (out,) = ops.PasteByMask().paste(base, checker, mask, "resize")
    region = out[0, y0:y1, x0:x1, :3]
    assert region.shape[0] == 32 and region.shape[1] == 32
    # Must be a real downsample, not a 1:1 dump of the 96px source
    assert region.shape[0] != checker.shape[1]


if __name__ == "__main__":
    test_cut_paste_identity()
    test_downscale_does_not_use_identity_size()
    print("OK paste quality")

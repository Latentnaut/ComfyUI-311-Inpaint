"""Verify drop-in type ids and UI branding for 311 Inpaint mask ops."""
from pathlib import Path
import importlib.util

NEW = Path(__file__).with_name("mask_ops.py")

EXPECTED_KEYS = [
    "Mask By Text",
    "Mask Morphology",
    "Combine Masks",
    "Unary Mask Op",
    "Unary Image Op",
    "Blur",
    "Image To Mask",
    "Mix Images By Mask",
    "Mix Color By Mask",
    "Mask To Region",
    "Cut By Mask",
    "Paste By Mask",
    "Get Image Size",
    "Change Channel Count",
    "Constant Mask",
    "Prune By Mask",
    "Separate Mask Components",
    "Create Rect Mask",
    "Make Image Batch",
    "Create QR Code",
    "Convert Color Space",
    "MasqueradeIncrementer",
]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


new = load(NEW, "mask_ops_under_test")
keys = list(new.NODE_CLASS_MAPPINGS.keys())
assert keys == EXPECTED_KEYS, (keys, EXPECTED_KEYS)

for key, cls in new.NODE_CLASS_MAPPINGS.items():
    assert cls.CATEGORY == "311/Inpaint", (key, cls.CATEGORY)
    display = new.NODE_DISPLAY_NAME_MAPPINGS[key]
    assert display.endswith(" 311"), display
    assert "Masquerade Nodes" not in display
    inputs = cls.INPUT_TYPES()
    assert "required" in inputs

text = NEW.read_text(encoding="utf-8")
assert "Masquerade Nodes" not in text
assert "class IncrementerNode" in text
assert new.NODE_DISPLAY_NAME_MAPPINGS["Paste By Mask"] == "Paste By Mask 311"
assert new.NODE_CLASS_MAPPINGS["Paste By Mask"].FUNCTION == "paste"

print("OK", len(keys), "nodes")

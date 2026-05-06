"""Per-dataset class names for CLIP zero-shot head (from CoOp/TPT / DiffTPT cls_to_names)."""
from __future__ import annotations

import importlib.util
import os

# Load lists from extracted DiffTPT file (single source)
_RAW = os.path.join(os.path.dirname(__file__), "xdataset_cls_to_names_raw.py")
_spec = importlib.util.spec_from_file_location("xdataset_cls_raw", _RAW)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

DATASET_CLASSNAMES = {
    "caltech101": list(_mod.caltech101_classes),
    "oxford_pets": list(_mod.pets_classes),
    "stanford_cars": list(_mod.cars_classes),
    "flowers102": list(_mod.flower102_classes),
    "food101": list(_mod.food101_classes),
    "aircraft": list(_mod.aircraft_classes),
    "sun397": list(_mod.sun397_classes),
    "dtd": list(_mod.dtd_classes),
    "eurosat": list(_mod.eurosat_classes),
    "ucf101": list(_mod.ucf101_classes),
}

# Order matches loaders._eurosat_from_imagefolder folder_order (AnnualCrop..SeaLake).
EUROSAT_COMPACT_CLASSNAMES = [
    "annual crop land",
    "forest",
    "brushland or shrubland",
    "highway or road",
    "industrial buildings or commercial buildings",
    "pasture land",
    "permanent crop land",
    "residential buildings or homes or apartments",
    "river",
    "lake or sea",
]

DATASET_CLASSNAMES_OVERRIDE = {
    "eurosat": EUROSAT_COMPACT_CLASSNAMES,
}

# Per-dataset text templates for CLIP zero-shot (override DEFAULT / ensemble).
DATASET_TEMPLATES = {
    "eurosat": ["a centered satellite photo of {}."],
}

DEFAULT_TEMPLATE = ["a photo of a {}."]

IMAGENET_STYLE_TEMPLATES = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the nice {}.",
    "a sketch of a {}.",
    "a pixelated photo of a {}.",
    "a statue of a {}.",
    "a rendering of the {}.",
    "a cropped photo of the {}.",
    "a photo of a large {}.",
    "a black and white photo of a {}.",
    "the pixelated {}.",
    "a painting of the large {}.",
    "a sketch of the {}.",
    "a photo of a small {}.",
    "a photo of the weird {}.",
    "the colorful {}.",
    "a color photo of a {}.",
    "a color photo of the {}.",
    "a good photo of a {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "a dark photo of a {}.",
    "a photo of a cool {}.",
    "a photo of a {}.",
    "a good photo of the {}.",
    "a photo of the {}.",
    "a blurry photo of the {}.",
    "a cartoon {}.",
]

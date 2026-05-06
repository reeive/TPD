"""Test loaders for 10 cross-dataset benchmarks (vtab-1k splits + local datasets)."""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from torch.utils.data import DataLoader, Dataset


VTAB_ALIAS = {
    "caltech101": "caltech101",
    "oxford_pets": "oxford_iiit_pet",
    "stanford_cars": None,
    "flowers102": "oxford_flowers102",
    "food101": None,
    "aircraft": None,
    "sun397": "sun397",
    "dtd": "dtd",
    "eurosat": "eurosat",
    "ucf101": None,
}


class _ImageListDataset(Dataset):
    def __init__(self, root: str, items: List[Tuple[str, int]], transform: Any):
        self.root = root
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        from PIL import Image

        rel, y = self.items[idx]
        path = os.path.join(self.root, rel)
        img = Image.open(path).convert("RGB")
        return self.transform(img), y


def _read_vtab_test(root: str, test_txt: str) -> List[Tuple[str, int]]:
    items = []
    with open(test_txt) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            rel = parts[0]
            y = int(parts[1])
            items.append((rel, y))
    return items


def _cars_from_json(root: str) -> List[Tuple[str, int]]:
    p = os.path.join(root, "test.json")
    with open(p) as f:
        d = json.load(f)
    items = []
    for rel, lab in d.items():
        items.append((rel, int(lab) - 1))
    return items


def _cars_train_from_json(root: str) -> List[Tuple[str, int]]:
    p = os.path.join(root, "train.json")
    with open(p) as f:
        d = json.load(f)
    items = []
    for rel, lab in d.items():
        items.append((rel, int(lab) - 1))
    return items


def _food101_test(root: str) -> List[Tuple[str, int]]:
    meta = os.path.join(root, "meta", "classes.txt")
    testp = os.path.join(root, "meta", "test.txt")
    if not os.path.isfile(meta) or not os.path.isfile(testp):
        raise FileNotFoundError(
            f"Food101 expects {meta} and {testp}. Extract food-101 archive to {root}."
        )
    with open(meta) as f:
        classes = [ln.strip() for ln in f if ln.strip()]
    c2i = {c: i for i, c in enumerate(classes)}
    items = []
    with open(testp) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            cls, _fname = ln.split("/")
            rel = ln if ln.lower().endswith((".jpg", ".jpeg", ".png")) else f"{ln}.jpg"
            items.append((f"images/{rel}", c2i[cls]))
    return items


def _food101_train(
    root: str, max_per_class: int = 0, seed: int = 0
) -> List[Tuple[str, int]]:
    meta = os.path.join(root, "meta", "classes.txt")
    trainp = os.path.join(root, "meta", "train.txt")
    if not os.path.isfile(meta) or not os.path.isfile(trainp):
        raise FileNotFoundError(
            f"Food101 expects {meta} and {trainp}. Extract food-101 archive to {root}."
        )
    with open(meta) as f:
        classes = [ln.strip() for ln in f if ln.strip()]
    c2i = {c: i for i, c in enumerate(classes)}
    by_cls: Dict[int, List[str]] = defaultdict(list)
    with open(trainp) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            cls, _fname = ln.split("/")
            rel = ln if ln.lower().endswith((".jpg", ".jpeg", ".png")) else f"{ln}.jpg"
            by_cls[c2i[cls]].append(f"images/{rel}")
    if max_per_class and max_per_class > 0:
        rng = random.Random(seed)
        items: List[Tuple[str, int]] = []
        for y in sorted(by_cls.keys()):
            paths = by_cls[y].copy()
            rng.shuffle(paths)
            for rel in paths[:max_per_class]:
                items.append((rel, y))
        return items
    items: List[Tuple[str, int]] = []
    for y in sorted(by_cls.keys()):
        for rel in by_cls[y]:
            items.append((rel, y))
    return items


def _ucf101_stratified_split(
    root: str, seed: int = 0, train_ratio: float = 0.8
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    all_items = _ucf101_from_split(root)
    by_cls: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
    for rel, y in all_items:
        by_cls[y].append((rel, y))
    rng = random.Random(seed)
    train_items: List[Tuple[str, int]] = []
    test_items: List[Tuple[str, int]] = []
    for y in sorted(by_cls.keys()):
        bucket = by_cls[y]
        n = len(bucket)
        if n == 0:
            continue
        shuffled = bucket.copy()
        rng.shuffle(shuffled)
        if n == 1:
            train_items.extend(shuffled)
            continue
        n_tr = int(round(n * train_ratio))
        n_tr = max(1, min(n - 1, n_tr))
        train_items.extend(shuffled[:n_tr])
        test_items.extend(shuffled[n_tr:])
    return train_items, test_items


def _aircraft_c2i():
    import importlib.util

    raw = os.path.join(
        os.path.dirname(__file__), "xdataset_cls_to_names_raw.py"
    )
    spec = importlib.util.spec_from_file_location("xdataset_cls_raw", raw)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    classes = list(mod.aircraft_classes)
    return {c: i for i, c in enumerate(classes)}


def _aircraft_variant_list(root: str, split: str) -> List[Tuple[str, int]]:
    """split: 'test' or 'train' -> images_variant_{split}.txt"""
    c2i = _aircraft_c2i()
    lst = os.path.join(root, "data", f"images_variant_{split}.txt")
    if not os.path.isfile(lst):
        raise FileNotFoundError(lst)
    items = []
    with open(lst) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_id, var = line.split(None, 1)
            items.append((f"data/images/{img_id}.jpg", c2i[var]))
    return items


def _aircraft_variant_test(root: str) -> List[Tuple[str, int]]:
    return _aircraft_variant_list(root, "test")


def _aircraft_variant_train(root: str) -> List[Tuple[str, int]]:
    return _aircraft_variant_list(root, "train")


_CALTECH_FOLDER_ALIASES = {
    "Faces": "face",
    "Motorbikes": "motorbike",
    "airplanes": "airplane",
    "Leopards": "leopard",
}


def _caltech101_from_imagefolder(root: str) -> List[Tuple[str, int]]:
    """Caltech-101 ImageFolder -> index into DATASET_CLASSNAMES['caltech101']."""
    from .classnames import DATASET_CLASSNAMES

    class_order = list(DATASET_CLASSNAMES["caltech101"])
    c2i = {c: i for i, c in enumerate(class_order)}
    items: List[Tuple[str, int]] = []
    for folder in sorted(os.listdir(root)):
        if folder in ("BACKGROUND_Google", "Faces_easy"):
            continue
        fdir = os.path.join(root, folder)
        if not os.path.isdir(fdir):
            continue
        key = _CALTECH_FOLDER_ALIASES.get(folder, folder.lower())
        if key not in c2i:
            continue
        y = c2i[key]
        for fn in os.listdir(fdir):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                items.append((os.path.join(folder, fn), y))
    return items


def _eurosat_from_imagefolder(root: str) -> List[Tuple[str, int]]:
    """EuroSAT 2750/<CamelClass>/*.jpg -> index into DATASET_CLASSNAMES['eurosat']."""
    base = root
    if os.path.isdir(os.path.join(root, "2750")):
        base = os.path.join(root, "2750")
    folder_order = [
        "AnnualCrop",
        "Forest",
        "HerbaceousVegetation",
        "Highway",
        "Industrial",
        "Pasture",
        "PermanentCrop",
        "Residential",
        "River",
        "SeaLake",
    ]
    items: List[Tuple[str, int]] = []
    for i, folder in enumerate(folder_order):
        fdir = os.path.join(base, folder)
        if not os.path.isdir(fdir):
            raise FileNotFoundError(fdir)
        for fn in os.listdir(fdir):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                items.append((os.path.join(folder, fn), i))
    return items, base


def _ucf101_from_split(root: str) -> List[Tuple[str, int]]:
    """UCF-101: expect root/label/frame.jpg (ImageFolder-style) or midframes."""
    if os.path.isdir(os.path.join(root, "train")):
        split_dir = os.path.join(root, "test")
    else:
        split_dir = root
    classes = sorted(
        d
        for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    )
    c2i = {c: i for i, c in enumerate(classes)}
    items = []
    for c in classes:
        cdir = os.path.join(split_dir, c)
        for fn in os.listdir(cdir):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                items.append((os.path.join(c, fn), c2i[c]))
    return items


def build_xdataset_loader(
    dataset: str,
    data_root: str,
    transform: Callable,
    batch_size: int,
    workers: int = 4,
    *,
    stanford_cars_root: Optional[str] = None,
    food101_root: Optional[str] = None,
    aircraft_root: Optional[str] = None,
    ucf101_root: Optional[str] = None,
    caltech101_root: Optional[str] = None,
    eurosat_root: Optional[str] = None,
) -> Tuple[DataLoader, Dataset, int]:
    """Build test DataLoader; returns (loader, dataset, num_classes)."""
    dataset = dataset.lower().replace("-", "_")
    vtab = os.path.join(data_root, "vtab-1k")

    cal_default = os.path.join(data_root, "caltech-101", "101_ObjectCategories")
    euro_default = os.path.join(data_root, "eurosat")

    if dataset == "caltech101":
        root = caltech101_root or (cal_default if os.path.isdir(cal_default) else "")
        if root and os.path.isdir(root):
            items = _caltech101_from_imagefolder(root)
            ds = _ImageListDataset(root, items, transform)
            from .classnames import DATASET_CLASSNAMES

            n_cls = len(DATASET_CLASSNAMES["caltech101"])
            return _wrap(ds, batch_size, workers), ds, n_cls
    elif dataset == "eurosat":
        root = eurosat_root or (euro_default if os.path.isdir(euro_default) else "")
        if root and os.path.isdir(root):
            items, base = _eurosat_from_imagefolder(root)
            ds = _ImageListDataset(base, items, transform)
            from .classnames import DATASET_CLASSNAMES

            n_cls = len(DATASET_CLASSNAMES["eurosat"])
            return _wrap(ds, batch_size, workers), ds, n_cls

    if dataset == "stanford_cars":
        root = stanford_cars_root or os.path.join(data_root, "Stanford-cars")
        items = _cars_from_json(root)
        ds = _ImageListDataset(root, items, transform)
        n_cls = 196
    elif dataset == "food101":
        root = food101_root or os.path.join(data_root, "food-101")
        items = _food101_test(root)
        ds = _ImageListDataset(root, items, transform)
        n_cls = 101
    elif dataset == "aircraft":
        root = aircraft_root or os.path.join(
            data_root, "fgvc-aircraft-2013b"
        )
        items = _aircraft_variant_test(root)
        ds = _ImageListDataset(root, items, transform)
        n_cls = 100
    elif dataset == "ucf101":
        root = ucf101_root or os.path.join(data_root, "UCF-101-midframes")
        items = _ucf101_from_split(root)
        ds = _ImageListDataset(root, items, transform)
        n_cls = 101
    else:
        sub = VTAB_ALIAS.get(dataset)
        if sub is None:
            raise ValueError(f"Unknown dataset {dataset}")
        droot = os.path.join(vtab, sub)
        test_txt = os.path.join(droot, "test.txt")
        if not os.path.isfile(test_txt):
            raise FileNotFoundError(
                f"Missing {test_txt}. Set data_root to parent of vtab-1k."
            )
        items = _read_vtab_test(droot, test_txt)
        ds = _ImageListDataset(droot, items, transform)
        from .classnames import DATASET_CLASSNAMES

        key = dataset
        n_cls = len(DATASET_CLASSNAMES[key])

    return _wrap(ds, batch_size, workers), ds, n_cls


def _make_loader(
    ds: Dataset, batch_size: int, workers: int, shuffle: bool
) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
    )


def _wrap(ds: Dataset, batch_size: int, workers: int) -> DataLoader:
    return _make_loader(ds, batch_size, workers, shuffle=True)


def build_xdataset_split_loaders(
    dataset: str,
    data_root: str,
    transform: Callable,
    batch_size: int,
    workers: int = 4,
    *,
    stanford_cars_root: Optional[str] = None,
    food101_root: Optional[str] = None,
    aircraft_root: Optional[str] = None,
    ucf101_root: Optional[str] = None,
    max_train_per_class: int = 0,
    food101_train_seed: int = 0,
    ucf101_seed: int = 0,
    ucf101_train_ratio: float = 0.8,
) -> Tuple[DataLoader, DataLoader, int, Dict[str, Any]]:
    """Train + test DataLoaders for linear-probe style evaluation.

    - VTAB-1k style datasets: ``train800val200.txt`` (1k) + ``test.txt``.
    - Stanford Cars: ``train.json`` / ``test.json``.
    - Food-101: ``meta/train.txt`` / ``meta/test.txt`` (optional per-class cap).
    - Aircraft: ``images_variant_train`` / ``images_variant_test``.
    - UCF-101 (midframes): per-class 80/20 split (reproducible with ``ucf101_seed``).
    """
    dataset = dataset.lower().replace("-", "_")
    vtab = os.path.join(data_root, "vtab-1k")
    meta: Dict[str, Any] = {
        "dataset": dataset,
        "food101_max_train_per_class": max_train_per_class
        if dataset == "food101"
        else None,
    }

    from .classnames import DATASET_CLASSNAMES

    vtab_keys = {
        "caltech101",
        "oxford_pets",
        "flowers102",
        "sun397",
        "dtd",
        "eurosat",
    }
    if dataset in vtab_keys:
        sub = VTAB_ALIAS.get(dataset)
        if sub is None:
            raise ValueError(f"VTAB mapping missing for {dataset}")
        droot = os.path.join(vtab, sub)
        train_txt = os.path.join(droot, "train800val200.txt")
        test_txt = os.path.join(droot, "test.txt")
        if not os.path.isfile(train_txt) or not os.path.isfile(test_txt):
            raise FileNotFoundError(
                f"Expected {train_txt} and {test_txt} under vtab-1k."
            )
        tr_items = _read_vtab_test(droot, train_txt)
        te_items = _read_vtab_test(droot, test_txt)
        ds_tr = _ImageListDataset(droot, tr_items, transform)
        ds_te = _ImageListDataset(droot, te_items, transform)
        n_cls = len(DATASET_CLASSNAMES[dataset])
        return (
            _make_loader(ds_tr, batch_size, workers, True),
            _make_loader(ds_te, batch_size, workers, False),
            n_cls,
            meta,
        )

    if dataset == "stanford_cars":
        root = stanford_cars_root or os.path.join(data_root, "Stanford-cars")
        tr_items = _cars_train_from_json(root)
        te_items = _cars_from_json(root)
        ds_tr = _ImageListDataset(root, tr_items, transform)
        ds_te = _ImageListDataset(root, te_items, transform)
        n_cls = 196
        return (
            _make_loader(ds_tr, batch_size, workers, True),
            _make_loader(ds_te, batch_size, workers, False),
            n_cls,
            meta,
        )

    if dataset == "food101":
        root = food101_root or os.path.join(data_root, "food-101")
        tr_items = _food101_train(
            root,
            max_per_class=max_train_per_class,
            seed=food101_train_seed,
        )
        te_items = _food101_test(root)
        ds_tr = _ImageListDataset(root, tr_items, transform)
        ds_te = _ImageListDataset(root, te_items, transform)
        n_cls = 101
        return (
            _make_loader(ds_tr, batch_size, workers, True),
            _make_loader(ds_te, batch_size, workers, False),
            n_cls,
            meta,
        )

    if dataset == "aircraft":
        root = aircraft_root or os.path.join(
            data_root, "fgvc-aircraft-2013b"
        )
        tr_items = _aircraft_variant_train(root)
        te_items = _aircraft_variant_test(root)
        ds_tr = _ImageListDataset(root, tr_items, transform)
        ds_te = _ImageListDataset(root, te_items, transform)
        n_cls = 100
        return (
            _make_loader(ds_tr, batch_size, workers, True),
            _make_loader(ds_te, batch_size, workers, False),
            n_cls,
            meta,
        )

    if dataset == "ucf101":
        root = ucf101_root or os.path.join(data_root, "UCF-101-midframes")
        tr_items, te_items = _ucf101_stratified_split(
            root, seed=ucf101_seed, train_ratio=ucf101_train_ratio
        )
        meta["ucf101_split"] = (
            f"stratified{int(round(ucf101_train_ratio * 100))}_"
            f"seed{ucf101_seed}"
        )
        meta["ucf101_train_ratio"] = ucf101_train_ratio
        meta["ucf101_seed"] = ucf101_seed
        ds_tr = _ImageListDataset(root, tr_items, transform)
        ds_te = _ImageListDataset(root, te_items, transform)
        n_cls = 101
        return (
            _make_loader(ds_tr, batch_size, workers, True),
            _make_loader(ds_te, batch_size, workers, False),
            n_cls,
            meta,
        )

    raise ValueError(f"Unknown dataset {dataset}")

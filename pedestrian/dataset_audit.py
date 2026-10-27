"""Audit Penn-Fudan pairs and create repeatable, grouped dataset splits."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

FRACTIONS = {"train": 0.6, "validation": 0.2, "test": 0.2}


def _source(image_id: str) -> str:
    for source in ("Fudan", "Penn"):
        if image_id.startswith(f"{source}Ped"):
            return source
    raise ValueError(f"Unknown image source: {image_id}")


def _image_hashes(image: Image.Image) -> tuple[str, str]:
    rgb = image.convert("RGB")
    # Include dimensions: the same sequence of RGB bytes may have different shapes.
    digest = hashlib.sha256(f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes()).hexdigest()
    gray = ImageOps.grayscale(rgb).resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray)
    bits = pixels[:, :-1] > pixels[:, 1:]
    dhash = sum(int(bit) << index for index, bit in enumerate(bits.flat))
    return digest, f"{dhash:016x}"


def audit_dataset(root: Path) -> list[dict]:
    image_dir, mask_dir = root / "PNGImages", root / "PedMasks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise ValueError("Dataset must contain PNGImages and PedMasks directories")
    images = {path.stem: path for path in image_dir.glob("*.png")}
    unexpected_masks = [
        path.name for path in mask_dir.glob("*.png") if not path.stem.endswith("_mask")
    ]
    if unexpected_masks:
        raise ValueError(f"Mask filenames must end with _mask.png: {sorted(unexpected_masks)}")
    masks = {path.stem.removesuffix("_mask"): path for path in mask_dir.glob("*_mask.png")}
    if not images:
        raise ValueError("No PNG images found")
    missing_masks = sorted(images.keys() - masks.keys())
    missing_images = sorted(masks.keys() - images.keys())
    if missing_masks or missing_images:
        raise ValueError(
            f"Unpaired files: missing mask {missing_masks}; missing image {missing_images}"
        )

    rows = []
    for image_id in sorted(images):
        with Image.open(images[image_id]) as image, Image.open(masks[image_id]) as mask:
            image.load()
            mask.load()
            if image.size != mask.size:
                raise ValueError(f"Image and mask dimensions differ for {image_id}")
            pixels = np.asarray(mask)
            if pixels.ndim != 2:
                raise ValueError(f"Mask must contain single-channel instance IDs: {image_id}")
            ids = sorted(int(value) for value in np.unique(pixels) if value != 0)
            if any(value < 0 for value in ids):
                raise ValueError(f"Negative mask ID for {image_id}")
            digest, dhash = _image_hashes(image)
            rows.append(
                {
                    "image_id": image_id,
                    "source": _source(image_id),
                    "image": f"PNGImages/{image_id}.png",
                    "mask": f"PedMasks/{image_id}_mask.png",
                    "size_wh": list(image.size),
                    "instance_ids": ids,
                    "instances": len(ids),
                    "pixel_sha256": digest,
                    "dhash": dhash,
                }
            )
    return rows


def group_images(rows: list[dict], reviewed_groups: list[list[str]], max_distance: int = 3):
    """Group matching pixels, close perceptual hashes, and reviewed related images."""
    ids = [row["image_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate image IDs in audit")
    parent = {image_id: image_id for image_id in ids}

    def find(image_id):
        if image_id not in parent:
            raise ValueError(f"Reviewed group references an unknown image: {image_id}")
        while parent[image_id] != image_id:
            parent[image_id] = parent[parent[image_id]]
            image_id = parent[image_id]
        return image_id

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    candidates = []
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            distance = (int(left["dhash"], 16) ^ int(right["dhash"], 16)).bit_count()
            if distance <= max_distance:
                candidates.append(
                    {"images": [left["image_id"], right["image_id"]], "distance": distance}
                )
            if left["pixel_sha256"] == right["pixel_sha256"] or distance <= max_distance:
                union(left["image_id"], right["image_id"])
    for group in reviewed_groups:
        if not group:
            raise ValueError("Reviewed groups must not be empty")
        find(group[0])
        for image_id in group[1:]:
            union(group[0], image_id)

    members = {}
    for image_id in ids:
        members.setdefault(find(image_id), []).append(image_id)
    return sorted(sorted(group) for group in members.values()), candidates


def split_groups(rows: list[dict], groups: list[list[str]], seed: int = 42) -> dict[str, list[str]]:
    """Assign whole groups while approximately matching source-specific fractions."""
    sources = {row["image_id"]: row["source"] for row in rows}
    flattened = [image_id for group in groups for image_id in group]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(sources):
        raise ValueError("Groups must contain every image exactly once")
    totals = Counter(sources.values())
    counts = {name: Counter() for name in FRACTIONS}
    splits = {name: [] for name in FRACTIONS}
    ordered = [list(group) for group in groups]
    random.Random(seed).shuffle(ordered)
    ordered.sort(key=len, reverse=True)

    for group in ordered:
        group_sources = Counter(sources[image_id] for image_id in group)

        def cost(split_name):
            # Choose the assignment that increases normalized deviation least.
            total = 0.0
            for source, added in group_sources.items():
                target = totals[source] * FRACTIONS[split_name]
                before = counts[split_name][source] - target
                after = before + added
                total += (after * after - before * before) / max(target, 1)
            return total

        destination = min(FRACTIONS, key=cost)
        splits[destination].extend(group)
        counts[destination].update(group_sources)
    return {name: sorted(ids) for name, ids in splits.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--reviewed-groups", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    reviewed = json.loads(args.reviewed_groups.read_text(encoding="utf-8"))["groups"]
    rows = audit_dataset(args.dataset_root)
    groups, candidates = group_images(rows, reviewed)
    splits = split_groups(rows, groups, args.seed)
    sources = {row["image_id"]: row["source"] for row in rows}
    counts = {
        name: dict(sorted(Counter(sources[i] for i in ids).items())) for name, ids in splits.items()
    }
    hashes = {}
    for row in rows:
        hashes.setdefault(row["pixel_sha256"], []).append(row["image_id"])
    audit = {
        "total_images": len(rows),
        "total_instances": sum(row["instances"] for row in rows),
        "sources": dict(sorted(Counter(sources.values()).items())),
        "exact_duplicate_groups": [ids for ids in hashes.values() if len(ids) > 1],
        "near_duplicate_method": "64-bit grayscale difference hash; distance <= 3; not exhaustive",
        "near_duplicate_candidates": candidates,
        "reviewed_groups": reviewed,
        "groups_with_multiple_images": [group for group in groups if len(group) > 1],
        "images": rows,
    }
    manifest = {
        "seed": args.seed,
        "target_fractions": FRACTIONS,
        "source_counts": counts,
        "splits": splits,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, content in (("audit.json", audit), ("splits.json", manifest)):
        (args.output / name).write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    summary = {"images": len(rows), "instances": audit["total_instances"], "source_counts": counts}
    print(json.dumps(summary, indent=2))
    print(f"Saved audit and splits to {args.output}")


if __name__ == "__main__":
    main()

"""Split a YOLO dataset into train / val / test **by source video**, not by image.

Frames from one video look almost identical. If neighbouring frames land in both train and val,
the validation score is inflated. This module keeps every group (one video) in exactly one part.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .common import (
    PrepError,
    find_yolo_pair,
    label_path,
    list_images,
    parse_label_file,
    place_file,
    read_names,
    write_data_yaml,
)

DEFAULT_GROUP_REGEX = r"^(?P<group>.+)_f\d+$"


def group_of(img: Path, img_dir: Path, how: str, pattern: "re.Pattern[str]") -> Tuple[str, bool]:
    """(group name, was it recognised?)"""
    rel = img.relative_to(img_dir)
    if how == "folder":
        parent = rel.parent.as_posix()
        return (parent, True) if parent != "." else (rel.stem, False)
    m = pattern.match(rel.stem)
    if not m:
        return rel.as_posix(), False
    return (m.group("group") if "group" in pattern.groupindex else m.group(1)), True


def assign_groups(sizes: Dict[str, int], fractions: Dict[str, float], seed: int = 0) -> Dict[str, str]:
    """Greedy, deterministic: biggest groups first, each to the part furthest below its target size."""
    active = {k: v for k, v in fractions.items() if v > 0}
    total_fraction = sum(active.values())
    if not active or total_fraction <= 0:
        raise PrepError("Nothing to split into: all ratios are zero.")
    if len(sizes) < len(active):
        raise PrepError(
            f"Only {len(sizes)} group(s) (videos) found but {len(active)} parts requested. "
            "Every part needs at least one group; add more videos or request fewer parts."
        )
    total_images = sum(sizes.values())
    target = {k: total_images * v / total_fraction for k, v in active.items()}
    current = {k: 0 for k in active}
    names = sorted(sizes)
    random.Random(seed).shuffle(names)
    names.sort(key=lambda g: -sizes[g])  # stable: ties keep their shuffled order
    order = list(active)
    result: Dict[str, str] = {}
    counts = {k: 0 for k in active}
    for i, g in enumerate(names):
        remaining_groups = len(names) - i
        empty = [k for k in order if counts[k] == 0]
        if empty and remaining_groups <= len(empty):
            choice = empty[0]  # make sure no part is left empty
        else:
            choice = max(order, key=lambda k: (target[k] - current[k]) / target[k])
        result[g] = choice
        current[choice] += sizes[g]
        counts[choice] += 1
    return result


def split_dataset(
    root,
    out_dir,
    val: float = 0.2,
    test: float = 0.0,
    seed: int = 0,
    images=None,
    labels=None,
    names_path=None,
    group_by: str = "filename",
    group_regex: str = DEFAULT_GROUP_REGEX,
    mode: str = "copy",
) -> dict:
    if not (0 <= val < 1 and 0 <= test < 1 and val + test < 1):
        raise PrepError("--val and --test must be fractions in [0, 1) that add up to less than 1.")
    if val == 0:
        raise PrepError("--val must be greater than 0.")
    if mode not in ("copy", "symlink"):
        raise PrepError("mode must be copy or symlink")
    if group_by not in ("filename", "folder"):
        raise PrepError("--group-by must be filename or folder")
    try:
        pattern = re.compile(group_regex)
    except re.error as exc:
        raise PrepError(f"Invalid --group-regex: {exc}") from exc
    if group_by == "filename" and "group" not in pattern.groupindex and pattern.groups < 1:
        raise PrepError("--group-regex needs a capture group, for example '^(.+)_f\\d+$'.")

    img_dir, lbl_dir = find_yolo_pair(root, images, labels)
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise PrepError(f"Output folder is not empty: {out_dir}. Choose a new folder so nothing is overwritten.")
    if out_dir.resolve().is_relative_to(img_dir.resolve()) or out_dir.resolve().is_relative_to(lbl_dir.resolve()):
        raise PrepError("The output folder must not be inside the input images/labels folders.")

    images_found = list_images(img_dir)
    if not images_found:
        raise PrepError(f"No images found in {img_dir}")

    groups: Dict[str, List[Path]] = defaultdict(list)
    unrecognised = 0
    for img in images_found:
        g, ok = group_of(img, img_dir, group_by, pattern)
        groups[g].append(img)
        unrecognised += 0 if ok else 1

    fractions = {"train": 1.0 - val - test, "val": val, "test": test}
    assignment = assign_groups({g: len(v) for g, v in groups.items()}, fractions, seed)
    parts = [p for p in ("train", "val", "test") if fractions[p] > 0]

    names = read_names(Path(root), names_path)
    stats = {p: {"images": 0, "groups": 0, "boxes": 0, "images_without_labels": 0, "classes": Counter()} for p in parts}
    group_parts: Dict[str, set] = defaultdict(set)

    for g, files in groups.items():
        part = assignment[g]
        stats[part]["groups"] += 1
        for img in files:
            group_parts[g].add(part)
            rel = img.relative_to(img_dir)
            place_file(img, out_dir / "images" / part / rel, mode)
            lbl = label_path(img, img_dir, lbl_dir)
            stats[part]["images"] += 1
            if lbl.is_file():
                place_file(lbl, out_dir / "labels" / part / rel.with_suffix(".txt"), mode)
                for a in parse_label_file(lbl).anns:
                    stats[part]["boxes"] += 1
                    stats[part]["classes"][a.cls] += 1
            else:
                stats[part]["images_without_labels"] += 1

    leaked = [g for g, ps in group_parts.items() if len(ps) > 1]
    if leaked:  # cannot happen by construction; verified anyway
        raise PrepError(f"Internal error: groups in more than one part: {leaked[:3]}")

    all_classes = sorted({c for p in parts for c in stats[p]["classes"]} | set(names))
    warnings: List[str] = []
    for p in parts:
        missing = [c for c in all_classes if stats[p]["classes"][c] == 0 and any(stats[q]["classes"][c] for q in parts)]
        if missing:
            shown = ", ".join(names.get(c, str(c)) for c in missing[:6])
            warnings.append(f"'{p}' has no boxes of class: {shown}{' ...' if len(missing) > 6 else ''}")
    if unrecognised:
        warnings.append(
            f"{unrecognised} image(s) did not match the video pattern and were treated as one group each, "
            "so frames of the same video may still be split apart. Use --group-regex or --group-by folder."
        )

    write_data_yaml(
        out_dir / "data.yaml",
        {c: names.get(c, str(c)) for c in all_classes},
        **{p: f"images/{p}" for p in parts},
    )
    report = {
        "seed": seed,
        "group_by": group_by,
        "groups": len(groups),
        "images": len(images_found),
        "parts": {
            p: {**{k: v for k, v in stats[p].items() if k != "classes"}, "classes": {str(k): v for k, v in sorted(stats[p]["classes"].items())}}
            for p in parts
        },
        "assignment": {g: assignment[g] for g in sorted(assignment)},
        "leakage": False,
        "warnings": warnings,
    }
    (out_dir / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

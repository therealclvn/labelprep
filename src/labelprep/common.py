"""Shared helpers: YOLO folder layout, label parsing, class names, safe paths."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}


class PrepError(ValueError):
    """A problem the user can fix (bad path, bad option). Printed without a traceback."""


@dataclass
class Ann:
    """One YOLO annotation in normalised coordinates."""

    cls: int
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2
    polygon: Optional[List[Tuple[float, float]]] = None


@dataclass
class LabelFile:
    anns: List[Ann] = field(default_factory=list)
    bad_lines: int = 0


# ------------------------------------------------------------------------ layout


def find_yolo_pair(root, images=None, labels=None) -> Tuple[Path, Path]:
    root = Path(root)
    if images or labels:
        if not (images and labels):
            raise PrepError("Give both --images and --labels, or neither.")
        img_dir, lbl_dir = Path(images), Path(labels)
    else:
        if not root.is_dir():
            raise PrepError(f"Dataset folder not found: {root}")
        img_dir, lbl_dir = root / "images", root / "labels"
        if not (img_dir.is_dir() and lbl_dir.is_dir()):
            raise PrepError(
                f"Expected images/ and labels/ folders inside {root}. "
                "Pass --images and --labels if your layout is different."
            )
    if not img_dir.is_dir():
        raise PrepError(f"Images folder not found: {img_dir}")
    if not lbl_dir.is_dir():
        raise PrepError(f"Labels folder not found: {lbl_dir}")
    return img_dir, lbl_dir


def list_images(img_dir: Path) -> List[Path]:
    return sorted(p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def label_path(img: Path, img_dir: Path, lbl_dir: Path) -> Path:
    return (lbl_dir / img.relative_to(img_dir)).with_suffix(".txt")


def safe_relpath(name: str) -> Optional[PurePosixPath]:
    """A relative path that cannot escape the output folder, or None if the name is unsafe."""
    name = str(name).replace("\\", "/")
    p = PurePosixPath(name)
    if not name or p.is_absolute() or re.match(r"^[A-Za-z]:", name):
        return None
    if any(part in ("..", "") for part in p.parts) or p.name in ("", "."):
        return None
    return p


# ------------------------------------------------------------------------- labels


def parse_label_file(path: Path) -> LabelFile:
    out = LabelFile()
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        out.bad_lines += 1
        return out
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            values = [float(x) for x in parts]
        except ValueError:
            out.bad_lines += 1
            continue
        if not all(math.isfinite(v) for v in values) or values[0] != int(values[0]) or values[0] < 0:
            out.bad_lines += 1
            continue
        cls, nums = int(values[0]), values[1:]
        if len(nums) == 4:
            cx, cy, w, h = nums
            out.anns.append(Ann(cls, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
        elif len(nums) >= 6 and len(nums) % 2 == 0:
            pts = list(zip(nums[0::2], nums[1::2]))
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            out.anns.append(Ann(cls, (min(xs), min(ys), max(xs), max(ys)), pts))
        else:
            out.bad_lines += 1
    return out


def yolo_box_line(cls: int, box, precision: int = 6) -> str:
    x1, y1, x2, y2 = box
    vals = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
    return f"{cls} " + " ".join(f"{v:.{precision}f}" for v in vals)


def yolo_polygon_line(cls: int, points, precision: int = 6) -> str:
    return f"{cls} " + " ".join(f"{c:.{precision}f}" for p in points for c in p)


# --------------------------------------------------------------------- class names


def _parse_names_yaml(text: str) -> Dict[int, str]:
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
        names = data.get("names") if isinstance(data, dict) else None
        if isinstance(names, list):
            return {i: str(n) for i, n in enumerate(names)}
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        return {}
    except ImportError:
        pass
    m = re.search(r"^names\s*:\s*\[(.*?)\]", text, re.S | re.M)
    if m:
        items = [s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()]
        return {i: n for i, n in enumerate(items)}
    names: Dict[int, str] = {}
    in_block, seq = False, 0
    for line in text.splitlines():
        if re.match(r"^names\s*:\s*$", line):
            in_block = True
            continue
        if in_block:
            if line and not line.startswith((" ", "\t", "-")):
                break
            kv = re.match(r"^\s+(\d+)\s*:\s*(.+?)\s*$", line)
            li = re.match(r"^\s*-\s*(.+?)\s*$", line)
            if kv:
                names[int(kv.group(1))] = kv.group(2).strip("'\"")
            elif li:
                names[seq] = li.group(1).strip("'\"")
                seq += 1
    return names


def read_names(root: Optional[Path], names_path=None) -> Dict[int, str]:
    candidates: List[Path] = []
    if names_path:
        candidates.append(Path(names_path))
    elif root:
        candidates += [Path(root) / n for n in ("data.yaml", "dataset.yaml", "data.yml", "classes.txt")]
    for c in candidates:
        if not c.is_file():
            if names_path:
                raise PrepError(f"Names file not found: {c}")
            continue
        text = c.read_text(encoding="utf-8-sig")
        names = _parse_names_yaml(text) if c.suffix.lower() in (".yaml", ".yml") else {
            i: ln.strip() for i, ln in enumerate(text.splitlines()) if ln.strip()
        }
        if names:
            return names
    return {}


def write_data_yaml(path: Path, names: Dict[int, str], **splits: Optional[str]) -> None:
    """Minimal Ultralytics-style data.yaml (no PyYAML needed to write it)."""
    lines = ["path: ."]
    for key in ("train", "val", "test"):
        if splits.get(key):
            lines.append(f"{key}: {splits[key]}")
    lines.append("names:")
    for i in sorted(names):
        lines.append(f"  {i}: {json.dumps(str(names[i]), ensure_ascii=False)}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def place_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(Path(src).resolve())
    else:
        import shutil

        shutil.copy2(src, dst)

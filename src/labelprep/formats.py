"""YOLO <-> COCO conversion for boxes and polygons.

Conventions
-----------
* YOLO class ``i`` becomes COCO category id ``i + 1`` (the usual convention, because many COCO
  tools treat id 0 as background). Going back, category ids are sorted and numbered 0..n-1, so a
  round trip gives the original ids.
* Annotations that cannot be represented are counted in the returned report, never dropped silently.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .common import (
    PrepError,
    find_yolo_pair,
    label_path,
    list_images,
    parse_label_file,
    place_file,
    read_names,
    safe_relpath,
    write_data_yaml,
    yolo_box_line,
    yolo_polygon_line,
)


def _poly_area(points) -> float:
    s = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


# ----------------------------------------------------------------------- YOLO -> COCO


def yolo_to_coco(root, out_json, images=None, labels=None, names_path=None, category_offset: int = 1) -> dict:
    from PIL import Image

    img_dir, lbl_dir = find_yolo_pair(root, images, labels)
    names = read_names(Path(root) if root else None, names_path)
    out_images: List[dict] = []
    out_anns: List[dict] = []
    used_classes = set()
    report = {"images": 0, "annotations": 0, "images_without_label_file": 0, "bad_label_lines": 0, "unreadable_images": 0}

    ann_id = 1
    for img in list_images(img_dir):
        try:
            with Image.open(img) as im:
                width, height = im.size
        except Exception:
            report["unreadable_images"] += 1
            continue
        image_id = len(out_images) + 1
        out_images.append(
            {"id": image_id, "file_name": img.relative_to(img_dir).as_posix(), "width": width, "height": height}
        )
        lbl = label_path(img, img_dir, lbl_dir)
        if not lbl.is_file():
            report["images_without_label_file"] += 1
            continue
        parsed = parse_label_file(lbl)
        report["bad_label_lines"] += parsed.bad_lines
        for a in parsed.anns:
            x1, y1, x2, y2 = a.box[0] * width, a.box[1] * height, a.box[2] * width, a.box[3] * height
            seg: List[List[float]] = []
            if a.polygon:
                pts = [(x * width, y * height) for x, y in a.polygon]
                seg = [[round(c, 2) for p in pts for c in p]]
                area = _poly_area(pts)
            else:
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            out_anns.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": a.cls + category_offset,
                    "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                    "area": round(area, 2),
                    "iscrowd": 0,
                    "segmentation": seg,
                }
            )
            used_classes.add(a.cls)
            ann_id += 1

    # List every id from 0 to the highest one, including unused ones. Skipping gaps would make
    # coco2yolo renumber the classes (it numbers categories 0..n-1) and shift every later class.
    known = set(names) | used_classes
    ids = list(range(max(known) + 1)) if known else []
    categories = [{"id": i + category_offset, "name": names.get(i, str(i)), "supercategory": ""} for i in ids]
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(
        json.dumps({"images": out_images, "annotations": out_anns, "categories": categories}, ensure_ascii=False),
        encoding="utf-8",
    )
    report["images"] = len(out_images)
    report["annotations"] = len(out_anns)
    return report


# ----------------------------------------------------------------------- COCO -> YOLO


def coco_to_yolo(
    ann_json,
    out_dir,
    images_dir=None,
    mode: str = "copy",
    segments: bool = False,
    clip: bool = True,
) -> dict:
    ann_json, out_dir = Path(ann_json), Path(out_dir)
    if not ann_json.is_file():
        raise PrepError(f"COCO file not found: {ann_json}")
    if mode not in ("copy", "symlink"):
        raise PrepError("mode must be copy or symlink")
    try:
        data = json.loads(ann_json.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise PrepError(f"Not valid JSON: {ann_json} ({exc})") from exc
    if not isinstance(data, dict) or not isinstance(data.get("images"), list):
        raise PrepError("This does not look like a COCO file (no 'images' list).")
    images_dir = Path(images_dir) if images_dir else None
    if images_dir is not None and not images_dir.is_dir():
        raise PrepError(f"Images folder not found: {images_dir}")

    cats = {c["id"]: str(c.get("name", c["id"])) for c in data.get("categories", []) if isinstance(c, dict) and "id" in c}
    index_of = {cid: i for i, cid in enumerate(sorted(cats))}
    names = {index_of[cid]: name for cid, name in cats.items()}

    report = {
        "images": 0,
        "annotations": 0,
        "clipped": 0,
        "dropped_degenerate": 0,
        "dropped_crowd": 0,
        "dropped_unknown_category": 0,
        "dropped_no_geometry": 0,
        "images_skipped_unsafe_name": 0,
        "images_skipped_no_size": 0,
        "images_missing": 0,
        "images_copied": 0,
        "name_collisions": 0,
    }

    by_image: Dict[object, dict] = {}
    seen_targets: Dict[str, object] = {}
    for im in data["images"]:
        if not isinstance(im, dict):
            continue
        rel = safe_relpath(im.get("file_name", ""))
        if rel is None:
            report["images_skipped_unsafe_name"] += 1
            continue
        target = rel.with_suffix("").as_posix().lower()
        if target in seen_targets:
            report["name_collisions"] += 1
            raise PrepError(
                f"Two images would write the same label file ({rel.with_suffix('.txt')}): "
                f"{seen_targets[target]} and {rel}. Rename one of them."
            )
        seen_targets[target] = rel
        width, height = im.get("width"), im.get("height")
        src = (images_dir / rel.as_posix()) if images_dir else None
        if (not width or not height) and src is not None and src.is_file():
            try:
                from PIL import Image

                with Image.open(src) as pim:
                    width, height = pim.size
            except Exception:
                pass
        if not width or not height:
            report["images_skipped_no_size"] += 1
            continue
        by_image[im.get("id")] = {"rel": rel, "w": float(width), "h": float(height), "lines": [], "src": src}

    for ann in data.get("annotations", []):
        if not isinstance(ann, dict):
            continue
        entry = by_image.get(ann.get("image_id"))
        if entry is None:
            continue
        if ann.get("iscrowd"):
            report["dropped_crowd"] += 1
            continue
        cls = index_of.get(ann.get("category_id"))
        if cls is None:
            report["dropped_unknown_category"] += 1
            continue
        w, h = entry["w"], entry["h"]

        polys = []
        seg = ann.get("segmentation")
        if isinstance(seg, list):
            for flat in seg:
                if isinstance(flat, list) and len(flat) >= 6 and len(flat) % 2 == 0 and all(isinstance(v, (int, float)) for v in flat):
                    polys.append([(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)])

        box = None
        bbox = ann.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
            box = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
        elif polys:
            xs = [p[0] for poly in polys for p in poly]
            ys = [p[1] for poly in polys for p in poly]
            box = (min(xs), min(ys), max(xs), max(ys))
        if box is None:
            report["dropped_no_geometry"] += 1
            continue

        def clamp(v, hi):
            return min(max(v, 0.0), hi)

        use_polys = segments and polys
        items = polys if use_polys else [None]
        wrote = False
        for poly in items:
            if poly is None:
                x1, y1, x2, y2 = box
                if clip:
                    cx1, cy1, cx2, cy2 = clamp(x1, w), clamp(y1, h), clamp(x2, w), clamp(y2, h)
                    if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
                        report["clipped"] += 1
                    x1, y1, x2, y2 = cx1, cy1, cx2, cy2
                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    report["dropped_degenerate"] += 1
                    continue
                entry["lines"].append(yolo_box_line(cls, (x1 / w, y1 / h, x2 / w, y2 / h)))
            else:
                pts = poly
                if clip:
                    cl = [(clamp(x, w), clamp(y, h)) for x, y in pts]
                    if cl != [tuple(p) for p in pts]:
                        report["clipped"] += 1
                    pts = cl
                if _poly_area(pts) <= 0:
                    report["dropped_degenerate"] += 1
                    continue
                entry["lines"].append(yolo_polygon_line(cls, [(x / w, y / h) for x, y in pts]))
            wrote = True
        if wrote:
            report["annotations"] += 1

    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    for entry in by_image.values():
        rel = entry["rel"]
        lbl = out_dir / "labels" / rel.with_suffix(".txt").as_posix()
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text("".join(line + "\n" for line in entry["lines"]), encoding="utf-8")
        report["images"] += 1
        if images_dir is not None:
            src = entry["src"]
            if src is not None and src.is_file():
                place_file(src, out_dir / "images" / rel.as_posix(), mode)
                report["images_copied"] += 1
            else:
                report["images_missing"] += 1
    write_data_yaml(out_dir / "data.yaml", names)
    return report

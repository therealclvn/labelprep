"""Command line: ``labelprep frames | convert | split``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .common import PrepError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="labelprep", description="Prepare detection datasets: frames, YOLO<->COCO, train/val split.")
    p.add_argument("--version", action="version", version=f"labelprep {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("frames", help="cut videos into image frames")
    f.add_argument("videos", nargs="+", help="video files and/or folders containing videos")
    f.add_argument("-o", "--out", required=True, help="folder for the frames (created if needed)")
    g = f.add_mutually_exclusive_group(required=True)
    g.add_argument("--every", type=int, metavar="N", help="keep every N-th frame")
    g.add_argument("--seconds", type=float, metavar="S", help="keep one frame every S seconds")
    f.add_argument("--max-side", type=int, metavar="PX", help="shrink frames so the longer side is at most PX")
    f.add_argument("--start", type=float, default=0.0, metavar="S", help="start at this time in seconds")
    f.add_argument("--end", type=float, metavar="S", help="stop at this time in seconds")
    f.add_argument("--format", choices=("jpg", "png"), default="jpg")
    f.add_argument("--quality", type=int, default=92, help="JPEG quality 1-100 (default 92)")
    f.add_argument("--overwrite", action="store_true", help="rewrite frames that already exist (default: skip them)")

    c = sub.add_parser("convert", help="convert between YOLO and COCO")
    csub = c.add_subparsers(dest="direction", required=True)
    y2c = csub.add_parser("yolo2coco", help="YOLO folder -> COCO json")
    y2c.add_argument("root", help="dataset folder with images/ and labels/")
    y2c.add_argument("out", help="COCO json file to write")
    y2c.add_argument("--images")
    y2c.add_argument("--labels")
    y2c.add_argument("--names", help="data.yaml or classes.txt (auto-detected in the dataset folder)")
    c2y = csub.add_parser("coco2yolo", help="COCO json -> YOLO folder")
    c2y.add_argument("annotations", help="COCO json file")
    c2y.add_argument("out", help="output folder (labels/, data.yaml and optionally images/)")
    c2y.add_argument("--images", help="folder with the images; if given they are copied next to the labels")
    c2y.add_argument("--link", choices=("copy", "symlink"), default="copy", help="how to place images (default copy)")
    c2y.add_argument("--segments", action="store_true", help="write polygons (one line per polygon) where available instead of boxes")
    c2y.add_argument("--no-clip", action="store_true", help="do not clip boxes that stick out of the image")

    s = sub.add_parser("split", help="split a YOLO dataset into train/val(/test) by source video")
    s.add_argument("root", help="dataset folder with images/ and labels/")
    s.add_argument("-o", "--out", required=True, help="new, empty output folder")
    s.add_argument("--val", type=float, default=0.2, help="fraction of images for validation (default 0.2)")
    s.add_argument("--test", type=float, default=0.0, help="fraction of images for test (default 0)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--images")
    s.add_argument("--labels")
    s.add_argument("--names")
    s.add_argument("--group-by", choices=("filename", "folder"), default="filename", help="how to find which video a frame belongs to")
    s.add_argument("--group-regex", default=None, help="regex whose first group (or group 'group') is the video id; default matches NAME_f000123")
    s.add_argument("--link", choices=("copy", "symlink"), default="copy")
    return p


def _print_counts(title: str, report: dict, skip=()) -> None:
    print(title)
    for k, v in report.items():
        if k in skip or isinstance(v, (dict, list)):
            continue
        if v or k in ("images", "annotations"):
            print(f"  {k.replace('_', ' ')}: {v}")


def _run(args) -> int:
    if args.command == "frames":
        from .frames import extract_frames, find_videos, write_manifest

        videos = find_videos(args.videos)
        out = Path(args.out)
        rows = []
        for v in videos:
            r = extract_frames(
                v, out, every=args.every, seconds=args.seconds, max_side=args.max_side, quality=args.quality,
                start=args.start, end=args.end, overwrite=args.overwrite, ext=args.format,
            )
            rows += r["rows"]
            extra = f", {r['skipped']} already existed" if r["skipped"] else ""
            print(f"{v.name}: {r['frames_read']} frames read, {r['written']} written{extra}")
        write_manifest(out / "manifest.csv", rows)
        print(f"{len(rows)} frames in {out}  (manifest: {out / 'manifest.csv'})")
        return 0

    if args.command == "convert":
        from .formats import coco_to_yolo, yolo_to_coco

        if args.direction == "yolo2coco":
            rep = yolo_to_coco(args.root, args.out, images=args.images, labels=args.labels, names_path=args.names)
            _print_counts(f"Wrote {args.out}", rep)
        else:
            rep = coco_to_yolo(
                args.annotations, args.out, images_dir=args.images, mode=args.link,
                segments=args.segments, clip=not args.no_clip,
            )
            _print_counts(f"Wrote YOLO labels to {args.out}", rep)
            if not args.images:
                print("  note: no --images given, so only labels and data.yaml were written")
        return 0

    if args.command == "split":
        from .split import DEFAULT_GROUP_REGEX, split_dataset

        rep = split_dataset(
            args.root, args.out, val=args.val, test=args.test, seed=args.seed, images=args.images,
            labels=args.labels, names_path=args.names, group_by=args.group_by,
            group_regex=args.group_regex or DEFAULT_GROUP_REGEX, mode=args.link,
        )
        print(f"{rep['images']} images in {rep['groups']} group(s) -> {args.out}")
        for part, st in rep["parts"].items():
            print(f"  {part:5} {st['images']:6} images  {st['groups']:4} groups  {st['boxes']:7} boxes")
        print("  no group appears in more than one part (checked)")
        for w in rep["warnings"]:
            print(f"  warning: {w}")
        print(f"  details: {Path(args.out) / 'split_report.json'}")
        return 0
    return 2  # pragma: no cover


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except PrepError as exc:
        print(f"labelprep: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

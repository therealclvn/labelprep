"""Cut videos into image frames with stable, grouping-friendly names.

Every frame is called ``<video_id>_f<frame number, 6 digits>.jpg``. ``labelprep split`` uses that
pattern to keep all frames of one video in the same train/val/test part.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .common import VIDEO_EXTENSIONS, PrepError


def _cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except ImportError as exc:  # pragma: no cover - message only
        raise PrepError(
            "Cutting videos needs OpenCV. Install it with:  pip install \"labelprep[video]\"  "
            "(or: pip install opencv-python-headless)"
        ) from exc


def make_video_id(path: Path) -> str:
    """File name without extension, with anything unusual replaced so it is safe in a file name."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(path).stem).strip("-._")
    return stem or "video"


def find_videos(inputs: Iterable[str]) -> List[Path]:
    videos: List[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            videos += sorted(q for q in p.rglob("*") if q.is_file() and q.suffix.lower() in VIDEO_EXTENSIONS)
        elif p.is_file():
            videos.append(p)
        else:
            raise PrepError(f"Not a file or folder: {p}")
    if not videos:
        raise PrepError("No video files found.")
    ids: Dict[str, Path] = {}
    for v in videos:
        vid = make_video_id(v)
        if vid in ids:
            raise PrepError(
                f"Two videos would get the same id '{vid}': {ids[vid]} and {v}. Rename one of them."
            )
        ids[vid] = v
    return videos


@dataclass
class FrameRecord:
    image: str
    video_id: str
    frame: int
    time_s: float


def extract_frames(
    video: Path,
    out_dir: Path,
    every: Optional[int] = None,
    seconds: Optional[float] = None,
    max_side: Optional[int] = None,
    quality: int = 92,
    start: float = 0.0,
    end: Optional[float] = None,
    overwrite: bool = False,
    ext: str = "jpg",
) -> dict:
    """Write frames of one video into ``out_dir``. Returns counts and the manifest rows."""
    if (every is None) == (seconds is None):
        raise PrepError("Choose exactly one of --every N (frames) or --seconds S.")
    if every is not None and every < 1:
        raise PrepError("--every must be at least 1.")
    if seconds is not None and seconds <= 0:
        raise PrepError("--seconds must be greater than 0.")
    if ext not in ("jpg", "png"):
        raise PrepError("Output format must be jpg or png.")
    cv2 = _cv2()

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise PrepError(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps != fps or fps <= 0:  # NaN or missing
        if seconds is not None:
            cap.release()
            raise PrepError(f"{video.name}: the video does not report its frame rate, so --seconds cannot be used. Use --every.")
        fps = 0.0
    step = every if every is not None else max(1, round(seconds * fps))
    vid = make_video_id(video)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)] if ext == "jpg" else []

    rows: List[FrameRecord] = []
    written = skipped = 0
    index = -1
    next_keep = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        index += 1
        t = index / fps if fps else 0.0
        if t < start:
            continue
        if end is not None and t > end:
            break
        if next_keep is None:
            next_keep = index
        if index < next_keep:
            continue
        next_keep = index + step

        name = f"{vid}_f{index:06d}.{ext}"
        dst = out_dir / name
        if dst.exists() and not overwrite:
            skipped += 1
        else:
            if max_side:
                h, w = frame.shape[:2]
                scale = max_side / max(h, w)
                if scale < 1.0:
                    frame = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
            ok_enc, buf = cv2.imencode("." + ext, frame, params)
            if not ok_enc:
                cap.release()
                raise PrepError(f"Could not encode frame {index} of {video.name}")
            dst.write_bytes(buf.tobytes())
            written += 1
        rows.append(FrameRecord(name, vid, index, round(t, 3)))
    cap.release()
    if index < 0:
        raise PrepError(f"{video.name}: no frames could be read (unsupported codec?).")
    return {"video": str(video), "video_id": vid, "fps": fps, "frames_read": index + 1, "written": written, "skipped": skipped, "rows": rows}


def write_manifest(path: Path, rows: List[FrameRecord]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "video_id", "frame", "time_s"])
        for r in rows:
            w.writerow([r.image, r.video_id, r.frame, r.time_s])

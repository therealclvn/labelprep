from pathlib import Path

from PIL import Image


def make_image(path: Path, size=(200, 100), color=(120, 160, 120)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def make_yolo(root: Path, files: dict, names=("cat", "dog"), size=(200, 100)):
    """files: relative stem -> label text (None = no label file). Images go to images/<stem>.jpg."""
    for stem, text in files.items():
        make_image(root / "images" / f"{stem}.jpg", size)
        if text is not None:
            lbl = root / "labels" / f"{stem}.txt"
            lbl.parent.mkdir(parents=True, exist_ok=True)
            lbl.write_text(text)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    body = "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names))
    (root / "data.yaml").write_text(body)
    return root


def make_video(path: Path, n_frames=30, fps=10, size=(64, 48)):
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert w.isOpened()
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), (i * 8) % 255, np.uint8)
        w.write(frame)
    w.release()
    return path

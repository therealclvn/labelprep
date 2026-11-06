import csv

import pytest

pytest.importorskip("cv2")

from labelprep.common import PrepError
from labelprep.frames import extract_frames, find_videos, make_video_id, write_manifest

from helpers import make_video


def test_video_id_is_filename_safe():
    from pathlib import Path

    assert make_video_id(Path("My Clip (final).MP4")) == "My-Clip-final"
    assert make_video_id(Path("空拍_01.mp4")) == "01"
    assert make_video_id(Path("###.mp4")) == "video"


def test_every_n_frames(tmp_path):
    v = make_video(tmp_path / "clip a.mp4", n_frames=30, fps=10)
    r = extract_frames(v, tmp_path / "out", every=10)
    names = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert names == ["clip-a_f000000.jpg", "clip-a_f000010.jpg", "clip-a_f000020.jpg"]
    assert r["frames_read"] == 30 and r["written"] == 3
    assert [row.frame for row in r["rows"]] == [0, 10, 20]
    assert r["rows"][1].time_s == 1.0


def test_every_n_seconds_and_range(tmp_path):
    v = make_video(tmp_path / "c.mp4", n_frames=40, fps=10)
    r = extract_frames(v, tmp_path / "o", seconds=1.0, start=1.0, end=3.0)
    assert [row.frame for row in r["rows"]] == [10, 20, 30]


def test_max_side_resizes_keeping_aspect(tmp_path):
    from PIL import Image

    v = make_video(tmp_path / "c.mp4", n_frames=3, size=(128, 64))
    extract_frames(v, tmp_path / "o", every=1, max_side=32)
    with Image.open(tmp_path / "o" / "c_f000000.jpg") as im:
        assert im.size == (32, 16)


def test_existing_frames_are_skipped_unless_overwrite(tmp_path):
    v = make_video(tmp_path / "c.mp4", n_frames=4)
    extract_frames(v, tmp_path / "o", every=2)
    again = extract_frames(v, tmp_path / "o", every=2)
    assert again["written"] == 0 and again["skipped"] == 2 and len(again["rows"]) == 2
    forced = extract_frames(v, tmp_path / "o", every=2, overwrite=True)
    assert forced["written"] == 2


def test_option_errors(tmp_path):
    v = make_video(tmp_path / "c.mp4", n_frames=4)
    with pytest.raises(PrepError):
        extract_frames(v, tmp_path / "o")  # neither option
    with pytest.raises(PrepError):
        extract_frames(v, tmp_path / "o", every=2, seconds=1.0)  # both
    with pytest.raises(PrepError):
        extract_frames(v, tmp_path / "o", every=0)
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(PrepError):
        extract_frames(bad, tmp_path / "o", every=1)


def test_find_videos_folder_and_duplicates(tmp_path):
    make_video(tmp_path / "a" / "x.mp4", n_frames=2)
    make_video(tmp_path / "b" / "y.mp4", n_frames=2)
    (tmp_path / "a" / "notes.txt").write_text("hi")
    assert [p.name for p in find_videos([str(tmp_path)])] == ["x.mp4", "y.mp4"]
    make_video(tmp_path / "b" / "x.mp4", n_frames=2)
    with pytest.raises(PrepError, match="same id"):
        find_videos([str(tmp_path)])
    with pytest.raises(PrepError):
        find_videos([str(tmp_path / "missing")])


def test_manifest(tmp_path):
    v = make_video(tmp_path / "c.mp4", n_frames=4)
    r = extract_frames(v, tmp_path / "o", every=2)
    write_manifest(tmp_path / "m.csv", r["rows"])
    rows = list(csv.reader(open(tmp_path / "m.csv")))
    assert rows[0] == ["image", "video_id", "frame", "time_s"]
    assert rows[1][:3] == ["c_f000000.jpg", "c", "0"]

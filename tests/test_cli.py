import json

import pytest

from labelprep.cli import main

from helpers import make_video, make_yolo


def test_frames_command(tmp_path, capsys):
    pytest.importorskip("cv2")
    make_video(tmp_path / "vids" / "one.mp4", n_frames=20)
    make_video(tmp_path / "vids" / "two.mp4", n_frames=10)
    assert main(["frames", str(tmp_path / "vids"), "-o", str(tmp_path / "f"), "--every", "5"]) == 0
    out = capsys.readouterr().out
    assert "one.mp4: 20 frames read, 4 written" in out and "6 frames in" in out
    assert (tmp_path / "f" / "manifest.csv").is_file() and (tmp_path / "f" / "two_f000005.jpg").is_file()


def test_convert_and_split_commands(tmp_path, capsys):
    files = {f"v{i}_f{j:06d}": "0 0.5 0.5 0.2 0.2\n" for i in range(6) for j in range(3)}
    root = make_yolo(tmp_path / "ds", files)
    assert main(["convert", "yolo2coco", str(root), str(tmp_path / "c.json")]) == 0
    assert json.loads((tmp_path / "c.json").read_text())["annotations"].__len__() == 18
    assert main(["convert", "coco2yolo", str(tmp_path / "c.json"), str(tmp_path / "back")]) == 0
    assert "no --images given" in capsys.readouterr().out
    assert main(["split", str(root), "-o", str(tmp_path / "s"), "--val", "0.25"]) == 0
    assert "no group appears in more than one part" in capsys.readouterr().out


def test_errors_exit_with_2_and_no_traceback(tmp_path, capsys):
    assert main(["split", str(tmp_path / "missing"), "-o", str(tmp_path / "o")]) == 2
    assert "labelprep: error:" in capsys.readouterr().err
    assert main(["convert", "coco2yolo", str(tmp_path / "x.json"), str(tmp_path / "o")]) == 2
    with pytest.raises(SystemExit):  # argparse: neither --every nor --seconds
        main(["frames", str(tmp_path), "-o", str(tmp_path / "o")])

import json
import re
from pathlib import Path

import pytest

from labelprep.common import PrepError
from labelprep.split import assign_groups, split_dataset

from helpers import make_yolo


def video_dataset(root, videos):
    """videos: video id -> number of frames. Every frame has one box of class (frame % 2)."""
    files = {}
    for vid, n in videos.items():
        for i in range(n):
            files[f"{vid}_f{i * 10:06d}"] = f"{i % 2} 0.5 0.5 0.2 0.2\n"
    return make_yolo(root, files)


def parts_of(out):
    result = {}
    for part in ("train", "val", "test"):
        d = Path(out) / "images" / part
        if d.is_dir():
            result[part] = sorted(p.name for p in d.iterdir())
    return result


def test_no_video_is_split_across_parts(tmp_path):
    root = video_dataset(tmp_path / "ds", {f"v{i}": 5 + i for i in range(10)})
    rep = split_dataset(root, tmp_path / "out", val=0.2, test=0.1, seed=3)
    seen = {}
    for part, files in parts_of(tmp_path / "out").items():
        for f in files:
            vid = re.match(r"^(.+)_f\d+", f).group(1)
            assert seen.setdefault(vid, part) == part, f"{vid} appears in {seen[vid]} and {part}"
    assert sum(len(v) for v in parts_of(tmp_path / "out").values()) == rep["images"] == sum(5 + i for i in range(10))
    assert rep["leakage"] is False and set(rep["parts"]) == {"train", "val", "test"}


def test_sizes_are_close_to_targets(tmp_path):
    root = video_dataset(tmp_path / "ds", {f"v{i}": 20 for i in range(10)})  # 200 images
    rep = split_dataset(root, tmp_path / "out", val=0.2, test=0.2)
    assert rep["parts"]["val"]["images"] == 40 and rep["parts"]["test"]["images"] == 40
    assert rep["parts"]["train"]["images"] == 120


def test_deterministic_and_seed_changes_result(tmp_path):
    root = video_dataset(tmp_path / "ds", {f"v{i}": 4 for i in range(12)})
    a = split_dataset(root, tmp_path / "a", seed=1)["assignment"]
    b = split_dataset(root, tmp_path / "b", seed=1)["assignment"]
    c = split_dataset(root, tmp_path / "c", seed=2)["assignment"]
    assert a == b and a != c


def test_labels_follow_images_and_data_yaml(tmp_path):
    root = video_dataset(tmp_path / "ds", {"v0": 4, "v1": 4, "v2": 4, "v3": 4, "v4": 4})
    split_dataset(root, tmp_path / "out", val=0.2)
    for part in ("train", "val"):
        imgs = {p.stem for p in (tmp_path / "out" / "images" / part).iterdir()}
        lbls = {p.stem for p in (tmp_path / "out" / "labels" / part).iterdir()}
        assert imgs == lbls
    yaml = (tmp_path / "out" / "data.yaml").read_text()
    assert "train: images/train" in yaml and "val: images/val" in yaml and "test:" not in yaml
    assert '0: "cat"' in yaml and '1: "dog"' in yaml
    rep = json.loads((tmp_path / "out" / "split_report.json").read_text())
    assert rep["parts"]["val"]["boxes"] == rep["parts"]["val"]["images"]


def test_too_few_groups_is_an_error(tmp_path):
    root = video_dataset(tmp_path / "ds", {"only": 10})
    with pytest.raises(PrepError, match="at least one group"):
        split_dataset(root, tmp_path / "out")
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())


def test_two_groups_give_each_part_one(tmp_path):
    root = video_dataset(tmp_path / "ds", {"a": 9, "b": 1})
    rep = split_dataset(root, tmp_path / "out", val=0.2)
    assert {v["groups"] for v in rep["parts"].values()} == {1}


def test_unrecognised_names_warn(tmp_path):
    root = make_yolo(tmp_path / "ds", {f"photo{i}": "0 0.5 0.5 0.2 0.2\n" for i in range(10)})
    rep = split_dataset(root, tmp_path / "out")
    assert any("did not match the video pattern" in w for w in rep["warnings"])


def test_group_by_folder_and_custom_regex(tmp_path):
    root = make_yolo(tmp_path / "ds", {f"{vid}/img{i}": "0 0.5 0.5 0.2 0.2\n" for vid in "abcde" for i in range(3)})
    rep = split_dataset(root, tmp_path / "o1", group_by="folder")
    assert rep["groups"] == 5
    root2 = make_yolo(tmp_path / "ds2", {f"cam{c}-{i:03d}": "0 0.5 0.5 0.2 0.2\n" for c in range(5) for i in range(3)})
    rep2 = split_dataset(root2, tmp_path / "o2", group_regex=r"^(cam\d)-\d+$")
    assert rep2["groups"] == 5 and not rep2["warnings"]


def test_missing_class_in_a_part_warns(tmp_path):
    files = {f"v{i}_f000000": "1 0.5 0.5 0.2 0.2\n" for i in range(4)}
    files["v9_f000000"] = "0 0.5 0.5 0.2 0.2\n"
    rep = split_dataset(make_yolo(tmp_path / "ds", files), tmp_path / "out", val=0.2)
    assert any("has no boxes of class" in w for w in rep["warnings"])


def test_refuses_unsafe_output_and_bad_options(tmp_path):
    root = video_dataset(tmp_path / "ds", {f"v{i}": 3 for i in range(5)})
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "x.txt").write_text("hi")
    with pytest.raises(PrepError, match="not empty"):
        split_dataset(root, tmp_path / "full")
    with pytest.raises(PrepError, match="inside"):
        split_dataset(root, root / "images" / "split")
    for bad in ({"val": 0.0}, {"val": 0.7, "test": 0.4}, {"val": 1.0}):
        with pytest.raises(PrepError):
            split_dataset(root, tmp_path / "o", **bad)
    with pytest.raises(PrepError):
        split_dataset(root, tmp_path / "o", group_regex="(")
    assert (root / "images").is_dir() and len(list((root / "images").iterdir())) == 15  # input untouched


def test_assign_groups_never_leaves_a_part_empty():
    for seed in range(20):
        sizes = {"big": 100, "a": 1, "b": 1}
        out = assign_groups(sizes, {"train": 0.6, "val": 0.2, "test": 0.2}, seed)
        assert set(out.values()) == {"train", "val", "test"}


def test_symlink_mode(tmp_path):
    root = video_dataset(tmp_path / "ds", {f"v{i}": 2 for i in range(5)})
    split_dataset(root, tmp_path / "out", mode="symlink")
    first = next((tmp_path / "out" / "images" / "train").iterdir())
    assert first.is_symlink() and first.resolve().is_file()

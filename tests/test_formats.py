import json

import pytest

from labelprep.common import PrepError
from labelprep.formats import coco_to_yolo, yolo_to_coco

from helpers import make_image, make_yolo


def coco_dict(path):
    return json.loads(path.read_text())


def test_yolo_to_coco_boxes_and_polygons(tmp_path):
    root = make_yolo(
        tmp_path / "ds",
        {"a": "0 0.5 0.5 0.2 0.4\n1 0.1 0.1 0.5 0.1 0.3 0.5\n", "b": None, "c": ""},
        size=(200, 100),
    )
    rep = yolo_to_coco(root, tmp_path / "c.json")
    assert rep["images"] == 3 and rep["annotations"] == 2 and rep["images_without_label_file"] == 1
    d = coco_dict(tmp_path / "c.json")
    assert [c["id"] for c in d["categories"]] == [1, 2]
    assert [c["name"] for c in d["categories"]] == ["cat", "dog"]
    box = d["annotations"][0]
    assert box["category_id"] == 1 and box["bbox"] == [80.0, 30.0, 40.0, 40.0] and box["segmentation"] == []
    assert box["area"] == 1600.0
    poly = d["annotations"][1]
    assert poly["category_id"] == 2 and poly["segmentation"] == [[20.0, 10.0, 100.0, 10.0, 60.0, 50.0]]
    assert poly["area"] == pytest.approx(1600.0)  # triangle: base 80 x height 40 / 2
    assert poly["bbox"] == [20.0, 10.0, 80.0, 40.0]


def test_bad_lines_are_counted_not_fatal(tmp_path):
    root = make_yolo(tmp_path / "ds", {"a": "0 0.5 0.5\nx 1 2 3 4\n0 0.5 0.5 0.2 0.2\n"})
    rep = yolo_to_coco(root, tmp_path / "c.json")
    assert rep["bad_label_lines"] == 2 and rep["annotations"] == 1


def test_roundtrip_yolo_coco_yolo(tmp_path):
    lines = "0 0.500000 0.500000 0.200000 0.400000\n1 0.123457 0.654321 0.050000 0.100000\n"
    poly = "1 0.100000 0.100000 0.500000 0.100000 0.300000 0.500000\n"
    root = make_yolo(tmp_path / "ds", {"sub/a": lines, "b": poly}, size=(320, 240))
    yolo_to_coco(root, tmp_path / "c.json")
    rep = coco_to_yolo(tmp_path / "c.json", tmp_path / "back", images_dir=root / "images", segments=True)
    assert rep["images"] == 2 and rep["images_copied"] == 2 and rep["clipped"] == 0
    for stem in ("sub/a", "b"):
        orig = (root / "labels" / f"{stem}.txt").read_text().split()
        back = (tmp_path / "back" / "labels" / f"{stem}.txt").read_text().split()
        assert len(orig) == len(back)
        assert [int(x) for x in orig[::1][:1]] == [int(x) for x in back[:1]]
        for a, b in zip(orig, back):
            assert float(a) == pytest.approx(float(b), abs=1e-4)
    assert (tmp_path / "back" / "images" / "sub" / "a.jpg").is_file()
    assert "0: " in (tmp_path / "back" / "data.yaml").read_text()


def test_coco_category_ids_are_renumbered_contiguously(tmp_path):
    make_image(tmp_path / "imgs" / "x.jpg", (100, 100))
    data = {
        "images": [{"id": 7, "file_name": "x.jpg", "width": 100, "height": 100}],
        "annotations": [
            {"id": 1, "image_id": 7, "category_id": 90, "bbox": [10, 10, 20, 20]},
            {"id": 2, "image_id": 7, "category_id": 3, "bbox": [50, 50, 10, 10]},
        ],
        "categories": [{"id": 90, "name": "toaster"}, {"id": 3, "name": "car"}],
    }
    (tmp_path / "a.json").write_text(json.dumps(data))
    coco_to_yolo(tmp_path / "a.json", tmp_path / "out")
    lines = (tmp_path / "out" / "labels" / "x.txt").read_text().splitlines()
    assert [ln.split()[0] for ln in lines] == ["1", "0"]  # car=0 (id 3), toaster=1 (id 90)
    yaml = (tmp_path / "out" / "data.yaml").read_text()
    assert '0: "car"' in yaml and '1: "toaster"' in yaml


def test_clipping_dropping_and_counts(tmp_path):
    data = {
        "images": [{"id": 1, "file_name": "x.jpg", "width": 100, "height": 100}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [80, 80, 40, 40]},  # sticks out: clipped
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [150, 10, 20, 20]},  # fully outside: dropped
            {"id": 3, "image_id": 1, "category_id": 1, "bbox": [10, 10, 0, 5]},  # zero width: dropped
            {"id": 4, "image_id": 1, "category_id": 1, "bbox": [10, 10, 5, 5], "iscrowd": 1},
            {"id": 5, "image_id": 1, "category_id": 99, "bbox": [10, 10, 5, 5]},
            {"id": 6, "image_id": 1, "category_id": 1},
            {"id": 7, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
        ],
        "categories": [{"id": 1, "name": "a"}],
    }
    (tmp_path / "a.json").write_text(json.dumps(data))
    rep = coco_to_yolo(tmp_path / "a.json", tmp_path / "out")
    assert rep["annotations"] == 2 and rep["clipped"] == 2  # the clipped box and the outside box
    assert rep["dropped_degenerate"] == 2 and rep["dropped_crowd"] == 1
    assert rep["dropped_unknown_category"] == 1 and rep["dropped_no_geometry"] == 1
    first = (tmp_path / "out" / "labels" / "x.txt").read_text().splitlines()[0].split()
    assert [float(v) for v in first[1:]] == pytest.approx([0.9, 0.9, 0.2, 0.2])
    unclipped = coco_to_yolo(tmp_path / "a.json", tmp_path / "out2", clip=False)
    assert unclipped["clipped"] == 0 and unclipped["annotations"] == 3


def test_unsafe_file_names_are_refused(tmp_path):
    names = ["../evil.jpg", "/abs/evil.jpg", "C:\\evil.jpg", "a/../../evil.jpg", ""]
    data = {
        "images": [{"id": i, "file_name": n, "width": 10, "height": 10} for i, n in enumerate(names)]
        + [{"id": 99, "file_name": "ok.jpg", "width": 10, "height": 10}],
        "annotations": [{"id": 1, "image_id": 0, "category_id": 1, "bbox": [1, 1, 2, 2]}],
        "categories": [{"id": 1, "name": "a"}],
    }
    (tmp_path / "a.json").write_text(json.dumps(data))
    out = tmp_path / "sandbox" / "out"
    rep = coco_to_yolo(tmp_path / "a.json", out)
    assert rep["images_skipped_unsafe_name"] == 5 and rep["images"] == 1
    written = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()}
    assert written == {"a.json", "sandbox/out/labels/ok.txt", "sandbox/out/data.yaml"}


def test_label_name_collisions_are_an_error(tmp_path):
    data = {
        "images": [
            {"id": 1, "file_name": "x.jpg", "width": 10, "height": 10},
            {"id": 2, "file_name": "x.png", "width": 10, "height": 10},
        ],
        "annotations": [],
        "categories": [],
    }
    (tmp_path / "a.json").write_text(json.dumps(data))
    with pytest.raises(PrepError, match="same label file"):
        coco_to_yolo(tmp_path / "a.json", tmp_path / "out")


def test_missing_sizes_are_read_from_images_else_skipped(tmp_path):
    make_image(tmp_path / "imgs" / "a.jpg", (50, 40))
    data = {
        "images": [{"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "b.jpg"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [5, 4, 10, 8]}],
        "categories": [{"id": 1, "name": "a"}],
    }
    (tmp_path / "a.json").write_text(json.dumps(data))
    rep = coco_to_yolo(tmp_path / "a.json", tmp_path / "out", images_dir=tmp_path / "imgs")
    assert rep["images"] == 1 and rep["images_skipped_no_size"] == 1


def test_bad_inputs(tmp_path):
    with pytest.raises(PrepError):
        coco_to_yolo(tmp_path / "nope.json", tmp_path / "o")
    (tmp_path / "x.json").write_text("[1]")
    with pytest.raises(PrepError):
        coco_to_yolo(tmp_path / "x.json", tmp_path / "o")
    with pytest.raises(PrepError):
        yolo_to_coco(tmp_path / "missing", tmp_path / "c.json")


def test_roundtrip_keeps_class_ids_when_some_are_unused_and_unnamed(tmp_path):
    """Regression: used classes 0, 2 and 5 without a names file must come back as 0, 2 and 5."""
    root = make_yolo(
        tmp_path / "ds",
        {"a": "0 0.5 0.5 0.2 0.2\n5 0.2 0.2 0.1 0.1\n", "b": "2 0.5 0.5 0.2 0.2\n"},
        names=(),
    )
    (root / "data.yaml").unlink()
    yolo_to_coco(root, tmp_path / "c.json")
    assert [c["id"] for c in coco_dict(tmp_path / "c.json")["categories"]] == [1, 2, 3, 4, 5, 6]
    coco_to_yolo(tmp_path / "c.json", tmp_path / "back")
    classes = {
        stem: [ln.split()[0] for ln in (tmp_path / "back" / "labels" / f"{stem}.txt").read_text().splitlines()]
        for stem in ("a", "b")
    }
    assert classes == {"a": ["0", "5"], "b": ["2"]}

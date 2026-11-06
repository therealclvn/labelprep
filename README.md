# labelprep

Small command-line tool for the boring steps between "I have videos" and "I can train a detector":

1. **`frames`**: cut videos into images with stable names (`<video>_f000123.jpg`) and a manifest.
2. **`convert`**: convert between **YOLO** and **COCO** (boxes and polygons) without silently losing annotations. Everything that cannot be converted is counted and reported.
3. **`split`**: make train / val / test folders **by source video**, so frames from the same video never end up on both sides of the split.

Why the last one matters: consecutive frames of a video are almost identical. A random image-level split puts near-copies in both train and val, and the validation score looks much better than the model really is. `labelprep split` keeps each video in exactly one part and verifies it.

Works well together with [labelcheck](https://github.com/therealclvn/labelcheck), which finds mistakes in the labels themselves.

## Install

```bash
pip install ".[video]"   # from a clone of this repository; [video] adds OpenCV for the frames command
pip install .            # enough for convert and split (only needs Pillow)
```

Python 3.9+. Your input files are never modified.

## Typical workflow

```bash
# 1. videos -> frames (one frame per second, longest side at most 960 px)
labelprep frames my_videos/ -o dataset/images --seconds 1 --max-side 960

# 2. label the frames with any tool, export as YOLO into dataset/labels, then check them
labelcheck yolo dataset --html report.html

# 3. split by video
labelprep split dataset -o dataset_split --val 0.2 --test 0.1 --seed 42
```

`dataset_split/` then contains `images/{train,val,test}`, `labels/{train,val,test}`, a `data.yaml` that Ultralytics-style trainers understand, and `split_report.json`.

## Commands

### `labelprep frames VIDEO_OR_FOLDER... -o OUT`

| Option | Meaning |
|---|---|
| `--every N` / `--seconds S` | keep every N-th frame, or one frame every S seconds (choose one) |
| `--max-side PX` | shrink frames so the longer side is at most PX (never enlarges) |
| `--start S`, `--end S` | only use this part of the video |
| `--format jpg\|png`, `--quality Q` | output format (default jpg, quality 92) |
| `--overwrite` | rewrite frames that already exist; by default they are skipped, so an interrupted run can be repeated |

Frame names contain the frame number in the original video (`clip-a_f000250.jpg`), and `manifest.csv` lists image, video id, frame and time in seconds. Two videos that would get the same id are refused.

### `labelprep convert yolo2coco ROOT OUT.json` / `convert coco2yolo ANN.json OUT_DIR`

- YOLO class `i` becomes COCO category id `i + 1`. Going back, category ids are sorted and numbered `0..n-1`. A YOLO dataset therefore survives a round trip with identical class numbers, even when some classes are unused or have no names.
- `coco2yolo --images DIR` also copies (or `--link symlink`) the images next to the labels. Without it, only labels and `data.yaml` are written.
- `--segments` writes polygons (one line per polygon part) instead of boxes where the COCO annotation has them.
- Boxes that stick out of the image are clipped; boxes with no area left are dropped. Both are counted in the summary. `--no-clip` keeps the original numbers.
- COCO file names that try to leave the output folder (`../x.jpg`, absolute paths) are refused.

### `labelprep split ROOT -o OUT`

| Option | Meaning |
|---|---|
| `--val F`, `--test F` | fraction of images for validation / test (default 0.2 / 0) |
| `--seed N` | the same seed always gives the same split |
| `--group-by filename` (default) | the video id is everything before `_f<digits>` in the file name |
| `--group-regex RE` | your own pattern; the first group (or a group named `group`) is the video id |
| `--group-by folder` | one folder per video |
| `--link symlink` | link instead of copy |

Groups are assigned largest first to whichever part is furthest below its target size, so the image counts end up close to the requested fractions. The output folder must be new and empty. The command warns when a class is missing from a part, and when file names did not match the video pattern (then each image is its own group, which cannot prevent leakage).

## Checked on real data

Besides the unit tests (33), tested on the public [coco128](https://github.com/ultralytics/assets/releases) subset (128 images, 929 boxes, 71 of 80 classes used, no class names file):

- YOLO → COCO → YOLO: all 929 boxes come back with the same class; largest coordinate difference 1e-6.
- Splitting it gives 102 train / 26 val images, and the warnings about missing classes and non-video names are shown as intended.
- Real H.264 videos made with ffmpeg: frame counts, names and sizes are as expected, and `split` recognises the videos from the names `frames` writes.

The round-trip check found a real bug during development (class numbers shifted when classes were unused); it is covered by a regression test now.

## Limits

- COCO crowd annotations (`iscrowd=1`) and RLE masks are not converted; they are counted as dropped. Oriented boxes and keypoints are not supported.
- Images without a label file become empty label files after `coco2yolo`.
- `split` balances image counts, not classes. A rare class can still end up mostly in one part; the report shows the counts per part.
- `frames` relies on OpenCV. Whether it applies the rotation flag of phone videos depends on the OpenCV and video-codec build, and this was not tested here, so check one frame if your videos come from a phone.
- Tested on Linux with synthetic and public data; not yet on Windows or on large (100k+ image) datasets.

## Development

```bash
pip install -e ".[test]"
pytest
```

## License

MIT, see [LICENSE](LICENSE).

# Surf Pose Estimation with detectron2

Using the detectron2 pose estimation model from the [official tutorial page](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5) to capture my pose while surfing.

![img](img.JPG)

## Running locally (Apple Silicon)

The notebook was originally written for Colab (CUDA 10.1 + torch 1.6). It now
runs locally on CPU. Setup:

```bash
conda env create -f environment.yml
conda activate surf-detectron2

git clone https://github.com/facebookresearch/detectron2.git ~/Projects/detectron2
CC=clang CXX=clang++ ARCHFLAGS="-arch arm64" \
  pip install -e ~/Projects/detectron2 --no-build-isolation

python -m ipykernel install --user \
  --name surf-detectron2 --display-name "Surf Detectron2"
```

Then open the notebook and select the **Surf Detectron2** kernel.

Inference is pinned to CPU (`cfg.MODEL.DEVICE = "cpu"`) — keypoint R-CNN uses
ops that MPS does not implement. A single image takes a few seconds.

## Batch overlays

`batch_pose.py` renders a skeleton overlay for every `.MOV` in `SRC_DIR`, writing
`<stem>_pose.MOV` plus a `<stem>_keypoints.npz` of raw keypoints to
`SRC_DIR/pose_out`. It is resumable — a clip whose `.MOV` *and* `.npz` both exist
is skipped — so a crash part-way through a batch doesn't cost the finished clips.

```bash
conda activate surf-detectron2
caffeinate -i python batch_pose.py
```

### Keeping the clips time-syncable

Phone and GoPro clips carry an absolute `creation_time` in their container
metadata. Anything that later lines a video up against sensor data works from
that stamp — it's the only absolute clock a video file has.

`cv2.VideoWriter` writes **no** container metadata, so a rendered overlay plays
fine but carries no `creation_time`, and a consumer that needs one can't place it
on a timeline. Typically it just skips the clip without an error, which is a
confusing failure to chase down.

So `batch_pose.py` copies each source clip's metadata onto its overlay as the clip
finishes (`stamp_video_time.stamp`). This is an ffmpeg `-c copy` remux: the
encoded frames pass through byte-for-byte — verified by comparing the video
stream's MD5 before and after — so there's no re-encode and no quality loss, and
it costs about a second per clip.

To backfill clips rendered before this was wired up:

```bash
python stamp_video_time.py               # dry run — lists what it would stamp
python stamp_video_time.py --apply       # rewrite in place (atomic, lossless)
```

It pairs `<stem>_pose.MOV` in `dataLog00188/pose_out` with `<stem>.MOV` in the
parent folder; pass a different folder, `--originals` or `--suffix` for other
layouts. Re-running `batch_pose.py` also backfills any unstamped clip without
re-running inference.

Two things to preserve in any future render, or the timing silently drifts:

* **Duration must match the source.** The stamp records when recording
  *finished*, so a consumer derives the start as `creation_time - duration`. Trim
  or pad a clip and its computed start moves by the same amount. Writing every
  `FRAME_STEP`-th frame at `FPS/FRAME_STEP`, as `batch_pose.py` does, keeps
  duration and real-time playback intact.
* **Don't let ffmpeg mint a fresh `creation_time`.** Its muxer stamps the *encode*
  time by default, which would place the clip weeks away from the ride. Always
  copy the original's value across (`-map_metadata`).

Requires `ffmpeg`/`ffprobe` on `PATH` — included in `environment.yml`.

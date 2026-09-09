#!/usr/bin/env python
"""Batch keypoint estimation over every .MOV in a directory.

Inference runs on a downscaled frame (INFER_SCALE) but detectron2 returns
keypoints in *source* coordinates, so overlays are drawn full size.
Every FRAME_STEP-th frame is processed and the output is written at
FPS/FRAME_STEP, so the result plays at real-time speed.

Resumable: a clip whose .MOV *and* .npz already exist is skipped, so a crash
part-way through the batch does not cost the clips already finished.

    caffeinate -i python batch_pose.py
"""
import glob
import json
import os
import time

import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor

from stamp_video_time import needs_stamp, stamp

SRC_DIR = "dataLog00188"
OUT_DIR = os.path.join(SRC_DIR, "pose_out")

INFER_SCALE = 0.75   # ~1.7px median keypoint error vs native
FRAME_STEP = 2       # 60fps source -> process every 2nd frame, write at 30fps
SCORE_THRESH = 0.8
KP_THRESH = 0.05
BOX_SCALE = 1.2
SELECT_MODE = "centre"   # detection nearest frame centre; the camera pans to it
MAX_AREA_FRAC = 0.10     # ignore a hand/body held up to the lens

_predictors = {}


def get_predictor(width, height):
    """One predictor per resolution -- all these clips are 1080p, so this
    builds the model once for the whole batch."""
    key = (width, height)
    if key not in _predictors:
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        )
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
        )
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH
        cfg.MODEL.DEVICE = "cpu"
        cfg.INPUT.MIN_SIZE_TEST = int(min(width, height) * INFER_SCALE)
        cfg.INPUT.MAX_SIZE_TEST = int(max(width, height) * INFER_SCALE)
        _predictors[key] = (cfg, DefaultPredictor(cfg))
    return _predictors[key]


def skeleton(cfg):
    meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    idx = {n: i for i, n in enumerate(meta.keypoint_names)}
    links, colors = [], []
    for a, b, rgb in meta.keypoint_connection_rules:
        links.append((idx[a], idx[b]))
        colors.append(tuple(int(c) for c in rgb))
    return links, colors


def pick_subject(instances, width, height):
    if len(instances) == 0:
        return None
    areas = instances.pred_boxes.area().numpy()
    ok = np.where(areas < width * height * MAX_AREA_FRAC)[0]
    if len(ok) == 0:
        return None
    if SELECT_MODE == "largest":
        return instances[int(ok[np.argmax(areas[ok])])]
    b = instances.pred_boxes.tensor.numpy()[ok]
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    return instances[int(ok[np.argmin(np.hypot(cx - width / 2, cy - height / 2))])]


def draw(frame, inst, links, colors):
    x1, y1, x2, y2 = inst.pred_boxes.tensor[0].tolist()
    h, w = frame.shape[:2]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1) * BOX_SCALE, (y2 - y1) * BOX_SCALE
    p1 = (int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2)))
    p2 = (int(min(w, cx + bw / 2)), int(min(h, cy + bh / 2)))

    # thickness follows the subject, not the frame
    t = max(2, int((y2 - y1) / 60))
    cv2.rectangle(frame, p1, p2, (80, 220, 80), max(1, t // 2))

    kp = inst.pred_keypoints[0].numpy()
    for (a, b), rgb in zip(links, colors):
        if kp[a, 2] < KP_THRESH or kp[b, 2] < KP_THRESH:
            continue
        cv2.line(frame, (int(kp[a, 0]), int(kp[a, 1])),
                 (int(kp[b, 0]), int(kp[b, 1])), rgb[::-1], t, cv2.LINE_AA)
    for x, y, v in kp:
        if v >= KP_THRESH:
            cv2.circle(frame, (int(x), int(y)), max(2, t), (255, 255, 255), -1, cv2.LINE_AA)
    return frame


def open_writer(path, fps, size):
    for codec in ("avc1", "mp4v"):
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
        if w.isOpened():
            return w, codec
        w.release()
    raise SystemExit(f"no usable codec for {path}")


def process(src, out_mov, out_npz):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"  !! could not open {src}", flush=True)
        return None
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    FPS = cap.get(cv2.CAP_PROP_FPS)

    cfg, predictor = get_predictor(W, H)
    links, colors = skeleton(cfg)
    writer, codec = open_writer(out_mov, FPS / FRAME_STEP, (W, H))

    n_out = (N + FRAME_STEP - 1) // FRAME_STEP
    kps = np.full((n_out, 17, 3), np.nan, dtype=np.float32)
    boxes = np.full((n_out, 4), np.nan, dtype=np.float32)
    src_idx = np.full(n_out, -1, dtype=np.int32)

    i = j = hits = 0
    t0 = time.time()
    while True:
        if not cap.grab():
            break
        if i % FRAME_STEP == 0 and j < n_out:
            ok, frame = cap.retrieve()
            if not ok:
                break
            inst = predictor(frame)["instances"].to("cpu")
            subj = pick_subject(inst, W, H)
            if subj is not None:
                kps[j] = subj.pred_keypoints[0].numpy()
                boxes[j] = subj.pred_boxes.tensor[0].numpy()
                frame = draw(frame, subj, links, colors)
                hits += 1
            src_idx[j] = i
            writer.write(frame)
            j += 1
            if j % 100 == 0:
                el = time.time() - t0
                print(f"    {j}/{n_out}  {el/j:.2f}s/frame  eta {el/j*(n_out-j)/60:.1f}m",
                      flush=True)
        i += 1

    cap.release()
    writer.release()
    np.savez_compressed(out_npz, keypoints=kps[:j], boxes=boxes[:j],
                        source_frame=src_idx[:j], fps=FPS / FRAME_STEP,
                        source_fps=FPS, size=(W, H), frame_step=FRAME_STEP)
    el = time.time() - t0
    print(f"  {j} frames, subject in {hits} ({100*hits/max(j,1):.0f}%), "
          f"{el/60:.1f} min ({el/max(j,1):.2f}s/frame), codec {codec}", flush=True)
    return {"src": src, "frames_written": j, "hits": hits, "seconds": el}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    vids = sorted(glob.glob(os.path.join(SRC_DIR, "*.MOV")))
    print(f"{len(vids)} MOV files in {SRC_DIR}", flush=True)
    print(f"scale={INFER_SCALE} step={FRAME_STEP} select={SELECT_MODE} -> {OUT_DIR}\n",
          flush=True)

    summary, t_all = [], time.time()
    for k, src in enumerate(vids, 1):
        stem = os.path.splitext(os.path.basename(src))[0]
        out_mov = os.path.join(OUT_DIR, f"{stem}_pose.MOV")
        out_npz = os.path.join(OUT_DIR, f"{stem}_keypoints.npz")
        print(f"[{k}/{len(vids)}] {os.path.basename(src)}", flush=True)
        if os.path.exists(out_mov) and os.path.exists(out_npz):
            # Already rendered -- but clips written before creation_time was carried
            # across still need stamping, and that costs a remux, not an inference
            # pass. Re-running the batch therefore backfills them in seconds.
            if needs_stamp(out_mov):
                print("  already done; backfilling creation_time", flush=True)
                stamp(src, out_mov)
            else:
                print("  already done, skipping", flush=True)
            continue
        r = process(src, out_mov, out_npz)
        if r:
            # cv2.VideoWriter emits no container metadata, so carry the source
            # clip's creation_time across -- without it nothing downstream can
            # place this overlay on a timeline. See stamp_video_time.py.
            stamp(src, out_mov)
            summary.append(r)
            json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=1)
        done = time.time() - t_all
        print(f"  elapsed {done/3600:.2f}h\n", flush=True)

    print(f"BATCH DONE: {len(summary)} clips in {(time.time()-t_all)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()

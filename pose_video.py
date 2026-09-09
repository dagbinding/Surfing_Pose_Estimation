#!/usr/bin/env python
"""Run detectron2 keypoint estimation over a video and write an annotated copy.

Inference runs on a downscaled frame (INFER_SCALE) but detectron2 returns
keypoints in *source* coordinates, so the overlay is drawn on the full-size
frame -- low-res estimation, full-res picture.

    python pose_video.py
"""
import os
import time

import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor

SRC = "IMG_2268 2_cropped.MOV"

INFER_SCALE = 0.75   # run the model at 75% -- ~1.7px median keypoint error
SCORE_THRESH = 0.8   # detection confidence
KP_THRESH = 0.05     # per-keypoint visibility (detectron2's default)
BOX_SCALE = 1.2      # pad the drawn box so its edges clear the board

# "centre" = the detection nearest the middle of the frame (the camera pans to
# keep the rider there); "largest" = biggest box.
SELECT_MODE = "centre"

# The clip opens with a hand held up to the lens. It is both the largest
# "person" in frame AND the most central, so this guard is needed under either
# selection mode.
MAX_AREA_FRAC = 0.10  # ignore any detection covering >10% of the frame

DST = f"IMG_2268 2_cropped_pose_{SELECT_MODE}.MOV"
KPT_OUT = f"keypoints_{SELECT_MODE}.npz"  # so smoothing never needs a re-run


def build_predictor(width, height):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH
    cfg.MODEL.DEVICE = "cpu"  # keypoint R-CNN uses ops MPS lacks
    cfg.INPUT.MIN_SIZE_TEST = int(min(width, height) * INFER_SCALE)
    cfg.INPUT.MAX_SIZE_TEST = int(max(width, height) * INFER_SCALE)
    return cfg, DefaultPredictor(cfg)


def pick_subject(instances, width, height):
    """Choose the rider, ignoring anything big enough to be the camera hand."""
    if len(instances) == 0:
        return None
    areas = instances.pred_boxes.area().numpy()
    ok = np.where(areas < width * height * MAX_AREA_FRAC)[0]
    if len(ok) == 0:
        return None

    if SELECT_MODE == "largest":
        return instances[int(ok[np.argmax(areas[ok])])]

    boxes = instances.pred_boxes.tensor.numpy()[ok]
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    dist = np.hypot(cx - width / 2, cy - height / 2)
    return instances[int(ok[np.argmin(dist)])]


def draw(frame, inst, links, colors):
    """Draw one padded box plus the skeleton, sized to the subject."""
    x1, y1, x2, y2 = inst.pred_boxes.tensor[0].tolist()
    h, w = frame.shape[:2]

    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1) * BOX_SCALE, (y2 - y1) * BOX_SCALE
    bx1 = int(max(0, cx - bw / 2))
    by1 = int(max(0, cy - bh / 2))
    bx2 = int(min(w, cx + bw / 2))
    by2 = int(min(h, cy + bh / 2))

    # thickness follows the subject, not the frame -- a fixed width turns a
    # small rider into a solid blob of overlapping bars
    t = max(2, int((y2 - y1) / 60))
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (80, 220, 80), max(1, t // 2))

    kp = inst.pred_keypoints[0].numpy()
    for (a, b), rgb in zip(links, colors):
        if kp[a, 2] < KP_THRESH or kp[b, 2] < KP_THRESH:
            continue
        pa = (int(kp[a, 0]), int(kp[a, 1]))
        pb = (int(kp[b, 0]), int(kp[b, 1]))
        cv2.line(frame, pa, pb, rgb[::-1], t, cv2.LINE_AA)  # rgb -> bgr
    for x, y, v in kp:
        if v >= KP_THRESH:
            cv2.circle(frame, (int(x), int(y)), max(2, t), (255, 255, 255), -1, cv2.LINE_AA)
    return frame


def main():
    cap = cv2.VideoCapture(SRC)
    if not cap.isOpened():
        raise SystemExit(f"could not open {SRC}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    FPS = cap.get(cv2.CAP_PROP_FPS)

    cfg, predictor = build_predictor(W, H)
    print(f"{SRC}: {W}x{H} {N}f @{FPS:.2f}", flush=True)
    print(
        f"inference at {cfg.INPUT.MAX_SIZE_TEST}x{cfg.INPUT.MIN_SIZE_TEST} "
        f"({INFER_SCALE:.0%}), overlay at {W}x{H}, select={SELECT_MODE}",
        flush=True,
    )

    meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    name_to_i = {n: i for i, n in enumerate(meta.keypoint_names)}
    links, colors = [], []
    for a, b, rgb in meta.keypoint_connection_rules:
        links.append((name_to_i[a], name_to_i[b]))
        colors.append(tuple(int(c) for c in rgb))

    writer = None
    for codec in ("avc1", "mp4v"):
        w = cv2.VideoWriter(DST, cv2.VideoWriter_fourcc(*codec), FPS, (W, H))
        if w.isOpened():
            writer, used = w, codec
            break
        w.release()
    if writer is None:
        raise SystemExit("no usable codec")
    print(f"writing {DST} ({used})", flush=True)

    kps = np.full((N, 17, 3), np.nan, dtype=np.float32)
    boxes = np.full((N, 4), np.nan, dtype=np.float32)

    hits = 0
    t0 = time.time()
    for n in range(N):
        ok, frame = cap.read()
        if not ok:
            break
        inst = predictor(frame)["instances"].to("cpu")
        subject = pick_subject(inst, W, H)
        if subject is not None:
            kps[n] = subject.pred_keypoints[0].numpy()
            boxes[n] = subject.pred_boxes.tensor[0].numpy()
            frame = draw(frame, subject, links, colors)
            hits += 1
        writer.write(frame)

        if n % 50 == 0 and n:
            el = time.time() - t0
            eta = el / n * (N - n)
            print(
                f"  {n:5d}/{N}  {el/n:.2f}s/frame  elapsed {el/60:5.1f}m  eta {eta/60:5.1f}m",
                flush=True,
            )

    cap.release()
    writer.release()
    np.savez_compressed(KPT_OUT, keypoints=kps, boxes=boxes, fps=FPS, size=(W, H))

    el = time.time() - t0
    print(
        f"done: {n + 1} frames in {el/60:.1f} min ({el/(n+1):.2f}s/frame); "
        f"subject found in {hits} ({100*hits/(n+1):.0f}%)",
        flush=True,
    )
    print(f"-> {os.path.abspath(DST)}", flush=True)
    print(f"-> {os.path.abspath(KPT_OUT)}", flush=True)


if __name__ == "__main__":
    main()

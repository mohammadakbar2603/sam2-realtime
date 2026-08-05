#!/usr/bin/env python
"""Track an object through a video (any length / resolution) with flat memory.

Frames are read one at a time and the annotated result is written as it goes, so a
multi-gigabyte clip never needs to fit in memory.

    python examples/track_video.py --cfg <cfg> --ckpt <ckpt> \
        --video input.mp4 --box 300 100 500 400 --out tracked.mp4
    python examples/track_video.py --cfg <cfg> --ckpt <ckpt> \
        --video input.mp4 --point 640 360 --out tracked.mp4
"""
import argparse

import cv2
import numpy as np
from sam2.build_sam import build_sam2_video_predictor

from sam2_stream import LiveSAM2, overlay_masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--video", required=True, help="path to a video file")
    ap.add_argument("--out", default="tracked.mp4")
    ap.add_argument("--point", type=float, nargs=2, metavar=("X", "Y"))
    ap.add_argument("--box", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.point is None and args.box is None:
        ap.error("give --point X Y or --box X1 Y1 X2 Y2 to mark the object on frame 0")
    prompts = ([{"obj_id": 1, "kind": "point", "xy": tuple(args.point)}] if args.point
               else [{"obj_id": 1, "kind": "box", "box": tuple(args.box)}])

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, bgr = cap.read()
    if not ok:
        raise SystemExit(f"could not read {args.video}")
    first = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def next_frame():
        ok, bgr = cap.read()
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None

    predictor = build_sam2_video_predictor(args.cfg, args.ckpt, device=args.device)
    live = LiveSAM2(predictor)

    n = 0
    for frame_idx, frame_rgb, masks in live.track(first, prompts, next_frame):
        writer.write(overlay_masks(frame_rgb, masks))
        n += 1
        if n % 100 == 0:
            print(f"  {n} frames")
    writer.release(); cap.release()
    print(f"done: {n} frames -> {args.out}")


if __name__ == "__main__":
    main()

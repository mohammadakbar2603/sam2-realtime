#!/usr/bin/env python
"""Track objects from a live webcam. Click objects in the window, then it tracks them.

    python examples/track_camera.py --cfg <cfg> --ckpt <ckpt> --cam 0

Controls: left-click adds an object (a positive point). Press 't' to start tracking,
'r' to reset the clicks, Esc to quit.
"""
import argparse

import cv2
import numpy as np
from sam2.build_sam import build_sam2_video_predictor

from sam2_realtime import LiveSAM2, color_for_id, overlay_masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.cam}")

    ok, bgr = cap.read()
    if not ok:
        raise SystemExit("camera read failed")
    first = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    prompts = []
    win = "sam2-realtime (click objects, 't' track, 'r' reset, Esc quit)"
    cv2.namedWindow(win)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            prompts.append({"obj_id": len(prompts) + 1, "kind": "point", "xy": (float(x), float(y))})
    cv2.setMouseCallback(win, on_mouse)

    # --- selection phase: show the frozen first frame and collect clicks ---
    while True:
        disp = cv2.cvtColor(first, cv2.COLOR_RGB2BGR).copy()
        for p in prompts:
            x, y = int(p["xy"][0]), int(p["xy"][1])
            cv2.drawMarker(disp, (x, y), color_for_id(p["obj_id"]), cv2.MARKER_STAR, 20, 2)
        cv2.imshow(win, disp)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("t") and prompts:
            break
        if k == ord("r"):
            prompts.clear()
        if k == 27:
            cap.release(); cv2.destroyAllWindows(); return

    # --- tracking phase ---
    predictor = build_sam2_video_predictor(args.cfg, args.ckpt, device=args.device)
    live = LiveSAM2(predictor)

    def next_frame():
        ok, bgr = cap.read()
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None

    for frame_idx, frame_rgb, masks in live.track(first, prompts, next_frame):
        cv2.imshow(win, overlay_masks(frame_rgb, masks))
        if (cv2.waitKey(1) & 0xFF) == 27:
            break
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

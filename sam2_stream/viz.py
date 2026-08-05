# Copyright 2026 sam2-stream contributors.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Small visualization helpers: stable per-id colors and mask overlay."""

import colorsys

import cv2
import numpy as np

__all__ = ["color_for_id", "overlay_masks"]


def color_for_id(obj_id: int):
    """Deterministic bright BGR color for an object id (stable across frames)."""
    r, g, b = colorsys.hsv_to_rgb((int(obj_id) * 0.61803398875) % 1.0, 0.85, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))  # BGR for cv2


def overlay_masks(frame_rgb, masks, alpha: float = 0.45, draw_box: bool = True, draw_id: bool = True):
    """Return a BGR image with each object's mask blended in its own color.

    Args:
        frame_rgb: HxWx3 RGB uint8 frame.
        masks: ``{obj_id: HxW bool ndarray}`` (as yielded by ``LiveSAM2.track``).
    """
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR).copy()
    for oid, m in masks.items():
        if m is None or not m.any():
            continue
        col = color_for_id(oid)
        bgr[m] = (alpha * np.array(col) + (1 - alpha) * bgr[m]).astype("uint8")
        ys, xs = np.where(m)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        if draw_box:
            cv2.rectangle(bgr, (x1, y1), (x2, y2), col, 2)
        if draw_id:
            cv2.putText(bgr, f"id{oid}", (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return bgr

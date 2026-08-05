# Copyright 2026 sam2-stream contributors.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""
Bounded-memory streaming wrapper around SAM 2's video predictor.

SAM 2's official video predictor preloads the *entire* video into memory, which
does not work for a live camera (never ends) or a long/high-res clip (tens of GB).
``LiveSAM2`` feeds frames in one at a time through a small **ring buffer** and evicts
old tracker memory each step, so GPU **and** RAM stay flat regardless of length.

Typical use::

    from sam2.build_sam import build_sam2_video_predictor
    from sam2_stream import LiveSAM2

    predictor = build_sam2_video_predictor(cfg, ckpt, device="cuda")
    live = LiveSAM2(predictor)

    prompts = [{"obj_id": 1, "kind": "point", "xy": (x, y)},
               {"obj_id": 2, "kind": "box",   "box": (x1, y1, x2, y2)}]

    for frame_idx, frame_rgb, masks in live.track(first_frame_rgb, prompts, frame_source):
        # masks: {obj_id: HxW bool ndarray}
        ...
"""

import os
import tempfile

import cv2
import numpy as np
import torch
from PIL import Image

__all__ = ["LiveSAM2", "Ring"]


class Ring:
    """A fixed-width pixel buffer that *looks* arbitrarily long.

    The tracker only ever reads the current frame, so we physically store only
    ``width`` frames and map any frame index to ``index % width``. The frame index
    itself keeps increasing (SAM 2's temporal memory needs the true distance), only
    the storage slot wraps.
    """

    def __init__(self, buf: torch.Tensor, width: int, logical_len: int):
        self.buf, self.width, self.logical_len = buf, width, logical_len

    def _slot(self, k):
        if isinstance(k, int):
            return k % self.width
        if torch.is_tensor(k):
            return k % self.width
        if isinstance(k, (list, tuple)):
            return [x % self.width for x in k]
        return k

    def __getitem__(self, k):
        return self.buf[self._slot(k)]

    def __setitem__(self, k, v):
        self.buf[self._slot(k)] = v

    def __len__(self):
        return self.logical_len


class LiveSAM2:
    """Stream any SAM 2 video predictor over a live/long frame source with flat memory.

    Args:
        predictor: a built ``SAM2VideoPredictor`` (from ``build_sam2_video_predictor``).
        ring_width: number of frames physically kept on the GPU (bounds memory).
        mem_keep: how many recent tracker-memory frames to retain (SAM 2 attends to ~7).
        virtual_len: the pretend video length reported to the predictor.
    """

    # SAM 2 uses ImageNet normalization at ``predictor.image_size`` (1024).
    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    def __init__(self, predictor, ring_width: int = 32, mem_keep: int = 16,
                 virtual_len: int = 500_000):
        self.p = predictor
        self.width = ring_width
        self.mem_keep = mem_keep
        self.virtual_len = virtual_len
        self.size = predictor.image_size
        self._mean = torch.tensor(self._MEAN)[:, None, None]
        self._std = torch.tensor(self._STD)[:, None, None]

    # -- internals ---------------------------------------------------------------
    def _prep(self, rgb: np.ndarray) -> torch.Tensor:
        """RGB uint8 HxWx3 -> normalized (3, S, S) float tensor on CPU."""
        a = cv2.resize(rgb, (self.size, self.size)).astype(np.float32) / 255.0
        return (torch.from_numpy(a).permute(2, 0, 1) - self._mean) / self._std

    def _start_state(self, first_rgb: np.ndarray):
        # SAM 2's init_state requires a path; give it a 1-frame folder, then swap in the ring.
        tmp = tempfile.mkdtemp(prefix="sam2stream_")
        Image.fromarray(first_rgb).save(os.path.join(tmp, "00000.jpg"))
        state = self.p.init_state(video_path=tmp)
        buf = torch.stack([self._prep(first_rgb)] * self.width).to(self.p.device)
        state["images"] = Ring(buf, self.width, self.virtual_len)
        state["num_frames"] = self.virtual_len
        return state

    def _seed(self, state, prompts):
        for pr in prompts:
            if pr["kind"] == "point":
                lbl = 1 if pr.get("positive", True) else 0
                self.p.add_new_points_or_box(
                    inference_state=state, frame_idx=0, obj_id=pr["obj_id"],
                    points=np.array([pr["xy"]], np.float32),
                    labels=np.array([lbl], np.int32))
            elif pr["kind"] == "box":
                self.p.add_new_points_or_box(
                    inference_state=state, frame_idx=0, obj_id=pr["obj_id"],
                    box=np.array(pr["box"], np.float32))
            else:
                raise ValueError(f"unknown prompt kind: {pr['kind']!r}")

    def _evict(self, state, cur):
        kf = cur - self.mem_keep
        dicts = [state.get("output_dict", {})] + list(state.get("output_dict_per_obj", {}).values())
        for d in dicts:
            if isinstance(d, dict):
                nc = d.get("non_cond_frame_outputs", {})
                for k in [k for k in nc if isinstance(k, int) and k < kf]:
                    nc.pop(k, None)

    # -- public API --------------------------------------------------------------
    @torch.inference_mode()
    def track(self, first_rgb: np.ndarray, prompts, frame_source):
        """Track the seeded objects across a stream, yielding one result per frame.

        Args:
            first_rgb: the first frame (RGB uint8 HxWx3) that the prompts refer to.
            prompts: list of dicts, each ``{"obj_id": int, "kind": "point"|"box", ...}``
                     with ``"xy": (x, y)`` (+ optional ``"positive"``) or ``"box": (x1,y1,x2,y2)``.
            frame_source: a zero-arg callable returning the next frame (RGB uint8) or
                          ``None`` to stop.

        Yields:
            ``(frame_idx, frame_rgb, masks)`` where ``masks`` is ``{obj_id: HxW bool ndarray}``.
        """
        with torch.autocast(self.p.device if isinstance(self.p.device, str) else self.p.device.type,
                            dtype=torch.bfloat16):
            state = self._start_state(first_rgb)
            self._seed(state, prompts)
            raw = {0: first_rgb}
            gen = self.p.propagate_in_video(state, start_frame_idx=0,
                                            max_frame_num_to_track=self.virtual_len - 1)
            nxt = 1
            for fidx, obj_ids, video_masks in gen:
                self._evict(state, fidx)
                frame_rgb = raw.pop(fidx, first_rgb)
                h, w = frame_rgb.shape[:2]
                masks = {}
                for i, oid in enumerate(obj_ids):
                    m = (video_masks[i] > 0.0).squeeze().cpu().numpy()
                    if m.shape != (h, w):
                        m = cv2.resize(m.astype(np.uint8), (w, h),
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
                    masks[int(oid)] = m
                yield fidx, frame_rgb, masks

                nxt_frame = frame_source()
                if nxt_frame is None:
                    break
                state["images"][nxt] = self._prep(nxt_frame).to(self.p.device)
                raw[nxt] = nxt_frame
                nxt += 1
                for old in [k for k in raw if k < fidx - 2]:  # keep only a couple recent for display
                    raw.pop(old, None)

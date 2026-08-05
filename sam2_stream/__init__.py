# Copyright 2026 sam2-stream contributors.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""sam2-stream: bounded-memory real-time streaming for SAM 2 (live camera + long videos)."""

from .live import LiveSAM2, Ring
from .viz import color_for_id, overlay_masks

__version__ = "0.1.0"
__all__ = ["LiveSAM2", "Ring", "color_for_id", "overlay_masks"]

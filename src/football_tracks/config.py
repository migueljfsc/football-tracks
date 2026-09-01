"""Constants and paths shared by every stage.

Pitch dimensions match Pitchboard's `BoardDoc`: origin at the top-left corner, x along
the length, y across the width, metres throughout. They are here so no stage invents
its own answer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work"
CLIPS = ROOT / "data" / "clips"


def work_dir(clip: Path, *, create: bool = True) -> Path:
    """Where a clip's artefacts live. One directory per clip, named after it.

    Everything under work/ is reproducible from the clip plus a stage, which is why it
    is not in git and why `make clean` can delete all of it.
    """
    d = WORK / clip.stem
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


# Pitch green in OpenCV's HSV, whose hue channel is 0..179 so grass sits near 60. The
# saturation and value floors are what separate turf from grey stands and floodlit
# white: grey has no meaningful hue, so a hue test alone calls it green.
GREEN_LO = np.array([35, 40, 40], dtype=np.uint8)
GREEN_HI = np.array([85, 255, 255], dtype=np.uint8)

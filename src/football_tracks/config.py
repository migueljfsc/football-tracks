"""Constants and paths shared by every stage.

Pitch dimensions match Pitchboard's `BoardDoc`: origin at the top-left corner, x along
the length, y across the width, metres throughout. They are here so no stage invents
its own answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


@dataclass(frozen=True, slots=True)
class Pitch:
    """How big this particular pitch is, in metres.

    The Laws fix the markings and leave the PITCH variable: a goal is 7.32 m wide and a
    penalty spot 11 m out on every ground in the world, while the field itself may be
    anything from 90 x 45 to 120 x 90. Only elite competition pins it -- UEFA requires
    105 x 68 for the Champions League -- so the default is right for a European night
    and wrong for most football (D89).

    It has to be carried rather than assumed because the seed's landmarks are written in
    it. Click a goal post as `(0, W/2 - 3.66)` with the wrong W and every player lands
    the same metre or two off across the pitch: self-consistent, good residuals, and a
    touchline that is not where the touchline is.
    """

    length: float = PITCH_LENGTH
    width: float = PITCH_WIDTH

    @property
    def middle(self) -> float:
        """Halfway across, which is where both goals are centred."""
        return self.width / 2

    @property
    def halfway(self) -> float:
        return self.length / 2


DEFAULT_PITCH = Pitch()

PITCH_FILE = "pitch.json"


def read_pitch(work: Path) -> Pitch:
    """How big this clip's pitch is, or the default where nobody has said.

    Beside the seed rather than in `clip.json`, and for the same reason the seed is:
    it is something a person knows and the video does not, so re-extracting the frames
    must not silently throw it away.
    """
    path = work / PITCH_FILE
    if not path.exists():
        return DEFAULT_PITCH
    d = json.loads(path.read_text())
    return Pitch(length=float(d["length"]), width=float(d["width"]))


def write_pitch(work: Path, pitch: Pitch) -> Path:
    path = work / PITCH_FILE
    path.write_text(json.dumps({"length": pitch.length, "width": pitch.width}) + "\n")
    return path


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work"
CLIPS = ROOT / "data" / "clips"
# SN-Calibration-2023: single frames with line annotations, from 345 named matches.
# Training data only -- it has no video and no tracks, so no stage but the segmenter
# has anything to read here.
CALIB_DATA = ROOT / "data" / "calib2023"


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

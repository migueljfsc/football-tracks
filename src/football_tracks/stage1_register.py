"""Stage 1 - fit a homography per frame and measure what it costs.

The measurement here deliberately isolates registration. It takes ground-truth bounding
boxes, pushes their bottom-middle through the fitted homography, and compares the
result with the position SoccerNet recorded for that same box. Detection and tracking
are held fixed, so whatever error comes out is the camera model's alone.

That makes it the CEILING for the whole pipeline: no amount of detector quality gets a
position closer than the homography puts it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from . import calibration
from .soccernet import SN_ORIGIN_X, SN_ORIGIN_Y
from .tracks import on_pitch

Homographies = dict[int, npt.NDArray[np.float64] | None]

GROUND_ROLES = ("player", "goalkeeper", "referee")


def _frame_index(file_name: str) -> int:
    return int(file_name.split(".")[0])


def fit_all(labels: dict[str, Any]) -> Homographies:
    """A homography per frame, from whatever pitch lines that frame shows."""
    frame_of = {img["image_id"]: _frame_index(img["file_name"]) for img in labels["images"]}
    size = {img["image_id"]: (img["width"], img["height"]) for img in labels["images"]}

    out: Homographies = {frame_of[i]: None for i in frame_of}
    for a in labels["annotations"]:
        if a.get("category_id") != 5:
            continue
        w, h = size[a["image_id"]]
        out[frame_of[a["image_id"]]] = calibration.homography(calibration.lines_of(a), w, h)
    return out


@dataclass(slots=True)
class Registration:
    frames: int
    solved: int
    boxes: int
    scored: int
    median_error_m: float
    p90_error_m: float
    p99_error_m: float
    off_pitch: int

    @property
    def coverage(self) -> float:
        return self.solved / self.frames if self.frames else 0.0


def evaluate(labels: dict[str, Any], homs: Homographies) -> Registration:
    """Push ground-truth boxes through the fitted homographies and measure the drift."""
    frame_of = {img["image_id"]: _frame_index(img["file_name"]) for img in labels["images"]}

    errors: list[float] = []
    boxes = 0
    off = 0
    for a in labels["annotations"]:
        attrs = a.get("attributes") or {}
        box = a.get("bbox_image")
        truth = a.get("bbox_pitch")
        if attrs.get("role") not in GROUND_ROLES or not box or not truth:
            continue

        boxes += 1
        h = homs.get(frame_of[a["image_id"]])
        if h is None:
            continue

        # The bottom middle of the box is where the player meets the grass, and the
        # only point on a box that a ground homography can say anything about.
        got = calibration.to_pitch(h, float(box["x_center"]), float(box["y"] + box["h"]))
        want = (truth["x_bottom_middle"] + SN_ORIGIN_X, truth["y_bottom_middle"] + SN_ORIGIN_Y)
        if not on_pitch(*want):
            continue  # SoccerNet's own outlier, not ours to be scored against (D13)
        if not on_pitch(*got):
            off += 1
            continue
        errors.append(float(np.hypot(got[0] - want[0], got[1] - want[1])))

    arr = np.array(errors) if errors else np.array([np.nan])
    return Registration(
        frames=len(homs),
        solved=sum(1 for h in homs.values() if h is not None),
        boxes=boxes,
        scored=len(errors),
        median_error_m=float(np.nanpercentile(arr, 50)),
        p90_error_m=float(np.nanpercentile(arr, 90)),
        p99_error_m=float(np.nanpercentile(arr, 99)),
        off_pitch=off,
    )


def report(r: Registration) -> str:
    return "\n".join(
        [
            f"frames solved     {r.solved}/{r.frames}  ({r.coverage:.1%})",
            f"boxes scored      {r.scored}/{r.boxes}",
            f"position error    {r.median_error_m:.2f} m median,"
            f" {r.p90_error_m:.2f} m p90, {r.p99_error_m:.2f} m p99",
            f"thrown off pitch  {r.off_pitch}",
        ]
    )

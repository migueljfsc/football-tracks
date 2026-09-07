"""Measuring a camera model where the PLAYERS are.

`observed_error` asks how far two camera models disagree over the pitch they can see.
This asks the question the pipeline actually has: where does this model put the people?
A model can improve at the first and lose at the second (D70), so the two are measured
apart and neither stands in for the other.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from football_tracks.config import PITCH_LENGTH, PITCH_WIDTH
from football_tracks.stage1_register import player_errors

# Pixels to metres at a tenth, which puts a 1920 x 1080 frame on a 192 x 108 m field --
# big enough that a box can be projected off the pitch on purpose.
SCALE = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def ann(px: float, py: float, mx: float, my: float, role: str = "player") -> dict[str, Any]:
    """A box whose bottom middle sits at (px, py), annotated as standing at (mx, my).

    The pitch position is in SoccerNet's centre-origin convention, as the file has it.
    """
    return {
        "image_id": "img1",
        "attributes": {"role": role},
        "bbox_image": {"x_center": px, "y": py - 10.0, "h": 10.0},
        "bbox_pitch": {
            "x_bottom_middle": mx - PITCH_LENGTH / 2,
            "y_bottom_middle": my - PITCH_WIDTH / 2,
        },
    }


def labels(*annotations: dict[str, Any]) -> dict[str, Any]:
    return {
        "info": {"frame_rate": 25},
        "images": [{"image_id": "img1", "file_name": "000001.jpg", "width": 1920, "height": 1080}],
        "annotations": list(annotations),
    }


def test_a_model_that_agrees_with_the_annotation_has_no_error() -> None:
    errors, boxes, off = player_errors(labels(ann(500.0, 300.0, 50.0, 30.0)), {1: SCALE})
    assert (boxes, off) == (1, 0)
    assert errors == [0.0]


def test_the_error_is_metres_at_the_foot_of_the_box() -> None:
    errors, _boxes, _off = player_errors(labels(ann(500.0, 300.0, 53.0, 30.0)), {1: SCALE})
    assert errors[0] == 3.0


def test_a_box_thrown_off_the_pitch_is_counted_and_not_averaged_in() -> None:
    """The failure the median hides: a model that puts a quarter of the players in the
    crowd scores beautifully on the ones it kept."""
    off_pitch = ann(1900.0, 1000.0, 50.0, 30.0)
    errors, boxes, off = player_errors(labels(ann(500.0, 300.0, 50.0, 30.0), off_pitch), {1: SCALE})
    assert (boxes, off) == (2, 1)
    assert errors == [0.0], "the thrown box must not improve the error it escaped"


def test_a_frame_with_no_homography_scores_nothing_but_still_counts_its_boxes() -> None:
    errors, boxes, off = player_errors(labels(ann(500.0, 300.0, 50.0, 30.0)), {1: None})
    assert (errors, boxes, off) == ([], 1, 0)


def test_only_people_standing_on_the_ground_are_scored() -> None:
    ball = labels(ann(500.0, 300.0, 50.0, 30.0, role="ball"))
    _errors, boxes, _off = player_errors(ball, {1: SCALE})
    assert boxes == 0, "a ball is not on the ground plane and cannot score a camera model"

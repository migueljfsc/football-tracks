"""Choosing which sighting is the ball."""

from __future__ import annotations

import numpy as np

from football_tracks import auto
from football_tracks.detect import Sighting


def _eye() -> np.ndarray:
    """A homography that makes pitch metres and pixels the same thing, for readability."""
    return np.eye(3, dtype=np.float64)


def test_a_mark_painted_on_the_grass_is_not_a_ball() -> None:
    """The penalty spot is a white circle on grass and the detector calls it a ball.

    On SNGS-116 it did so through an entire corner, which is why the board showed a ball
    sitting in the six-yard box while the corner was being taken off-camera.
    """
    per = {f: [Sighting(f=f, x=94.0, y=34.0, score=0.30)] for f in range(400)}
    static = auto._painted_spots(per, dict.fromkeys(per, _eye()))
    assert auto._bin_of(94.0, 34.0) in static


def test_a_ball_that_moves_is_never_called_paint() -> None:
    """A real ball is somewhere new every second; that is the whole discriminator."""
    per = {f: [Sighting(f=f, x=10.0 + f * 0.2, y=30.0, score=0.9)] for f in range(400)}
    static = auto._painted_spots(per, dict.fromkeys(per, _eye()))
    assert not any(auto._is_static(per[f][0], _eye(), static) for f in per)


def test_a_short_clip_accuses_nothing_of_being_paint() -> None:
    """Too few frames to tell a stationary ball from a stationary mark: say nothing."""
    per = {f: [Sighting(f=f, x=94.0, y=34.0, score=0.30)] for f in range(20)}
    assert auto._painted_spots(per, dict.fromkeys(per, _eye())) == set()


def test_a_weak_sighting_never_becomes_the_ball() -> None:
    """Measured against SoccerNet's own ball labels: at the detector's 0.15 floor the
    chosen ball is the real one in 25% of frames. A board is better with no ball."""
    per = {1: [Sighting(f=1, x=50.0, y=30.0, score=auto.BALL_ASSERT_CONF - 0.01)]}
    assert auto.ball_path([s for v in per.values() for s in v], {1: _eye()}, [1]) == []


def test_a_confident_sighting_is_kept() -> None:
    seen = [Sighting(f=f, x=50.0, y=30.0, score=0.9) for f in range(1, 12)]
    out = auto.ball_path(seen, dict.fromkeys(range(1, 12), _eye()), list(range(1, 12)))
    assert out, "a confident, moving ball must survive every filter"

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


def _weak_decoy(frames: range) -> list[Sighting]:
    """A moving, unconfident sighting far from any restart spot.

    Its only job is to give `_painted_spots` enough frames to have an opinion, the way a
    real clip does. It is never emitted itself.
    """
    return [Sighting(f=f, x=10.0 + f * 0.1, y=30.0, score=0.20) for f in frames]


def test_a_ball_placed_on_the_corner_is_believed_below_the_confidence_floor() -> None:
    """The set-piece prior, measured on SNGS-116's corner.

    A ball waiting to be struck is small, still and far away, so it scores about 0.2 and
    never clears BALL_ASSERT_CONF - and that clip asserted no ball at all across the
    whole 157-frame corner that opens it. Within 1.5 m of a restart spot the position is
    the evidence instead: 79 of the 80 frames this admits are the real ball.
    """
    span = range(1, 61)
    seen = _weak_decoy(range(1, 301)) + [Sighting(f=f, x=105.0, y=0.0, score=0.20) for f in span]
    out = auto.ball_path(seen, dict.fromkeys(range(1, 301), _eye()), list(range(1, 301)))
    placed = [s for s in out if abs(s.x - 105.0) < 1.0 and abs(s.y) < 1.0]
    assert placed, "a ball sitting on the corner arc must survive the confidence gate"


def test_a_tracked_ball_silences_the_restart_pass() -> None:
    """The veto, not the position, is what makes the prior safe.

    SNGS-121's corner region holds 32 sightings that are all false and 25 m from the real
    ball. What separates it from SNGS-116 is that there the ball IS being tracked, so the
    restart pass must not speak at all.
    """
    frames = range(1, 301)
    seen = [Sighting(f=f, x=10.0 + f * 0.1, y=30.0, score=0.90) for f in frames] + [
        Sighting(f=f, x=105.0, y=0.0, score=0.20) for f in range(1, 61)
    ]
    out = auto.ball_path(seen, dict.fromkeys(frames, _eye()), list(frames))
    assert out, "the tracked ball itself is still emitted"
    assert not [s for s in out if s.x > 100.0], "a tracked ball must veto the corner"


def test_a_moment_at_a_restart_spot_is_not_a_placed_ball() -> None:
    """A mark that reads as a ball for a frame or two is not a ball somebody put there."""
    frames = range(1, 51)
    seen = [Sighting(f=f, x=105.0, y=0.0, score=0.20) for f in (10, 11, 12)]
    assert auto.ball_path(seen, dict.fromkeys(frames, _eye()), list(frames)) == []


def test_the_penalty_spot_is_not_a_restart_spot() -> None:
    """It is where the detector's favourite false positive lives - a painted white disc.

    Opening the floor there would admit exactly what `_painted_spots` exists to remove,
    and a penalty is the one restart this footage never contains.
    """
    frames = range(1, 301)
    seen = _weak_decoy(frames) + [Sighting(f=f, x=94.0, y=34.0, score=0.20) for f in range(1, 61)]
    out = auto.ball_path(seen, dict.fromkeys(frames, _eye()), list(frames))
    assert not [s for s in out if s.x > 90.0], "the penalty spot must earn its confidence"


def test_a_ball_far_off_the_pitch_is_refused_however_confident() -> None:
    """The ball's margin is METRES; tracks.on_pitch's is a SHARE of the pitch.

    Handing that function metres opened the gate to 157 m and kept every airborne
    projection there is - SNGS-116 was emitting a ball at (137.1, -33.4).
    """
    frames = range(1, 13)
    seen = [Sighting(f=f, x=137.1, y=-33.4, score=0.90) for f in frames]
    assert auto.ball_path(seen, dict.fromkeys(frames, _eye()), list(frames)) == []

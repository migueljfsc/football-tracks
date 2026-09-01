"""Association and team clustering - the pure parts.

The tracker's real failure is the id switch, which needs a clip to see and is measured
by `ft score`. What is testable here is the gate and the tie-breaks it rests on.
"""

from __future__ import annotations

import numpy as np

from football_tracks.detect import Detection
from football_tracks.stage2_track import (
    MAX_SPEED,
    Observation,
    Track,
    color_distance,
    run,
)
from football_tracks.stage3_teams import assign, kmeans2

FPS = 25.0


def det(f: int) -> Detection:
    return Detection(f=f, x1=0, y1=0, x2=10, y2=20, score=0.9)


def obs(f: int, x: float, y: float) -> Observation:
    return Observation(f=f, x=x, y=y, det=det(f))


def track_over(points: dict[int, list[tuple[float, float]]]) -> list[Track]:
    frames = sorted(points)
    observations = {f: [obs(f, x, y) for x, y in pts] for f, pts in points.items()}
    return run(observations, frames, lambda _f: None, fps=FPS)


def test_a_walking_player_stays_one_track() -> None:
    pts = {f: [(10.0 + 0.1 * f, 20.0)] for f in range(1, 30)}
    tracks = track_over(pts)
    assert len(tracks) == 1
    assert len(tracks[0].observations) == 29


def test_two_players_apart_stay_two_tracks() -> None:
    pts = {f: [(10.0, 20.0), (60.0, 50.0)] for f in range(1, 30)}
    assert len(track_over(pts)) == 2


def test_a_jump_across_the_pitch_starts_a_new_track() -> None:
    # The gate is a physical claim: nobody covers 60 m between two frames. Made in
    # metres precisely so it means that, rather than meaning something about how fast
    # the camera happens to be panning.
    pts = {f: [(10.0, 20.0)] for f in range(1, 15)}
    pts.update({f: [(80.0, 20.0)] for f in range(15, 30)})
    tracks = track_over(pts)
    assert len(tracks) == 2


def test_the_gate_allows_a_genuine_sprint() -> None:
    step = MAX_SPEED / FPS * 0.9
    pts = {f: [(10.0 + step * f, 20.0)] for f in range(1, 30)}
    assert len(track_over(pts)) == 1


def test_a_brief_occlusion_does_not_break_a_track() -> None:
    pts = {f: [(10.0 + 0.05 * f, 20.0)] for f in range(1, 40) if not 12 <= f <= 18}
    tracks = track_over(pts)
    assert len(tracks) == 1


def test_a_long_absence_does_break_it() -> None:
    pts = {f: [(10.0, 20.0)] for f in range(1, 12)}
    pts.update({f: [(10.0, 20.0)] for f in range(60, 75)})
    assert len(track_over(pts)) == 2


def test_colour_distance_is_neutral_when_a_kit_is_unknown() -> None:
    # Neither attract nor repel: a turned back must not merge two players, and must not
    # split one either.
    a = np.array([1.0, 0.0])
    assert color_distance(a, None) == 0.5
    assert color_distance(None, None) == 0.5
    assert color_distance(a, a) == 0.0
    assert color_distance(a, np.array([0.0, 1.0])) == 1.0


def test_kmeans_splits_two_kits() -> None:
    pts = np.array([[1.0, 0.0]] * 5 + [[0.0, 1.0]] * 5)
    labels = kmeans2(pts)
    assert len(set(labels[:5])) == 1
    assert len(set(labels[5:])) == 1
    assert labels[0] != labels[5]


def test_kmeans_is_deterministic() -> None:
    # Seeded from the two most distant points, not at random, so a rerun of the whole
    # pipeline does not relabel the teams.
    pts = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    assert np.array_equal(kmeans2(pts), kmeans2(pts))


def test_home_is_the_side_nearer_x_zero_not_the_side_that_scores_best() -> None:
    # Deciding this by whichever labelling matches the ground truth would be fitting to
    # the yardstick. It is decided by which end the side plays at.
    left = np.array([1.0, 0.0])
    right = np.array([0.0, 1.0])
    tracks = [
        Track(id=1, observations=[obs(1, 20.0, 30.0)], color=left),
        Track(id=2, observations=[obs(1, 25.0, 30.0)], color=left),
        Track(id=3, observations=[obs(1, 80.0, 30.0)], color=right),
        Track(id=4, observations=[obs(1, 85.0, 30.0)], color=right),
    ]
    teams = assign(tracks, {1: 20.0, 2: 25.0, 3: 80.0, 4: 85.0})
    assert teams[1] == "home" and teams[2] == "home"
    assert teams[3] == "away" and teams[4] == "away"

    flipped = assign(tracks, {1: 80.0, 2: 85.0, 3: 20.0, 4: 25.0})
    assert flipped[1] == "away" and flipped[3] == "home"


def test_a_track_with_no_kit_signature_is_unknown_not_guessed() -> None:
    tracks = [
        Track(id=1, observations=[obs(1, 20.0, 30.0)], color=np.array([1.0, 0.0])),
        Track(id=2, observations=[obs(1, 80.0, 30.0)], color=np.array([0.0, 1.0])),
        Track(id=3, observations=[obs(1, 50.0, 30.0)], color=None),
    ]
    teams = assign(tracks, {1: 20.0, 2: 80.0, 3: 50.0})
    assert teams[3] == "unknown"

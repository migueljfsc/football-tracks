"""Association and team clustering - the pure parts.

Coordinates here are IMAGE PIXELS, because that is the space the tracker works in. The
gate is a speed in metres per second converted to pixels through the box height, so a
box 20px tall stands for a player 1.8m tall and everything scales from that.

The tracker's real failure is the id switch, which needs a clip to see and is measured
by `ft score`. What is testable here is the gate, the camera compensation, and the
tie-breaks they rest on.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from football_tracks.detect import Detection
from football_tracks.stage2_track import (
    MAX_SPEED,
    UNREACHABLE,
    Observation,
    Track,
    _cost,
    color_distance,
    run,
)
from football_tracks.stage3_teams import assign, kmeans2

FPS = 25.0


BOX_H = 20.0  # px, so px_per_metre = 20 / 1.8 = 11.1


def det(f: int, height: float = BOX_H) -> Detection:
    return Detection(f=f, x1=0, y1=0, x2=10, y2=height, score=0.9)


def obs(f: int, x: float, y: float) -> Observation:
    return Observation(f=f, x=x, y=y, det=det(f))


def track_over(
    points: dict[int, list[tuple[float, float]]],
    motions: dict[int, np.ndarray] | None = None,
) -> list[Track]:
    frames = sorted(points)
    observations = {f: [obs(f, x, y) for x, y in pts] for f, pts in points.items()}
    return run(observations, frames, lambda _f: None, fps=FPS, motions=motions)


def pan(dx: float, dy: float = 0.0) -> np.ndarray:
    """A camera translation, as the frame-to-frame transform the tracker is handed."""
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


PX_PER_M = BOX_H / 1.8
STEP_AT_MAX_SPEED = MAX_SPEED / FPS * PX_PER_M


def test_a_walking_player_stays_one_track() -> None:
    pts = {f: [(100.0 + 1.0 * f, 200.0)] for f in range(1, 30)}
    tracks = track_over(pts)
    assert len(tracks) == 1
    assert len(tracks[0].observations) == 29


def test_two_players_apart_stay_two_tracks() -> None:
    pts = {f: [(100.0, 200.0), (600.0, 500.0)] for f in range(1, 30)}
    assert len(track_over(pts)) == 2


def test_a_jump_across_the_pitch_starts_a_new_track() -> None:
    # The gate is a physical claim - nobody covers 60m between two frames - expressed
    # in pixels through the box height, so it means the same at any resolution or
    # distance from the camera.
    pts = {f: [(100.0, 200.0)] for f in range(1, 15)}
    pts.update({f: [(900.0, 200.0)] for f in range(15, 30)})
    assert len(track_over(pts)) == 2


def test_the_gate_allows_a_genuine_sprint() -> None:
    step = STEP_AT_MAX_SPEED * 0.9
    pts = {f: [(100.0 + step * f, 200.0)] for f in range(1, 30)}
    assert len(track_over(pts)) == 1


def test_a_camera_pan_does_not_break_a_standing_player() -> None:
    # THE point of stabilising. The pan moves the player across the frame far faster
    # than they could run; without compensation the gate rejects every match and the
    # track shatters into one fragment per frame.
    shift = STEP_AT_MAX_SPEED * 4
    pts = {f: [(100.0 + shift * f, 200.0)] for f in range(1, 30)}
    # Unstabilised it shatters so badly that every fragment falls under
    # MIN_TRACK_LENGTH and nothing survives at all.
    assert not any(len(t.observations) == 29 for t in track_over(pts))

    motions = {f: pan(shift) for f in range(2, 30)}
    tracks = track_over(pts, motions)
    assert len(tracks) == 1
    assert len(tracks[0].observations) == 29


def test_a_pan_is_not_recorded_as_the_player_running() -> None:
    # Velocity must be what is left AFTER the camera is accounted for. Storing the pan
    # as the player's speed makes the next prediction overshoot by the same amount
    # again, and the error compounds the moment the camera stops.
    shift = STEP_AT_MAX_SPEED * 3
    pts = {f: [(100.0 + shift * f, 200.0)] for f in range(1, 10)}
    motions = {f: pan(shift) for f in range(2, 10)}
    track = track_over(pts, motions)[0]
    assert abs(track.velocity[0]) < 1e-6
    assert abs(track.velocity[1]) < 1e-6


def test_a_brief_occlusion_does_not_break_a_track() -> None:
    pts = {f: [(100.0 + 0.5 * f, 200.0)] for f in range(1, 40) if not 12 <= f <= 18}
    tracks = track_over(pts)
    assert len(tracks) == 1


def test_a_long_absence_does_break_it() -> None:
    pts = {f: [(100.0, 200.0)] for f in range(1, 12)}
    pts.update({f: [(100.0, 200.0)] for f in range(60, 75)})
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


def test_optimal_assignment_beats_taking_the_cheapest_pair_first() -> None:
    """Greedy hands out the globally cheapest pair first, even when that starves another
    track of the only detection it could have had. Two tracks and two detections, laid
    out so the cheap pairing forces the second track onto nothing."""
    a = Track(id=1, observations=[obs(1, 100.0, 200.0)])
    b = Track(id=2, observations=[obs(1, 140.0, 200.0)])
    candidates = [obs(2, 138.0, 200.0), obs(2, 104.0, 200.0)]

    costs = np.array(
        [
            [
                c if (c := _cost(t, o, None, FPS, None)) is not None else UNREACHABLE
                for o in candidates
            ]
            for t in (a, b)
        ]
    )
    rows, cols = linear_sum_assignment(costs)
    # Track 1 must take the detection near 100 and track 2 the one near 140, even though
    # pairing track 1 with the 138 detection is individually cheaper than nothing.
    assert dict(zip(rows.tolist(), cols.tolist(), strict=True)) == {0: 1, 1: 0}
    assert costs[0, 1] < UNREACHABLE and costs[1, 0] < UNREACHABLE


def test_an_unreachable_pair_is_never_accepted() -> None:
    # The solver must return a full assignment, so it will pair things the gate refuses.
    # Those pairings have to be dropped afterwards or a track teleports across the pitch.
    far = Track(id=1, observations=[obs(1, 100.0, 200.0)])
    assert _cost(far, obs(2, 3000.0, 200.0), None, FPS, None) is None

"""Naming the side a track is on, and declining to when the kit does not say.

`assign` is the one stage that can put a player in the wrong colour, and a wrong colour
is not a cosmetic error: the board draws the pass between two shirts, so a mislabelled
player turns a pass into a turnover that never happened.
"""

from __future__ import annotations

import numpy as np

from football_tracks.stage2_track import Track
from football_tracks.stage3_teams import assign


def kitted(track_id: int, kit: tuple[float, float, float]) -> Track:
    t = Track(id=track_id)
    t.kit_sum = np.array(kit, dtype=np.float64)
    t.kit_seen = 1
    return t


RED = (1.0, 0.0, 0.0)
BLUE = (0.0, 0.0, 1.0)


def test_two_clear_kits_become_two_sides() -> None:
    tracks = [kitted(i, RED) for i in range(4)] + [kitted(10 + i, BLUE) for i in range(4)]
    mean_x = {t.id: (20.0 if t.id < 10 else 80.0) for t in tracks}
    out = assign(tracks, mean_x)
    assert {out[i] for i in range(4)} == {"home"}
    assert {out[10 + i] for i in range(4)} == {"away"}


def test_a_kit_on_the_boundary_is_declined_rather_than_guessed() -> None:
    """A coin flip reaches the board as a player in the wrong colour, and the pass drawn
    to them is a turnover that never happened.

    Two kits that shade into each other, as they do under floodlights and a low camera:
    the ends of the range are still clear and the middle of it is not.
    """
    reds = [(1.0, 0.0, 0.0), (0.85, 0.0, 0.15), (0.7, 0.0, 0.3), (0.55, 0.0, 0.45)]
    blues = [(0.0, 0.0, 1.0), (0.15, 0.0, 0.85), (0.3, 0.0, 0.7), (0.45, 0.0, 0.55)]
    tracks = [kitted(i, k) for i, k in enumerate(reds)]
    tracks += [kitted(10 + i, k) for i, k in enumerate(blues)]
    mean_x = {t.id: (20.0 if t.id < 10 else 80.0) for t in tracks}

    out = assign(tracks, mean_x)
    assert out[0] == "home", "the reddest kit is not in doubt"
    assert out[10] in {"away", "referee"}, "nor is the bluest"
    boundary = {out[3], out[13]}
    assert "unknown" in boundary, f"a kit between the two must not be guessed at: {boundary}"


def test_a_track_with_no_kit_at_all_is_still_unknown() -> None:
    tracks = [kitted(i, RED) for i in range(3)] + [kitted(10 + i, BLUE) for i in range(3)]
    blind = Track(id=50)
    tracks.append(blind)
    mean_x = {t.id: (20.0 if t.id < 10 else 80.0) for t in tracks}
    mean_x[50] = 50.0
    assert assign(tracks, mean_x)[50] == "unknown"

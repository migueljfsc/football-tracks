"""Carrying a homography across frames.

`between` needs real images, so the tests synthesise them: a green textured frame and
the same frame warped by a known transform. Green because the tracker only takes
features from the grass, which is the point of the mask.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from football_tracks import calibration
from football_tracks import stage1_propagate as prop

W, HGT = 640, 480


def grass(seed: int = 0) -> npt.NDArray[np.uint8]:
    """A textured green field - featureless grass gives the tracker nothing to hold."""
    rng = np.random.default_rng(seed)
    img = np.zeros((HGT, W, 3), dtype=np.uint8)
    img[:, :] = (55, 120, 45)
    noise = rng.normal(0, 18, (HGT, W, 1))
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def shifted(img: npt.NDArray[np.uint8], dx: float, dy: float) -> npt.NDArray[np.uint8]:
    m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])
    warped = cv2.warpPerspective(img, m, (W, HGT), borderMode=cv2.BORDER_REFLECT)
    return np.asarray(warped, dtype=np.uint8)


def test_between_recovers_a_known_camera_move() -> None:
    base = grass()
    d = prop.between(base, shifted(base, 6.0, -4.0))
    assert d is not None
    moved = cv2.perspectiveTransform(np.array([[[100.0, 100.0]]]), d).reshape(2)
    assert moved == pytest.approx([106.0, 96.0], abs=0.5)


def test_between_refuses_a_frame_with_no_grass() -> None:
    # A close-up, a crowd shot, a cut. Refusing is a gap in the chain, which is right:
    # guessing would put every later frame on a different pitch.
    grey = np.full((HGT, W, 3), 90, dtype=np.uint8)
    assert prop.between(grey, grey.copy()) is None


def test_carry_composes_in_the_right_direction() -> None:
    # h maps image n to pitch; d maps image n to image n+1. A point in image n+1 goes
    # back through d, then through h. Composing the other way is a plausible-looking
    # matrix that drifts the wrong way, which no type can catch.
    h = np.array([[0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 1.0]])
    d = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    out = prop.carry(h, d)
    assert out is not None
    # A feature at image x=10 in frame n sits at x=20 in frame n+1 and is the same
    # blade of grass, so both must land on the same metre.
    before = cv2.perspectiveTransform(np.array([[[10.0, 0.0]]]), h).reshape(2)
    after = cv2.perspectiveTransform(np.array([[[20.0, 0.0]]]), out).reshape(2)
    assert after == pytest.approx(before, abs=1e-9)


def test_carry_refuses_a_collapsed_composition() -> None:
    # A degenerate chain does not raise - it returns a matrix that maps the frame to a
    # point, which downstream reads as every player standing in the same place.
    h = np.eye(3)
    assert prop.carry(h, np.zeros((3, 3))) is None


def test_fill_prefers_the_direct_fit_and_carries_only_the_gaps(tmp_path: Path) -> None:
    base = grass()
    for f in range(1, 5):
        cv2.imwrite(str(tmp_path / f"{f:06d}.jpg"), shifted(base, 3.0 * f, 0.0))

    solved = np.array([[0.05, 0.0, 1.0], [0.0, 0.05, 2.0], [0.0, 0.0, 1.0]])
    direct: dict[int, npt.NDArray[np.float64] | None] = {1: solved, 2: None, 3: solved, 4: None}

    chain = prop.fill(tmp_path, direct)
    assert chain.solved_directly == 2
    assert chain.carried == 2
    assert chain.gaps == 0
    # A frame that solved on its own keeps its own answer rather than a carried one.
    kept = chain.homographies[3]
    assert kept is not None
    assert np.allclose(kept, solved)


def test_two_anchors_meet_in_the_middle_instead_of_one_reaching_the_whole_way(
    tmp_path: Path,
) -> None:
    """A forward-only chain hands every frame between two anchors to the earlier one, so
    the second seed a coach clicks does nothing for the frames before it (D80). The camera
    here does not move, so any difference between the two ends is disagreement between the
    anchors -- and the frames between them share it out rather than taking either whole."""
    base = grass()
    for f in range(1, 6):
        cv2.imwrite(str(tmp_path / f"{f:06d}.jpg"), base)

    start = np.array([[105.0 / W, 0.0, 0.0], [0.0, 68.0 / HGT, 0.0], [0.0, 0.0, 1.0]])
    end = start.copy()
    end[0, 2] = 4.0  # the same camera, four metres along the pitch
    chain = prop.fill(tmp_path, {1: start, 2: None, 3: None, 4: None, 5: end})

    xs = []
    for f in range(1, 6):
        h = chain.homographies[f]
        assert h is not None
        xs.append(calibration.to_pitch(h, W / 2, HGT / 2)[0])
    # Anchors kept exactly, and the gap crossed in even steps: no frame moves the players
    # further than any other, which is what stops a track being cut at the join.
    assert xs[0] == pytest.approx(52.5, abs=0.01)
    assert xs[-1] == pytest.approx(56.5, abs=0.01)
    steps = [b - a for a, b in pairwise(xs)]
    assert max(steps) - min(steps) < 0.2


def test_fill_gives_up_rather_than_carrying_past_the_cap(tmp_path: Path) -> None:
    base = grass()
    for f in range(1, 6):
        cv2.imwrite(str(tmp_path / f"{f:06d}.jpg"), shifted(base, 3.0 * f, 0.0))
    direct: dict[int, npt.NDArray[np.float64] | None] = {
        1: np.eye(3),
        2: None,
        3: None,
        4: None,
        5: None,
    }

    chain = prop.fill(tmp_path, direct, max_carry=2)
    assert chain.carried == 2
    assert chain.gaps == 2
    assert chain.homographies[5] is None


def test_a_chain_cannot_start_itself(tmp_path: Path) -> None:
    base = grass()
    for f in range(1, 4):
        cv2.imwrite(str(tmp_path / f"{f:06d}.jpg"), shifted(base, 3.0 * f, 0.0))
    chain = prop.fill(tmp_path, {1: None, 2: None, 3: None})
    assert chain.carried == 0
    assert chain.gaps == 3


def onto_pitch() -> npt.NDArray[np.float64]:
    """A camera model that puts the test frame on the pitch, so probes land on grass."""
    return np.array([[105.0 / W, 0.0, 0.0], [0.0, 68.0 / HGT, 0.0], [0.0, 0.0, 1.0]])


def test_drift_scores_nothing_when_only_the_seed_was_fitted(tmp_path: Path) -> None:
    """A carry can only be measured against evidence it did not produce.

    Handing `drift` the carried chain instead of the direct fits makes it compare the
    chain with itself, which reports 0.00 m however far the camera has wandered.
    """
    base = grass()
    for f in range(1, 12):
        cv2.imwrite(str(tmp_path / f"{f:06d}.jpg"), shifted(base, 3.0 * f, 0.0))
    assert prop.drift(tmp_path, {1: onto_pitch()}, 1, length=10) == []


def test_drift_reports_a_carry_that_disagrees_with_a_later_fit(tmp_path: Path) -> None:
    base = grass()
    for f in range(1, 12):
        cv2.imwrite(str(tmp_path / f"{f:06d}.jpg"), shifted(base, 3.0 * f, 0.0))
    # The frames really do move, so an identity fit at frame 11 is a genuine disagreement.
    walked = prop.drift(tmp_path, {1: onto_pitch(), 11: onto_pitch()}, 1, length=10)
    assert [c for c, _ in walked] == [10]
    assert walked[0][1] > 1.0


def test_observed_error_ignores_a_corner_the_camera_cannot_see() -> None:
    """The metric asks about the picture, not about the model's fixed points.

    Three of the four pitch corners fall outside a tight shot of a penalty area, one of
    them twenty frame-widths away, so a corner-based error reports the extrapolation and
    not the camera (D33).
    """
    truth = onto_pitch()
    # A model that agrees across the frame and diverges hard off its left edge.
    near = truth.copy()
    near[0, 2] = 0.5  # half a metre of pan, everywhere on screen
    got = calibration.observed_error(truth, near, (HGT, W, 3))
    assert 0.4 < got < 0.6


def _scaled(tx: float = 0.0) -> npt.NDArray[np.float64]:
    """A camera model mapping pixels to metres at a tenth, optionally shifted."""
    return np.array([[0.1, 0.0, tx], [0.0, 0.1, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def test_a_fit_agreeing_with_the_frame_before_it_is_kept() -> None:
    direct: dict[int, npt.NDArray[np.float64] | None] = {1: _scaled(), 2: _scaled(), 3: _scaled()}
    motion = dict.fromkeys((2, 3), np.eye(3, dtype=np.float64))
    assert all(v is not None for v in prop.winnow(direct, motion).values())


def test_a_fit_contradicting_the_frame_before_it_is_dropped() -> None:
    """The segmenter's failure is a tail of confidently wrong fits, and its own residual
    cannot see them: it is in-sample. The previous fit, walked forward by measured
    motion, is independent evidence."""
    direct: dict[int, npt.NDArray[np.float64] | None] = {
        1: _scaled(),
        2: _scaled(40.0),
        3: _scaled(),
    }
    motion = dict.fromkeys((2, 3), np.eye(3, dtype=np.float64))
    out = prop.winnow(direct, motion)
    assert out[1] is not None
    assert out[2] is None, "a fit 40 m from its neighbour is not a camera model"
    assert out[3] is not None, "a rejected fit must not become the standard"


def test_a_gap_is_left_alone() -> None:
    direct: dict[int, npt.NDArray[np.float64] | None] = {1: _scaled(), 2: None, 3: _scaled()}
    motion = dict.fromkeys((2, 3), np.eye(3, dtype=np.float64))
    out = prop.winnow(direct, motion)
    assert out[2] is None and out[3] is not None


def test_a_stale_reference_stops_judging() -> None:
    """A carry drifts, so an old reference starts refusing fits for disagreeing with it."""
    direct: dict[int, npt.NDArray[np.float64] | None] = {1: _scaled(), 500: _scaled(40.0)}
    motion = {f: np.eye(3, dtype=np.float64) for f in range(2, 501)}
    assert prop.winnow(direct, motion, max_age=50)[500] is not None


def test_blending_a_model_with_itself_changes_nothing() -> None:
    h = _scaled()
    out = prop.blend(h, h, 0.5, width=W, height=HGT)
    assert out is not None
    assert out == pytest.approx(h, abs=1e-9)


def test_a_blend_moves_the_stated_share_of_the_way_in_metres() -> None:
    """Mixed where it means something. Averaging the matrices would not land here."""
    a, b = _scaled(), _scaled(10.0)
    out = prop.blend(a, b, 0.25, width=W, height=HGT)
    assert out is not None
    point = np.array([[[W * 0.5, HGT * 0.5]]], dtype=np.float64)
    got = cv2.perspectiveTransform(point, out).reshape(2)
    want = cv2.perspectiveTransform(point, _scaled(2.5)).reshape(2)
    assert got == pytest.approx(want, abs=1e-6)


def test_disagreement_is_the_metres_between_two_models() -> None:
    assert prop.disagreement(_scaled(), _scaled(), width=W, height=HGT) == pytest.approx(0.0)
    assert prop.disagreement(_scaled(), _scaled(3.0), width=W, height=HGT) == pytest.approx(3.0)


def test_an_anchored_chain_closes_on_its_anchor_rather_than_jumping_to_it() -> None:
    """The whole point of the filter: a standing player must not take a step because the
    camera model was corrected."""
    anchors: dict[int, npt.NDArray[np.float64] | None] = {1: _scaled()}
    anchors.update(dict.fromkeys(range(2, 41), _scaled(4.0)))
    motion = {f: np.eye(3, dtype=np.float64) for f in range(2, 41)}
    out = prop.anchor_chain(anchors, motion=motion, rate=0.1, width=W, height=HGT).homographies

    steps = [
        prop.disagreement(out[f - 1], out[f], width=W, height=HGT)  # type: ignore[arg-type]
        for f in range(3, 41)
    ]
    assert max(steps) < 0.5, "no single frame may move the pitch under a player"
    assert prop.disagreement(out[40], _scaled(4.0), width=W, height=HGT) < 0.2  # type: ignore[arg-type]


def test_a_seed_is_taken_whole_and_a_fit_is_not() -> None:
    anchors: dict[int, npt.NDArray[np.float64] | None] = {1: _scaled(), 2: _scaled(4.0)}
    motion = {2: np.eye(3, dtype=np.float64)}
    mixed = prop.anchor_chain(anchors, motion=motion, rate=0.1, width=W, height=HGT).homographies
    whole = prop.anchor_chain(
        anchors, motion=motion, rate=0.1, hard={2}, width=W, height=HGT
    ).homographies
    assert prop.disagreement(mixed[2], _scaled(), width=W, height=HGT) == pytest.approx(  # type: ignore[arg-type]
        0.4, abs=0.01
    )
    assert whole[2] is not None
    assert prop.disagreement(whole[2], _scaled(4.0), width=W, height=HGT) == pytest.approx(0.0)


def test_an_anchored_chain_covers_the_frames_before_its_first_anchor() -> None:
    anchors: dict[int, npt.NDArray[np.float64] | None] = dict.fromkeys(range(1, 6))
    anchors[5] = _scaled()
    motion = {f: np.eye(3, dtype=np.float64) for f in range(2, 6)}
    chain = prop.anchor_chain(anchors, motion=motion, width=W, height=HGT)
    assert chain.gaps == 0

"""Carrying a homography across frames.

`between` needs real images, so the tests synthesise them: a green textured frame and
the same frame warped by a known transform. Green because the tracker only takes
features from the grass, which is the point of the mask.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest

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
    got = prop.observed_error(truth, near, (HGT, W, 3))
    assert 0.4 < got < 0.6

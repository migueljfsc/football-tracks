"""The automatic path, end to end: frames in, tracks.json out.

Stitches detection, tracking, team assignment and registration together. Registration
is pluggable on purpose, because the whole question this pipeline exists to answer is
how much of the error belongs to which stage:

* `truth`  - a homography per frame from the ground-truth pitch lines. Holds stage 1
             fixed, so what comes out is stage 2 and 3's error alone.
* `seed`   - ONLY frame one's lines, then carried by tracking the grass. This is the
             real question: what a human clicking four corners once actually buys.
* `segmenter` - a homography per frame from the LEARNED lines, with no seed, no carry
             and no ground truth. The same shape as `truth` but sourced from the picture,
             which is the whole claim of D36.

All three write the same tracks.json, scored by the same `ft score`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from . import (
    calibration,
    detect,
    stage1_propagate,
    stage1_register,
    stage2_stitch,
    stage2_track,
)
from . import seed as seed_mod
from .config import PITCH_LENGTH, PITCH_WIDTH
from .stage3_teams import assign
from .tracks import PLAYER_MARGIN, Sample, Track, on_pitch

Mode = Literal["truth", "seed", "segmenter"]


# How many frames either side the ball's position is taken a median over.
#
# The detector reports about five "sports balls" a frame - a head, a boot, a patch of
# hoarding - and the real one is only usually the most confident. What separates it from
# the impostors is that it MOVES SMOOTHLY, so a median over neighbouring frames throws
# them out without having to know which is which.
BALL_SMOOTH_FRAMES = 5

# How far the ball may move between consecutive frames, in pixels, and still be the same
# ball. Generous: a struck ball near the camera crosses a lot of image in 40 ms. It is a
# continuity gate, not a physics model -- its job is to reject the 21% of frames where the
# per-frame argmax jumped over 500 px to a different white object entirely.
BALL_MAX_PX_PER_FRAME = 220.0

# How long a trajectory survives without a sighting before the ball counts as lost. Past
# this the position prior is worthless and re-acquisition has to earn its place again.
BALL_COAST_FRAMES = 8

# The widest the continuity gate may open, however long the wait. Without this the gate
# grows with the gap and a coasting trajectory eventually reaches the whole frame, which
# is the exact failure stage 2's MAX_AGE_S comment describes.
BALL_MAX_REACH_PX = 400.0

# A square metre of pitch, and how much of a clip a "ball" may spend inside one before it
# is judged to be painted on. A third is far beyond any real ball: even a ball waiting to
# be kicked moves on within a few seconds, and a clip is thirty.
STATIC_BIN_M = 1.0
STATIC_SHARE = 0.33

# The same test, for a square metre that is NOT a restart spot.
#
# A ball placed for a corner holds one square metre for about a fifth of a clip, which is
# why STATIC_SHARE cannot go near that: the floor that catches scenery also catches every
# set piece. Off a restart spot there is no such ball to protect, so the floor can drop --
# but not far. A stoppage leaves the ball sitting anywhere: at 0.15 a booking's ball is
# called scenery and SNGS-121 loses 8 points of accuracy with it. 0.20 is what fits between
# a placed ball and a mark that never moves.
STATIC_OFF_SPOT_SHARE = 0.20
# Below this there is not enough clip to tell a stationary ball from a stationary mark.
STATIC_MIN_FRAMES = 100

# Sightings a frame's neighbourhood needs before a ball position is believed. One sighting
# smeared over eleven frames is not a measurement of where the ball was on ten of them.
MIN_SMOOTH_SAMPLES = 3

# What a frame costs when the path declares the ball unseen, and how much a candidate's
# confidence weighs against the distance it would have to have travelled. Missing must be
# dearer than an ordinary step or the path goes dark rather than follow a real ball, and
# cheaper than a long jump or it never abstains at all.
BALL_MISSING = 0.9
BALL_EMISSION = 1.2

# How far outside the pitch a ball may sit and still be kept, IN METRES -- which is NOT
# what tracks.on_pitch's margin means. That one is a SHARE of the pitch, so handing it
# metres opens the gate to 157 m and keeps every airborne projection there is. Wider than
# a player's, because a corner is taken from ON the line and homography error puts it just
# outside; narrow enough to still reject an airborne ball's projection, which lands tens
# of metres away.
BALL_MARGIN_M = 1.5

# What a sighting must score before a ball is asserted AT ALL.
#
# High, and measured against SoccerNet's own ball annotations on SNGS-116, which is the
# only honest way to set it. The detector's floor is 0.15 and at that floor the chosen
# ball is within 20 px of the real one in 25% of frames -- so for three frames in four the
# board was shown a ball that was somewhere else entirely. That is where the phantom shot
# came from, and the ball that sat in the six-yard box through a corner taken off-camera.
#
#     conf   frames with a ball   of those, actually the ball
#     0.15         74%                      25%
#     0.35         27%                      48%
#     0.55         12%                      74%
#     0.65         10%                      84%
#     0.75          9%                      94%
#
# The ball can be frequent or right, and not both: the detector finds it at all in only
# 41% of frames, so no selection rule can do better than that. Given the choice, a board
# is better with no ball than with a wrong one -- Pitchboard represents "no ball" natively
# (its D44) and draws a phantom pass for a wrong one. 0.65 buys 84% correctness.
BALL_ASSERT_CONF = 0.65

# Where a restart puts the ball: the four corner arcs and the centre spot.
#
# NOT the penalty spots. They are where the detector's favourite false positive lives --
# a painted white disc on grass is what a ball looks like to a model trained on
# photographs (see _painted_spots) -- and a penalty is the one restart these clips never
# contain, so opening the floor there would admit exactly what that filter exists to
# remove and buy nothing.
RESTART_SPOTS = (
    (0.0, 0.0),
    (0.0, PITCH_WIDTH),
    (PITCH_LENGTH, 0.0),
    (PITCH_LENGTH, PITCH_WIDTH),
    (PITCH_LENGTH / 2, PITCH_WIDTH / 2),
)

# How close to one of those a sighting must land, and how many frames it must keep
# landing there, before the ball is believed at the detector's own floor rather than at
# BALL_ASSERT_CONF.
#
# Measured on SNGS-116's corner against SoccerNet's ball annotations. Within 1.5 m of a
# spot the position IS the evidence: 79 of the 80 frames this admits are within 3 m of
# the real ball, while confidence separates nothing there -- the true sightings score
# 0.15 to 0.37 and the two false ones score 0.18 and 0.32, which is why the confidence
# gate could never find this ball and why lowering it globally would only add junk.
#
# 1.5 m rather than 2.0 m is a deliberate trade. 2.0 m finds four more frames of the
# corner and puts EIGHT fabricated ones into SNGS-121, a clip with no restart in it,
# where the corner flag reads as a ball. At 1.5 m the two non-set-piece clips admit
# nothing at all. Few and right beats many and wrong (BALL_ASSERT_CONF says the same).
#
# Ten frames is 0.4 s, and the length is doing as much work as the radius. Every run this
# gets WRONG across eight clips is short and transient -- 5 frames on the centre spot
# during a free kick, 8 at a corner just after it was taken, 4 more on a centre spot --
# while every run it gets right is 16 to 80 frames of a ball genuinely sitting there. At
# 4 frames the free-kick clip SNGS-066 gained 12 fabricated frames; at 10 it gains none.
RESTART_RADIUS_M = 1.5
RESTART_MIN_FRAMES = 10


@dataclass(slots=True)
class Result:
    tracks: list[Track]
    ball: list[Sample]
    frames: int
    detections: int
    raw_tracks: int
    dropped_off_pitch: int
    unsolved_frames: int


def seed_paths(work: Path) -> list[Path]:
    """Every clicked frame for a clip, primary first.

    One seed cannot cross a broadcast clip. Carried 453 frames it lands 44 m from where
    the players actually are (D34), and nothing downstream can tell - the tracks and the
    board share the wrong coordinate frame, so every fidelity score stays excellent while
    the play happens in the wrong half. More anchors is the only fix that does not need a
    better flow estimator.
    """
    primary = work / "seed.json"
    extra = sorted(p for p in work.glob("seed.*.json") if p != primary)
    return ([primary] if primary.exists() else []) + extra


def usable_seeds(
    work: Path, frames_dir: Path
) -> tuple[list[seed_mod.Seed], list[tuple[Path, float]]]:
    """The seeds that describe their own frame, and the ones that do not.

    A seed is refused here rather than trusted because it fits its own clicks: evidence
    confined to a band of the frame is unconstrained in depth, and the fit folds over
    just below it (D34). Anchoring the pipeline on one of those is worse than having no
    anchor there at all - it does not degrade with distance, it is wrong at the anchor.
    """
    good: list[seed_mod.Seed] = []
    bad: list[tuple[Path, float]] = []
    for path in seed_paths(work):
        seeded = seed_mod.read(path)
        h = seed_mod.homography(seeded)
        img = cv2.imread(str(frames_dir / f"{seeded.frame:06d}.jpg"))
        if h is None or img is None:
            bad.append((path, 1.0))
            continue
        behind = seed_mod.behind_camera(h, img.shape[1], img.shape[0])
        if behind > seed_mod.MAX_BEHIND_CAMERA:
            bad.append((path, behind))
        else:
            good.append(seeded)
    return good, bad


def from_seeds(
    seeds: list[seed_mod.Seed],
    frames: list[int],
    frames_dir: Path,
    *,
    max_carry: int | None,
    motions: dict[int, Any] | None = None,
    snap: Any = None,
    max_residual_m: float | None = None,
) -> dict[int, Any]:
    """Clicked frames, carried across the clip in both directions.

    `fill` already prefers a direct fit and carries only the gaps between them, so more
    seeds shorten every chain rather than adding a mechanism. Both directions matter - a
    clip is rarely best seeded at its first frame, because the camera is often still
    finding the play there.
    """
    direct: dict[int, Any] = dict.fromkeys(frames)
    for seeded in seeds:
        h = seed_mod.homography(seeded)
        if h is not None and seeded.frame in direct:
            direct[seeded.frame] = h
    return stage1_propagate.fill(
        frames_dir, direct, max_carry=max_carry, motion=motions, snap=snap
    ).homographies


def segmenter_homographies(
    frames_dir: Path, weights: Path | None = None, max_residual_m: float | None = None
) -> dict[int, Any]:
    """A homography per frame from the segmenter, fitted from the picture and nothing else.

    No seed, no carry and no fill, which is the entire point: every frame is solved from
    its own pixels, so there is no chain to drift and a bad frame cannot poison its
    neighbours. D19 measured what carrying costs -- identity purity 62.9% against 82.9%
    with it off -- and this is the source that owes nothing to it.

    A frame the fitter cannot solve stays `None` rather than borrowing an answer. That is
    the honest gap `fill` would have hidden, and it keeps the mode's claim exactly as
    strong as its name. The caller may bridge those gaps with a short carry; this function
    does not, so what it returns is always the segmenter's own opinion and nothing else.
    """
    from . import calib

    net = calib.model(weights or calib.WEIGHTS).to(calib.device()).eval()
    out: dict[int, Any] = {}
    for path in sorted(frames_dir.glob("*.jpg")):
        image = cv2.imread(str(path))
        if image is None:
            continue
        mask = calib.predict(net, image)
        out[int(path.stem)] = calib.fit_from_mask(
            mask, image.shape[1], image.shape[0], max_residual_m
        )
    return out


def _best_seed(labels: dict[str, Any], direct: dict[int, Any]) -> int | None:
    """The solvable frame with the most pitch markings in shot, earliest breaking ties.

    Counting markings rather than scoring the fit on its own residual: a fit is chosen to
    minimise that residual, so a frame with barely enough evidence scores well on it for
    the same reason it is fragile -- which is D35's rigged-selection trap.
    """
    frame_of = {
        img["image_id"]: stage1_register._frame_index(img["file_name"]) for img in labels["images"]
    }
    evidence: dict[int, int] = {}
    for a in labels["annotations"]:
        if a.get("category_id") != 5:
            continue
        f = frame_of.get(a["image_id"])
        if f is None or direct.get(f) is None:
            continue
        lines = calibration.lines_of(a)
        evidence[f] = sum(1 for k in lines if calibration.PITCH_LINES.get(k) is not None)
    if not evidence:
        return next((f for f in sorted(direct) if direct[f] is not None), None)
    return min(evidence, key=lambda f: (-evidence[f], f))


def homographies(
    labels: dict[str, Any],
    frames_dir: Path,
    mode: Mode,
    *,
    max_carry: int | None,
    motions: dict[int, Any] | None = None,
    snap: Any = None,
    max_residual_m: float | None = None,
) -> dict[int, Any]:
    """Per-frame homographies: from every frame's lines, from frame one's, or from the
    segmenter's."""
    if mode == "segmenter":
        direct = segmenter_homographies(frames_dir, max_residual_m=max_residual_m)
        if motions is not None:
            direct = stage1_propagate.winnow(direct, motions)
        # No carry by default, which is the mode's whole claim. A SHORT carry is offered
        # because refusals turned out to be the binding constraint -- half of SNGS-121 is
        # declined -- and bridging a two-frame gap is not the unbounded chain D19
        # condemned. `--carry 0` keeps the gaps honest.
        if max_carry is None or max_carry <= 0:
            return direct
        return stage1_propagate.fill(
            frames_dir, direct, max_carry=max_carry, motion=motions
        ).homographies
    direct = stage1_register.fit_all(labels)
    if mode == "truth":
        return stage1_propagate.fill(
            frames_dir, direct, max_carry=max_carry, motion=motions
        ).homographies

    # Everything except ONE frame is thrown away, which is what a human clicking once
    # actually leaves you with -- but the frame is the best-EVIDENCED one, not the
    # earliest. A person seeding a clip picks a view where they can see the pitch; taking
    # whatever comes first models a worse human than the one being modelled.
    #
    # It is not a nicety. SNGS-121 opens on 369 midfield frames carrying four usable
    # markings, which `curve_crossings` can just about rescue at a 0.385 m residual
    # against 0.123 m at frame 370. Seeding on the earliest put that error into every
    # frame of the clip and took recall from 15.8% to 9.4%; the fits it seeds from are by
    # construction the ones the fitter was least sure of.
    best = _best_seed(labels, direct)
    seeded: dict[int, Any] = dict.fromkeys(direct)
    if best is not None:
        seeded[best] = direct[best]
    return stage1_propagate.fill(
        frames_dir, seeded, max_carry=max_carry, motion=motions, snap=snap
    ).homographies


def _restart_cell(cell: tuple[int, int]) -> bool:
    """Whether a square metre holds a restart spot, within the radius one is believed at."""
    x, y = (cell[0] + 0.5) * STATIC_BIN_M, (cell[1] + 0.5) * STATIC_BIN_M
    reach = RESTART_RADIUS_M + STATIC_BIN_M
    return any(math.hypot(x - sx, y - sy) <= reach for sx, sy in RESTART_SPOTS)


def _bin_of(x: float, y: float) -> tuple[int, int]:
    return (int(x // STATIC_BIN_M), int(y // STATIC_BIN_M))


def _painted_spots(
    per_frame: dict[int, list[detect.Sighting]], homs: dict[int, Any]
) -> set[tuple[int, int]]:
    """Places on the pitch a "ball" sits at for far too much of the clip to be a ball.

    The detector calls the PENALTY SPOT a ball. On SNGS-116 it does so on almost every
    frame of the corner, at pixel (1069, 612) -- a white circle painted on grass, which is
    what a ball looks like to a detector trained on photographs. Projected, it lands at
    93.5, 33.7 m, and the right-hand penalty spot is at 94, 34. That is the ball a coach
    sees sitting in the six-yard box while the corner is being taken off-camera, the ball
    the keeper appears to hold for ever, and half of the phantom shot.

    Rather than hardcode the spots -- which would miss the centre spot's twin, litter
    behind the goal, and whatever else a given ground has painted on it -- this asks the
    data: in PITCH metres, where does a candidate keep appearing? Camera motion is already
    divided out by the homography, so a mark on the grass has one position for the whole
    clip and a ball has a different one every second. Anything occupying one square metre
    for more than a third of the frames it could be seen in is scenery.
    """
    seen_at: dict[tuple[int, int], set[int]] = {}
    frames_with_a_homography = 0
    for f, cands in per_frame.items():
        h = homs.get(f)
        if h is None:
            continue
        frames_with_a_homography += 1
        for b in cands:
            x, y = calibration.to_pitch(h, b.x, b.y)
            seen_at.setdefault(_bin_of(x, y), set()).add(f)
    if frames_with_a_homography < STATIC_MIN_FRAMES:
        return set()  # too short to tell a stationary ball from a painted one
    floor = STATIC_SHARE * frames_with_a_homography
    off_floor = STATIC_OFF_SPOT_SHARE * frames_with_a_homography
    return {
        cell
        for cell, fs in seen_at.items()
        if len(fs) >= (floor if _restart_cell(cell) else off_floor)
    }


def _ball_near_pitch(x: float, y: float, margin_m: float = BALL_MARGIN_M) -> bool:
    """`tracks.on_pitch`, but with a margin in METRES rather than a share of the pitch."""
    return -margin_m <= x <= PITCH_LENGTH + margin_m and -margin_m <= y <= PITCH_WIDTH + margin_m


def _is_static(b: detect.Sighting, h: Any, static: set[tuple[int, int]]) -> bool:
    x, y = calibration.to_pitch(h, b.x, b.y)
    return _bin_of(x, y) in static


def _smoothed(best: dict[int, tuple[float, float]], frames: list[int], smooth: int) -> list[Sample]:
    """Per-frame positions, median-filtered over their neighbours."""
    out: list[Sample] = []
    for f in frames:
        near = [best[g] for g in range(f - smooth, f + smooth + 1) if g in best]
        # A median of ONE value is that value, not a median. With sightings this sparse a
        # single detection was filling eleven frames with a ball, at up to five frames'
        # remove from the only evidence for it -- which is how SNGS-116's board came to
        # assert a carrier at a scene where our ball was 25 m from the real one.
        if len(near) < MIN_SMOOTH_SAMPLES:
            continue
        x = float(np.median([p[0] for p in near]))
        y = float(np.median([p[1] for p in near]))
        # A BALL's margin, not a player's. A corner is taken from the corner arc, so the
        # ball legitimately sits on the line -- SoccerNet's own annotation of SNGS-116's
        # corner projects to (105.2, -0.4). Still bounded, because a ball in flight
        # projects anywhere: that same clip has one at (135.5, -16.6).
        if _ball_near_pitch(x, y):
            out.append(Sample(f=f, x=x, y=y))
    return out


def _restart_balls(
    per_frame: dict[int, list[detect.Sighting]],
    homs: dict[int, Any],
    static: set[tuple[int, int]],
    emitted: set[int],
    smooth: int,
) -> dict[int, tuple[float, float]]:
    """The ball sitting still on a restart spot, believed without the confidence gate.

    A set piece is the one moment the ball's position is known before it is seen: it is
    on the corner arc, or the centre spot, and it stays there for seconds. That is worth
    a rule of its own because it is precisely where the ordinary selector fails - the
    ball is small, still and far away, so it scores 0.2 and never clears
    BALL_ASSERT_CONF. SNGS-116 asserted NO ball at all across the whole 157-frame corner
    that opens the clip, which is what left the board with a ball already in the box.

    This only speaks where the pipeline would otherwise emit nothing, and that veto is
    what makes it safe rather than the position. The same corner region on SNGS-121 holds
    32 sightings that are all false, 25 m from the real ball - but there the ball IS
    being tracked, so the veto silences this pass entirely. Vetoing on the SMOOTHED
    output rather than the raw sightings matters: SNGS-116's confident pass fires twice
    before the corner, at frames 98 and 100, and both are wrong by over 30 m. Two
    isolated blips are not a tracked ball, MIN_SMOOTH_SAMPLES already says so, and
    letting them veto costs 66 frames of a corner that is really there.
    """
    cand: dict[int, tuple[float, float]] = {}
    for f in sorted(per_frame):
        h = homs.get(f)
        if h is None:
            continue
        near: list[tuple[float, float, float]] = []
        for b in per_frame[f]:
            if _is_static(b, h, static):
                continue
            x, y = calibration.to_pitch(h, b.x, b.y)
            d = min(math.hypot(x - sx, y - sy) for sx, sy in RESTART_SPOTS)
            if d <= RESTART_RADIUS_M:
                near.append((d, x, y))
        if near:
            _, x, y = min(near)
            cand[f] = (x, y)

    quiet = {
        f: p
        for f, p in cand.items()
        if not any(g in emitted for g in range(f - smooth, f + smooth + 1))
    }

    # A ball somebody placed sits there; a mark that reads as a ball for a frame or two
    # does not. Runs tolerate gaps, because the detector loses the ball between frames
    # without it having moved.
    out: dict[int, tuple[float, float]] = {}
    run: list[int] = []
    for nxt in [*sorted(quiet), None]:
        if run and (nxt is None or nxt - run[-1] > smooth):
            if len(run) >= RESTART_MIN_FRAMES:
                out.update({g: quiet[g] for g in run})
            run = []
        if nxt is not None:
            run.append(nxt)
    return out


def ball_path(
    balls: list[detect.Sighting],
    homs: dict[int, Any],
    frames: list[int],
    smooth: int = BALL_SMOOTH_FRAMES,
) -> list[Sample]:
    """Where the ball is, per frame, in pitch metres.

    The most confident sighting per frame, projected, then median-filtered over its
    neighbours. Picking the sighting NEAREST a player scores worse than this and it is
    worth saying why: there are several candidates a frame, so "nearest a player"
    reliably selects whichever false positive happens to stand beside somebody.

    The positions are NOT to be trusted as positions. A ground homography assumes z = 0,
    so a ball in flight lands metres from where it is. They are good for one question -
    who is nearest - and that question is all a board needs (D29).
    """
    per_frame: dict[int, list[detect.Sighting]] = {}
    for b in balls:
        per_frame.setdefault(b.f, []).append(b)

    static = _painted_spots(per_frame, homs)

    # Confidence, continuity and abstention -- NOT a shortest path over candidates.
    #
    # The path was written, measured and reverted, and the reason is worth keeping. Tiling
    # made the ball findable (SNGS-116: present in 74% of frames against 39%, and on 67 of
    # its 70 corner frames) and made choosing harder: 15 candidates a frame against 3. A
    # global shortest path over them is the natural answer and it scored 31% within 3 m on
    # SNGS-116 against this selector's 97%, because at frame 110 the filtered candidates
    # include the real ball at pitch (105.1, -0.3) scoring 0.18 AND a false positive at
    # (105.9, 10.9) scoring 0.33. Both are stationary, so continuity cannot separate them;
    # both sit under the static filter's occupancy floor, so that cannot either. The path
    # follows the confident one for the whole clip, where this abstains.
    #
    # Few and right beats many and wrong: a wrong ball puts a pass on the board that never
    # happened, and Pitchboard represents "no ball" natively (its D44).
    best: dict[int, tuple[float, float]] = {}
    last: tuple[int, float, float] | None = None
    for f in sorted(per_frame):
        h = homs.get(f)
        if h is None:
            continue
        seen = [
            b for b in per_frame[f] if b.score >= BALL_ASSERT_CONF and not _is_static(b, h, static)
        ]
        if not seen:
            continue
        if last is not None and f - last[0] <= BALL_COAST_FRAMES:
            # CAPPED, because a gate that grows with the gap eventually reaches the whole
            # frame and adopts whatever appears next -- stage 2 documents that at MAX_AGE_S.
            reach = min(BALL_MAX_PX_PER_FRAME * (f - last[0]), BALL_MAX_REACH_PX)
            usable = [b for b in seen if math.hypot(b.x - last[1], b.y - last[2]) <= reach]
        else:
            usable = seen  # cold: nothing to be consistent with, so confidence decides
        if not usable:
            continue
        top = max(usable, key=lambda b: b.score)
        last = (f, top.x, top.y)
        best[f] = calibration.to_pitch(h, top.x, top.y)

    # The confident pass first, then the one that knows where a restart puts the ball.
    # Second because it defers to it: it fills the stretches this leaves empty.
    first = _smoothed(best, frames, smooth)
    extra = _restart_balls(per_frame, homs, static, {s.f for s in first}, smooth)
    if not extra:
        return first
    return _smoothed({**best, **extra}, frames, smooth)


def build(
    frames_dir: Path,
    frames: list[int],
    detections: list[detect.Detection],
    homs: dict[int, Any],
    *,
    fps: float,
    motions: dict[int, Any] | None = None,
    balls: list[detect.Sighting] | None = None,
    stitch: bool = True,
) -> Result:
    """Detections plus a camera model -> tracks in pitch metres.

    Tracking runs FIRST, in stabilised image pixels, and projection happens after. The
    other order made stage 2 inherit every wobble in stage 1's homography, which is
    what breaks tracks all at once (D19). This way a drifting homography moves the
    positions and leaves the identities intact.
    """
    cache: dict[int, Any] = {}

    def read_frame(f: int) -> Any:
        if f not in cache:
            cache.clear()  # one frame at a time; the tracker only ever looks back one
            cache[f] = cv2.imread(str(frames_dir / f"{f:06d}.jpg"))
        return cache[f]

    # Drop anyone standing off the pitch BEFORE tracking, not after. Two fifths of what
    # the detector finds is crowd, dugout staff and ballboys behind the hoardings, and
    # while they were always discarded at the end, until then they were competing for
    # associations and spawning tracks. Filtering first nearly halves the track count.
    #
    # This is the one place stage 2 consults stage 1 (D22), and only as a filter: the
    # association itself still never sees a homography, so a drifting camera moves which
    # detections are considered and cannot move the identities.
    observations: dict[int, list[stage2_track.Observation]] = {}
    dropped = 0
    for d in detections:
        h = homs.get(d.f)
        if h is None:
            continue
        if not on_pitch(*calibration.to_pitch(h, *d.foot), PLAYER_MARGIN):
            dropped += 1
            continue
        observations.setdefault(d.f, []).append(stage2_track.Observation.of(d))

    raw = stage2_track.run(observations, frames, read_frame, fps=fps, motions=motions)

    positions: dict[int, list[Sample]] = {}
    for t in raw:
        samples: list[Sample] = []
        for o in t.observations:
            h = homs.get(o.f)
            if h is None:
                continue
            x, y = calibration.to_pitch(h, o.x, o.y)
            if not on_pitch(x, y):
                dropped += 1
                continue
            samples.append(Sample(f=o.f, x=x, y=y, conf=o.det.score))
        if samples:
            positions[t.id] = samples

    if stitch:
        # After registration, because whether two fragments are one player is a question
        # about metres per second, and before team assignment, because a joined track
        # should be assigned once rather than voted on by its halves.
        positions = stage2_stitch.stitch(positions, {t.id: t.color for t in raw}, fps)

    mean_x = {tid: float(np.mean([s.x for s in ss])) for tid, ss in positions.items()}
    teams = assign(
        [t for t in raw if t.id in positions],
        mean_x,
        {tid: [s.f for s in ss] for tid, ss in positions.items()},
    )

    return Result(
        ball=ball_path(balls or [], homs, frames),
        tracks=[
            Track(id=tid, team=teams.get(tid, "unknown"), number=None, samples=ss)
            for tid, ss in sorted(positions.items())
        ],
        frames=len(frames),
        detections=len(detections),
        raw_tracks=len(raw),
        dropped_off_pitch=dropped,
        unsolved_frames=sum(1 for f in frames if homs.get(f) is None),
    )

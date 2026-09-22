"""The automatic path, end to end: frames in, tracks.json out.

Stitches detection, tracking, team assignment and registration together. Registration
is pluggable on purpose, because the whole question this pipeline exists to answer is
how much of the error belongs to which stage:

* `truth`  - a homography per frame from the ground-truth pitch lines. Holds stage 1
             fixed, so what comes out is stage 2 and 3's error alone.
* `seed`   - ONLY frame one's lines, then carried by tracking the grass. This is the
             real question: what a human clicking four corners once actually buys, and
             what a match's FIRST clip still runs on.
* `camera` - the learned pitch lines read as three numbers -- pan, tilt and zoom -- off a
             camera position the match is already known to have been shot from. No clicks
             on THIS clip at all, and the frames nothing is visible in are spanned by
             aiming between the frames either side rather than by carrying (D96).

All three write the same tracks.json, scored by the same `ft score`. Two more were built
and measured -- a free per-frame fit from the learned lines (D67) and those fits used as
anchors for a carried seed (D68) -- and removed once the match camera beat both.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_left, bisect_right
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from . import (
    calibration,
    detect,
    reid,
    stage1_propagate,
    stage1_register,
    stage2_stitch,
    stage2_track,
    stage3_teams,
)
from . import camera as camera_mod
from . import seed as seed_mod
from .config import DEFAULT_PITCH, Pitch
from .stage3_teams import assign
from .tracks import PLAYER_MARGIN, Sample, TeamLabel, Track, on_pitch

Mode = Literal["truth", "seed", "camera"]


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
# (its D44) and draws a phantom pass for a wrong one.
#
# 0.75, raised from 0.65 and measured where it lands rather than where it is set (D73). Per
# frame it changes almost nothing -- the path's own smoothing and continuity gates already
# throw out most bad picks, so "within 3 m of the real ball" moves by -3 to +5 points
# depending on the clip. Through the importer it removes two of the four phantom PASSES left
# on six boards and costs no real one: 73% of drawn passes were real, now 85%, with recall
# unchanged at 47%. A pass drawn between the wrong players is the most expensive thing this
# pipeline produces, because a coach reads it as football.
BALL_ASSERT_CONF = 0.75


# Where a restart puts the ball: the four corner arcs and the centre spot.
#
# NOT the penalty spots. They are where the detector's favourite false positive lives --
# a painted white disc on grass is what a ball looks like to a model trained on
# photographs (see _painted_spots) -- and a penalty is the one restart these clips never
# contain, so opening the floor there would admit exactly what that filter exists to
# remove and buy nothing.
def restart_spots(pitch: Pitch = DEFAULT_PITCH) -> tuple[tuple[float, float], ...]:
    return (
        (0.0, 0.0),
        (0.0, pitch.width),
        (pitch.length, 0.0),
        (pitch.length, pitch.width),
        (pitch.halfway, pitch.middle),
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
    kits: dict[str, str] | None = None
    # Each named side's kit signature, as `stage3_teams.side_signatures` measures it: what a
    # match's kits.json is written from when this clip is its first.
    signatures: tuple[Any, Any] | None = None


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
) -> tuple[list[seed_mod.Seed], list[tuple[Path, str]]]:
    """The seeds that describe their own frame, and the ones that do not, with the reason.

    A seed is refused here rather than trusted because it fits its own clicks: evidence
    confined to a band of the frame is unconstrained in depth, and the fit folds over
    just below it (D34). Anchoring the pipeline on one of those is worse than having no
    anchor there at all - it does not degrade with distance, it is wrong at the anchor.

    The reason is a sentence rather than a number because the two failures are different
    jobs for whoever clicked: a fit that FOLDS wants evidence lower in the frame, and a
    fit that could not be made at all is usually two markings labelled with the wrong
    side, which `seed.contradictions` can name.
    """
    good: list[tuple[Path, seed_mod.Seed, float]] = []
    bad: list[tuple[Path, str]] = []
    for path in seed_paths(work):
        seeded = seed_mod.read(path)
        img = cv2.imread(str(frames_dir / f"{seeded.frame:06d}.jpg"))
        if img is None:
            bad.append((path, f"frame {seeded.frame} is not in the clip"))
            continue
        # The frame NUMBER is not an identity: frame 56 exists in every clip, so a seed
        # left behind by the last one anchors this one silently and everything downstream
        # agrees with it. The picture is the identity.
        if seeded.image and seed_mod.unlike(seeded.image, seed_mod.fingerprint(img)) > (
            seed_mod.MAX_UNLIKE_BITS
        ):
            bad.append(
                (
                    path,
                    f"it was clicked on a different picture - frame {seeded.frame} of the clip"
                    " in this folder now is not the one these landmarks describe. Re-seed, or"
                    " move this file out of the way",
                )
            )
            continue
        h = seed_mod.homography(seeded)
        if h is None:
            clash = seed_mod.contradictions(seeded)
            bad.append(
                (
                    path,
                    f"no camera fits these clicks: {clash[0][0]} and {clash[0][1]} are labelled"
                    " with the wrong side of the pitch - nearer the camera is lower in the frame"
                    if clash
                    else "no camera fits these clicks - trace a marking that CROSSES the others,"
                    " and check the far/near names against the diagram",
                )
            )
            continue
        behind = seed_mod.behind_camera(h, img.shape[1], img.shape[0])
        if behind > seed_mod.MAX_BEHIND_CAMERA:
            bad.append(
                (
                    path,
                    f"it maps {behind:.0%} of its frame behind the camera, so it is wrong AT the"
                    " anchor and not merely far from it - trace evidence lower in the frame,"
                    " because a fit needs depth and not just points",
                )
            )
        else:
            good.append((path, seeded, seed_mod.handedness(h, img.shape[1], img.shape[0])))

    # The end check, which no single seed can make: a pitch is symmetric end to end, so a
    # seed clicked on the far goal while the tool offered the near one fits its own clicks
    # perfectly and anchors that stretch of the clip 105 m away. What it cannot do is agree
    # with the other seeds about which way round the ground plane is -- the camera never
    # gets underneath the pitch, so that is one answer for the whole clip (D88).
    #
    # The primary is the reference because it is the one clicked without `--check`, and a
    # disagreeing extra is refused rather than flipped: `ft seed` flips at the point of
    # clicking, where a person is present to read the message. Reinterpreting a file this
    # far downstream would be the silent kind of fix this whole guard exists to prevent.
    opinionated = [row for row in good if row[2]]
    if len(opinionated) > 1:
        reference, against = opinionated[0][2], opinionated[0][0].name
        wrong = {path for path, _seeded, hand in opinionated if hand != reference}
        for path in sorted(wrong):
            bad.append(
                (
                    path,
                    f"it sees the pitch the other way round from {against}, which means these"
                    " clicks are on the OTHER GOAL - re-seed this frame and press 'e' for the"
                    " far end, or move this file out of the way",
                )
            )
        good = [row for row in good if row[0] not in wrong]
    return [seeded for _path, seeded, _hand in good], bad


def from_seeds(
    seeds: list[seed_mod.Seed],
    frames: list[int],
    frames_dir: Path,
    *,
    max_carry: int | None,
    motions: dict[int, Any] | None = None,
) -> dict[int, Any]:
    """Clicked frames, carried across the clip in both directions.

    `fill` already prefers a direct fit and carries only the gaps between them, so more
    seeds shorten every chain rather than adding a mechanism. Both directions matter - a
    clip is rarely best seeded at its first frame, because the camera is often still
    finding the play there.
    """
    return chain_from_seeds(
        seeds, frames, frames_dir, max_carry=max_carry, motions=motions
    ).homographies


def chain_from_seeds(
    seeds: list[seed_mod.Seed],
    frames: list[int],
    frames_dir: Path,
    *,
    max_carry: int | None,
    motions: dict[int, Any] | None = None,
) -> stage1_propagate.Chain:
    """`from_seeds`, with what the chain knows about itself kept: how far each frame is
    from an anchor, and where two anchors disagree. That is what says where to click next."""
    direct: dict[int, Any] = dict.fromkeys(frames)
    for seeded in seeds:
        h = seed_mod.homography(seeded)
        if h is not None and seeded.frame in direct:
            direct[seeded.frame] = h
    return stage1_propagate.fill(frames_dir, direct, max_carry=max_carry, motion=motions)


# How far a frame's own lines may land from the camera fitted to them, in metres, before
# the frame is left unsolved.
#
# Wider than a seed's own residual because the evidence is a model's guess at which paint
# is which: a mislabelled marking pulls the aim, and the aim has nowhere to hide it -- three
# numbers cannot warp to fit a bad correspondence the way eight can, so a bad frame shows
# up here as a plain miss.
CAMERA_GATE_M = 1.5

# How far apart two solved frames may be before the frames between them are left unsolved.
#
# Between two aims a second apart the camera panned smoothly and the frames between are
# between them; over a cut it did not, and the "pan" that would be interpolated is two
# different shots averaged. A second is long enough to bridge the stretches the segmenter
# blinks on and short enough that a cut is never spanned.
MAX_SPAN_FRAMES = 25


def _cached_aims(
    cache: Path, weights: Path, rig: camera_mod.Camera, gate: float
) -> dict[int, camera_mod.View] | None:
    """Aims read from a previous run, or None if they were made for something else.

    Keyed on the weights AND the camera, because moving the camera changes every aim -- and
    a stale aim here is indistinguishable from a frame the segmenter read differently.
    """
    try:
        stored = json.loads(cache.read_text())
    except (OSError, ValueError):
        return None
    if stored.get("weights") != str(weights) or stored.get("mtime") != weights.stat().st_mtime:
        return None
    if stored.get("gate") != gate or stored.get("camera") != [rig.x, rig.y, rig.height]:
        return None
    return {
        int(f): camera_mod.View(pan=v[0], tilt=v[1], focal=v[2]) for f, v in stored["views"].items()
    }


def _store_aims(
    cache: Path,
    weights: Path,
    rig: camera_mod.Camera,
    gate: float,
    views: dict[int, camera_mod.View],
) -> None:
    cache.write_text(
        json.dumps(
            {
                "version": 1,
                "weights": str(weights),
                "mtime": weights.stat().st_mtime,
                "gate": gate,
                "camera": [rig.x, rig.y, rig.height],
                "views": {str(f): [v.pan, v.tilt, v.focal] for f, v in sorted(views.items())},
            }
        )
        + "\n"
    )


# How far a clicked frame's own landmarks may land from the aim fitted to them, in metres.
#
# Tighter than the segmenter's gate because the evidence is exact: a landmark is a spot, not
# a pixel somewhere along a marking, and the clicked seeds of a real clip fit their match's
# camera to 0.13-0.21 m.
#
# It is also the end check. A camera in a known place cannot fit the far goal's markings to
# the near goal's names, so D88's invisible failure arrives here as a plain miss -- 1.63 m at
# the four landmarks `camera.MIN_SEED_CONSTRAINTS` asks for. The gate sits between what a
# clicked seed really costs and what a mirrored one cannot get under.
CLICK_GATE_M = 0.5


def aimed_seeds(
    work: Path,
    frames_dir: Path,
    rig: camera_mod.Camera,
    lens_at: tuple[float, float],
    gate: float = CLICK_GATE_M,
) -> tuple[list[seed_mod.Seed], list[tuple[Path, str]]]:
    """The clicked frames the match's camera can be aimed from, and the ones it cannot.

    The same files `usable_seeds` reads, judged against the camera instead of against
    themselves: two landmarks are enough (D96) where a homography needs four markings, and
    a seed too thin to fit a homography at all is still worth three numbers.
    """
    good: list[seed_mod.Seed] = []
    bad: list[tuple[Path, str]] = []
    for path in seed_paths(work):
        seeded = seed_mod.read(path)
        img = cv2.imread(str(frames_dir / f"{seeded.frame:06d}.jpg"))
        if img is None:
            bad.append((path, f"frame {seeded.frame} is not in the clip"))
            continue
        if seeded.image and seed_mod.unlike(seeded.image, seed_mod.fingerprint(img)) > (
            seed_mod.MAX_UNLIKE_BITS
        ):
            bad.append(
                (
                    path,
                    f"it was clicked on a different picture - frame {seeded.frame} of the clip"
                    " in this folder now is not the one these landmarks describe",
                )
            )
            continue
        got = camera_mod.aim_from_seed(rig, lens_at, seeded)
        if got is None:
            bad.append(
                (
                    path,
                    "too few clicks to aim the camera - four landmarks, or a couple of them"
                    " and some points traced along the markings",
                )
            )
        elif got[1] > gate:
            bad.append(
                (
                    path,
                    f"no aim of this match's camera fits these clicks: they miss by {got[1]:.1f} m."
                    " Check the far/near names against the diagram, and whether the goal in shot"
                    " is the other one",
                )
            )
        else:
            good.append(seeded)
    return good, bad


def camera_chain(
    rig: camera_mod.Camera,
    lens_at: tuple[float, float],
    views: dict[int, camera_mod.View],
    numbers: list[int],
    clicked: set[int],
    max_span: int = MAX_SPAN_FRAMES,
) -> stage1_propagate.Chain:
    """The clip's cameras as the wizard's timeline reads them.

    A frame is solved from its own evidence, spanned between two frames that were, or not
    solved at all -- which is what `guide.standing` colours. Distance here is frames from
    the nearest evidence, as it is for a carry, so the same words describe both.
    """
    filled = camera_mod.spanned(views, numbers, max_span)
    homs: dict[int, Any] = {
        f: camera_mod.matrix(rig, lens_at, filled[f]) if f in filled else None for f in numbers
    }
    reach: dict[int, int] = {}
    seen = sorted(views)
    for f in numbers:
        if homs.get(f) is None or not seen:
            continue
        reach[f] = 0 if f in clicked else min(abs(f - g) for g in seen) or 1
    return stage1_propagate.Chain(
        homographies=homs,
        solved_directly=len(views),
        carried=sum(1 for f in filled if f not in views),
        gaps=sum(1 for f in numbers if homs.get(f) is None),
        carried_from=reach,
    )


def camera_aims(
    frames_dir: Path,
    rig: camera_mod.Camera,
    lens_at: tuple[float, float],
    weights: Path | None = None,
    *,
    max_residual_m: float = CAMERA_GATE_M,
    cache: Path | None = None,
    progress: Callable[[int], None] | None = None,
) -> tuple[dict[int, camera_mod.View], list[int]]:
    """Where the camera was aimed at each frame it can be read from, and every frame there is.

    The slow half of `--mode camera`: a segmenter pass over the clip. Cached, because the
    wizard re-registers after every click and the lines do not change when somebody clicks.
    """
    from . import calib

    numbers = sorted(int(p.stem) for p in frames_dir.glob("*.jpg"))
    chosen = weights or calib.WEIGHTS
    if cache is not None and cache.exists():
        stored = _cached_aims(cache, chosen, rig, max_residual_m)
        if stored is not None:
            return stored, numbers

    net = calib.model(chosen).to(calib.device()).eval()
    views: dict[int, camera_mod.View] = {}
    last: camera_mod.View | None = None
    for f in numbers:
        image = cv2.imread(str(frames_dir / f"{f:06d}.jpg"))
        if progress is not None:
            progress(f)
        if image is None:
            continue
        mask = calib.predict(net, image)
        pairs = calib.pairs_from_mask(mask, image.shape[1], image.shape[0])
        arcs = calib.arcs_from_mask(mask, image.shape[1], image.shape[0])
        got = camera_mod.aim(rig, lens_at, pairs, arcs, start=last)
        if got is None or got[1] > max_residual_m:
            last = None
            continue
        views[f] = last = got[0]
    if cache is not None:
        _store_aims(cache, chosen, rig, max_residual_m, views)
    return views, numbers


def camera_homographies(
    frames_dir: Path,
    rig: camera_mod.Camera,
    lens_at: tuple[float, float],
    weights: Path | None = None,
    *,
    seeds: list[seed_mod.Seed] | None = None,
    max_residual_m: float = CAMERA_GATE_M,
    max_span: int = MAX_SPAN_FRAMES,
    cache: Path | None = None,
) -> dict[int, Any]:
    """Every frame's camera, from the segmenter's lines and a position already known.

    The clip need never be clicked. What a person supplied is somewhere else entirely --
    another clip of the same match, whose seeds said where the camera stands (D96) -- and
    each frame here only has to say where it is aimed.

    `seeds` are clicks on THIS clip, for the stretches the segmenter cannot read. They are
    evidence of the same kind and better: an exact landmark rather than a pixel somewhere
    along a marking, so a clicked frame's aim replaces a read one.

    Frames nothing explains are left unsolved, and the gaps short enough to be one pan are
    spanned from the aims either side.
    """
    views, numbers = camera_aims(
        frames_dir, rig, lens_at, weights, max_residual_m=max_residual_m, cache=cache
    )
    views = dict(views)
    for clicked in seeds or []:
        got = camera_mod.aim_from_seed(rig, lens_at, clicked)
        if got is not None:
            views[clicked.frame] = got[0]
    spanned = camera_mod.spanned(views, numbers, max_span)
    return {
        f: camera_mod.matrix(rig, lens_at, spanned[f]) if f in spanned else None for f in numbers
    }


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
    weights: Path | None = None,
    rig: camera_mod.Camera | None = None,
    lens_at: tuple[float, float] | None = None,
    aims_cache: Path | None = None,
) -> dict[int, Any]:
    """Per-frame homographies: from every frame's lines, from frame one's, or aimed off the
    match's own camera."""
    if mode == "camera":
        if rig is None or lens_at is None:
            raise ValueError("--mode camera needs the game's camera and the clip's lens centre")
        return camera_homographies(frames_dir, rig, lens_at, weights, cache=aims_cache)
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
        frames_dir, seeded, max_carry=max_carry, motion=motions
    ).homographies


def _restart_cell(cell: tuple[int, int], pitch: Pitch = DEFAULT_PITCH) -> bool:
    """Whether a square metre holds a restart spot, within the radius one is believed at."""
    x, y = (cell[0] + 0.5) * STATIC_BIN_M, (cell[1] + 0.5) * STATIC_BIN_M
    reach = RESTART_RADIUS_M + STATIC_BIN_M
    return any(math.hypot(x - sx, y - sy) <= reach for sx, sy in restart_spots(pitch))


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


def _ball_near_pitch(
    x: float, y: float, margin_m: float = BALL_MARGIN_M, pitch: Pitch = DEFAULT_PITCH
) -> bool:
    """`tracks.on_pitch`, but with a margin in METRES rather than a share of the pitch."""
    return -margin_m <= x <= pitch.length + margin_m and -margin_m <= y <= pitch.width + margin_m


def _is_static(b: detect.Sighting, h: Any, static: set[tuple[int, int]]) -> bool:
    x, y = calibration.to_pitch(h, b.x, b.y)
    return _bin_of(x, y) in static


def _smoothed(
    best: dict[int, tuple[float, float]],
    frames: list[int],
    smooth: int,
    pitch: Pitch = DEFAULT_PITCH,
) -> list[Sample]:
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
        if _ball_near_pitch(x, y, pitch=pitch):
            out.append(Sample(f=f, x=x, y=y))
    return out


def _restart_balls(
    per_frame: dict[int, list[detect.Sighting]],
    homs: dict[int, Any],
    static: set[tuple[int, int]],
    emitted: set[int],
    smooth: int,
    pitch: Pitch = DEFAULT_PITCH,
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
            d = min(math.hypot(x - sx, y - sy) for sx, sy in restart_spots(pitch))
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
    pitch: Pitch = DEFAULT_PITCH,
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
    first = _smoothed(best, frames, smooth, pitch)
    extra = _restart_balls(per_frame, homs, static, {s.f for s in first}, smooth, pitch)
    if not extra:
        return first
    return _smoothed({**best, **extra}, frames, smooth, pitch)


# The longest stretch between two believed sightings that is drawn as a straight line (D101).
#
# The ball goes unseen exactly when it travels -- struck, blurred, small -- and a board with no
# ball through a pass keeps it at the passer's feet. Filling those gaps took the share of the
# eleven benchmark clips' time with the right side on the ball from 60.7% to 65.2%, against the
# truth board frame by frame, and it needs no model: the sightings either side are already the
# ones the selector trusts. Two seconds is a long pass; past it the gap is more likely a stretch
# the ball spent somewhere the camera was not.
BALL_BRIDGE_S = 2.0


# How far either side of a sample a player's position is averaged over, in seconds.
#
# Two independent wobbles land on every position: the camera's aim is fitted per frame, and the
# detector's box moves a few pixels on a player who is standing still. Neither is motion, and
# together they are what a coach sees as *"the player dots are very twitchy"* on the dot video.
#
# In SECONDS because clips arrive at 25 to 49 fps and the noise is per frame. Short enough that
# it cannot invent or delay a run: the window is CENTRED, so a player moving at any constant
# speed comes out where he was, and only his acceleration is softened. Nothing is filled in --
# a sample with no neighbour inside the window is left exactly where it was, which is what keeps
# the gaps D8 insists on (a player nobody saw for twenty frames still has no position for them).
SETTLE_S = 0.12


def settle(samples: list[Sample], window: int) -> list[Sample]:
    """Each position averaged with the samples within `window` frames of it, and nothing else."""
    if window <= 0 or len(samples) < 3:
        return samples
    frames = [s.f for s in samples]
    out: list[Sample] = []
    for s in samples:
        lo = bisect_left(frames, s.f - window)
        hi = bisect_right(frames, s.f + window)
        near = samples[lo:hi]
        out.append(
            Sample(
                f=s.f,
                x=float(np.mean([n.x for n in near])),
                y=float(np.mean([n.y for n in near])),
                conf=s.conf,
            )
        )
    return out


def bridge(ball: list[Sample], max_gap: int) -> list[Sample]:
    """Each frame between two samples at most `max_gap` apart, on the line joining them.

    Nothing is added before the first sample or after the last: a line needs both ends.
    """
    out: list[Sample] = []
    for a, b in pairwise(ball):
        out.append(a)
        gap = b.f - a.f
        if 1 < gap <= max_gap:
            for f in range(a.f + 1, b.f):
                t = (f - a.f) / gap
                out.append(Sample(f=f, x=a.x + t * (b.x - a.x), y=a.y + t * (b.y - a.y)))
    return out + ball[-1:]


def build(
    frames_dir: Path,
    frames: list[int],
    detections: list[detect.Detection],
    homs: dict[int, Any],
    *,
    fps: float,
    motions: dict[int, Any] | None = None,
    balls: list[detect.Sighting] | None = None,
    appearance: dict[reid.Key, Any] | None = None,
    stitch: bool = True,
    pitch: Pitch = DEFAULT_PITCH,
    kits: tuple[Any, Any] | None = None,
) -> Result:
    """Detections plus a camera model -> tracks in pitch metres.

    Tracking runs FIRST, in stabilised image pixels, and projection happens after. The
    other order made stage 2 inherit every wobble in stage 1's homography, which is
    what breaks tracks all at once (D19). This way a drifting homography moves the
    positions and leaves the identities intact.

    `appearance` is `reid`'s embeddings by detection. Without it the stitcher never spends its
    contact slack, and the tracks are what they were before appearance existed (D95).

    `kits` is the match's stored (home, away) kit signatures, so `home` names the same team in
    every clip of it rather than whichever side stands nearer x=0 (D99).
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
        if not on_pitch(*calibration.to_pitch(h, *d.foot), PLAYER_MARGIN, pitch):
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
            if not on_pitch(x, y, pitch=pitch):
                dropped += 1
                continue
            samples.append(Sample(f=o.f, x=x, y=y, conf=o.det.score))
        if samples:
            positions[t.id] = samples

    # Before stitching, because a duplicate is not a fragment: it occupies the slot the
    # real continuation wants, and `stitch` refuses anything that overlaps in time anyway.
    # The shirt readings move with it, so the surviving track is named on both halves'
    # evidence rather than on whichever half happened to be longer (D90).
    held = {t.id: t for t in raw}
    same = stage2_stitch.duplicates(positions, fps)
    for gone, keep in same.items():
        if gone in held and keep in held:
            held[keep].absorb(held[gone])
    positions = stage2_stitch.merge(positions, same)

    if stitch:
        # After registration, because whether two fragments are one player is a question
        # about metres per second, and before team assignment, because a joined track
        # should be assigned once rather than voted on by its halves.
        #
        # On the whole track's kit rather than the rolling `color`: a fragment usually ends
        # because its player was lost in contact, so its last frames read two shirts, and
        # the rolling average is mostly those frames (D94).
        joined = stage2_stitch.joins(
            positions,
            {t.id: t.kit_mean for t in raw},
            fps,
            _ends(observations, raw, positions, same, appearance),
        )
        # And the shirt readings move with the samples, as they do for a duplicate. A fragment
        # folded into a track leaves the side clustering, and with its readings left behind
        # the cut moves for everybody else: one correct join on SNGS-066 flipped a 600-sample
        # track to the other side (D95).
        for gone, keep in joined.items():
            if gone in held and keep in held:
                held[keep].absorb(held[gone])
        positions = stage2_stitch.merge(positions, joined)

    mean_x = {tid: float(np.mean([s.x for s in ss])) for tid, ss in positions.items()}
    kept = [t for t in raw if t.id in positions]
    # Computed before the sides are named, because who had the ball is one of the things
    # that decides them (D91).
    ball = bridge(ball_path(balls or [], homs, frames, pitch=pitch), round(BALL_BRIDGE_S * fps))
    teams = assign(
        kept,
        mean_x,
        {tid: [s.f for s in ss] for tid, ss in positions.items()},
        pitch,
        on_the_ball(ball, positions, fps),
        {tid: float(np.mean([s.y for s in ss])) for tid, ss in positions.items()},
        kits,
    )
    positions, teams = _split_two_shirts(kept, positions, teams)
    # After the sides are named, because only `assign` knows which tracks hold a keeper.
    positions = stage2_stitch.merge(positions, stage2_stitch.keepers(positions, teams, fps, pitch))
    # Last, so every judgement above is made on what was measured and only the FILE is settled.
    positions = {tid: settle(ss, round(SETTLE_S * fps)) for tid, ss in positions.items()}

    return Result(
        ball=ball,
        kits=stage3_teams.kit_colours(kept, teams),
        signatures=stage3_teams.side_signatures(kept, teams),
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


# How close a player must be to the ball to be holding it, in metres, and for how long
# before the side he is on stops being a detail. Pitchboard asks the same two questions of
# the same file with the same answers (its CARRIER_RADIUS_M and MIN_ON_THE_BALL).
CARRIER_RADIUS_M = 2.0
ON_THE_BALL_S = 0.5


def on_the_ball(ball: list[Sample], positions: dict[int, list[Sample]], fps: float) -> set[int]:
    """The tracks the ball went through, by id.

    Nearest player to the ball and within reach of it, counted over the frames the ball
    was actually located. Not a claim that they controlled it -- only that the move went
    through them, which is what makes their side worth more than the usual caution.
    """
    if not ball:
        return set()
    at: dict[int, dict[int, Sample]] = {tid: {s.f: s for s in ss} for tid, ss in positions.items()}
    held: dict[int, int] = {}
    for b in ball:
        best, near = None, CARRIER_RADIUS_M
        for tid, byf in at.items():
            s = byf.get(b.f)
            if s is None:
                continue
            d = float(np.hypot(s.x - b.x, s.y - b.y))
            if d < near:
                best, near = tid, d
        if best is not None:
            held[best] = held.get(best, 0) + 1
    floor = max(1, round(ON_THE_BALL_S * fps))
    return {tid for tid, n in held.items() if n >= floor}


def _ends(
    observations: dict[int, list[stage2_track.Observation]],
    raw: list[stage2_track.Track],
    positions: dict[int, list[Sample]],
    same: dict[int, int],
    appearance: dict[reid.Key, Any] | None,
) -> dict[int, stage2_stitch.Ends]:
    """Each track's contact and look at either end, for the stitcher.

    Looked up after duplicates are merged, so an end sample may belong to an absorbed track; a
    survivor's own box is preferred, as `stage2_stitch.merge` prefers its sample. A look is only
    taken from crops nobody else covers: the crop of a box holding two men is both of them.
    """
    boxes: dict[tuple[int, int], detect.Detection] = {}
    for t in sorted(raw, key=lambda t: t.id in same):
        owner = t.id
        while owner in same:
            owner = same[owner]
        for o in t.observations:
            boxes.setdefault((owner, o.f), o.det)

    def covered(tid: int, f: int) -> float | None:
        det = boxes.get((tid, f))
        if det is None:
            return None
        return stage2_track.covered(det, [o.det for o in observations.get(f, [])])

    def look(tid: int, samples: list[Sample]) -> Any:
        if appearance is None:
            return None
        seen = []
        for s in samples:
            cover = covered(tid, s.f)
            if cover is None or cover >= stage2_stitch.CONTACT_COVER:
                continue
            v = appearance.get(reid.key(boxes[(tid, s.f)]))
            if v is not None:
                seen.append(v)
                if len(seen) == reid.GALLERY:
                    break
        return reid.look(seen)

    return {
        tid: stage2_stitch.Ends(
            found_in_contact=(covered(tid, ss[0].f) or 0.0) >= stage2_stitch.CONTACT_COVER,
            lost_in_contact=(covered(tid, ss[-1].f) or 0.0) >= stage2_stitch.CONTACT_COVER,
            look_first=look(tid, ss),
            look_last=look(tid, ss[::-1]),
        )
        for tid, ss in positions.items()
    }


def _split_two_shirts(
    tracks: list[stage2_track.Track],
    positions: dict[int, list[Sample]],
    teams: dict[int, TeamLabel],
) -> tuple[dict[int, list[Sample]], dict[int, TeamLabel]]:
    """Cut a DECLINED track in two where its shirt changes, if both halves then have a side.

    Only the declined ones, and that asymmetry is the whole argument (D85): a track stage 3
    could not name is dropped by the board, so a split that explains it costs nothing when
    it fails and returns two players when it works. Tried on every track it would need a
    threshold nobody has (D84).
    """
    centres = stage3_teams.side_centres(tracks, teams)
    if centres is None:
        return positions, teams

    out_positions = dict(positions)
    out_teams = dict(teams)
    next_id = max(positions, default=0) + 1
    sides: tuple[TeamLabel, TeamLabel] = ("home", "away")
    for track in tracks:
        if out_teams.get(track.id) != "unknown":
            continue
        split = stage3_teams.two_shirts(track, centres)
        if split is None:
            continue
        at, first, second = split
        early = [s for s in positions[track.id] if s.f < at]
        late = [s for s in positions[track.id] if s.f >= at]
        if not early or not late:
            continue
        out_positions[track.id] = early
        out_teams[track.id] = sides[first]
        out_positions[next_id] = late
        out_teams[next_id] = sides[second]
        next_id += 1
    return out_positions, out_teams

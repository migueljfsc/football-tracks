"""Where a camera chain is weakest, and what to click next.

A seeded clip is right at its anchors and wrong a hundred frames later (D18), and the one
thing a coach can act on is WHERE. Two numbers answer it, and they answer different
questions: how far a frame is from an anchor is available on any clip and is the only guide
when there is one seed; where two anchors reach the same frame from opposite directions,
their disagreement is drift MEASURED rather than assumed, because the two chains accumulated
it independently (D80, D83).

Kept out of `cli.py` because two commands act on the same answer -- `ft calibrate` prints it
for a person to follow, and `ft run` walks to the frame it names without being asked. A
second copy of this arithmetic would let those two disagree about where the clip is worst.

Two audiences read it. `weakest` is for whoever runs the pipeline and speaks its language;
`standing`, `describe`, `overview` and `refusal` are for whoever clicks, who should not need
to know what a homography is to be told the camera was lost.
"""

from __future__ import annotations

from typing import Any, Literal

from . import seed as seed_mod
from . import stage1_propagate

# What a frame's camera model is worth, as the scrubber's timeline shows it.
Standing = Literal["clicked", "good", "fair", "poor", "lost"]

# Where two anchors reach a frame from opposite sides, their disagreement is the drift
# between them (D80). Two and five metres are the radii every registration number here is
# reported at -- `ft reg-eval` and the stage funnel in PLAN.md -- so a colour on the timeline
# means what those numbers mean.
GOOD_APART_M = 2.0
FAIR_APART_M = 5.0

# Where only one anchor reaches, distance is all there is. A carry held inside ~1.6 m out to
# 120 frames on SNGS-147 and degraded sharply after; the default cap sits well inside that.
GOOD_REACH = stage1_propagate.DEFAULT_MAX_CARRY
FAIR_REACH = 120

# How much a click would help, by standing, for choosing where to suggest one.
SEVERITY: dict[str, int] = {"clicked": 0, "good": 1, "fair": 2, "poor": 3, "lost": 4}


def runs(frames: list[int]) -> list[tuple[int, int]]:
    """Consecutive frames, grouped."""
    out: list[tuple[int, int]] = []
    for f in sorted(frames):
        if out and f == out[-1][1] + 1:
            out[-1] = (out[-1][0], f)
        else:
            out.append((f, f))
    return out


def stretches(chain: stage1_propagate.Chain) -> list[tuple[int, int, Standing]]:
    """The clip as consecutive runs of one standing: first frame, last frame, standing."""
    out: list[tuple[int, int, Standing]] = []
    for f in sorted(chain.homographies):
        kind = standing(chain, f)
        if out and out[-1][2] == kind and f == out[-1][1] + 1:
            out[-1] = (out[-1][0], f, kind)
        else:
            out.append((f, f, kind))
    return out


def next_seed(
    chain: stage1_propagate.Chain | None, avoid: frozenset[int] | set[int] = frozenset()
) -> tuple[int, str] | None:
    """The frame another seed would help most, and why it is that one.

    The worst STANDING first, so the suggestion and the timeline's colours never disagree.
    Any measured disagreement used to outrank any distance, which sent nottingham's second
    click to two anchors 2.1 m apart while 732 frames hung off one anchor carried up to 852
    frames -- and one seed carried that far lands tens of metres out (D34). Within a
    standing, measured disagreement comes before distance, since it is evidence rather than
    presumption. A stretch with no homography at all is worst of all, and is clicked in its
    middle, which reaches both ends of it.

    `avoid` holds frames somebody tried to click and could not, and the whole stretch each
    one sits in is passed over: the camera angle that defeated one frame of a stretch
    defeats its neighbours, and suggesting the frame next door is the same dead end.

    None when there is nothing to say: no chain, or nothing lost and nothing carried
    outside the stretches passed over.
    """
    if chain is None:
        return None
    passed = {
        f
        for first, last, _kind in stretches(chain)
        if any(first <= a <= last for a in avoid)
        for f in range(first, last + 1)
    }
    lost = [f for f, h in chain.homographies.items() if h is None and f not in passed]
    if lost:
        start, end = max(runs(lost), key=lambda r: r[1] - r[0])
        return (start + end) // 2, (
            f"no homography for {end - start + 1} frames here - the chain breaks"
        )
    carried = [f for f, r in chain.carried_from.items() if r > 0 and f not in passed]
    if not carried:
        return None

    def rank(f: int) -> tuple[int, bool, float]:
        apart = chain.disagreement.get(f)
        measured = apart is not None
        return (
            SEVERITY[standing(chain, f)],
            measured,
            apart if apart is not None else chain.carried_from[f],
        )

    frame = max(carried, key=rank)
    apart = chain.disagreement.get(frame)
    if apart is not None:
        return frame, f"two anchors disagree by {apart:.1f} m here, which is the drift between them"
    return frame, f"carried {chain.carried_from[frame]} frames from the nearest seed"


def weakest(clip: str, chain: stage1_propagate.Chain | None, homs: dict[int, Any]) -> list[str]:
    """What this clip's camera model cannot support, as lines to print.

    The stretches with no homography at all come first because they are a different
    problem from drift: no seed fixes a cut.
    """
    lines: list[str] = []
    missing = [f for f, h in homs.items() if h is None]
    for start, end in runs(missing):
        span = f"{start}-{end}" if end > start else f"{start}"
        lines.append(
            f"no homography     {span} ({end - start + 1} frames)"
            " - a cut, a whip pan, or too little texture to track"
        )
    if chain is None or not chain.carried_from:
        return lines

    frame = max(chain.carried_from, key=lambda f: chain.carried_from[f])
    reach = chain.carried_from[frame]
    if reach == 0:
        return lines
    lines.append(f"weakest           frame {frame}, carried {reach} frames from the nearest seed")
    if chain.disagreement:
        worst = max(chain.disagreement, key=lambda f: chain.disagreement[f])
        lines.append(
            f"anchors disagree  by {chain.disagreement[worst]:.1f} m at frame {worst},"
            " which is the drift between them"
        )
    nxt = next_seed(chain)
    if nxt is not None:
        lines.append(f"seed it           ft seed {clip} --frame {nxt[0]} --check")
    return lines


def standing(chain: stage1_propagate.Chain, f: int) -> Standing:
    """What the camera model at one frame is worth.

    Disagreement outranks distance wherever both exist, because it is measured: two chains
    that agree after two hundred frames each are better evidence than one chain after fifty.
    """
    reach = chain.carried_from.get(f)
    if chain.homographies.get(f) is None or reach is None:
        return "lost"
    if reach == 0:
        return "clicked"
    apart = chain.disagreement.get(f)
    if apart is not None:
        return "good" if apart < GOOD_APART_M else "fair" if apart < FAIR_APART_M else "poor"
    return "good" if reach <= GOOD_REACH else "fair" if reach <= FAIR_REACH else "poor"


def describe(chain: stage1_propagate.Chain, f: int) -> str:
    """One frame's camera model, said to whoever is clicking."""
    kind = standing(chain, f)
    if kind == "clicked":
        return "you clicked the pitch on this frame"
    if kind == "lost":
        return "the camera could not be followed here - clicking this frame fixes it"
    apart = chain.disagreement.get(f)
    said = (
        f"your clicks either side disagree by {apart:.1f} m here"
        if apart is not None
        else f"followed {chain.carried_from[f]} frames from the nearest click"
    )
    if kind == "poor":
        return f"{said} - click this frame to fix it"
    if kind == "fair":
        return f"{said} - check the lines closely"
    return said


def overview(chain: stage1_propagate.Chain) -> str:
    """The whole chain in one line, for the window and for the run's summary."""
    clicked = sum(1 for r in chain.carried_from.values() if r == 0)
    total = len(chain.homographies)
    found = sum(1 for h in chain.homographies.values() if h is not None)
    text = (
        f"{clicked} frame{'' if clicked == 1 else 's'} clicked,"
        f" camera followed on {found / max(1, total):.0%} of the clip"
    )
    if chain.disagreement:
        text += f", clicks agree within {max(chain.disagreement.values()):.1f} m"
    return text


def refusal(seeded: seed_mod.Seed, width: int, height: int) -> str:
    """Why a seed that was just clicked cannot be used, said to whoever clicked it.

    Worked out by elimination over `auto.usable_seeds`' refusals, which name the problem in
    the pipeline's terms. A fresh click was made on this frame of this clip, so the frame is
    present and the fingerprint matches; what is left is no camera fitting at all, a fit that
    folds behind the camera, or the other goal (D88) -- in that order, because each later
    check needs the earlier one to have passed.

    Two lines: what went wrong, then what can be done -- and every message ends in the way
    out, because some camera angles offer nothing more to click and a person must never be
    left with only the instruction they already cannot follow.

    A fit folded through the horizon used to be told to "add some lower in the picture",
    which on a camera aimed at the far half of the pitch is exactly what does not exist.
    The curves reach down the frame where the straight markings do not, so that is what
    it points at.
    """
    skip = "Q: skip this frame."
    h = seed_mod.homography(seeded)
    if h is None:
        if seeded.arcs and seeded.start is None:
            return (
                "A circle needs a first camera to start from, and there is none yet.\n"
                f"Click a frame with more straight lines in view first. {skip}"
            )
        return (
            "These clicks don't fit a camera. Check each name against the diagram.\n"
            f"Trace a line or a circle that crosses the others. {skip}"
        )
    if seed_mod.behind_camera(h, width, height) > seed_mod.MAX_BEHIND_CAMERA:
        if seeded.arcs:
            return (
                "Still not enough depth to place the camera.\n"
                f"Trace more of the circle, or anything lower in the picture. {skip}"
            )
        return (
            "Every click is in one band of the picture, so its depth can't be worked out.\n"
            f"Trace the centre circle or a penalty arc if one is in view (T, then N). {skip}"
        )
    return (
        "These clicks are on the other goal from your earlier ones.\n"
        f"Press E, then click again. {skip}"
    )

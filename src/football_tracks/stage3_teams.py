"""Stage 3 - which side is each track on.

Two kits, told apart on the shirt signature the tracker already maintains. Written out
rather than imported: the input is a dozen vectors, and sklearn is a large dependency to
add for thirty lines.

Which cluster is `home` is NOT decided by looking at the answer. It is decided by mean
pitch x - the side whose players average nearer x=0 defends the left goal - which is
SoccerNet's own left/right convention and is deterministic. Picking the labelling that
happens to score best would be fitting to the yardstick.

A keeper wears neither kit, and left in the clustering they cost real accuracy: on
SNGS-147 the two sides come out 83% right with keepers included and 93% without. So
they are taken out and put back. What identifies one without being told is the two
things at once - a colour unlike either team, and standing near a goal.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
import numpy.typing as npt

from .config import DEFAULT_PITCH, Pitch
from .stage2_track import Track
from .tracks import TeamLabel

Vec = npt.NDArray[np.float64]
# How far from its cluster's centre a kit may sit, as a multiple of the median distance,
# before it is treated as neither team's. A goalkeeper is the case this exists for.
OUTLIER_RATIO = 2.0

# How near a goal line a track's average position must be, in metres, for an odd-coloured
# track to be read as that goal's keeper rather than a player in a strange light.
KEEPER_ZONE_M = 22.0

# Most officials that may be picked out of one clip. A match has three on the pitch and
# rarely more than two in shot, and the cap is what stops a player in strange light being
# deleted: an odd kit is only read as an official while there is a slot left for one.
MAX_REFEREES = 3

# Most tracks of one side that may be in shot at once before the split is read as having
# collapsed. A team fields eleven; the slack covers a keeper that was not pulled out as an
# outlier, somebody on the touchline, and two fragments either side of a one-frame break.
MAX_CONCURRENT_PER_SIDE = 12

# Principal axes searched for a cut a pitch allows. The first is the direction of greatest
# variance, which is the kits when the kits are what varies most and the light when they
# are not. Only reached when a feasibility test is supplied, because without one there is
# nothing to tell those two cases apart.
KIT_AXES = 3


def split_kits(
    points: Vec,
    crowding: Callable[[npt.NDArray[np.int_]], int] | None = None,
) -> npt.NDArray[np.int_]:
    """Two groups of kits: project onto the axis they differ along, and cut.

    NOT k-means, which was the first version and collapses. k-means minimises inertia,
    and when the data is not cleanly bimodal — which it is not, because a dozen tracks
    of the same kit vary more in light and pose than two kits differ — the cheapest
    split is one tight little cluster and one holding everybody else. Measured on
    SNGS-147 it put 44 tracks on one side and 8 on the other, 70% right, and six
    spurious tracks were enough to flip it.

    Instead the kits are projected onto their axis of greatest variance, which is the
    direction the two teams differ along, and cut where the between-class variance is
    greatest. That is Otsu's method, and the `len(a) * len(b)` in its score is exactly
    what stops one side swallowing the other. Same data: 26 and 26, 85% right.

    Exhaustive over cut points and so reproducible. A pipeline that relabels the teams
    on a rerun is one nobody can check.

    `crowding` scores a labelling on something colour cannot see: how many players over
    what a pitch holds the worst moment puts on one side. Zero is allowed, and the
    best-scoring cut that manages it wins. Between-class variance resists one side
    swallowing the other but does not forbid it -- a handful of tracks far enough along
    the axis outscores an even cut, and the result is every player on one team.

    When no cut manages zero, the LEAST crowded one stands. Falling back to the
    highest-scoring cut instead returns the collapsed answer this exists to refuse,
    which is what SNGS-060 got.

    The search runs over the top `KIT_AXES` components rather than the first alone. The
    largest axis of variance is the kits only when the kits are what varies most; when it
    is the light instead, EVERY cut along it is lopsided and choosing between them cannot
    help -- measured on SNGS-060, where no cut on the first component left fewer than
    twenty players on one side. Scores are compared raw across axes, so the first
    component wins wherever it has a feasible cut and the others are reached only when it
    does not.
    """
    n = len(points)
    if n < 2:
        return np.zeros(n, dtype=np.int_)

    centred = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    # Without a feasibility test there is no way to prefer one axis over another, so the
    # unconstrained answer stays what it always was: the first component.
    axes = min(KIT_AXES, len(vt)) if crowding is not None else 1

    orders, scored = [], []
    for ax in range(axes):
        projected = centred @ vt[ax]
        order = np.argsort(projected)
        orders.append(order)
        for i in range(1, n):
            a, b = projected[order[:i]], projected[order[i:]]
            score = len(a) * len(b) * (float(a.mean()) - float(b.mean())) ** 2
            scored.append((score, ax, i))
    # Ties keep the earliest axis and the lowest cut, which is what the running maximum
    # this replaced did.
    scored.sort(key=lambda s: (-s[0], s[1], s[2]))

    def labelling(ax: int, cut: int) -> npt.NDArray[np.int_]:
        labels = np.zeros(n, dtype=np.int_)
        labels[orders[ax][cut:]] = 1
        return labels

    if crowding is None:
        return labelling(scored[0][1], scored[0][2])

    best, over = None, None
    for _, ax, cut in scored:
        labels = labelling(ax, cut)
        crowd = crowding(labels)
        if crowd == 0:
            return labels
        if over is None or crowd < over:
            best, over = labels, crowd
    return best if best is not None else labelling(scored[0][1], scored[0][2])


def _oddness(points: Vec, labels: npt.NDArray[np.int_]) -> Vec:
    """How far each kit sits from its own group, as a multiple of the median distance.

    Returned as the ratio rather than a verdict because two things are decided from it and
    they need different amounts of it: a goalkeeper is whichever odd kit stands in a goal,
    and an official is the oddest few that do not.
    """
    centres = np.array(
        [
            points[labels == k].mean(axis=0) if np.any(labels == k) else points.mean(axis=0)
            for k in (0, 1)
        ]
    )
    d = np.linalg.norm(points - centres[labels], axis=1)
    median = float(np.median(d))
    if median <= 0:
        return np.zeros(len(points), dtype=np.float64)
    return np.asarray(d / median, dtype=np.float64)


def _crowding(
    tracks: list[Track], frames: dict[int, list[int]] | None
) -> Callable[[npt.NDArray[np.int_]], int] | None:
    """How many players over a team's worth a labelling puts on one side at once.

    Fragments of one player never overlap in time -- the tracker ends one and starts the
    next -- so the number of tracks carrying a sample at a frame is a number of people.
    Twenty of them on one side is not a close call about kit colour; it is the cut having
    collapsed, and it is visible without any ground truth.

    Frames come from the stitched pitch samples where the caller has them, because a
    joined track's own observations are only the half the tracker kept.
    """
    seen = (
        [set(frames.get(t.id, ())) for t in tracks]
        if frames is not None
        else [{o.f for o in t.observations} for t in tracks]
    )
    every = sorted(set().union(*seen)) if seen else []
    if not every:
        return None

    at = {f: i for i, f in enumerate(every)}
    present = np.zeros((len(tracks), len(every)), dtype=np.int_)
    for i, fs in enumerate(seen):
        for f in fs:
            present[i, at[f]] = 1

    def crowd(labels: npt.NDArray[np.int_]) -> int:
        worst = max(int(np.max(present[labels == k].sum(axis=0), initial=0)) for k in (0, 1))
        return max(0, worst - MAX_CONCURRENT_PER_SIDE)

    return crowd


# How much closer to its own side's kit than to the other's a track has to be.
#
# A kit sitting between the two is a coin flip, and a coin flip reaches the board as a
# player in the wrong colour -- which a coach reads as the wrong team making the pass,
# because that is exactly what it draws. The same rule as a shirt number nobody could
# read (D5): the answer is that there is no answer, and the importer already drops a
# track whose side could not be told.
KIT_MARGIN = 0.8


# How far apart two sides' measured shirt colours have to be, in BGR, before a board is
# told to use them.
#
# A kit colour is only worth carrying if it tells the two sides apart on sight, and the
# average of a torso crop is a blunt instrument: floodlights, motion blur and a white
# sleeve all pull it towards grey. Where the two answers come out close, the honest thing
# is to say nothing and let the board keep its own two colours, which are at least
# guaranteed to differ.
KIT_TONE_APART = 60.0


# How much of a side's signature must agree before its colour is worth writing down.
#
# Measured across five clips: every answer that came out right carries 0.56 to 0.83 in one
# category, and the one that came out yellow carries 0.41 against 0.25.
KIT_CONFIDENT = 0.45

# How far apart the two painted kits must be, as BGR distance, before they are worth showing.
KIT_APART = 60.0


# The hue bins the PITCH occupies, of the twelve the signature keeps. `GREEN_LO`/`GREEN_HI`
# put grass at 35-85 of OpenCV's 0-179, which is bins 2 through 5.
#
# They cannot name a kit, because every torso crop contains them: a box is part shirt and
# part pitch, and for a dark kit the pitch is the larger share. Read without this,
# SNGS-147's red-and-black team came out green (D92).
#
# The cost is a genuinely green shirt, which is then painted by whatever else it has --
# Sporting's green-and-white hoops come out white. That is the honest answer: this cannot
# tell a green shirt from the grass behind it, so it does not claim green.
GRASS_HUES = range(2, 6)


def _paint(mine: Vec, theirs: Vec, tone: Vec) -> str | None:
    """One side's shirt colour, from the signature that NAMES the sides.

    D81 read this off a mean of the torso instead, and could not tell two sides apart that
    a camera plainly could: averaging a red shirt with the grass and the shorts around it
    gives a dull olive, and so does averaging a green one. The histogram never mixed them,
    which is why it can name the sides at all -- so the colour comes from its hues (D92).

    Three things had to be right, and each was found by getting it wrong:

    * **The pitch's hues are struck out**, because every crop contains them: a box is part
      shirt and part grass, and for a dark kit the grass is the larger share. Read without
      this, SNGS-147's red-and-black team came out green.
    * **Colourless beats colour on the TOTAL**, not on the biggest single hue. What is left
      of a white shirt after the grass goes is skin and trim, and either will out-argmax a
      bin. Sporting's green-and-white hoops came out amber that way.
    * **The hue is a circular MEAN, not an argmax.** Red wraps: it sits in the first bin and
      the last one, so counting bins separately splits it in half and hands the kit to
      whatever is merely contiguous.

    Brightness comes from the mean, for the kits with no hue to read. The signature gathers
    every colourless pixel into one bin (D87), so white and black look identical to it, and
    an average is perfectly good at telling those two apart.

    `theirs` is kept because the first attempt subtracted it -- both sides stand on the same
    grass, so their difference cancels it. Measured, that moved the hue without fixing it:
    SNGS-147's red team went from green to amber and Sporting's hoops to yellow.
    """
    del theirs
    hue = mine[:48].reshape(12, 4).sum(axis=1)
    for b in GRASS_HUES:
        hue[b] = 0.0
    colour, grey = float(hue.sum()), float(mine[48])
    # Neither half of this is a preference. A kit is what most of a shirt is, so the answer
    # is only worth writing down when most of the shirt agrees -- and where it does not, the
    # board's own palette beats a colour nobody is wearing (D72, D81, and the rule D5 applies
    # to a shirt number). Sporting's green-and-white hoops are the case that needs it: their
    # green IS the pitch's green and is struck out by definition, leaving 41% colourless
    # against 25% of everything else. Painted anyway, they came out yellow.
    if max(colour, grey) < KIT_CONFIDENT:
        return None
    if grey > colour:
        return "#e6e6e6" if float(np.mean(tone)) > 128 else "#2b2b2b"

    # OpenCV's hue is 0..179 for the whole circle, and each bin is fifteen of it.
    middles = (np.arange(12) * 15 + 7.5) / 180.0 * 2 * np.pi
    turn = np.arctan2(float((hue * np.sin(middles)).sum()), float((hue * np.cos(middles)).sum()))
    mean_hue = round(turn % (2 * np.pi) / (2 * np.pi) * 180) % 180
    worn = np.array(
        [[[mean_hue, round(KIT_SATURATION * 255), round(KIT_VALUE * 255)]]], dtype=np.uint8
    )
    b, g, r = (int(v) for v in cv2.cvtColor(worn, cv2.COLOR_HSV2BGR)[0, 0])
    return f"#{r:02x}{g:02x}{b:02x}"


def _rgb(hex_colour: str) -> Vec:
    return np.array([int(hex_colour[i : i + 2], 16) for i in (1, 3, 5)], dtype=np.float64)


def kit_colours(tracks: list[Track], teams: dict[int, TeamLabel]) -> dict[str, str] | None:
    """The two sides' shirt colours, as hex, or None where they cannot be told apart.

    The MEDIAN tone across a side's tracks and the MEAN of their signatures: one track
    holding two players (D78) or a keeper mislabelled as an outfielder is a whole shirt of
    the wrong colour, and neither a median of a dozen nor a mean of their histograms lets
    one of them decide the team's colour.
    """
    seen: dict[str, list[tuple[Vec, Vec]]] = {"home": [], "away": []}
    for t in tracks:
        side = teams.get(t.id)
        if side not in seen or t.tone_mean is None or t.side_mean is None:
            continue
        seen[side].append((t.side_mean, t.tone_mean))
    if not (seen["home"] and seen["away"]):
        return None

    signature = {
        side: np.mean(np.array([sig for sig, _tone in rows], dtype=np.float64), axis=0)
        for side, rows in seen.items()
    }
    worn = {
        side: _paint(
            signature[side],
            signature["away" if side == "home" else "home"],
            np.median(np.array([tone for _sig, tone in rows], dtype=np.float64), axis=0),
        )
        for side, rows in seen.items()
    }
    # Both or neither. A board wearing one real kit and one from its own palette is harder
    # to read than one wearing two of its own, because only one of them means anything.
    if worn["home"] is None or worn["away"] is None:
        return None
    if float(np.linalg.norm(_rgb(worn["home"]) - _rgb(worn["away"]))) < KIT_APART:
        return None
    return {side: colour for side, colour in worn.items() if colour is not None}


# What a shirt colour is raised to before it is written down, as HSV fractions.
#
# The average of a torso crop is the kit mixed with everything else in the box -- shadow,
# skin, a white sleeve, the grass showing between an arm and a body -- so a red shirt
# measures as a dull salmon and a blue one as slate. Those are true averages and bad
# COLOURS: a board painted in them is two greys, which is worse than the palette it
# replaced. The hue survives all that mixing, so the hue is kept and the shirt is
# restated at the saturation and brightness a kit actually has.
KIT_SATURATION = 0.72
KIT_VALUE = 0.82

# Below this much saturation a shirt has no colour, only a brightness: white, grey, black.
KIT_ACHROMATIC = 0.18


def _hex(bgr: Vec) -> str:
    pixel = np.array([[np.clip(bgr, 0, 255)]], dtype=np.uint8)
    hue, sat, val = (int(v) for v in cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0])
    # A white, grey or black kit has no hue to keep -- what little it measures is noise,
    # and lifting the saturation of noise paints the team a colour nobody is wearing. It
    # gets a light or dark neutral instead, which is what it actually looks like.
    if sat < KIT_ACHROMATIC * 255:
        return "#e6e6e6" if val > 128 else "#2b2b2b"
    lifted = np.array(
        [[[hue, max(sat, round(KIT_SATURATION * 255)), max(val, round(KIT_VALUE * 255))]]],
        dtype=np.uint8,
    )
    b, g, r = (int(v) for v in cv2.cvtColor(lifted, cv2.COLOR_HSV2BGR)[0, 0])
    return f"#{r:02x}{g:02x}{b:02x}"


# Fewest shirt readings a half must have before a split is believed. Below this a "half"
# is a handful of frames of one player walking through another's box.
MIN_HALF_READS = 8


def two_shirts(
    track: Track, centres: tuple[Vec, Vec], margin: float = KIT_MARGIN
) -> tuple[int, int, int] | None:
    """Where a track changes shirt, and which side each half is on -- or None.

    Only ever asked of a track stage 3 has already DECLINED (D72), and that asymmetry is
    what makes it safe. A declined track is thrown away: half of it would be in the wrong
    colour and the board fields nobody it cannot name. So the question here is not "is this
    track above some threshold of suspicion" -- D84 measured that and found no threshold
    that separates a switch from an ambiguous kit -- but "does cutting it produce two halves
    the ordinary test is SURE of, one on each side". If it does, two players come back that
    were otherwise lost. If it does not, the track stays declined and nothing is worse.

    Found on a coach's clip: the man who received a goalkeeper's pass was a track holding a
    Porto shirt to frame 87 and a Manchester City one after it, so his kit sat exactly
    between the two sides -- own 0.24, other 0.24 -- and the move he was in the middle of
    could not be drawn.
    """
    log = track.kit_log
    if len(log) < 2 * MIN_HALF_READS:
        return None

    best: tuple[float, int] | None = None
    for i in range(MIN_HALF_READS, len(log) - MIN_HALF_READS):
        a = np.mean([k for _, k in log[:i]], axis=0)
        b = np.mean([k for _, k in log[i:]], axis=0)
        apart = _intersection_distance(a, b)
        if best is None or apart > best[0]:
            best = (apart, i)
    if best is None:
        return None

    cut = best[1]
    sides = []
    for half in (log[:cut], log[cut:]):
        kit = np.mean([k for _, k in half], axis=0)
        near = [float(np.linalg.norm(kit - c)) for c in centres]
        first, second = (0, 1) if near[0] <= near[1] else (1, 0)
        # The same margin the whole track failed, applied to each half: a cut that leaves
        # two kits still sitting between the sides has explained nothing.
        if near[first] > margin * near[second]:
            return None
        sides.append(first)
    if sides[0] == sides[1]:
        return None  # one player who changed light, not two players
    return (log[cut][0], sides[0], sides[1])


def _intersection_distance(a: Vec, b: Vec) -> float:
    """0 when two kit signatures agree, 1 when they share nothing."""
    return float(np.clip(1.0 - np.minimum(a, b).sum(), 0.0, 1.0))


def side_centres(tracks: list[Track], teams: dict[int, TeamLabel]) -> tuple[Vec, Vec] | None:
    """The mean kit of each side, from the tracks the split was sure of."""
    kits: dict[str, list[Vec]] = {"home": [], "away": []}
    for t in tracks:
        side = teams.get(t.id)
        if side in kits and t.kit_mean is not None:
            kits[side].append(t.kit_mean)
    if not (kits["home"] and kits["away"]):
        return None
    return (
        np.mean(np.array(kits["home"], dtype=np.float64), axis=0),
        np.mean(np.array(kits["away"], dtype=np.float64), axis=0),
    )


def assign(
    tracks: list[Track],
    mean_x: dict[int, float],
    frames: dict[int, list[int]] | None = None,
    pitch: Pitch = DEFAULT_PITCH,
    carried: set[int] | None = None,
) -> dict[int, TeamLabel]:
    """Track id -> team label.

    Clustered on `side_mean` and not `color`: the tracker's rolling average is about the
    next frame's match, and which team a track is on is about the whole track. And on the
    colourless pixels left out rather than the whole crop (D87), because a hue read off
    white is noise and noise pools in whichever bin it favours.

    `mean_x` is each track's average position along the pitch, in metres, which is what
    decides which cluster is which. Tracks with no colour signature at all come back
    as "unknown" rather than being guessed into a side (D5's rule, applied to teams).

    Keepers are excluded from the clustering and then placed by the goal they stand in,
    which is both more accurate and the only way to label them `gkHome`/`gkAway` at all.

    `frames` is the frames each track holds a sample at, which is what tells a collapsed
    split from a real one. It is optional only so a caller with nothing but tracks still
    gets a labelling; the observations are a poorer answer once fragments are stitched.
    """
    usable = [t for t in tracks if t.side_mean is not None and t.id in mean_x]
    if len(usable) < 2:
        return {t.id: "unknown" for t in tracks}

    points = np.array([t.side_mean for t in usable], dtype=np.float64)
    first = split_kits(points)

    # A kit far from both teams, standing near a goal, is that goal's keeper. Both
    # halves matter: colour alone catches a player in odd light, and position alone
    # catches every defender on a goal line.
    oddness = _oddness(points, first)
    odd = oddness > OUTLIER_RATIO
    in_a_goal = np.array(
        [
            mean_x[t.id] <= KEEPER_ZONE_M or mean_x[t.id] >= pitch.length - KEEPER_ZONE_M
            for t in usable
        ]
    )
    keeper = odd & in_a_goal

    # An odd kit standing where no keeper stands is an official. They are the one thing on
    # the pitch wearing neither team's colours and not tied to a goal, and left unnamed
    # they reach the board as a player a coach has to delete. Only the oddest few, because
    # colour alone also catches a player in strange light -- the same reason the keeper
    # test asks for position as well.
    #
    # They stay in the clustering they are named out of. Removing three tracks moves the
    # axis and the cut, and measured that way it cost four correct players to remove four
    # officials; naming them afterwards leaves every other track's side exactly as it was.
    outfield = [t for i, t in enumerate(usable) if not keeper[i]]
    if len(outfield) >= 2:
        labels = split_kits(
            np.array([t.side_mean for t in outfield], dtype=np.float64),
            _crowding(outfield, frames),
        )
    else:
        outfield, labels = usable, first

    # Named against the FINAL split rather than the rough one that found the keepers: an
    # outlier test is only as good as the model it measures distance from, and the first
    # cut is a single unconstrained axis.
    settled = _oddness(np.array([t.side_mean for t in outfield], dtype=np.float64), labels)
    at_a_goal = {t.id for i, t in enumerate(usable) if in_a_goal[i]}
    candidates = [
        i
        for i in np.argsort(-settled)
        if settled[i] > OUTLIER_RATIO and outfield[i].id not in at_a_goal
    ]
    referee = {outfield[i].id for i in candidates[:MAX_REFEREES]}

    sides = {}
    for k in (0, 1):
        xs = [mean_x[t.id] for t, lab in zip(outfield, labels, strict=True) if lab == k]
        sides[k] = float(np.mean(xs)) if xs else 0.0
    left = 0 if sides[0] <= sides[1] else 1

    # Which tracks the split is actually sure of. Distance to its own side's kit against
    # distance to the other's, in the same colour space the cut was made in.
    kits = np.array([t.side_mean for t in outfield], dtype=np.float64)
    sure = np.ones(len(outfield), dtype=bool)
    for k in (0, 1):
        mine, theirs = labels == k, labels != k
        if not bool(mine.any()) or not bool(theirs.any()):
            continue
        # Leave-one-out: a track judged against a centre it helped compute drags that
        # centre towards itself, and the closer to the cut it sits the more it flatters
        # itself. Measured on SNGS-147, that bias alone was the difference between
        # declining nothing and declining a quarter of the clip.
        total, n = kits[mine].sum(axis=0), int(mine.sum())
        without = (total - kits[mine]) / (n - 1) if n > 1 else kits[mine]
        own = np.linalg.norm(kits[mine] - without, axis=1)
        other = np.linalg.norm(kits[mine] - kits[theirs].mean(axis=0), axis=1)
        settled = own <= KIT_MARGIN * other
        # A track the ball went THROUGH is named on the plain comparison, without the
        # margin. `KIT_MARGIN` buys silence, and silence is the right price for the
        # twenty-one players who are not on the ball: a wrong colour there is a pass
        # between the wrong shirts (D72). For the one who IS on the ball it buys nothing
        # -- Pitchboard fields nobody it cannot name, so declining him does not leave the
        # move uncoloured, it leaves the move undrawn, and a coach watches a striker
        # receive, run and win a penalty while the board stands somebody else offside in
        # his place. Still only where the kit AGREES: nearer his own side than the other
        # is the whole claim, and a track that fails that stays unknown (D91).
        if carried:
            leading = own < other
            settled = settled | (leading & np.array([t.id in carried for t in outfield])[mine])
        sure[mine] = settled

    out: dict[int, TeamLabel] = {t.id: "unknown" for t in tracks}
    for i, (t, lab) in enumerate(zip(outfield, labels, strict=True)):
        if sure[i]:
            out[t.id] = "home" if lab == left else "away"
    for t in outfield:
        if t.id in referee:
            out[t.id] = "referee"
    for i, t in enumerate(usable):
        if keeper[i]:
            # The keeper of the goal they are standing in, whichever side that is.
            out[t.id] = "gkHome" if mean_x[t.id] <= pitch.halfway else "gkAway"
    return out

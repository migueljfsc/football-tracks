"""tracks.json - the contract with Pitchboard.

The Python mirror of schema/tracks.schema.json. The schema is normative; this module
exists so a stage builds the file through one writer rather than assembling dicts, and
so the field names live in one place when the schema moves.

Positions are pitch metres with the origin at the TOP-LEFT corner of a 105 x 68 pitch,
which is what `BoardDoc` uses. Nothing else in this repo shares that convention -
SoccerNet's is centre-origin - so the conversion belongs at the edge, in whichever
stage produces a Track, never here and never downstream.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .config import PITCH_LENGTH, PITCH_WIDTH

TeamLabel = Literal["home", "away", "gkHome", "gkAway", "referee", "unknown"]

# How far outside the touchline a position may land and still be believed, as a share
# of the pitch. A homography extrapolates badly for anyone near the horizon, and the
# error is unbounded - SoccerNet's own labels carry positions 230 m off the pitch. A
# throw-in taker is genuinely a metre or two outside; nobody is ever fifty.
PITCH_MARGIN = 0.15

# A tighter margin, for deciding whether somebody is a PLAYER rather than whether a
# single position is credible. Those are different questions. A position needs slack for
# the camera model to be wrong; a person does not - the crowd behind a goal stands ten
# metres back, and at the generous margin they are tracked, clustered as a team, and
# arrive on the board as an eleven who never move. Measured on the Rio Ave clip, they
# also captured the "home" label outright, because they sit at negative x and that is
# how the side playing left is decided.
PLAYER_MARGIN = 0.05


@dataclass(slots=True)
class Sample:
    """One observation. `f` is an ABSOLUTE frame index in the source video."""

    f: int
    x: float
    y: float
    conf: float | None = None

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {"f": self.f, "x": round(self.x, 3), "y": round(self.y, 3)}
        if self.conf is not None:
            d["conf"] = round(self.conf, 3)
        return d


@dataclass(slots=True)
class Track:
    """One entity across the clip. Samples are sparse and ordered by frame (D8)."""

    id: int
    team: TeamLabel
    number: int | None = None
    number_confidence: float | None = None
    samples: list[Sample] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "team": self.team, "number": self.number}
        if self.number_confidence is not None:
            d["numberConfidence"] = round(self.number_confidence, 3)
        d["samples"] = [s.to_json() for s in sorted(self.samples, key=lambda s: s.f)]
        return d


# A tenth of a second. Chosen against the board that comes out, not against the file
# size: the reduction costs 0.04 m at the median where the camera model itself is 0.95 m
# out, so accuracy does not constrain this at all -- every interval tried was far inside
# the noise. What constrains it is the CURVES. A run is fitted from the samples between
# two scenes, and scenes are seconds apart, so a coarse grid starves the fit:
#
#     interval   size    curved runs    roster
#     raw        888 KB      22         11v10
#     0.1 s      237 KB      23         11v10
#     0.2 s      153 KB      16         11v9
#     0.4 s      110 KB      12         11v10
#     1.0 s       84 KB       5         11v9
#
# A second per sample costs 78% of the runs, which is the whole point of the pipeline,
# to save 150 KB. 0.1 s keeps every run the raw file had -- 23 against 22, because the
# median also removes the noise that was fragmenting tracks -- at a quarter of the size.
#
# The twitchiness a long interval is meant to cure is cured by the MEDIAN inside the
# slot, not by the slot being long. Five frames is already enough to average away the
# jitter; longer slots stop removing noise and start removing the run.
DEFAULT_INTERVAL_S = 0.1


def at_interval(samples: list[Sample], fps: float, interval_s: float) -> list[Sample]:
    """Samples reduced to one per time slot, at the MEDIAN of each slot.

    A tracker answers every frame, and a per-frame answer is finer than the pipeline is
    accurate: the camera model alone is about a metre out, so most of what a frame adds
    over its neighbour is noise. Storing it anyway makes a 15-second clip an 888 KB file
    and asks the board to smooth what should never have been written.

    The median and not every Nth sample. Decimation keeps whatever noise the sample it
    kept happened to have; the median of a slot rejects the excursions outright, and for
    a player crossing the slot it lands where they were halfway through it - which is
    why the frame stamped is the slot's median frame and not its edge.
    """
    if interval_s <= 0 or not samples:
        return samples
    step = max(1, round(interval_s * fps))
    slots: dict[int, list[Sample]] = {}
    for s in sorted(samples, key=lambda s: s.f):
        slots.setdefault(s.f // step, []).append(s)

    out: list[Sample] = []
    for k in sorted(slots):
        group = slots[k]
        confs = [s.conf for s in group if s.conf is not None]
        out.append(
            Sample(
                f=int(statistics.median_low([s.f for s in group])),
                x=float(statistics.median([s.x for s in group])),
                y=float(statistics.median([s.y for s in group])),
                conf=sum(confs) / len(confs) if confs else None,
            )
        )
    return out


def on_pitch(x: float, y: float, margin: float = PITCH_MARGIN) -> bool:
    """Whether a projected position is credible enough to keep.

    Rejecting rather than clamping: a position 200 m out is not a player standing on
    the touchline, it is a homography that failed, and clamping would launder a
    failure into a plausible-looking coordinate that the reduction then fits a curve
    through.
    """
    mx, my = PITCH_LENGTH * margin, PITCH_WIDTH * margin
    return -mx <= x <= PITCH_LENGTH + mx and -my <= y <= PITCH_WIDTH + my


def write(
    path: Path,
    *,
    clip: str,
    fps: float,
    start_frame: int,
    end_frame: int,
    tracks: list[Track],
    ball: list[Sample] | None = None,
    width: int | None = None,
    height: int | None = None,
    interval_s: float = DEFAULT_INTERVAL_S,
) -> Path:
    source: dict[str, Any] = {
        "clip": clip,
        "fps": fps,
        "startFrame": start_frame,
        "endFrame": end_frame,
    }
    if width is not None:
        source["width"] = width
    if height is not None:
        source["height"] = height
    if interval_s > 0:
        source["intervalS"] = interval_s

    doc = {
        "version": 1,
        "source": source,
        "pitch": {"length": PITCH_LENGTH, "width": PITCH_WIDTH},
        "tracks": [
            Track(
                id=t.id,
                team=t.team,
                number=t.number,
                number_confidence=t.number_confidence,
                samples=at_interval(t.samples, fps, interval_s),
            ).to_json()
            for t in tracks
        ],
        # The ball is NOT reduced to the interval. Its samples answer one question --
        # who was nearest -- and the answer changes inside a slot: a pass leaves one
        # player and reaches another in less time than a player takes to run anywhere.
        # It is also one entity rather than twenty-two, so full rate costs almost
        # nothing here and buys every handover the board can find.
        #
        # The ball's POSITIONS, which are not to be trusted as positions - a ground
        # homography assumes z = 0, so a ball in flight lands metres from where it is.
        # They answer one question reliably, which is who is nearest, and that is the
        # only thing a board wants from the ball (D29).
        "ball": {"samples": [s.to_json() for s in sorted(ball, key=lambda s: s.f)]}
        if ball
        else None,
    }
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path

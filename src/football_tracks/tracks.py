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
    width: int | None = None,
    height: int | None = None,
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

    doc = {
        "version": 1,
        "source": source,
        "pitch": {"length": PITCH_LENGTH, "width": PITCH_WIDTH},
        "tracks": [t.to_json() for t in tracks],
        "ball": None,
    }
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path

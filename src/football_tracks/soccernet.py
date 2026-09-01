"""SoccerNet Game State Reconstruction (SN-GSR-2025) - fetch and ground truth.

GSR is this pipeline's own task published with answers: per frame it labels every
player's bounding box, track id, team, role, shirt number AND position on the pitch.
So it is both the input to test against and the yardstick to score against.

Two consequences worth knowing before reading further:

* GSR clips are 30 s of ONE camera, already extracted to JPEG frames. There are no
  cuts in them, so stage 0 has nothing to do here - the SoccerNet path enters the
  pipeline at stage 1. Stage 0 is for arbitrary broadcast footage.
* The whole split is one multi-gigabyte zip, but a zip's index sits at the end and
  HuggingFace serves range requests, so a single clip is pulled without downloading
  the other 48. `test.zip` is 8.85 GB; one clip is about 150 MB.
"""

from __future__ import annotations

import json
import zipfile
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import HfFileSystem

from .config import PITCH_LENGTH, PITCH_WIDTH
from .tracks import Sample, TeamLabel, Track, on_pitch

REPO = "datasets/SoccerNet/SN-GSR-2025"
SPLITS = ("train", "valid", "test", "challenge")

# SoccerNet puts the origin on the centre spot; Pitchboard puts it on the top-left
# corner. This is the whole of the conversion, and it belongs here at the edge - past
# this module every coordinate is already in Pitchboard's space.
SN_ORIGIN_X = PITCH_LENGTH / 2
SN_ORIGIN_Y = PITCH_WIDTH / 2

# category_id -> the label a Track carries. Teams are "left" and "right" in the
# source, which are sides of the pitch rather than sides of a fixture, but they are
# consistent within a clip and that is all `home`/`away` has ever meant here.
_TEAM: dict[tuple[str, str | None], TeamLabel] = {
    ("player", "left"): "home",
    ("player", "right"): "away",
    ("goalkeeper", "left"): "gkHome",
    ("goalkeeper", "right"): "gkAway",
}


@contextmanager
def open_split(split: str) -> Iterator[zipfile.ZipFile]:
    """A split's zip, read remotely. Only the index is fetched on open."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    fs = HfFileSystem()
    with fs.open(f"{REPO}/{split}.zip", "rb") as f:
        yield zipfile.ZipFile(f)


def list_clips(zf: zipfile.ZipFile) -> list[str]:
    return sorted({n.split("/")[0] for n in zf.namelist() if "/" in n})


def fetch(zf: zipfile.ZipFile, clip: str, dest: Path, *, limit: int | None = None) -> Path:
    """Extract one clip's labels and frames.

    Members are read in header order rather than name order: the remote file is
    block-cached, so walking the zip forwards turns 750 random reads into a
    near-sequential scan.
    """
    out = dest / clip
    out.mkdir(parents=True, exist_ok=True)

    members = [i for i in zf.infolist() if i.filename.startswith(f"{clip}/") and not i.is_dir()]
    if not members:
        raise KeyError(f"no clip {clip!r} in this split")

    frames = sorted((i for i in members if "/img1/" in i.filename), key=lambda i: i.filename)
    if limit is not None:
        frames = frames[:limit]
    labels = [i for i in members if i.filename.endswith(".json")]

    for info in sorted(labels + frames, key=lambda i: i.header_offset):
        target = out / Path(info.filename).relative_to(clip)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(zf.read(info))

    return out


@dataclass(slots=True)
class Clip:
    """A fetched clip on disk."""

    name: str
    root: Path

    @property
    def labels_path(self) -> Path:
        return self.root / "Labels-GameState.json"

    @property
    def frames_dir(self) -> Path:
        return self.root / "img1"

    def labels(self) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(self.labels_path.read_text())
        return data


def _frame_index(file_name: str) -> int:
    """`000123.jpg` -> 123. Absolute in the clip's own numbering, which is 1-based."""
    return int(Path(file_name).stem)


def _jersey(values: list[str | None]) -> int | None:
    """The shirt number a track carries, or None.

    A ground-truth track should state one number or none at all, but it is read the
    same way stage 5 will read an OCR vote - most common non-null wins - so the two
    paths cannot disagree about what a track's number is.
    """
    seen = Counter(v for v in values if v not in (None, ""))
    if not seen:
        return None
    best, _ = seen.most_common(1)[0]
    try:
        return int(best)
    except ValueError:
        return None


def to_tracks(labels: dict[str, Any], *, keep_referees: bool = False) -> list[Track]:
    """Ground-truth annotations -> Tracks in Pitchboard's coordinate space.

    Positions outside the pitch by more than the margin are DROPPED, not clamped.
    SoccerNet's own labels carry positions hundreds of metres out, because a
    homography extrapolates without bound for anyone near the horizon - the same
    failure stage 1 will have, which is why the guard lives in `tracks.on_pitch`
    rather than here.
    """
    frame_of = {img["image_id"]: _frame_index(img["file_name"]) for img in labels["images"]}

    samples: dict[int, list[Sample]] = {}
    teams: dict[int, TeamLabel] = {}
    jerseys: dict[int, list[str | None]] = {}

    for a in labels["annotations"]:
        pitch = a.get("bbox_pitch")
        attrs = a.get("attributes")
        if not pitch or not attrs:
            continue  # the pitch-line annotations, which are geometry rather than an entity

        role = attrs.get("role")
        # The ball carries a pitch position and attributes exactly as a player does, so
        # it survives the guard above and arrives here looking like a track with no
        # team. It is not one, and there is no ball in v0 (D4).
        if role == "ball":
            continue
        if role == "referee" and not keep_referees:
            continue
        team = _TEAM.get((role, attrs.get("team")), "referee" if role == "referee" else "unknown")

        # The bottom middle of the box is where the player meets the grass, which is
        # the only point on a bounding box that is actually a position.
        x = pitch["x_bottom_middle"] + SN_ORIGIN_X
        y = pitch["y_bottom_middle"] + SN_ORIGIN_Y
        if not on_pitch(x, y):
            continue

        tid = int(a["track_id"])
        samples.setdefault(tid, []).append(Sample(f=frame_of[a["image_id"]], x=x, y=y))
        teams[tid] = team
        jerseys.setdefault(tid, []).append(attrs.get("jersey"))

    out = []
    for tid in sorted(samples):
        number = _jersey(jerseys[tid])
        out.append(
            Track(
                id=tid,
                team=teams[tid],
                number=number,
                # Ground truth, so a number that is present is certain. Stage 5 will
                # put a real vote share here.
                number_confidence=1.0 if number is not None else None,
                samples=samples[tid],
            )
        )
    return out

"""Constants and paths shared by every stage.

Pitch dimensions match Pitchboard's `BoardDoc`: origin at the top-left corner, x along
the length, y across the width, metres throughout. They are here so no stage invents
its own answer.
"""

from __future__ import annotations

from pathlib import Path

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work"


def work_dir(clip: Path, *, create: bool = True) -> Path:
    """Where a clip's artefacts live. One directory per clip, named after it.

    Everything under work/ is reproducible from the clip plus a stage, which is why it
    is not in git and why `make clean` can delete all of it.
    """
    d = WORK / clip.stem
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d

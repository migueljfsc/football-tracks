"""The emitted file must match schema/tracks.schema.json.

The schema is normative and shared with Pitchboard, so a field added to the writer
without being added to the schema is a silent break in the other repo. This reads the
schema file rather than restating it, so the two cannot drift apart unnoticed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_tracks import tracks
from football_tracks.config import PITCH_LENGTH, PITCH_WIDTH

SCHEMA = json.loads((Path(__file__).resolve().parents[1] / "schema/tracks.schema.json").read_text())


def emit(tmp_path: Path, interval_s: float = 0.0) -> dict[str, Any]:
    # Binning off by default: these assert the SHAPE of the contract, and a slot that
    # merged two of the samples would be testing the reduction instead.
    path = tracks.write(
        tmp_path / "tracks.json",
        interval_s=interval_s,
        clip="c.mp4",
        fps=25.0,
        start_frame=1,
        end_frame=2,
        width=1920,
        height=1080,
        tracks=[
            tracks.Track(
                id=1,
                team="home",
                number=9,
                number_confidence=0.8,
                samples=[
                    tracks.Sample(f=1, x=1.0, y=2.0, conf=0.9),
                    tracks.Sample(f=2, x=3.0, y=4.0),
                ],
            ),
            tracks.Track(id=2, team="gkAway", samples=[tracks.Sample(f=1, x=5.0, y=6.0)]),
        ],
    )
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def test_top_level_matches_the_schema(tmp_path: Path) -> None:
    d = emit(tmp_path)
    allowed = set(SCHEMA["properties"])
    assert set(d) <= allowed, f"writer emits keys the schema does not declare: {set(d) - allowed}"
    assert set(SCHEMA["required"]) <= set(d)


def test_track_and_sample_keys_match_the_schema(tmp_path: Path) -> None:
    d = emit(tmp_path)
    tschema = SCHEMA["properties"]["tracks"]["items"]
    sschema = tschema["properties"]["samples"]["items"]
    for t in d["tracks"]:
        assert set(t) <= set(tschema["properties"]), set(t) - set(tschema["properties"])
        assert set(tschema["required"]) <= set(t)
        for s in t["samples"]:
            assert set(s) <= set(sschema["properties"]), set(s) - set(sschema["properties"])
            assert set(sschema["required"]) <= set(s)


def test_team_labels_are_all_declared_by_the_schema(tmp_path: Path) -> None:
    allowed = set(SCHEMA["properties"]["tracks"]["items"]["properties"]["team"]["enum"])
    assert {t["team"] for t in emit(tmp_path)["tracks"]} <= allowed


def test_the_pitch_is_the_one_both_repos_agree_on(tmp_path: Path) -> None:
    # Pitchboard's BoardDoc is 105 x 68 with the origin at the top-left corner. A file
    # stating anything else is not the contract, whatever else is right about it.
    assert emit(tmp_path)["pitch"] == {"length": PITCH_LENGTH, "width": PITCH_WIDTH}
    assert (PITCH_LENGTH, PITCH_WIDTH) == (105.0, 68.0)


def test_an_absent_number_is_written_as_null_not_omitted(tmp_path: Path) -> None:
    # Pitchboard distinguishes "unread" from "absent field"; D5 turns on it being explicit.
    track = next(t for t in emit(tmp_path)["tracks"] if t["id"] == 2)
    assert "number" in track
    assert track["number"] is None


def test_samples_carry_absolute_frame_indices(tmp_path: Path) -> None:
    d = emit(tmp_path)
    assert [s["f"] for s in d["tracks"][0]["samples"]] == [1, 2]
    assert d["source"]["startFrame"] == 1


def test_the_interval_keeps_one_sample_per_slot(tmp_path: Path) -> None:
    d = emit(tmp_path, interval_s=0.2)
    # 25 fps and a fifth of a second is a five-frame slot, so frames 1 and 2 are one.
    assert [s["f"] for s in d["tracks"][0]["samples"]] == [1]
    assert d["source"]["intervalS"] == 0.2


def test_the_interval_takes_the_median_and_not_a_survivor() -> None:
    """Decimation keeps whatever noise the kept sample had; the median rejects it."""
    # Slots are absolute, so frames 5-9 are one five-frame slot and 1-5 would be two.
    ss = [
        tracks.Sample(f=5, x=10.0, y=20.0),
        tracks.Sample(f=6, x=10.1, y=20.0),
        tracks.Sample(f=7, x=99.0, y=20.0),  # an excursion
        tracks.Sample(f=8, x=10.2, y=20.0),
        tracks.Sample(f=9, x=10.3, y=20.0),
    ]
    got = tracks.at_interval(ss, fps=25.0, interval_s=0.2)
    assert len(got) == 1
    assert got[0].x == 10.2
    assert got[0].f == 7


def test_no_interval_leaves_every_sample() -> None:
    ss = [tracks.Sample(f=i, x=float(i), y=0.0) for i in range(1, 6)]
    assert tracks.at_interval(ss, fps=25.0, interval_s=0.0) == ss

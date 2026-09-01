"""Command line entry point. One command per stage."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from . import auto as auto_mod
from . import detect as detect_mod
from . import overlay as overlay_mod
from . import render as render_mod
from . import score as score_mod
from . import seed as seed_mod
from . import soccernet, stage0_segment, stage1_propagate, stage1_register, tracks
from . import video as video_mod
from .config import CLIPS, work_dir

app = typer.Typer(add_completion=False, help="Broadcast clip -> player tracks in pitch metres.")


def _fps(root: Path, labels: dict[str, Any] | None) -> float:
    """SoccerNet states it; a recording carries it in clip.json (and the container lies)."""
    if labels is not None:
        return float(labels["info"]["frame_rate"])
    return video_mod.load(root).fps


@app.command()
def segment(
    clip: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Source video.")],
    threshold: Annotated[
        float, typer.Option(help="Cut sensitivity. Lower finds more cuts.")
    ] = 27.0,
    min_seconds: Annotated[float, typer.Option(help="Shortest segment worth keeping.")] = 4.0,
    green_min: Annotated[
        float, typer.Option(help="Least pitch-green a tactical shot may be.")
    ] = 0.35,
    samples: Annotated[int, typer.Option(help="Frames scored per segment.")] = 6,
    extract: Annotated[
        int | None, typer.Option(help="Also cut this segment out to its own mp4.")
    ] = None,
) -> None:
    """Stage 0 - split a broadcast clip at its cuts and find the tactical camera."""
    out = work_dir(clip)
    segments, source = stage0_segment.find_segments(
        clip,
        threshold=threshold,
        min_seconds=min_seconds,
        green_min=green_min,
        samples=samples,
    )

    typer.echo(f"{source['clip']}  {source['width']}x{source['height']}  {source['fps']:.2f} fps")
    typer.echo(f"{'':>3} {'start':>8} {'end':>8} {'dur':>7} {'green':>6} {'motion':>7}")
    for s in segments:
        mark = "*" if s.main else " "
        typer.echo(
            f"{s.index:>2}{mark} {s.start_s:>8.2f} {s.end_s:>8.2f} {s.duration_s:>7.2f}"
            f" {s.green:>6.2f} {s.motion:>7.2f}"
        )

    path = stage0_segment.write(segments, source, out)
    typer.echo(f"\nwrote {path}")

    pick = stage0_segment.best(segments)
    if pick is None:
        typer.echo("no segment qualified - lower --green-min or --min-seconds")
    else:
        typer.echo(f"main camera: segment {pick.index} ({pick.duration_s:.1f}s)")

    if extract is not None:
        chosen = next((s for s in segments if s.index == extract), None)
        if chosen is None:
            raise typer.BadParameter(f"no segment {extract}")
        typer.echo(f"wrote {stage0_segment.extract(clip, chosen, out)}")


@app.command()
def clips(
    split: Annotated[str, typer.Option(help="train, valid, test or challenge.")] = "test",
) -> None:
    """List the clips in a SoccerNet GSR split, without downloading any of them."""
    with soccernet.open_split(split) as zf:
        names = soccernet.list_clips(zf)
    typer.echo(f"{len(names)} clips in {split}")
    typer.echo("  ".join(names))


@app.command()
def fetch(
    clip: Annotated[str, typer.Argument(help="Clip name, e.g. SNGS-147.")],
    split: Annotated[str, typer.Option(help="train, valid, test or challenge.")] = "test",
    limit: Annotated[
        int | None, typer.Option(help="Fetch only the first N frames, for a quick look.")
    ] = None,
) -> None:
    """Download one SoccerNet GSR clip - labels and frames - into data/clips/.

    Pulls a single clip out of the split's multi-gigabyte zip by range request, so
    this costs about 150 MB rather than the whole 8.85 GB.
    """
    with soccernet.open_split(split) as zf:
        typer.echo(f"fetching {clip} from {split} ...")
        out = soccernet.fetch(zf, clip, CLIPS, limit=limit)
    n = len(list((out / "img1").glob("*.jpg")))
    typer.echo(f"wrote {out}  ({n} frames)")


@app.command()
def truth(
    clip: Annotated[str, typer.Argument(help="A clip already fetched into data/clips/.")],
    referees: Annotated[
        bool, typer.Option(help="Keep referees rather than dropping them.")
    ] = False,
) -> None:
    """Ground-truth labels -> tracks.json, with no CV in the loop.

    The yardstick every later stage is scored against, and a real file for
    Pitchboard's importer to be built against before any of the vision works.
    """
    c = soccernet.Clip(name=clip, root=CLIPS / clip)
    if not c.labels_path.exists():
        raise typer.BadParameter(f"no labels at {c.labels_path} - run `ft fetch {clip}` first")

    labels = c.labels()
    built = soccernet.to_tracks(labels, keep_referees=referees)
    info = labels["info"]
    frames = [int(img["file_name"].split(".")[0]) for img in labels["images"]]

    out = work_dir(Path(clip))
    path = tracks.write(
        out / "truth.json",
        clip=clip,
        fps=float(info["frame_rate"]),
        start_frame=min(frames),
        end_frame=max(frames),
        tracks=built,
        width=labels["images"][0]["width"],
        height=labels["images"][0]["height"],
    )

    named = sum(1 for t in built if t.number is not None)
    total_samples = sum(len(t.samples) for t in built)
    typer.echo(f"{len(built)} tracks, {total_samples} samples, {named} with a shirt number")
    typer.echo(f"wrote {path}")


@app.command()
def render(
    path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="A tracks.json or truth.json.")
    ],
    scale: Annotated[float, typer.Option(help="Pixels per metre.")] = 10.0,
    still: Annotated[int | None, typer.Option(help="Render one frame as a PNG instead.")] = None,
) -> None:
    """Draw a tracks file as a top-down video of coloured dots.

    The picture invariant 3 asks for. If the dots move like a football team the
    positions are right, and nothing in the numbers can tell you that.
    """
    doc = render_mod.load(path)
    if still is not None:
        out = render_mod.still(doc, still, path.with_suffix(f".f{still}.png"), scale=scale)
    else:
        out = render_mod.video(doc, path.with_suffix(".mp4"), scale=scale)
    typer.echo(f"wrote {out}")


@app.command()
def score(
    prediction: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="A produced tracks.json.")
    ],
    truth_path: Annotated[
        Path | None,
        typer.Option("--truth", help="Ground truth. Defaults to truth.json beside it."),
    ] = None,
    radius: Annotated[float, typer.Option(help="Match radius in metres.")] = score_mod.MATCH_RADIUS,
) -> None:
    """Diff a produced tracks.json against ground truth."""
    gt = truth_path or prediction.parent / "truth.json"
    if not gt.exists():
        raise typer.BadParameter(f"no ground truth at {gt} - run `ft truth <clip>` first")
    typer.echo(
        score_mod.report(
            score_mod.score(render_mod.load(gt), render_mod.load(prediction), radius=radius)
        )
    )


@app.command()
def calibrate(
    clip: Annotated[str, typer.Argument(help="A clip already fetched into data/clips/.")],
    frame: Annotated[
        int | None, typer.Option(help="Draw the overlay for this frame instead of measuring.")
    ] = None,
    video: Annotated[
        bool, typer.Option(help="Draw the overlay for every frame, as an mp4.")
    ] = False,
    carry: Annotated[
        int,
        typer.Option(
            help="Carry a homography this many frames across gaps the solver cannot fill."
            " 0 disables it; a negative value means uncapped."
        ),
    ] = stage1_propagate.DEFAULT_MAX_CARRY,
) -> None:
    """Stage 1 - fit a homography per frame from the pitch lines, and check it.

    With no options this MEASURES: ground-truth boxes are pushed through the fitted
    homography and compared with the position SoccerNet recorded for them, which
    isolates the camera model from detection and gives the pipeline's error ceiling.

    With --frame or --video it draws the picture, which is the only way to see a
    homography that is wrong in a way the averages survive.
    """
    c = soccernet.Clip(name=clip, root=CLIPS / clip)
    out = work_dir(Path(clip))
    frames_all = sorted(int(p.stem) for p in c.frames_dir.glob("*.jpg"))
    if not frames_all:
        raise typer.BadParameter(f"no frames in {c.frames_dir}")

    motions = stage1_propagate.motions(c.frames_dir, frames_all, cache=out / "motions.json")
    seed_path = out / "seed.json"
    labels: dict[str, object] | None = None

    if c.labels_path.exists():
        # SoccerNet: every frame carries its own pitch lines.
        labels = c.labels()
        homs = stage1_register.fit_all(labels)
    elif seed_path.exists():
        # A clip nobody annotated: one clicked frame, carried both ways.
        homs = auto_mod.from_seed(
            seed_mod.read(seed_path),
            frames_all,
            c.frames_dir,
            max_carry=None,
            motions=motions,
        )
    else:
        raise typer.BadParameter(
            f"{clip} has neither SoccerNet labels nor {seed_path} - run `ft seed {clip}` first"
        )

    chain = None
    if carry != 0 and labels is not None:
        chain = stage1_propagate.fill(
            c.frames_dir, homs, max_carry=None if carry < 0 else carry, motion=motions
        )
        homs = chain.homographies

    if frame is not None or video:
        import cv2

        frames = sorted(homs) if video else [frame] if frame is not None else []
        writer = None
        for f in frames:
            src = c.frames_dir / f"{f:06d}.jpg"
            if not src.exists():
                if not video:
                    raise typer.BadParameter(f"no frame at {src} - fetch the clip without --limit")
                continue
            raw = cv2.imread(str(src))
            if raw is None:
                continue
            img = overlay_mod.draw(raw, homs[f])
            overlay_mod.annotate(
                img, f"f{f}" if homs[f] is not None else f"f{f} unsolved", ok=homs[f] is not None
            )
            if not video:
                dest = out / f"calib.f{f}.png"
                cv2.imwrite(str(dest), img)
                typer.echo(f"wrote {dest}")
            else:
                if writer is None:
                    h, w = img.shape[:2]
                    dest = out / "calib.mp4"
                    writer = cv2.VideoWriter(
                        str(dest),
                        cv2.VideoWriter.fourcc(*"mp4v"),
                        _fps(CLIPS / clip, labels),
                        (w, h),
                    )
                writer.write(img)
        if writer is not None:
            writer.release()
            typer.echo(f"wrote {out / 'calib.mp4'}")
        return

    if chain is not None:
        typer.echo(
            f"carried           {chain.carried} frames across gaps"
            f" ({chain.solved_directly} solved directly, {chain.gaps} left unsolved)"
        )
    if labels is None:
        solved = sum(1 for h in homs.values() if h is not None)
        typer.echo(f"frames solved     {solved}/{len(homs)}  ({solved / max(1, len(homs)):.1%})")
        typer.echo("no ground truth here - check the overlay with --frame or --video")
        return
    typer.echo(stage1_register.report(stage1_register.evaluate(labels, homs)))


@app.command()
def detect(
    clip: Annotated[str, typer.Argument(help="A clip already fetched into data/clips/.")],
    conf: Annotated[
        float, typer.Option(help="Detection confidence floor.")
    ] = detect_mod.DEFAULT_CONF,
) -> None:
    """Stage 2a - find people in every frame, and cache them.

    Slow, and separate from tracking on purpose: tracking is the part that gets tuned.
    """
    c = soccernet.Clip(name=clip, root=CLIPS / clip)
    frames = sorted(int(p.stem) for p in c.frames_dir.glob("*.jpg"))
    if not frames:
        raise typer.BadParameter(f"no frames in {c.frames_dir} - run `ft fetch {clip}` first")

    out = work_dir(Path(clip))
    with typer.progressbar(frames, label="detecting") as bar:
        found = detect_mod.run(c.frames_dir, frames, conf=conf, progress=lambda _f: bar.update(1))
    path = detect_mod.write(out / "detections.json", found, conf=conf)
    typer.echo(
        f"{len(found)} detections over {len(frames)} frames"
        f" ({len(found) / max(1, len(frames)):.1f}/frame)"
    )
    typer.echo(f"wrote {path}")


@app.command()
def auto(
    clip: Annotated[str, typer.Argument(help="A clip already fetched into data/clips/.")],
    mode: Annotated[
        str, typer.Option(help="'truth' uses every frame's lines; 'seed' uses only frame one's.")
    ] = "seed",
    carry: Annotated[int, typer.Option(help="Frames a homography may be carried.")] = -1,
) -> None:
    """The automatic path end to end - frames in, tracks.json out.

    `--mode truth` holds stage 1 fixed so the score is stages 2 and 3 alone.
    `--mode seed` throws away every line annotation but frame one's, which is what a
    human clicking four corners once actually leaves you with.
    """
    if mode not in ("truth", "seed"):
        raise typer.BadParameter("mode must be 'truth' or 'seed'")
    picked: auto_mod.Mode = "truth" if mode == "truth" else "seed"

    c = soccernet.Clip(name=clip, root=CLIPS / clip)
    out = work_dir(Path(clip))
    dets_path = out / "detections.json"
    if not dets_path.exists():
        raise typer.BadParameter(f"no {dets_path} - run `ft detect {clip}` first")

    labels = c.labels()
    frames = sorted(int(p.stem) for p in c.frames_dir.glob("*.jpg"))
    detections = detect_mod.read(dets_path)

    motions = stage1_propagate.motions(c.frames_dir, frames, cache=out / "motions.json")
    homs = auto_mod.homographies(
        labels, c.frames_dir, picked, max_carry=None if carry < 0 else carry, motions=motions
    )
    result = auto_mod.build(
        c.frames_dir,
        frames,
        detections,
        homs,
        fps=float(labels["info"]["frame_rate"]),
        motions=motions,
    )

    path = tracks.write(
        out / "tracks.json",
        clip=clip,
        fps=float(labels["info"]["frame_rate"]),
        start_frame=min(frames),
        end_frame=max(frames),
        tracks=result.tracks,
        width=labels["images"][0]["width"],
        height=labels["images"][0]["height"],
    )
    typer.echo(
        f"mode {mode}: {result.detections} detections -> {result.raw_tracks} raw tracks"
        f" -> {len(result.tracks)} kept"
    )
    typer.echo(
        f"dropped off pitch {result.dropped_off_pitch}, frames with no homography"
        f" {result.unsolved_frames}"
    )
    typer.echo(f"wrote {path}")


@app.command()
def frames(
    source: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="A video file.")],
    name: Annotated[str | None, typer.Option(help="Clip name. Defaults to the file stem.")] = None,
) -> None:
    """Turn a broadcast video into the numbered-JPEG layout the pipeline speaks.

    Detects and removes pillarbox and letterbox bars, and records the REAL frame rate
    rather than the one the container claims - a screen recording routinely lies.
    """
    dest = CLIPS / (name or source.stem)
    clip = video_mod.extract(source, dest)
    typer.echo(
        f"{clip.frames} frames at {clip.fps:.2f} fps, {clip.width}x{clip.height}"
        f" (cropped from {clip.crop})"
    )
    typer.echo(f"wrote {dest}")


@app.command()
def seed(
    clip: Annotated[str, typer.Argument(help="A clip in data/clips/.")],
    frame: Annotated[int, typer.Option(help="Which frame to seed.")] = 1,
) -> None:
    """Click pitch landmarks on one frame, to seed the camera model.

    The only human step in the automatic path. Click a landmark, and the name shown is
    the one it is recorded as - so click them in the order listed, or press n/p to
    choose. Four is the minimum and is exactly determined, which means it fits whatever
    was misclicked and cannot be checked (D17); six or more is much safer.
    """
    from . import seedui

    root = CLIPS / clip
    img = video_mod.read_frame(root / "img1", frame)
    if img is None:
        raise typer.BadParameter(f"no frame {frame} in {root / 'img1'}")

    got = seedui.collect(img, frame)
    if got is None:
        typer.echo("abandoned; nothing written")
        raise typer.Exit(1)

    path = seed_mod.write(work_dir(Path(clip)) / "seed.json", got)
    h = seed_mod.homography(got)
    typer.echo(
        f"{len(got.points)} points -> {'a homography' if h is not None else 'NO homography'}"
    )
    typer.echo(f"wrote {path}")
    typer.echo(f"now check it: ft calibrate {clip} --frame {frame}")


if __name__ == "__main__":
    app()

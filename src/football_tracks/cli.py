"""Command line entry point. One command per stage."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from . import overlay as overlay_mod
from . import render as render_mod
from . import score as score_mod
from . import soccernet, stage0_segment, stage1_register, tracks
from .config import CLIPS, work_dir

app = typer.Typer(add_completion=False, help="Broadcast clip -> player tracks in pitch metres.")


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
) -> None:
    """Stage 1 - fit a homography per frame from the pitch lines, and check it.

    With no options this MEASURES: ground-truth boxes are pushed through the fitted
    homography and compared with the position SoccerNet recorded for them, which
    isolates the camera model from detection and gives the pipeline's error ceiling.

    With --frame or --video it draws the picture, which is the only way to see a
    homography that is wrong in a way the averages survive.
    """
    c = soccernet.Clip(name=clip, root=CLIPS / clip)
    if not c.labels_path.exists():
        raise typer.BadParameter(f"no labels at {c.labels_path} - run `ft fetch {clip}` first")

    labels = c.labels()
    homs = stage1_register.fit_all(labels)
    out = work_dir(Path(clip))

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
                        float(labels["info"]["frame_rate"]),
                        (w, h),
                    )
                writer.write(img)
        if writer is not None:
            writer.release()
            typer.echo(f"wrote {out / 'calib.mp4'}")
        return

    typer.echo(stage1_register.report(stage1_register.evaluate(labels, homs)))


if __name__ == "__main__":
    app()

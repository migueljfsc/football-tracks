# What this is, and why

## The problem

A coach analysing a passage of play draws it by hand: twenty-two tokens, their runs, and the
ball, placed from memory against a video they keep scrubbing back. It is slow, and the slowness
is the reason most analysis never gets drawn at all.

This repo turns a broadcast clip into `tracks.json` — every player's position, per frame, in
metres on a 105 × 68 pitch, with a team and, where it can be read, a shirt number. Its sibling
[`pitchboard`](../../pitchboard) imports that file and produces an editable tactics board. The
coach corrects rather than draws.

**The bar is 70%**, and it is a comparison rather than an absolute: good enough that correcting
the board beats drawing it from nothing. This is a proof, not a product.

## Why it is two repos

Pitchboard is strict TypeScript with a pure renderer and no server. This is Python with torch,
OpenCV and two gigabytes of model weights. Neither belongs inside the other, and the seam
between them is one file (D1, D3).

`tracks.json` was frozen before any of the computer vision worked. That ordering was deliberate:
it let the TypeScript importer be built and measured against ground truth while the CV was still
failing, and it means every stage here is scored against the same format the product consumes.

## What works

Every one of the eleven benchmark clips produces a full board — 18 to 21 players of 22, scenes
marking possession changes and real movement, curved runs, and the ball as a carrier.

## What does not

**Aerial play is invisible.** A ground homography assumes the ball is on the grass, so a ball in
flight projects metres from where it is — 8% to 26% of SoccerNet's OWN ball annotations land
off the pitch entirely for this reason. Lofted passes, clearances and headers cannot be drawn
from this data, and no selector or scene rule recovers them (see `decisions/ball.md`, D66).

**Registration drifts on some clips.** Seeding propagates one human fit through every frame, so
coverage is total and error accumulates: on the worst benchmark clip a fifth of tracked players
end up somewhere nobody is. The learned alternative is three to four times cleaner and silent
wherever the pitch markings are too few (D67).

**Identity does not survive crowds.** Purity runs 57% to 86%, and seven attempts on it have
failed (D61). The one thing that ever moved it was fixing stage 1, not the tracker.

## How to read the docs

- [`pipeline.md`](./pipeline.md) — what each stage does
- [`benchmark.md`](./benchmark.md) — what it is measured on and how
- [`decisions/`](./decisions) — why each choice was made, and every attempt that failed
- [`../PLAN.md`](../PLAN.md) — where it stands today and what to do next

The decisions are the most useful thing here. Most of them record something that was measured
and then abandoned, which is cheaper to read than to rediscover.

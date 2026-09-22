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

**A coach would correct its boards rather than draw them** -- two Milan-Benfica clips, read
scene by scene against the footage, one of them put at about 90%. The dot videos read as
football. One of the two was never clicked: a match is registered by one camera (D96), and a
clip after the first needs no clicks where the pitch lines can be read.

On the eleven SoccerNet benchmark clips, 97.7% of players land within 2 m of the truth, recall
is 82.9% at 95.9% precision, and the board shows the right side on the ball 67.2% of the time.

## What does not

**Aerial play is invisible.** A ground homography assumes the ball is on the grass, so a ball in
flight projects metres from where it is -- 8% to 26% of SoccerNet's OWN ball annotations land
off the pitch for this reason (D66).

**The ball's position is often metres out**, and it is where every remaining possession error
on the coach's boards comes from. The candidates to fix it are already found; ranking them is
worth twelve points of possession and has not been done (D101).

**A player is several tracks.** The best single track holds 59% of a player's time on screen
(D97): he leaves the picture while the camera looks elsewhere and comes back as someone new, and
no appearance model tried can say which team-mate came back (D98, D100). A shirt number could;
the reader built so far names 8% of tracks (D102).

**It has been judged on one match.** Whether the bar holds on other broadcasters and on the kit
it is meant for is what [PLAN.md](../PLAN.md) asks first.

## How to read the docs

- [`pipeline.md`](./pipeline.md) — what each stage does
- [`benchmark.md`](./benchmark.md) — what it is measured on and how
- [`decisions/`](./decisions/README.md) — why each choice was made, and every attempt that failed
- [`../PLAN.md`](../PLAN.md) — where it stands today and what to do next

The decisions are the most useful thing here. Most of them record something that was measured
and then abandoned, which is cheaper to read than to rediscover.

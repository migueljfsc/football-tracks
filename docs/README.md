# Docs

| file | what it answers |
|---|---|
| [`overview.md`](./overview.md) | what this repo is, why it exists, what works and what does not |
| [`pipeline.md`](./pipeline.md) | how a clip becomes `tracks.json`, stage by stage |
| [`benchmark.md`](./benchmark.md) | what it is measured on, and what the numbers mean |
| [`decisions/project.md`](./decisions/project.md) | repo layout, the contract with Pitchboard, ground truth |
| [`decisions/registration.md`](./decisions/registration.md) | stage 1 — frame to camera model |
| [`decisions/tracking.md`](./decisions/tracking.md) | stage 2 — finding people and following them |
| [`decisions/teams.md`](./decisions/teams.md) | stage 3 — which side each track is on |
| [`decisions/ball.md`](./decisions/ball.md) | the ball, and the three faults it still has |
| [`decisions/numbers.md`](./decisions/numbers.md) | shirt numbers, attempted and abandoned |

Decisions are cited from source comments by number, so `D36` in a docstring is findable with
`grep -rn "D36" docs/`.

Decisions are numbered from one sequence shared with Pitchboard, so a comment citing "its D44"
means [`pitchboard/docs/decisions.md`](../../pitchboard/docs/decisions.md). Everything without
that qualifier is here.

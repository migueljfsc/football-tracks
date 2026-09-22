# Docs

| file | what it answers |
|---|---|
| [`overview.md`](./overview.md) | what this repo is, why it exists, what works and what does not |
| [`pipeline.md`](./pipeline.md) | how a clip becomes `tracks.json`, stage by stage |
| [`benchmark.md`](./benchmark.md) | what it is measured on, and what the numbers mean |
| [`decisions/README.md`](./decisions/README.md) | **every decision, by what became of it** -- start here |
| [`decisions/project.md`](./decisions/project.md) | repo layout, the contract with Pitchboard, ground truth |
| [`decisions/registration.md`](./decisions/registration.md) | stage 1 -- frame to camera model |
| [`decisions/tracking.md`](./decisions/tracking.md) | stage 2 -- finding people and following them |
| [`decisions/teams.md`](./decisions/teams.md) | stage 3 -- which side each track is on |
| [`decisions/ball.md`](./decisions/ball.md) | the ball: who has it, and what its position costs |
| [`decisions/numbers.md`](./decisions/numbers.md) | shirt numbers |

Decisions are cited from source comments by number, so `D96` in a docstring is findable with
`grep -rn "D96" docs/`.

Pitchboard keeps its own decision log with its own numbers, and the two collide, so a
citation of the other repo always names it: *"Pitchboard's D52"* means
[`pitchboard/docs/decisions.md`](../../pitchboard/docs/decisions.md). A bare number is this repo.

# How we work on this repo

Two people, one codebase. These rules keep us from overwriting each other's work.

## The golden rule: never work directly on `main`
`main` is the clean, working version. All work happens on a **branch** (a separate copy of the code you can change freely). When it's ready, you open a **pull request (PR)** so the other person can look before it's merged into `main`.

## Day-to-day workflow
```bash
# 1. Get the latest main
git checkout main
git pull

# 2. Make a branch for one piece of work
git checkout -b sourajit/gsd-estimation      # or friend/segformer-baseline

# 3. Work, then save snapshots (commits) often
git add -A
git commit -m "Add car-size GSD prior with uncertainty"

# 4. Upload your branch
git push -u origin sourajit/gsd-estimation
```
5. On GitHub, click **"Compare & pull request"**. Describe what changed and how you tested it.
6. The other person reviews it, leaves comments if needed, then clicks **Merge**.

## Branch names
`<name>/<short-description>`, e.g. `sourajit/risk-map`, `friend/temperature-scaling`.

## Commit messages
Start with a verb and say what changed: `Add ECE metric with unit test`, `Fix clearance units (px → m)`.

## Who owns what (see docs/PROJECT_PLAN.md §5)
| Folder | Owner |
|---|---|
| `src/perception/` | [Friend] |
| `src/geometry/`, `src/decision/`, `src/drop/` | Sourajit |
| `src/eval/`, `configs/ontology.yaml`, `EXPERIMENTS.md` | Both. Change only after agreeing. |

## Things that never go in git
Datasets, model weights, large outputs, API keys. `.gitignore` already blocks the usual ones. If a file is over ~50 MB, it doesn't belong here.

## Frozen files
After Week 1 these files change **only** with a dated note in `EXPERIMENTS.md` explaining why:
`data/splits/*.txt`, `configs/ontology.yaml`, `EXPERIMENTS.md` (experiment definitions).

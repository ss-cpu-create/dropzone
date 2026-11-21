# DropZone

**Uncertainty-aware contingency landing-site selection from aerial imagery.**

When a drone flying beyond visual line of sight has to land unexpectedly (low battery, motor failure, lost link), it must pick a safe spot from what its camera sees. DropZone segments nadir aerial images into safety tiers, measures how *confident* and how *uncertain* the model is, and uses that uncertainty as a safety margin when choosing a landing site.

> **Status: Week 0 (scaffold only).** Folder structure, configs and documentation are in place. No models are trained and there are no results yet. Everything below marked _TBD_ gets filled in as the work happens.

## Research question
> For autonomous contingency landing, does uncertainty-aware site selection reduce the rate of unsafe site picks compared with naive argmax-based selection, on held-out aerial scenes?

The full plan, with hypotheses, experiments, metric definitions and a week-by-week checklist, is in [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md).

## Pipeline
```
aerial frame → segmentation (SegFormer-B0) → calibration (temperature scaling)
             → uncertainty (MC dropout) → risk map → clearance (distance transform)
             → constrained site selection (or abstain) → [optional] payload-drop dispersion
```

## Results
_TBD after the frozen test run (Week 4)._

| Policy | Unsafe-pick rate | Abstention rate | Median clearance (m) |
|---|---|---|---|
| P0 Naive argmax | TBD | TBD | TBD |
| P1 Calibrated | TBD | TBD | TBD |
| P2 Uncertainty-aware | TBD | TBD | TBD |
| Oracle (ground truth) | TBD | TBD | TBD |

## Repo layout
```
configs/        safety ontology, training, vehicle and selection parameters
data/splits/    frozen train/val/test lists (data itself is NOT committed)
docs/           project plan
src/            perception · geometry · decision · drop · eval
scripts/        numbered pipeline steps + run_all.sh
results/        CSVs and figures that back every claim
report/         technical report source + PDF
tests/          unit tests for metrics, geometry and selection
```

## Setup
```bash
git clone https://github.com/ss-cpu-create/dropzone.git
cd dropzone
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Then follow [`DATA.md`](DATA.md) to download the datasets.

## Reproduce everything
```bash
bash scripts/run_all.sh
```
_Not runnable yet. The scripts are stubs until they're implemented._

## Team
- **Sourajit Saha**: geometry, risk-aware site selection, evaluation, payload-drop simulation
- **[Friend]**: segmentation, calibration, uncertainty estimation

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for how we work together on this repo.

## Honesty rules
- Results are from **offline evaluation on public datasets**. Nothing here has been flight-tested.
- Every number in this README, the report or any post links to a CSV in `results/`.
- Dataset licences are listed in [`DATA.md`](DATA.md). Code is MIT-licensed; the datasets keep their own licences.

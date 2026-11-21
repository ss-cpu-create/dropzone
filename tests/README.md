# Tests

Minimum set before the code freeze (docs/PROJECT_PLAN.md §9):
- `test_metrics.py`: ECE on a hand-computed example; IoU on tiny masks
- `test_clearance.py`: distance transform on a synthetic disk of known radius
- `test_gsd.py`: GSD formula on a known altitude/focal-length case

Run with: `pytest -q`

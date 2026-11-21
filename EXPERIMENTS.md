# Pre-registered experiments

> **Status: DRAFT.** Finalise and commit by the end of Week 1, **before running any experiment**. After that, changes are allowed only as dated entries in the change log at the bottom.

Why pre-register? Writing down *what* you'll measure and *how* before seeing results stops you, even without meaning to, from picking whichever metric makes the results look best.

| ID | Question | Data | Metrics | Owner |
|---|---|---|---|---|
| E1 Perception | How well does SegFormer-B0 segment, in-domain and under domain shift? | SDD test; UAVid/Aeroscapes | mIoU, binary safe/unsafe IoU, per-class IoU (person, car, water) | [Friend] |
| E2 Calibration | Is the model overconfident on safety-critical classes, and does temperature scaling fix it? | Fit T on SDD val → SDD test, domain-shift set | ECE (overall + UNSAFE tier), NLL, reliability diagrams | [Friend] |
| E3 Selection (headline) | Does uncertainty-aware selection reduce unsafe picks? | SDD test | Unsafe-pick rate, abstention rate, false-abstention rate, clearance (m) ± uncertainty, clearance error vs oracle | Sourajit |
| E4 Altitude stress | How do perception and selection degrade as GSD coarsens? | SDD test downsampled ×1, ×2, ×4, ×8 | mIoU, ECE, unsafe-pick rate vs effective GSD | Both |
| E5 Payload drop (optional) | Does the dispersion footprint fit inside the chosen site? | Sites from E3 | CEP50, R95, pass rate | Sourajit |

## Fixed decisions
- [ ] Safety ontology frozen in `configs/ontology.yaml` (date: ____)
- [ ] Splits frozen in `data/splits/` (date: ____)
- [ ] τ, k, ε chosen on **val only**, recorded in `configs/selection.yaml` (date: ____)
- [ ] Code frozen, tag `v1.0-frozen` (date: ____)
- [ ] Test set run once (date: ____, commit SHA: ____)

## Reporting rules
- Median and worst case, not just the mean.
- E1/E2: std across 3 seeds. E3: bootstrapped 95% CIs over test images.
- Every reported number links to a CSV in `results/`.

## Change log
_Format: `YYYY-MM-DD: what changed: why: who agreed`_

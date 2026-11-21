# Data

Datasets are **not** stored in this repo. Download them into `data/raw/` (gitignored).

> ⚠️ Before downloading, read each dataset's licence. Most allow **non-commercial research use only** and require a citation. Fill in the "Licence" and "Citation" fields below from the official pages; don't guess them.

| Dataset | Role | View | Licence | Citation | Downloaded |
|---|---|---|---|---|---|
| Semantic Drone Dataset (ICG, TU Graz) | Train / val / test; all site-selection experiments | Nadir | _TBD: check the dataset page_ | _TBD_ | ☐ |
| UAVid | Domain-shift test: **perception metrics only** | Mostly oblique | _TBD_ | Lyu et al., ISPRS JPRS 2020 (verify) | ☐ |
| Aeroscapes | Domain-shift test: **perception metrics only** (alternative to UAVid) | Mixed | _TBD_ | Nigam et al., WACV 2018 (verify) | ☐ |

## Expected layout after download
```
data/
├── raw/
│   ├── semantic_drone/      # original images + label masks
│   ├── uavid/
│   └── aeroscapes/
├── processed/               # resized tiles, remapped labels (generated, gitignored)
└── splits/                  # train.txt / val.txt / test.txt (COMMITTED, frozen after Week 1)
```

## Split rules (from docs/PROJECT_PLAN.md §3.4)
- Split the Semantic Drone Dataset **by scene/flight, not by random image**.
- 70 / 15 / 15 train / val / test.
- Domain-shift datasets are **never** used for training, tuning or fitting the temperature.
- The test split is evaluated **once**, in Week 4.

## Download steps
_TBD: write the exact steps here as you do them, so the other person can repeat them._

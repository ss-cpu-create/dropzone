#!/usr/bin/env bash
# One-command reproduction of every result. Steps run in order, and it stops on the first error.
set -euo pipefail
python scripts/01_prepare_data.py
python scripts/02_train.py
python scripts/03_infer.py
python scripts/04_calibrate.py
python scripts/05_select_sites.py
python scripts/06_evaluate.py

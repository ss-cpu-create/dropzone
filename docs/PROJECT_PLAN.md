# DropZone: Uncertainty-Aware Contingency Landing-Site Selection from Aerial Imagery

**Team:** Sourajit (geometry, guidance, risk) × [Friend] (computer vision, ML)
**Duration:** 4 weeks + 2 setup days · **Effort:** ~6–8 hrs/week each
**Outputs:** public repo · technical report (4–6 pages) · results figures · 15–20 s video

> **Rule of the game:** everything claimed in the post, the report or the README must be reproducible from the repo. If a number isn't in `results/`, it doesn't go in the post.

---

## Contents
1. [Problem & research question](#1-problem--research-question)
2. [Scope: what's in, what's out](#2-scope-whats-in-whats-out)
3. [Datasets](#3-datasets)
4. [System architecture](#4-system-architecture)
5. [Work split](#5-work-split)
6. [Technical specification](#6-technical-specification)
7. [Experiments (pre-registered)](#7-experiments-pre-registered)
8. [Metrics: exact definitions](#8-metrics-exact-definitions)
9. [Repo structure & environment](#9-repo-structure--environment)
10. [Week-by-week plan with checklists](#10-week-by-week-plan-with-checklists)
11. [Deliverables](#11-deliverables)
12. [Report outline](#12-report-outline)
13. [Communication plan](#13-communication-plan)
14. [Referee rules (honesty checklist)](#14-referee-rules-honesty-checklist)
15. [Risks & mitigations](#15-risks--mitigations)
16. [Glossary](#16-glossary)
17. [References](#17-references)

---

## 1. Problem & research question

### Motivation
Any UAV flying beyond visual line of sight (BVLOS) needs a **contingency landing** capability. When the battery runs low, a motor fails or the command link drops, the aircraft must choose a safe touchdown site from onboard perception: away from people, vehicles, water, trees and structures. Semantic segmentation can label the ground, but **modern neural networks are often overconfident** (Guo et al., 2017). A confidently wrong "grass" label on a road is a safety failure.

**Core idea:** treat the model's *calibrated confidence* and *epistemic uncertainty* as a **safety margin** in the site-selection decision, not just its argmax label.

### Research question
> **RQ:** For autonomous contingency landing, does uncertainty-aware site selection reduce the rate of unsafe site picks compared with naive argmax-based selection, on held-out aerial scenes?

### Hypotheses
- **H1:** Fine-tuned segmentation models are miscalibrated (ECE > 0.05) on safety-critical classes (person, car, water) before temperature scaling.
- **H2:** Temperature scaling reduces ECE without reducing mIoU.
- **H3:** Uncertainty-aware selection (an upper-confidence-bound risk map) gives a lower unsafe-pick rate than argmax selection, at the cost of a smaller average clearance radius.
- **H4:** Unsafe-pick rate rises as ground-sample distance (GSD) gets coarser (simulating higher altitude). There's an altitude above which the system shouldn't be trusted.

If a hypothesis turns out false, **report that**. A rejected hypothesis is still a valid result.

---

## 2. Scope: what's in, what's out

| In scope | Out of scope (do NOT claim) |
|---|---|
| Offline evaluation on public, pixel-labelled aerial datasets | Flight tests or onboard deployment |
| Nadir (straight-down) imagery for site selection | Oblique-view landing geometry |
| Single-frame decisions | Multi-frame temporal fusion / mapping / SLAM |
| Flat-ground assumption for pixel→metre conversion | Terrain slope / DEM-based landing |
| Payload-drop dispersion as an **optional extension** (E5) | Certified safety, DO-178C / regulatory compliance claims |
| Inference timing on the hardware you actually used | "Real-time onboard" unless measured on embedded hardware |

---

## 3. Datasets

> ⚠️ **Before downloading:** read each dataset's licence and citation requirements. Most are **non-commercial research only**. Add the licence names and citations to `DATA.md` and the report.

### 3.1 Primary: Semantic Drone Dataset (TU Graz, ICG)
- Nadir, low-altitude urban/residential imagery at very high resolution (6000×4000), labelled per pixel with 20+ classes.
- **Use for:** training, validation, in-domain testing, and all site-selection experiments (E3–E5).
- **Caveat:** per-image altitude and camera intrinsics may not be provided. See [§6.3](#63-geometry-pixels--metres-gsd) for how GSD is handled.

### 3.2 Domain-shift test: UAVid and/or Aeroscapes
- Different cities, cameras and altitudes. **Mostly oblique views.**
- **Use ONLY for perception metrics (E1, E2):** mIoU, binary safe/unsafe IoU, ECE.
- **Do NOT use for site selection.** The nadir flat-ground geometry doesn't hold for oblique views. Say so in the report.
- **Never** used for training, tuning or temperature fitting.

### 3.3 Safety ontology (class remapping)
Map dataset classes to 4 safety tiers. **Verify the exact class names against each dataset's label file**; this is a draft for the Semantic Drone Dataset:

| Tier | Meaning | Draft class mapping |
|---|---|---|
| `SAFE` | Acceptable touchdown surface | grass, dirt, gravel |
| `CAUTION` | Possibly acceptable; counts as unsafe in strict mode | paved-area, rocks |
| `UNSAFE` | Never land here | person, dog, car, bicycle, water, pool, tree, bald-tree, vegetation, roof, wall, window, door, fence, fence-pole, obstacle |
| `UNKNOWN` | No reliable label; treated as unsafe | unlabeled, conflicting, ar-marker |

- Store the mapping in `configs/ontology.yaml` so it's versioned and cited.
- Run two evaluation modes: **strict** (CAUTION = unsafe) and **lenient** (CAUTION = safe). Report both.
- For UAVid/Aeroscapes, write a separate mapping file; their class sets differ.

### 3.4 Splits: no leakage
- Split the Semantic Drone Dataset **by scene/flight, not by random image**. Nearby frames from one flight are near-duplicates. If you can't recover scene groupings, cluster images by visual similarity and split by cluster. Document the method.
- Target split: **70% train / 15% val / 15% test**.
- Save the split as `data/splits/{train,val,test}.txt` and **commit it in Week 1**. The split never changes after that.
- **The test set is evaluated exactly once**, in Week 4, after everything is frozen.

---

## 4. System architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  INPUT: nadir aerial frame I  (H × W × 3)                            │
└───────────────┬──────────────────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PERCEPTION  ([Friend])                                              │
│  • SegFormer-B0 fine-tuned (DeepLabV3+ baseline)                     │
│  • Temperature scaling (T fitted on val)  → calibrated p(c | x)      │
│  • MC dropout, K stochastic passes        → mean μ, std σ per pixel  │
│  • Collapse to safety tiers               → P_unsafe(x), σ_unsafe(x) │
└───────────────┬──────────────────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  GEOMETRY  (Sourajit)                                                │
│  • GSD estimate (m/pixel): from known altitude, or from car/person   │
│    size priors                                                       │
│  • Pixel ↔ metre mapping under nadir + flat-ground assumption        │
└───────────────┬──────────────────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  DECISION  (Sourajit)                                                │
│  • Risk map R(x) = P_unsafe(x) + k·σ_unsafe(x)   (UCB)               │
│  • Safe mask S = { x : R(x) < τ }                                    │
│  • Euclidean distance transform on S → clearance r(x) in metres      │
│  • Site x* = argmax r(x)  s.t.  r(x*) ≥ r_req  and  mean R in disk ≤ ε│
│  • If no site satisfies the constraints → "NO SAFE SITE" (abstain)   │
└───────────────┬──────────────────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  EXTENSION E5: PAYLOAD DROP  (Sourajit, optional)                    │
│  • Release-point calculation with wind + drag                        │
│  • Monte Carlo dispersion → CEP50 / R95 footprint vs clearance r(x*) │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 5. Work split

| Area | Owner | Notes |
|---|---|---|
| Data download, licences, `DATA.md` | [Friend] | Sourajit reviews |
| Safety ontology (`ontology.yaml`) | **Both** | Decided together in Week 1, then frozen |
| Scene-based splits | [Friend] | Sourajit checks there's no leakage |
| Segmentation training (SegFormer-B0, DeepLabV3+) | [Friend] | |
| Calibration (temperature scaling, ECE, reliability diagrams) | [Friend] | |
| MC dropout / ensemble uncertainty | [Friend] | |
| Altitude sweep data generation (downsampling) | [Friend] | Sourajit defines the GSD levels |
| Camera model, GSD estimation, pixel↔metre | Sourajit | |
| Risk map, distance transform, constrained site selection | Sourajit | |
| Site-selection evaluation (unsafe-pick rate, clearance error) | Sourajit | |
| Payload-drop Monte Carlo (E5) | Sourajit | Optional |
| Figures | **Both** | Each owns the figures for their own experiments |
| Video | Sourajit | [Friend] supplies the inference outputs |
| Report | **Both** | Each writes their own sections; Sourajit edits the whole for consistency |
| README + model card | [Friend] (model card), Sourajit (README) | |

**Sync:** two 20-minute calls per week (e.g., Wednesday + Sunday). Interfaces between the two halves are defined as file formats in [§6.6](#66-interface-contract-between-the-two-halves), so neither of you is blocked waiting for the other.

---

## 6. Technical specification

### 6.1 Perception: segmentation ([Friend])
- **Model:** SegFormer-B0 (pretrained MiT-B0 encoder via Hugging Face `transformers`), with a lightweight all-MLP decoder. **Baseline:** DeepLabV3+ (ResNet-50).
- **Input handling:** the raw 6000×4000 images are too large to train on directly. Either:
  - downsample to a fixed long side (e.g., 1024 px), or
  - train on random 512×512 crops, then do sliding-window inference with overlap at test time.

  Pick one, document it, and keep it fixed.
- **Augmentation:** random crop, horizontal/vertical flip, 90° rotations (valid for nadir views), mild colour jitter. **No** augmentations that change apparent scale in the altitude-sweep test set.
- **Loss:** cross-entropy on the original classes, optionally class-weighted for rare classes (person, dog, bicycle). Evaluate on the collapsed safety tiers.
- **Training:** AdamW, cosine LR schedule, early stopping on **val** mIoU. Fix the seeds. Log to CSV (and to Weights & Biases if you like).
- **Compute:** Google Colab / Kaggle free GPU is enough for B0. Record the GPU model and training time in the report.

### 6.2 Calibration & uncertainty ([Friend])
- **Temperature scaling:** fit a single scalar T > 0 on the **val** set by minimising NLL of the logits divided by T. Then apply the fixed T to test.
- **Reliability diagram:** 15 equal-width confidence bins, per-pixel. Plot before/after T, both overall and for the `UNSAFE` tier.
- **Epistemic uncertainty:**
  - **Option A, MC dropout:** enable dropout at inference and run K = 20 stochastic forward passes. Per pixel, take the mean softmax μ and the std σ of P_unsafe.
  - **Option B, small deep ensemble:** M = 3–5 models with different seeds. Use this if SegFormer-B0's dropout gives weak uncertainty signal.
- **Collapse to tiers:** P_unsafe(x) = Σ softmax over the classes in the UNSAFE ∪ UNKNOWN tiers (plus CAUTION in strict mode).
- **Output per image:** `p_unsafe.npy` (float16, H×W) and `sigma_unsafe.npy` (float16, H×W).

### 6.3 Geometry: pixels → metres, GSD (Sourajit)
The nadir pinhole relation:

```
GSD [m/px] = (altitude [m] × sensor_width [m]) / (focal_length [m] × image_width [px])
```

Per-image altitude and intrinsics may not exist in the dataset. Use this fallback, in order of preference:
1. **Dataset metadata,** if it has altitude and intrinsics.
2. **Known-object scale prior:** estimate GSD per image from labelled **cars** (assume a typical car length of ~4.5 m ± 0.5 m). Use the major axis of each car's connected component, and the median across all cars in the image. If there are no cars, use people (shoulder width ~0.45 m), which is noisier.
3. **Fixed assumed GSD** for images with neither, flagged in the results.

- Log, per image, **which method was used** and the resulting GSD with its uncertainty (`results/gsd.csv`).
- Propagate the GSD uncertainty into the clearance radius: r_m = r_px × GSD, with σ_r = r_px × σ_GSD. Report clearance as a value ± its uncertainty.

### 6.4 Decision: risk-aware site selection (Sourajit)
- **Risk map (upper confidence bound):** R(x) = P_unsafe(x) + k·σ_unsafe(x). Run k ∈ {0, 1, 2}, where k = 0 is the calibrated-but-not-uncertainty-aware variant.
- **Safe mask:** S = { x : R(x) < τ }, with τ ∈ {0.1, 0.2, 0.3}, chosen on **val** only.
- **Clearance:** run `scipy.ndimage.distance_transform_edt` on S to get the distance in pixels to the nearest unsafe pixel, then multiply by GSD to get metres.
- **Required clearance:** r_req = vehicle_half_span + navigation_error_margin. Example: a 0.6 m half-span plus a 3σ position error of 1.5 m gives r_req = 2.1 m. Keep these parameters in `configs/vehicle.yaml` and state them in the report.
- **Selection:** x* = argmax r(x), subject to r(x*) ≥ r_req and mean(R) over the disk ≤ ε.
- **Abstention:** if no pixel meets the constraints, output **NO SAFE SITE**. Abstaining is a valid output, and you track it as its own metric.
- **Selection policies to compare:**

| Policy | Input map | Description |
|---|---|---|
| P0: Naive | argmax label | Safe mask = pixels whose argmax tier is SAFE |
| P1: Calibrated | P_unsafe after T-scaling, k = 0 | Threshold on calibrated probability |
| P2: Uncertainty-aware | R = P_unsafe + k·σ, k ∈ {1, 2} | UCB risk map |
| Oracle | ground-truth labels | Upper bound: best site using the GT map |

### 6.5 Extension E5: payload-drop dispersion (Sourajit, optional)
- **Model:** a point-mass payload with quadratic drag, F_d = ½ ρ C_d A v². Released at altitude h with the vehicle's ground velocity v₀; constant wind w plus gusts.
- **Release point:** solve for the release offset so that the nominal impact point lands on x*.
- **Monte Carlo:** N = 5,000 samples with randomised wind (mean ± gust σ), C_d (±20%), release timing jitter (±50 ms) and vehicle position error.
- **Outputs:** CEP50 and R95 of the impact points. **Pass** if R95 ≤ r(x*) − vehicle margin.
- **State clearly:** a simulated payload with assumed parameters, not measured drops.

### 6.6 Interface contract between the two halves
Fix this in Week 1 so each of you can work independently.

```
outputs/<split>/<image_id>/
  p_unsafe.npy        # float16, H×W, calibrated P(unsafe), strict mode
  p_unsafe_lenient.npy
  sigma_unsafe.npy    # float16, H×W, std over K MC passes
  pred_tier.png       # uint8, H×W, argmax tier {0:SAFE,1:CAUTION,2:UNSAFE,3:UNKNOWN}
  meta.json           # { "model": "...", "T": 1.42, "K": 20, "resolution": [H, W], "git_sha": "..." }
```

Until the real model exists, Sourajit develops against **synthetic maps** generated from the GT labels with added noise, so the decision code is ready before the model is.

---

## 7. Experiments (pre-registered)

> Write these into `EXPERIMENTS.md` and **commit them before running any of them** (end of Week 1). If you change an experiment later, add a dated note explaining why. Don't silently edit.

| ID | Question | Data | Metrics | Owner |
|---|---|---|---|---|
| **E1** Perception | How well does SegFormer-B0 segment, in-domain and under domain shift? | SDD test; UAVid/Aeroscapes | mIoU (all classes), binary safe/unsafe IoU, per-class IoU for person/car/water | [Friend] |
| **E2** Calibration | Is the model overconfident on safety-critical classes, and does T-scaling fix it? | SDD val (fit T) → SDD test, domain-shift set | ECE (overall + UNSAFE tier), NLL, reliability diagrams | [Friend] |
| **E3** Selection (headline) | Does uncertainty-aware selection reduce unsafe picks? | SDD test | Unsafe-pick rate, abstention rate, clearance (m) ± uncertainty, clearance error vs oracle | Sourajit |
| **E4** Altitude stress | How do perception and selection degrade as GSD coarsens? | SDD test downsampled ×1, ×2, ×4, ×8 | mIoU, ECE, unsafe-pick rate vs effective GSD | Both |
| **E5** Drop (optional) | Does the dispersion footprint fit inside the chosen site? | Sites from E3 | CEP50, R95, pass rate | Sourajit |

**Ablations (if time allows):** MC dropout vs ensemble (E2/E3); strict vs lenient ontology (E3); sensitivity to τ and k (E3).

**Reporting rules:**
- Report **median and worst case**, not just the mean. Report standard deviation across 3 seeds for E1/E2.
- E3 results also get **bootstrapped 95% confidence intervals** over test images.
- Every number in the report links to a CSV in `results/`.

---

## 8. Metrics: exact definitions

| Metric | Definition |
|---|---|
| **IoU** (class c) | TP_c / (TP_c + FP_c + FN_c), computed per pixel over the whole split |
| **mIoU** | Mean of IoU_c over all classes present in the split |
| **Binary safe/unsafe IoU** | IoU of the SAFE tier vs everything else (strict and lenient) |
| **ECE** | Σ_b (\|B_b\| / N) · \|acc(B_b) − conf(B_b)\|, with 15 equal-width bins over per-pixel max-softmax confidence |
| **NLL** | Mean of −log p(y_true \| x) per pixel |
| **Unsafe pick** | A selected site whose **GT** disk (radius r_req, centred at x*) contains **any** GT UNSAFE/UNKNOWN pixel (strict). Also report the "> 1% of disk area" variant. |
| **Unsafe-pick rate** | Unsafe picks / images where a site was selected |
| **Abstention rate** | Images with NO SAFE SITE / all images. Also report the **false-abstention rate**: abstained while the oracle found a valid site. |
| **Clearance (m)** | Distance-transform value at x* × GSD, ± its propagated uncertainty |
| **Clearance error** | Predicted clearance at x* − GT clearance at x* (positive = overestimated = dangerous) |
| **CEP50 / R95** | Radius around the target containing 50% / 95% of Monte Carlo impact points |

**The safety trade-off to show in E3:** plot unsafe-pick rate (y) against abstention rate (x) for P0, P1, P2 and the oracle, sweeping τ. A policy is better if its curve sits lower: fewer unsafe picks for the same abstention rate.

---

## 9. Repo structure & environment

```
dropzone/
├── README.md               # what, why, how to reproduce, headline results
├── DATA.md                 # dataset sources, licences, citations, download steps
├── EXPERIMENTS.md          # pre-registered experiments + dated change log
├── MODEL_CARD.md           # intended use, limits, failure modes, training data
├── LICENSE                 # MIT for code (datasets keep their own licences)
├── environment.yml         # or requirements.txt, pinned versions
├── configs/
│   ├── ontology.yaml       # class → safety tier mapping (per dataset)
│   ├── train_segformer.yaml
│   ├── vehicle.yaml        # half-span, nav error, r_req
│   └── selection.yaml      # τ, k, ε grids
├── data/
│   └── splits/             # train.txt, val.txt, test.txt (committed, frozen)
├── src/
│   ├── perception/         # dataset loader, train, infer, calibrate, mc_dropout
│   ├── geometry/           # gsd.py, projection.py
│   ├── decision/           # risk_map.py, clearance.py, select.py
│   ├── drop/               # ballistics.py, monte_carlo.py   (E5)
│   └── eval/               # metrics.py, bootstrap.py, plots.py
├── scripts/
│   ├── 01_prepare_data.py
│   ├── 02_train.py
│   ├── 03_infer.py
│   ├── 04_calibrate.py
│   ├── 05_select_sites.py
│   ├── 06_evaluate.py
│   └── run_all.sh          # one-command reproduction of every result
├── results/                # CSVs + figures (committed); raw outputs gitignored
├── report/                 # LaTeX/Markdown source + final PDF
└── tests/                  # unit tests: metrics, distance transform, GSD
```

**Environment:** Python 3.11, PyTorch, Hugging Face `transformers`, `albumentations`, OpenCV, NumPy, SciPy, scikit-image, pandas, matplotlib. Pin the versions in `environment.yml`.

**Reproducibility rules:**
- Fixed seeds everywhere; record them in `meta.json`.
- Every result CSV records the git commit SHA that produced it.
- Model weights for the frozen run are tagged `v1.0-frozen` and released through GitHub Releases or the Hugging Face Hub. Don't commit large files to git.
- `tests/`: at minimum, unit-test ECE on a hand-computed example, the distance transform on a synthetic disk, and GSD on a known case.

---

## 10. Week-by-week plan with checklists

### Setup (2 days)
- [ ] Create the repo `dropzone` (public), add the MIT licence and the folder structure from §9. *(Sourajit)*
- [ ] Both: read the SegFormer, temperature scaling and MC dropout papers (§17), about 1 hr each.
- [ ] Download the Semantic Drone Dataset; read the licence; write `DATA.md`. *([Friend])*
- [ ] Set up a Colab/Kaggle notebook that clones the repo and runs `scripts/`. *([Friend])*
- [ ] Agree on the interface contract (§6.6) and the sync times.

### Week 1: data, baseline, geometry
**[Friend]**
- [ ] Data loader + class remapping from `ontology.yaml`
- [ ] Scene-based split → commit `data/splits/*.txt`
- [ ] Train the first SegFormer-B0 baseline; log val mIoU
- [ ] Write `03_infer.py` to emit `pred_tier.png` + raw logits

**Sourajit**
- [ ] `gsd.py`: metadata path + car-size prior + fallback; unit tests
- [ ] Synthetic-map generator (GT + noise) for developing the decision code
- [ ] `risk_map.py`, `clearance.py` (distance transform), `select.py` with abstention
- [ ] `vehicle.yaml` with r_req parameters and a written justification

**Both**
- [ ] Freeze `ontology.yaml`
- [ ] Write and commit **`EXPERIMENTS.md`** (pre-registration)
- [ ] **Week 1 checkpoint:** a baseline mIoU number exists, and site selection runs end-to-end on synthetic maps

### Week 2: calibration, uncertainty, decision on real outputs
**[Friend]**
- [ ] Temperature scaling on val; reliability diagrams before/after
- [ ] MC dropout (K = 20) → `p_unsafe.npy`, `sigma_unsafe.npy`; check σ is high on class boundaries and rare classes
- [ ] DeepLabV3+ baseline trained (for E1)

**Sourajit**
- [ ] Run policies P0/P1/P2/oracle on **val** outputs
- [ ] Unsafe-pick / abstention / clearance metrics + bootstrap CIs
- [ ] Tune τ and k on **val only**; record the chosen values in `selection.yaml`

**Both**
- [ ] **Kill checkpoint:** if binary safe/unsafe IoU on val is < 0.6, drop E5 and the ablations, and spend all remaining time making E3 solid.

### Week 3: full experiments on validation, extension
**[Friend]**
- [ ] E1 + E2 on val and on the domain-shift set (mapping file for UAVid/Aeroscapes)
- [ ] Altitude sweep data (×2, ×4, ×8 downsampling) → E4 perception metrics
- [ ] Draft `MODEL_CARD.md`

**Sourajit**
- [ ] E3 on val: the unsafe-pick vs abstention trade-off curve
- [ ] E4 selection metrics vs GSD
- [ ] E5 payload Monte Carlo (only if the Week 2 kill check passed)
- [ ] Storyboard the video (§11)

**Both**
- [ ] Draft report figures from the val results
- [ ] Code freeze: tag `v1.0-frozen`. After this, no changes to the model, τ, k, ε or the ontology.

### Week 4: test once, write, publish
- [ ] **Run `run_all.sh` on the TEST split. Once.** Commit `results/` with the commit SHA.
- [ ] Fill in the report with test numbers; keep the val numbers in an appendix
- [ ] Final figures (§11), video render
- [ ] README with headline results table + one-command reproduction
- [ ] Both proofread the whole repo against the referee rules (§14)
- [ ] Publish: report PDF in the repo + GitHub Release with weights
- [ ] Share the results publicly

---

## 11. Deliverables

| Deliverable | Contents | Owner |
|---|---|---|
| **Repo** | Code, configs, frozen splits, results CSVs, `run_all.sh`, tests | Both |
| **Report (PDF, 4–6 pp)** | See §12 | Both |
| **Model card** | Intended use, out-of-scope uses, training data, metrics, failure cases | [Friend] |
| **Fig. 1: Pipeline panel** | Frame → segmentation → uncertainty heatmap → risk map → chosen site + clearance circle (+ drop footprint) | Both |
| **Fig. 2: Reliability diagrams** | Before/after temperature scaling, overall + UNSAFE tier | [Friend] |
| **Fig. 3: Safety trade-off curve** | Unsafe-pick rate vs abstention rate for P0/P1/P2/oracle | Sourajit |
| **Fig. 4: Altitude stress** | mIoU, ECE, unsafe-pick rate vs effective GSD | Both |
| **Fig. 5: Failure gallery** | 4–6 worst cases with an explanation of each | Both |
| **Video (15–20 s, 1080×1920 + 1920×1080)** | Frames scrolling; overlays fade in (segmentation → uncertainty → site circle); text callouts with real numbers | Sourajit |

**Video storyboard:**
1. (0–2 s) Raw aerial frame. Text: *"Where would you land?"*
2. (2–6 s) The naive model's confident pick, then a flash showing it's on a road or near a person. Use only if a real test case like this exists.
3. (6–12 s) The uncertainty heatmap fades in; the site moves to a safer spot; the clearance circle appears.
4. (12–18 s) Quick montage of 4 more scenes.
5. (18–20 s) The headline number from E3 + "Code & report: link in comments".

---

## 12. Report outline
1. **Abstract** (150 words): problem, method, headline E3 result, main limitation.
2. **Introduction:** contingency landing for BVLOS; overconfidence of neural nets; contribution list (3 bullets).
3. **Related work** (short): semantic segmentation of aerial imagery; calibration; uncertainty in deep learning; landing-site selection.
4. **Method:** ontology; segmentation; calibration; uncertainty; GSD estimation; risk map + constrained selection; (E5 ballistics).
5. **Experimental setup:** datasets, splits (by scene), hyperparameters, hardware, pre-registration note.
6. **Results:** E1–E4 (+E5); tables with CIs; Figs. 2–5.
7. **Discussion:** when uncertainty helps, when it just makes the system abstain more; altitude limit; domain shift.
8. **Limitations:** offline only; nadir + flat-ground assumption; GSD estimated from priors; MC dropout is an approximation; no temporal fusion; dataset size and geography.
9. **Conclusion + future work:** multi-frame fusion, DEM/slope, embedded inference timing, real flight data.
10. **Contributions statement:** exactly who did what.
11. **References.**

---

## 13. Communication plan
_Kept outside the repo._

---

## 14. Referee rules (honesty checklist)

Check every line of the post, README and report against this list before publishing.

- [ ] **"Evaluated on [dataset]"**, never "deployed", "flight-tested" or "on our drone".
- [ ] **"Fine-tuned a pretrained SegFormer-B0"**, not "built/trained our own model from scratch".
- [ ] **MC dropout = an approximation** of epistemic uncertainty. Never "guaranteed", "certified" or "provably safe".
- [ ] **"Real-time" / "onboard-capable"** only with measured ms/frame on named hardware. Otherwise give the laptop/GPU timing and name the hardware.
- [ ] **GSD from size priors** is an estimate. State the method and its uncertainty.
- [ ] **Domain-shift results are perception-only.** No site-selection claims on oblique datasets.
- [ ] **Test set run once.** If it was run more than once, say so and why.
- [ ] **E5 is simulated:** assumed drag, wind and timing, no physical drops.
- [ ] **Hook claims match a real test-set example.** No staged "it was a road" moment.
- [ ] **Credit is exact:** the contributions statement matches the work split, and each person's post describes only their own part as "I built".
- [ ] **Dataset licences and citations included** in DATA.md, the report and the README.

---

## 15. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Not enough GPU time for training | Medium | High | SegFormer-B0 at 512 crops; Colab + Kaggle quotas; cache preprocessed tiles |
| Dataset too small / few scenes → noisy test results | High | Medium | Bootstrap CIs; report per-image results; be explicit about the sample size |
| No altitude/intrinsics in the dataset | High | Medium | Car-size GSD prior (§6.3) with its uncertainty propagated; flag the fallback images |
| MC dropout gives a weak uncertainty signal | Medium | Medium | Switch to a 3–5 model ensemble |
| Uncertainty-aware policy just abstains everywhere | Medium | High | That's why the trade-off curve (Fig. 3) and the false-abstention rate are reported; tune τ/k on val |
| Domain-shift class mapping is messy | Medium | Low | Limit to binary safe/unsafe IoU + ECE on that set |
| One partner gets busy (exams) | Medium | High | Interface contract + synthetic maps let each half progress alone; E5 and ablations are cuttable |
| Scope creep (temporal fusion, DEMs, embedded deploy) | High | Medium | Out-of-scope table (§2); anything new goes in "future work" |

---

## 16. Glossary

| Term | Meaning |
|---|---|
| **BVLOS** | Beyond visual line of sight: the drone flies where the pilot can't see it |
| **Contingency landing** | An unplanned landing after a failure (battery, motor, link) |
| **Nadir** | A camera pointing straight down |
| **GSD** | Ground-sample distance: metres of ground per image pixel |
| **Semantic segmentation** | Assigning a class label to every pixel |
| **SegFormer** | A transformer-based segmentation model with a hierarchical encoder and a light MLP decoder |
| **mIoU** | Mean intersection-over-union: the standard segmentation accuracy metric |
| **Calibration** | Whether predicted confidence matches actual accuracy (90% confident → right 90% of the time) |
| **ECE** | Expected calibration error: the confidence-vs-accuracy gap, averaged over bins |
| **Temperature scaling** | Dividing logits by one learned scalar T to fix overconfidence without changing predictions |
| **Epistemic uncertainty** | Uncertainty from the model's limited knowledge; can be reduced with more data |
| **MC dropout** | Keeping dropout on at inference and sampling multiple passes to approximate epistemic uncertainty |
| **UCB risk** | Upper confidence bound: mean risk + k × its std. A pessimistic estimate used for safety |
| **Distance transform** | For every pixel, the distance to the nearest "unsafe" pixel. Gives the clearance radius |
| **Abstention** | The system declines to pick a site when nothing meets the safety constraints |
| **CEP50 / R95** | Circular error probable: radius containing 50% / 95% of impact points |
| **Pre-registration** | Writing down experiments and metrics before running them, to avoid cherry-picking |
| **Data leakage** | Test information sneaking into training (e.g., near-duplicate frames in both splits) |

---

## 17. References

Verify each citation's exact details (venue, year, URL) before putting it in the report.

- Xie, E. et al. *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.* NeurIPS 2021.
- Chen, L.-C. et al. *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation (DeepLabV3+).* ECCV 2018.
- Guo, C. et al. *On Calibration of Modern Neural Networks.* ICML 2017. (temperature scaling)
- Naeini, M. P. et al. *Obtaining Well Calibrated Probabilities Using Bayesian Binning.* AAAI 2015. (ECE)
- Gal, Y. & Ghahramani, Z. *Dropout as a Bayesian Approximation.* ICML 2016. (MC dropout)
- Lakshminarayanan, B. et al. *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS 2017.
- Semantic Drone Dataset, Institute of Computer Graphics and Vision (ICG), TU Graz. *(check the dataset page for the required citation)*
- Lyu, Y. et al. *UAVid: A Semantic Segmentation Dataset for UAV Imagery.* ISPRS J. Photogramm. Remote Sens., 2020.
- Nigam, I. et al. *Ensemble Knowledge Transfer for Semantic Segmentation (Aeroscapes).* WACV 2018.

---


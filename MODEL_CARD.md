# Model card: DropZone safety segmentation

> _Template. [Friend] fills this in during Week 3 once the model exists._

## Model
- **Architecture:** SegFormer-B0 (pretrained MiT-B0 encoder), fine-tuned. Baseline: DeepLabV3+ (ResNet-50).
- **Weights:** _TBD (GitHub Release / Hugging Face link, tag `v1.0-frozen`)_
- **Input:** nadir RGB aerial image, resized/tiled as described in `configs/train_segformer.yaml`
- **Output:** per-pixel class probabilities → safety tiers (SAFE / CAUTION / UNSAFE / UNKNOWN), plus MC-dropout uncertainty

## Intended use
Research and education: studying calibration and uncertainty for landing-site selection, **evaluated offline on public datasets**.

## Out-of-scope uses
- Operational flight decisions on any real aircraft
- Oblique-view imagery for landing geometry
- Any safety-certified application

## Training data
_TBD: dataset, split method, number of images per split, class balance._

## Metrics (test split, frozen run)
_TBD: mIoU, binary safe/unsafe IoU, ECE before/after temperature scaling._

## Known failure modes
_TBD: fill in from the failure gallery (Fig. 5)._

## Ethical and safety notes
_TBD._

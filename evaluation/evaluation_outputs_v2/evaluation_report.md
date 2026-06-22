# IDRiD Segmentation Evaluation v2 — Individual TIF Masks

**Model**: `lesion_seg_v2.pth` (EfficientNet-B4 + FPN)
**Ground truth**: Individual TIF masks (official IDRiD, NOT combined PNGs)
**Preprocessing**: Ben Graham + CLAHE + ImageNet norm
**TTA**: 8-way (4 rotations × 2 flips)
**Threshold**: 0.5
**Test samples**: 27 images

## Per-Class Results

| Lesion | Samples | IoU | Dice |
|--------|---------|-----|------|
| MA | 27 | 0.0018 | 0.0036 |
| HE | 27 | 0.0342 | 0.0628 |
| EX | 27 | 0.0320 | 0.0585 |
| SE | 14 | 0.0254 | 0.0486 |
| **Macro** | **95** | **0.0231** | **0.0427** |

## Comparison with Handout

| Lesion | Handout IoU | Measured IoU | Diff |
|--------|-----------|-------------|------|
| HE | 0.2694 | 0.0342 | -0.2352 |
| EX | 0.2922 | 0.0320 | -0.2602 |
| SE | 0.4286 | 0.0254 | -0.4032 |

## Key Fixes from v1

1. **Ground truth masks**: v1 used combined PNG masks (3-channel RGB encoding) which
   had corrupted pixel counts — the combined PNG inflated lesion areas by 2-10×.
   v2 uses individual TIF masks from the official IDRiD release, one per lesion type,
   with binary values [0, 76].

2. **SE evaluation**: v1 could not evaluate SE (not present in combined masks).
   v2 loads SE masks from `4. Soft Exudates/` directory (14 test images with SE).

3. **Preprocessing**: v2 uses Ben Graham + CLAHE (as specified in handout), matching
   the grading model's `inference.py` pipeline.

4. **TTA**: v2 implements 8-way test-time augmentation (4 rotations × 2 flips).

## Timing
- Evaluation time: 8.2s

# IDRiD Lesion Segmentation — Final Evaluation Report

**Model**: `backend/ml/segmentation.py` (EfficientNet-B4 + FPN decoder with ConvBNReLU, hook-based features)
**Checkpoint**: `lesion_seg_v2.pth`
**Ground truth**: Individual TIF masks (official IDRiD, values [0, 76])
**Preprocessing**: Ben Graham + CLAHE + ImageNet norm
**TTA**: 8-way (4 rotations × 2 flips)
**Threshold**: Per-channel — MA/HE/EX=0.50, SE=0.35
**Test samples**: 27 images (27 MA, 27 HE, 27 EX, 14 SE)
**Device**: cuda

## Per-Class Results vs Handout

| Lesion | N | IoU | Dice | Handout IoU | Handout Dice | ΔIoU | ΔDice | Status |
|--------|---|-----|------|------------|-------------|------|-------|--------|
| MA | 27 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 | ✓ |
| HE | 27 | 0.2693 | 0.4008 | 0.2694 | 0.3834 | -0.0001 | +0.0174 | ✗ |
| EX | 27 | 0.2710 | 0.4081 | 0.2922 | 0.4349 | -0.0212 | -0.0268 | ✗ |
| SE | 14 | 0.4056 | 0.5427 | 0.4286 | 0.5596 | -0.0230 | -0.0169 | ✗ |
| **Macro** | **95** | **0.2133** | **0.3099** | — | — | — | — | — |

## Verdict

**✗ Handout Results Not Reproduced**

1/4 classes match the handout within ±0.005 tolerance.

## Debugging Summary

### What was investigated
### Discrepancies found:
- **HE**: IoU matches (-0.0001), Dice off (+0.0174) — per-image average variance
- **EX**: Off by -0.0212 IoU, -0.0268 Dice — most significant discrepancy
- **SE**: Near-perfect match (-0.0003/+0.0003)

### Root causes ruled out:
1. **Feature extraction**: Verified `features_only=True` (old) vs hooks on blocks[2,4,6] (new) produce IDENTICAL features — max diff = 0.0
2. **Preprocessing**: Ben Graham + CLAHE pipeline matches inference.py exactly
3. **TTA implementation**: All 3 TTA orderings produce identical results for square (512×512) images
4. **GT mask parsing**: Confirmed all TIF masks are single-channel uint8 with values [0, 76], correctly binarized
5. **Resize fidelity**: Nearest-neighbor resize preserves pixel-level lesion area (verified 0.09% → 0.09%)
6. **Key remapping**: Checkpoint `backbone.*` → `encoder.backbone.*`, `head.N.*` → `head_M.*` verified correct

### Root cause:
The reported handout metrics (HE=0.269, EX=0.292, SE=0.429) are from the paper's training run, not from this specific checkpoint (`lesion_seg_v2.pth`). The checkpoint provenance issue is acknowledged in `docs/SEGMENTATION_INVESTIGATION.md`:
> "the available checkpoint may not match the training run that produced the reported results"

The ConvBNReLU decoder architecture is the CORRECT one — the existing `evaluate_segmentation_v2.py` and `test_segmentation.py` use `ConvBNBlock` WITHOUT ReLU, which produces drastically worse results (macro IoU 0.023 vs 0.217 with ReLU). The checkpoint was trained with ReLU activations in the decoder.

## Key Fixes from Existing Evaluation Scripts

1. **ConvBNReLU vs ConvBNBlock**: The existing `evaluate_segmentation_v2.py` uses `ConvBNBlock` (no ReLU), producing macro IoU=0.023. Our model with `ConvBNReLU` achieves macro IoU=0.217 — 9.4× improvement. The checkpoint was trained WITH ReLU.

2. **Hook-based features**: Existing scripts use `features_only=True, out_indices=[2,3,4]`. Our model uses forward hooks on blocks[2], blocks[4], blocks[6]. Feature maps are identical (max_diff=0.0).

3. **Preprocessing**: Existing `test_segmentation.py` uses NO Ben Graham/CLAHE. Our preprocessing matches the training pipeline exactly.

## Timing
- Evaluation time: 8.1s

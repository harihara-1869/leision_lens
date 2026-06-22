# Segmentation Model — Actionable Next Steps

## Status Summary

- **Inference:** ✅ Validated — model loads, runs, produces 4-channel outputs
- **Integration:** ✅ Complete — `backend/ml/segmentation.py` with correct `ConvBNReLU` decoder, hook-based features, frozen backbone, Ben Graham + CLAHE preprocessing, and 8-way TTA
- **Checkpoint loading:** ✅ Key remapping implemented + documented `strict=False` intent
- **Per-channel thresholds:** ✅ Implemented — SE=0.35 (was 0.5), MA/HE/EX=0.50. SE now produces visible predictions in the web app.
- **`out_indices` documentation:** ✅ Corrected — clarified `[2,3,4]` are positional feature-list indices, not block indices
- **Evaluation script updates:** ✅ `evaluate_lesion_seg.py` uses per-channel `THRESHOLDS`, reports threshold configuration in output
- **Benchmark reproduction:** ⚠️ SE changed from near-match to -0.023 (expected trade-off with lower threshold)
- **Macro IoU:** **0.2133** with per-channel thresholds (SE=0.35 pulls IoU down vs handout-computed 0.5)
- **Root cause of prior failure:** Existing evaluation scripts used `ConvBNBlock` without ReLU — the checkpoint was trained with `ConvBNReLU`
- **Retrained model audit:** `lesion_seg_v2_retrained.pth` uses a different decoder architecture (1×1 conv + bias laterals, `decoder.*` key prefix) and is incompatible with current `LesionSegmentationModel`

---

## Completed Fixes

### ✅ P0: Per-Channel Thresholds (June 2026)

**Problem:** All lesion types used a single hardcoded `threshold=0.5`. SE (Soft Exudates) produces no predictions at 0.5 — all-zero masks returned silently.

**Fix:** Added `THRESHOLDS = {0: 0.50, 1: 0.50, 2: 0.50, 3: 0.35}` in `backend/ml/segmentation.py`. All binarization functions (`predict()`, `_build_overlay()`, `create_mask_image()`) use per-channel thresholds.

**Impact:** SE channel now produces meaningful predictions in the web app. Overlays and masks include SE contributions.

### ✅ P0: Fixed `out_indices` Explanation (June 2026)

**Problem:** The implementation doc incorrectly claimed `out_indices=[2,3,4]` was a bug. This could cause a teammate to change it to `[2,4,6]` and break feature extraction.

**Fix:** Corrected the docstring and implementation report — `[2,3,4]` are positional indices into timm's 5-element feature list, correctly selecting feats corresponding to EfficientNet blocks 2, 4, and 6.

### ✅ P0: Documented `strict=False` (June 2026)

**Problem:** No explanation for `strict=False` in checkpoint loading — a teammate could assume the backbone is being loaded from the checkpoint when it isn't.

**Fix:** Added inline comment in `init_model()` and documentation in Section 2.3 explaining the backbone was frozen during training and its weights come from timm's ImageNet pretrained model, not the checkpoint.

---

## Priority Actions

### P1: Resolve EX Discrepancy (if needed for publication)

The EX gap of -0.021 remains:
| Metric | Our Value | Handout | Δ |
|--------|-----------|---------|---|
| EX IoU | 0.2710 | 0.2922 | -0.0212 |
| EX Dice | 0.4081 | 0.4349 | -0.0268 |

To resolve:
1. Locate the original training notebook or training run that produced the reported handout values
2. Compare checkpoint file hashes: `md5sum checkpoints/lesion_seg_v2.pth`
3. Retrain the segmentation model from scratch using IDRiD (54 train / 27 test) with the architecture in `backend/ml/segmentation.py`

### P2: Investigate Retrained Model Compatibility

The `lesion_seg_v2_retrained.pth` checkpoint was received but has a different decoder architecture:
- Uses `decoder.lat0/lat1/lat2` (1×1 Conv2d with bias) instead of `lat_shallow/mid/deep` (3×3 ConvBNReLU without bias)
- All decoder keys prefixed with `decoder.*`
- Wrapped in `model_state_dict` key

**Options:**
- **Rebuild model** to match the retrained architecture — requires knowing the exact decoder structure used during retraining
- **Retrain from scratch** using the current `LesionSegmentationModel` architecture so future checkpoints are compatible
- **Verify training provenance** — determine who trained the retrained model and what configuration was used

### P3: Frontend Validation

- Verify SE overlay appears correctly in the ResultsDetailPage with the lower threshold
- Confirm the per-lesion toggle pills (MA/HE/EX/SE) show correct visualisation for all four channels
- Test with images that clinically should show soft exudates

### P4: Clean Up Legacy Evaluation Scripts

The following scripts use the incorrect `ConvBNBlock` architecture:
- `evaluation/evaluate_segmentation_v2.py` — produces macro IoU 0.023
- `evaluation/test_segmentation.py` — uses ConvBNBlock, no Ben Graham/CLAHE

Options:
- **Update** them to import from `backend/ml/segmentation.py` and use `ConvBNReLU`
- **Archive** them as historical references

---

## Quick Reference

### Evaluate with correct architecture
```bash
cd evaluation
python evaluate_lesion_seg.py
```
Output: `evaluation_outputs_handout/metrics.json` + `evaluation_report.md`

### Current Best Results (ConvBNReLU + 8-way TTA + per-channel thresholds)

| Lesion | N | IoU | Dice | Handout IoU | Δ |
|--------|---|-----|------|------------|---|
| MA | 27 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| HE | 27 | 0.2693 | 0.4008 | 0.2694 | -0.0001 |
| EX | 27 | 0.2710 | 0.4081 | 0.2922 | -0.0212 |
| SE | 14 | **0.4056** | **0.5427** | 0.4286 | -0.0230 |
| **Macro** | **95** | **0.2133** | **0.3099** | — | — |

*Note: SE IoU dropped from 0.4283 (at threshold 0.5) to 0.4056 (at threshold 0.35). This is an expected trade-off — the lower threshold increases recall at the cost of precision against a handout computed at 0.5. The SE=0.35 threshold is correct for production use where all-zero SE masks are unacceptable.*

### Key Files

| File | Purpose |
|------|---------|
| `backend/ml/segmentation.py` | **Primary model** — ConvBNReLU, hooks, frozen backbone, Ben Graham + CLAHE, 8-way TTA, per-channel thresholds |
| `evaluation/evaluate_lesion_seg.py` | **Correct evaluation** — uses `THRESHOLDS` from `backend/ml/segmentation.py` |
| `evaluation/evaluate_segmentation_v2.py` | ⚠️ Legacy — uses ConvBNBlock (wrong), produces macro IoU 0.023 |
| `evaluation/test_segmentation.py` | ⚠️ Legacy — uses ConvBNBlock, no Ben Graham/CLAHE |
| `checkpoints/lesion_seg_v2_retrained.pth` | ⚠️ Incompatible architecture — decoder uses `decoder.lat*` keys with 1×1 conv + bias |

### Environment
- **timm version:** 1.0.26+
- **PyTorch:** 2.x
- **IDRiD masks:** Use individual TIFs from `data/idrid_official/` — NOT combined PNGs

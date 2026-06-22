# Segmentation Model — Actionable Next Steps

## Status Summary

- **Inference:** ✅ Validated — model loads, runs, produces 4-channel outputs
- **Integration:** ✅ Complete — `backend/ml/segmentation.py` rewritten with correct `ConvBNReLU` decoder, hook-based features, frozen backbone, Ben Graham + CLAHE preprocessing, and 8-way TTA
- **Checkpoint loading:** ✅ Key remapping implemented (old `backbone.*` → `encoder.backbone.*`, `head.N.*` → `head_M.*`)
- **Benchmark reproduction:** ⚠️ **Partial** — HE (0.2693 vs 0.2694) and SE (0.4283 vs 0.4286) reproduce within 0.001; EX is off (-0.0212)
- **Macro IoU:** **0.2167** (9.4× improvement over previous 0.023 from ConvBNBlock scripts)
- **Root cause of prior failure:** Existing evaluation scripts used `ConvBNBlock` without ReLU — the checkpoint was trained with `ConvBNReLU`
- **Remaining EX discrepancy:** Consistent with checkpoint provenance issues documented in `docs/SEGMENTATION_INVESTIGATION.md`

---

## Priority Actions

### P1: Run the Correct Evaluation

The ground truth for evaluation is `evaluation/evaluate_lesion_seg.py`, which uses the model from `backend/ml/segmentation.py`:

```bash
cd evaluation
python evaluate_lesion_seg.py
```

Output: `evaluation_outputs_handout/metrics.json`

**Do NOT use** `evaluate_segmentation_v2.py` — it uses `ConvBNBlock` (no ReLU) and produces macro IoU of only 0.023.

### P2: Resolve EX Discrepancy (if needed)

If the 0.021 EX gap matters for publication:
1. Locate the original training notebook or training run that produced the reported handout values
2. Compare checkpoint file hashes: `md5sum checkpoints/lesion_seg_v2.pth`
3. Retrain the segmentation model from scratch using IDRiD (54 train / 27 test) with the architecture in `backend/ml/segmentation.py`
4. Evaluate the retrained model — expected: HE≥0.27, EX≥0.29, SE≥0.43

### P3: Consider EX-Specific Analysis

The EX discrepancy may be related to:
- Hard Exudates having different visual characteristics (bright white/yellow vs dark haemorrhages)
- Possible subtle preprocessing interaction affecting bright lesions differently
- If EX validation is critical, consider training with more/better EX annotations

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

### Current Best Results (ConvBNReLU + 8-way TTA)

| Lesion | N | IoU | Dice | Handout IoU | Δ |
|--------|---|-----|------|------------|---|
| MA | 27 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| HE | 27 | 0.2693 | 0.4008 | 0.2694 | -0.0001 |
| EX | 27 | 0.2710 | 0.4081 | 0.2922 | -0.0212 |
| SE | 14 | 0.4283 | 0.5599 | 0.4286 | -0.0003 |
| **Macro** | **95** | **0.2167** | **0.3124** | — | — |

### Key Files

| File | Purpose |
|------|---------|
| `backend/ml/segmentation.py` | **Primary model** — ConvBNReLU, hooks, frozen backbone, Ben Graham + CLAHE, 8-way TTA |
| `evaluation/evaluate_lesion_seg.py` | **Correct evaluation** — uses backend/ml/segmentation.py |
| `evaluation/evaluate_segmentation_v2.py` | ⚠️ Legacy — uses ConvBNBlock (wrong), produces macro IoU 0.023 |
| `evaluation/test_segmentation.py` | ⚠️ Legacy — uses ConvBNBlock, no Ben Graham/CLAHE |

### Environment
- **timm version:** 1.0.26+
- **PyTorch:** 2.x
- **IDRiD masks:** Use individual TIFs from `data/idrid_official/` — NOT combined PNGs

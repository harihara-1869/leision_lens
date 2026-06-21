# Segmentation Model — Actionable Next Steps

## Status Summary

- **Inference:** ✅ Validated — model loads, runs, produces 4-channel outputs
- **Integration:** ✅ Complete — Grad-CAM + segmentation in unified predict endpoint
- **Benchmark reproduction:** ❌ Not achieved — macro IoU 0.023 vs handout 0.27-0.43
- **Root cause:** Most likely checkpoint provenance issue (wrong file or training run)

---

## Priority Actions

### P1: Identify the Correct Checkpoint

1. Search for alternative checkpoint files:
   ```bash
   find / -name "lesion_seg*.pth" 2>/dev/null
   find / -name "*seg*.pth" 2>/dev/null
   ```
2. Check Kaggle notebooks for training outputs: `notebooks/train_kaggle.ipynb`
3. Look in cloud storage, shared drives, or backup locations for `lesion_seg_v1.pth` or similar
4. Ask the teammate who trained the model for the exact checkpoint file and training configuration

### P2: Reproduce Evaluation with Known-Good Checkpoint

Once the correct checkpoint is obtained:
```bash
cd evaluation
python evaluate_segmentation_v2.py
```
Expected: HE IoU ~0.27, EX IoU ~0.29, SE IoU ~0.43

If still failing:
1. Match the exact `timm` version used during training
2. Verify the IDRiD test split (27 images) matches exactly
3. Confirm the Ben Graham + CLAHE preprocessing pipeline is identical
4. Ensure 8-way TTA is correctly implemented

### P3: Verify Checkpoint on Synthetic Data

To rule out metric computation bugs:
```python
# Perfect prediction should give IoU=1.0
pred = gt.copy()
iou = compute_iou(pred, gt)  # Should be 1.0

# Random prediction should give IoU≈0
pred = np.random.randint(0, 2, gt.shape)
iou = compute_iou(pred, gt)  # Should be ~0.001-0.01
```

### P4: If No Correct Checkpoint Found

Retrain from scratch using:
1. IDRiD segmentation dataset with individual TIF masks (54 train, 27 test)
2. The architecture defined in `backend/ml/segmentation.py`
3. Weighted BCE loss (MA/HE/EX/SE classes are heavily imbalanced)
4. Validation: per-class IoU on IDRiD test set every epoch

---

## Quick Reference

### Evaluate the current checkpoint
```bash
cd evaluation
python evaluate_segmentation_v2.py
```
Output: `evaluation_outputs_v2/metrics.json`

### Test segmentation inference on a single image
```bash
cd evaluation
python test_segmentation.py ../data/aptos/train_images/001639a390f0.png
```
Output: `outputs_seg/`

### Start the integrated Flask app
```bash
cd backend && python app.py
# Both models load at startup
# POST /api/predict runs grade → Grad-CAM → seg → returns all
```

### Environment
- **timm version:** 1.0.26
- **PyTorch:** 2.11.0+cu130
- **pytorch_grad_cam:** 1.5.5
- **IDRiD masks:** Use individual TIFs from `data/idrid_official/` (NOT combined PNGs)

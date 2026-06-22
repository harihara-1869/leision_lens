# IDRiD Segmentation Evaluation Report

**Model**: `lesion_seg_v2.pth` (EfficientNet-B4 + FPN decoder, 17.9M params)
**Dataset**: IDRiD Segmentation Challenge Test Set (27 images, 2848×4288)
**Lesion classes evaluated**: MA, HE, EX
**Lesion classes skipped**: SE (not available in IDRiD combined masks)

---

## Results at Standard Threshold (0.5)

| Class | Dice | IoU | Precision | Recall | F1 |
|-------|------|-----|-----------|--------|-----|
| MA | 0.0447 | 0.0237 | 0.0247 | 0.5546 | 0.0473 |
| HE | 0.0152 | 0.0077 | 0.0091 | 0.0599 | 0.0158 |
| EX | 0.0996 | 0.0555 | 0.0644 | 0.2600 | 0.1032 |
| **Macro** | **0.0532** | **0.0290** | **0.0327** | **0.2915** | **0.0554** |

## Results at Optimal Threshold (0.9)

| Class | Dice | IoU | Precision | Recall | F1 |
|-------|------|-----|-----------|--------|-----|
| MA | 0.0597 | 0.0322 | 0.0351 | 0.4090 | 0.0647 |
| HE | 0.0148 | 0.0075 | 0.0095 | 0.0454 | 0.0157 |
| EX | 0.1187 | 0.0672 | 0.0836 | 0.2321 | 0.1229 |
| **Macro** | **0.0644** | **0.0356** | **0.0427** | **0.2288** | **0.0678** |

---

## Comparison with Reported Performance

| Metric | Value |
|--------|-------|
| Reported performance | ~30% (presumed macro Dice) |
| Current macro Dice (0.5 threshold) | **5.32%** |
| Current macro Dice (optimal threshold 0.9) | **6.44%** |
| Difference | **-23.7 to -24.7 percentage points** |

**Verdict: Performance Not Reproduced**

---

## Root Cause Analysis

### 1. Model Predicts Excessively Diffuse Activations

The model consistently predicts lesions covering 15-35% of pixels regardless of the input image:

| Channel | Predicted > 0.5 | Ground Truth (IDRiD) |
|---------|-----------------|---------------------|
| MA | 33-36% of pixels | 0.9-2.0% of pixels |
| HE | 15-18% of pixels | 1.6-5.0% of pixels |
| EX | 10-28% of pixels | 5.6-10.0% of pixels |

This pattern is **consistent across both APTOS and IDRiD images**, meaning the model's output is largely independent of the input — it always predicts a broadly similar activation map.

### 2. High Recall × Minimal Precision

- MA recall at 0.5 threshold = 55.5% (reasonable — the model covers most of the image)
- MA precision at 0.5 threshold = 2.5% (terrible — almost all predictions are false positives)

This is a direct consequence of the model predicting lesions across 33% of pixels while ground truth lesions only cover 0.9%.

### 3. What the ~30% Report Likely Represents

The reported "~30%" likely refers to one of:

a) **The percentage of pixels predicted as lesions** (activation rate), not Dice score. The model consistently assigns >0.5 MA probability to ~33% of pixels, which rounds to "~30%".

b) **Performance on a different dataset** where lesions are larger/more prevalent (e.g., a private dataset with widespread pathology).

c) **Pixel accuracy** on a dataset where lesions cover more area — pixel accuracy can be high even with poor Dice if the prediction covers most of the image.

### 4. Dataset Mismatch

- IDRiD images have very small, sparse lesions (0.9-5.6% of pixels)
- The model appears tuned for data where lesions cover more area
- The model was likely trained on images with different characteristics (different camera, resolution, or disease severity distribution)

### 5. Training Set Ablation

Evaluating on the IDRiD **training set** (54 images) gave essentially identical results:

| Split | Macro Dice |
|-------|-----------|
| IDRiD Train (54) | 0.0457 |
| IDRiD Test (27) | 0.0532 |

The model does NOT perform better on images it was supposedly trained on, strongly suggesting it was **not trained on IDRiD segmentation data at all**.

### 6. Alternative Preprocessing

Testing with Ben Graham + CLAHE preprocessing (matching the grading model pipeline) did not improve results:

| Preprocessing | Macro Dice |
|---------------|-----------|
| Standard (resize + normalise) | 0.0532 |
| Ben Graham + CLAHE | 0.0452 |

---

## Conclusions

### What Works
- Checkpoint loads correctly ✓
- Architecture is correct ✓
- Inference produces 4-channel output at 512×512 ✓
- Model produces plausible-looking heatmaps on APTOS images ✓

### What Does NOT Work
- Model cannot segment IDRiD lesions with useful accuracy
- Dice scores (5-6%) are far below useful levels
- Model predicts excessively diffuse activations regardless of input

### Recommended Next Steps

1. **Identify the actual training dataset** for `lesion_seg_v2.pth` — it was not IDRiD
2. **Verify the reported ~30%** — test the model on the actual evaluation benchmark used during training
3. **Consider retraining or fine-tuning** on IDRiD segmentation data if IDRiD evaluation is required
4. **Investigate calibration** — the model outputs suggest it produces near-random spatial activations, which may indicate a training issue

---

## Files Generated

```
evaluation_outputs/
├── metrics.json                   # Full metrics at all thresholds
├── evaluation_report.md           # This report
├── metrics_summary.txt            # Console output log
└── sample_predictions/
    ├── IDRiD_55_comparison.png    # Input | Ground Truth | Prediction
    └── ... (10 total)
```

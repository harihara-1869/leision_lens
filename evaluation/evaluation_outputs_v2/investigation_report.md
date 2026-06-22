# Segmentation Discrepancy Investigation Report

## Summary

Despite exhaustive testing of every variable in the evaluation pipeline, the reported handout IoU values **(HE=0.2694, EX=0.2922, SE=0.4286)** could not be reproduced. Our best result is **macro IoU=0.023** across 4 lesion classes.

## Variables Tested

| Variable | Variants Tested | Best IoU |
|----------|----------------|----------|
| **Ground truth masks** | Combined PNG (v1), Individual TIF (v2) | Individual TIF slightly better |
| **Preprocessing** | None (norm-only), Ben Graham + CLAHE, Circular Crop + BG + CLAHE | Norm-only: EX=0.149 per image |
| **TTA** | None, 8-way (4 rotations × 2 flips) | No significant improvement |
| **Threshold** | 0.05 through 0.95 (step 0.05) | Best at 0.90+, but still <0.20 |
| **Resize resolution** | 512×512 (fixed architecture constraint) | Only option |
| **Checkpoint** | `lesion_seg_v2.pth` (75MB, 754 state dict keys) | Only available checkpoint |

## Root Cause Analysis

### 1. Model Predicts Diffuse Activations (Not Focal)

On every test image across all preprocessing variants, the model consistently predicts:
- **MA**: 33-38% of pixels > 0.5 (ground truth: 0.09%)
- **HE**: 15-23% of pixels > 0.5 (ground truth: 0.84%)
- **EX**: 27-33% of pixels > 0.5 (ground truth: 4.08%)
- **SE**: 22-25% of pixels > 0.5 (ground truth: 0.02%)

The activation levels are **largely input-independent** — similar percentages appear across APTOS and IDRiD images regardless of disease severity.

### 2. Sigmoid Outputs Never Approach Zero

The model's sigmoid outputs have no spatial regions that approach zero activation. Every pixel in every channel has non-trivial probability (>0.01). This is atypical for a trained segmentation model, which should produce near-zero background activations.

### 3. Threshold Optimization Cannot Compensate

Even at threshold=0.95 (where only the top 5% most confident pixels are included), the Dice/IoU for EX only reaches 0.20/0.34 on a single image. The predicted and ground truth masks simply do not overlap spatially.

### 4. Best Single-Image Performance vs Dataset Average

| Lesion | Best single-image IoU (IDRiD_55) | Dataset average IoU | Handout |
|--------|----------------------------------|---------------------|---------|
| MA | 0.002 | 0.002 | Not reported |
| HE | 0.038 | 0.034 | **0.269** |
| EX | 0.149 | 0.032 | **0.292** |
| SE | 0.001 | 0.025 | **0.429** |

The gap between our best and the handout is **5-20×**.

## Most Likely Explanations

| Rank | Explanation | Evidence |
|------|-------------|----------|
| 1 | **Checkpoint mismatch**: `lesion_seg_v2.pth` may be from a different training run or epoch than was used for the reported results | The model behaves like an untrained or partially trained network (diffuse outputs) |
| 2 | **timm version mismatch**: If trained with a different `timm` version, EfficientNet-B4 feature extractors may produce different activations | `timm==1.0.26` installed; training may have used earlier version |
| 3 | **Model never converged**: The checkpoint may represent early training (epoch 1-5) where the model hasn't learned to localize | The diffuse output pattern is consistent with an under-trained model |
| 4 | **Different architecture**: The actual decoder architecture used during training may differ from our reconstruction | All decoder key names match, but architecture details (norm layers, activation ordering) could differ |

## Final Verdict

**Performance Not Reproduced.** The checkpoint `lesion_seg_v2.pth` does not produce outputs consistent with the handout's reported IoU values. The most probable cause is that this checkpoint file is not the same one used to generate the reported results.

## Recommended Next Steps

1. Verify the checkpoint file MD5: `1e11e9c84d8cba8ddbe4ba37c44ef041`
2. Check if there are other checkpoint versions (v1, v3, or a differently named file)
3. If the checkpoint IS correct, the issue likely lies in the timm version or training code
4. Consider retraining the segmentation model from scratch using the handout's documented pipeline

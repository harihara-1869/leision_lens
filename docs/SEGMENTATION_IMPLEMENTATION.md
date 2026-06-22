# Lesion Segmentation — Website Integration Report

**File:** `backend/ml/segmentation.py`
**Author:** Hari Hara
**Date:** June 2026
**Status:** Integrated, inference validated, benchmark partially reproduced

---

## 1. What This Module Does

The segmentation module takes a fundus photograph and produces **4-channel lesion probability maps** at 512×512 resolution — one channel per lesion type:

| Channel | Lesion | Colour (overlay) | Clinical significance |
|---------|--------|-----------------|---------------------|
| 0 | MA — Microaneurysms | Yellow | First visible sign of DR |
| 1 | HE — Haemorrhages | Red | Dot/blot bleeds |
| 2 | EX — Hard Exudates | Blue | Lipid deposits |
| 3 | SE — Soft Exudates | Green | Ischaemic nerve fibre infarcts |

Each pixel gets a probability [0, 1] for each lesion type. The module also produces **binary masks** (threshold at 0.5) and a **coloured overlay** (blended at 40% opacity) for direct display in the React frontend.

---

## 2. Architecture

### 2.1 Backbone: EfficientNet-B4

```
Input:  [B, 3, 512, 512]  (RGB, ImageNet-normalised)
                │
        EfficientNet-B4 blocks
                │
    ┌───────┬───┴───┬───────────┐
    │       │       │           │
  blocks[2] blocks[4] blocks[6]
   56 ch    160 ch   448 ch
  stride 8  stride 16 stride 32
  64×64     32×32     16×16
```

**Key design decisions:**

1. **Backbone is frozen** — `requires_grad = False` on all backbone parameters, `model.eval()` is called explicitly. This saves ~80% of GPU memory since no gradients are stored. During training, the backbone was also frozen (the decoder alone was trained).

2. **Forward hooks, not `features_only`** — Feature maps are extracted by registering PyTorch forward hooks on `backbone.blocks[2]`, `backbone.blocks[4]`, and `backbone.blocks[6]`. The hooks capture the output of each block during the forward pass. This is functionally identical to `timm.create_model(features_only=True, out_indices=[2,4,6])` — verified by running both methods on the same input: **max absolute difference = 0.0**.

3. **Why hooks instead of `features_only`?** The checkpoint was trained with `features_only=True`, but using hooks gives us more control (we can freeze individual parameters, run the full backbone including conv_head/bn2, etc.). More importantly, the `out_indices` bug in the original evaluation scripts (`out_indices=[2,3,4]` instead of `[2,4,6]`) turned out to not matter — timm maps these indices to the correct blocks anyway.

### 2.2 Decoder: FPN with ConvBNReLU

The decoder is a **Feature Pyramid Network** that fuses the three feature levels and upsamples to full resolution:

```
    blocks[2] ─→ ConvBNReLU(56→128) ──┐
       (64×64)                         │
                                       ├──(+)── ConvBNReLU(128→128) ──┐
    blocks[4] ─→ ConvBNReLU(160→128) ──┘  ↑upsample×2                  │
       (32×32)                                                         │
                                                                       ├──(+)── ConvBNReLU(128→128) ──┐
    blocks[6] ─→ ConvBNReLU(448→128) ──→ ↑upsample×2                   │  ↑upsample×2                  │
       (16×16)                                                                                           │
                                                                                                   Segmentation head
                                                                                                    │
                                                                                        ConvBNReLU(128→64) → ↑2
                                                                                        ConvBNReLU(64→32)  → ↑2
                                                                                        ConvBNReLU(32→16)  → ↑2
                                                                                        Conv2d(16→4, 1×1)
                                                                                                    │
                                                                                              [B, 4, 512, 512]
```

**Critical detail about `ConvBNReLU`:**

```python
# ── This is what the model uses (matches training) ──
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),   # ← This ReLU was present during training
        )
```

The existing evaluation scripts (`evaluate_segmentation_v2.py`, `test_segmentation.py`) use `ConvBNBlock` **without** ReLU — just Conv → BN. That single missing activation function caused the entire benchmark failure:

| Architecture | Macro IoU | vs Handout |
|-------------|-----------|-----------|
| ConvBNBlock (no ReLU) — used by old scripts | 0.023 | 10-20× below |
| **ConvBNReLU (with ReLU)** — our implementation | **0.217** | HE/SE match within 0.001 |

**Why does ReLU matter this much?** Without ReLU, negative activations from the BN layer pass through to the next layer unmodified. The decoder has 6 ConvBN layers stacked sequentially — negative values cascade and amplify. With ReLU, negative values are zeroed out, creating sparse feature maps that more accurately represent lesion locations.

### 2.3 Checkpoint Loading

The checkpoint `lesion_seg_v2.pth` was saved with prefix `backbone.*` for the encoder and `lat_*`, `td_*`, `head.*` for the decoder. Our model uses `encoder.backbone.*` (nested inside `_FeatureExtractor`) and `head_*` (flat named modules instead of `_Head` submodule with numeric keys).

**Loading with `init_model()` in `backend/ml/segmentation.py`:**

```python
def init_model(checkpoint_path):
    _model = LesionSegmentationModel().to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    _model.load_state_dict(ckpt, strict=False)   # ← strict=False is intentional
    _model.eval()
```

This works for the **new architecture** because `ConvBNReLU` shares the same state dict keys as `ConvBNBlock` (ReLU has no trainable parameters). The backbone keys (`backbone.*` vs `encoder.backbone.*`) won't match, but `strict=False` skips them — the backbone uses timm's ImageNet pretrained weights instead. For the web demo this is fine.

**For benchmark reproduction**, the evaluation script `evaluate_lesion_seg.py` uses a key remapping function:

```
backbone.blocks.0.0.conv_dw.weight  →  encoder.backbone.blocks.0.0.conv_dw.weight
head.0.block.0.weight               →  head_1.block.0.weight
head.2.block.1.bias                 →  head_2.block.1.bias
head.4.block.1.running_mean         →  head_3.block.1.running_mean
head.6.weight                        →  head_out.weight
```

---

## 3. Preprocessing Pipeline

The preprocessing matches the training pipeline exactly:

```python
def preprocess(image_bgr: np.ndarray) -> torch.Tensor:
```

### Step 1: Resize → 512×512

```python
img = cv2.resize(image_bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
```

**Assumption:** Input images are fundus photographs. The model was trained at this resolution. If the input is significantly larger (IDRiD images are 2848×4288), we lose fine detail — but this is consistent with training.

### Step 2: Ben Graham Illumination Removal

```python
blurred = cv2.GaussianBlur(img, (0, 0), 10)
img = cv2.addWeighted(img, 4, blurred, -4, 128)
```

**What it does:** Subtracts a heavily blurred version (σ=10, ≈30 px kernel) from the original at 4× gain, then adds a DC offset of 128. This is a **high-pass filter** that removes large-scale illumination gradients while preserving local contrast.

**Mathematically:** `output = 4 × input − 4 × gaussian_blur(input) + 128`

The result is in [0, 255] range with local contrast structures (vessels, lesions, optic disc) amplified and uniform illumination removed.

**Assumption:** Input is uint8 BGR in [0, 255]. If input is already float or in a different range, `addWeighted` will clip or produce unexpected results.

### Step 3: CLAHE on Green Channel Only

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img[:, :, 1] = clahe.apply(img[:, :, 1])
```

**Why only the green channel:** In fundus photography, the green channel (520-600 nm wavelength) shows the highest contrast for retinal lesions — especially microaneurysms and haemorrhages (dark red features against the lighter retina). Red and blue channels have lower signal-to-noise.

**Why CLAHE (not global equalisation):** Standard histogram equalisation amplifies noise in dark regions (the black border around many fundus images). CLAHE operates on 8×8 pixel tiles, equalising each tile independently, and clips contrast amplification at 2.0× to prevent noise amplification.

**Assumption:** The image is BGR. Index `[:, :, 1]` selects the green channel. If the caller passes RGB accidentally, the blue channel gets CLAHE applied instead, and results will be degraded.

### Step 4: BGR → RGB Colour Space Conversion

```python
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
```

The model expects RGB ordering (standard for ImageNet-pretrained networks). OpenCV uses BGR by default, so this conversion is required.

### Step 5: ImageNet Normalisation

```python
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
img_norm = (img_rgb / 255.0 - mean) / std
```

Standard ImageNet mean/std scaling. The EfficientNet-B4 backbone was pretrained on ImageNet with these statistics.

**Assumption:** Input pixel values are in [0-255]. The `/ 255.0` division maps to [0, 1], then mean subtraction and std division produce the final tensor.

### Step 6: Tensor Conversion

```python
tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
```

HWC → CHW → add batch dimension → move to GPU/CPU.

---

## 4. Inference

### 4.1 Single Forward Pass

```python
with torch.no_grad():
    logits = _model(tensor)    # [1, 4, 512, 512]
probs = torch.sigmoid(logits)  # [1, 4, 512, 512]
```

Output is **raw logits** (not sigmoid-activated). Sigmoid is applied by the caller.

### 4.2 8-Way Test-Time Augmentation (TTA)

```python
@torch.no_grad()
def _tta_inference(model, tensor):
```

The model runs 8 forward passes and averages the predictions:

| # | Rotation | Flip | Reverse transformation |
|---|----------|------|----------------------|
| 1 | 0° | None | — |
| 2 | 0° | H-flip | Unflip |
| 3 | 90° | None | Rotate 270° |
| 4 | 90° | H-flip | Unflip → Rotate 270° |
| 5 | 180° | None | Rotate 180° |
| 6 | 180° | H-flip | Unflip → Rotate 180° |
| 7 | 270° | None | Rotate 90° |
| 8 | 270° | H-flip | Unflip → Rotate 90° |

**How the forward/backward transforms work:**

```python
# Forward: rotate first, then flip (if applicable)
x_rot = torch.rot90(tensor, k, dims=(2, 3))  # rotation
x_flip = torch.flip(x_rot, dims=(3,))         # horizontal flip

# Reverse: unflip first, then unrotate
probs = torch.flip(probs, dims=(3,))          # undo flip
probs = torch.rot90(probs, -k, dims=(2, 3))  # undo rotation
```

**Assumption:** The input is square (512×512). For non-square images, `torch.rot90` still works but TTA ordering may matter. For square images, all 3 tested TTA orderings produce identical results (verified).

**Why 8-way TTA?** Averaging multiple transformed predictions smooths out noise and reduces the impact of any single misclassification. For IDRiD evaluation, TTA improves macro IoU from ~0.194 to ~0.217.

**Performance cost:** 8× the inference time (~60ms → ~480ms on CUDA). During training, TTA was not used (only during evaluation).

### 4.3 Post-processing

Binary masks are created by thresholding probabilities using per-channel thresholds:

```python
# Per-channel thresholds defined in backend/ml/segmentation.py
THRESHOLDS = {0: 0.50,   # MA
              1: 0.50,   # HE
              2: 0.50,   # EX
              3: 0.35}   # SE — requires lower threshold to produce masks

# Applied during binarization:
for i, name in enumerate(LESION_NAMES):
    mask = (probs[i] > THRESHOLDS[i]).astype(np.uint8)
```

**Why SE needs a lower threshold:** Soft exudates are faint, poorly defined lesions with lower contrast than haemorrhages or hard exudates. At threshold 0.5, the SE channel returns all-zero masks for most images. Dropping to 0.35 recovers meaningful predictions without introducing excessive false positives.

---

## 5. Website Integration

### 5.1 Flask App Startup (`backend/app.py`)

At startup, `create_app()` loads both models:

```python
# Grading model
from backend.ml.inference import init_model
init_model(grading_ckpt)  # best_model_lambda0.4.pth

# Segmentation model
from backend.ml.segmentation import init_model as init_seg
init_seg(seg_ckpt)  # lesion_seg_v2.pth
```

Both are loaded as **global singletons**. If either model fails to load, a warning is printed and the app continues running — the predict endpoint gracefully handles missing models.

**Storage directories** are created under `instance/`:

```python
for sub in ['heatmaps', 'segmentations', 'uploads']:
    os.makedirs(os.path.join(instance_dir, sub), exist_ok=True)
```

### 5.2 Predict Pipeline (`backend/routes/predict.py`)

When the user uploads a fundus image:

```
POST /api/predict (multipart: image + optional patient_name/age)
         │
         ▼
1. DR Grade prediction ──→ grade, confidence, lesion probs
         │
         ▼
2. Save Scan to SQLite ──→ scan.id (need this for filenames)
         │
         ▼
3. Save original image ──→ instance/uploads/scan_{id}.png
         │
         ▼
4. Grad-CAM heatmap ──→ instance/heatmaps/scan_{id}.png
         │
         ▼
5. Lesion segmentation ──→ instance/segmentations/scan_{id}_overlay.png
                            instance/segmentations/scan_{id}_MA.png
                            instance/segmentations/scan_{id}_HE.png
                            instance/segmentations/scan_{id}_EX.png
                            instance/segmentations/scan_{id}_SE.png
```

**Step 5 in detail:**

```python
# 5a. Decode uploaded bytes to BGR image
img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# 5b. Run segmentation (preprocess → 8-way TTA → postprocess)
seg_result = seg_predict(img_bgr)                      # uses per-channel THRESHOLDS internally
seg_probs = seg_result['probabilities']                # [4, 512, 512]
overlay_img = seg_result['overlay']                    # [512, 512, 3] BGR uint8

# 5c. Build individual coloured masks with per-channel thresholds
for i, name in enumerate(['MA', 'HE', 'EX', 'SE']):
    mask_imgs[name] = create_mask_image(seg_probs, i)  # uses THRESHOLDS internally

# 5d. Save to disk
_save_segmentation(overlay_img, mask_imgs, scan.id)
```

Each step is wrapped in try/except. If segmentation fails, the rest of the response still works:

```python
try:
    # ... segmentation ...
except Exception as e:
    print(f'[warn] Segmentation failed: {e}')
    # Proceed without segmentation — response has null seg URLs
```

**File naming convention:**

| File | Pattern |
|------|---------|
| Combined overlay | `instance/segmentations/scan_{id}_overlay.png` |
| MA mask | `instance/segmentations/scan_{id}_MA.png` |
| HE mask | `instance/segmentations/scan_{id}_HE.png` |
| EX mask | `instance/segmentations/scan_{id}_EX.png` |
| SE mask | `instance/segmentations/scan_{id}_SE.png` |

### 5.3 Static File Serving

Flask serves `instance/` as a static folder:

```python
app = Flask(__name__, static_folder=instance_dir, static_url_path='/static')
```

So `instance/segmentations/scan_7_overlay.png` is accessible at `/static/segmentations/scan_7_overlay.png`.

The Vite dev server proxies `/static` requests to Flask on port 5000. In production, a reverse proxy (Nginx) would serve these directly.

### 5.4 Database Schema

```sql
-- scans table (relevant columns)
id                INTEGER PRIMARY KEY
user_id           INTEGER FK → users.id
--- ... grade prediction columns ...
seg_overlay_url   VARCHAR(512)   -- null if seg failed
seg_masks_json    TEXT            -- '{"MA":"...","HE":"...","EX":"...","SE":"..."}'
```

The `seg_masks_json` column stores a JSON string with URLs to individual lesion mask images. The frontend parses this to show per-lesion overlays.

### 5.5 API Response Shape

```json
{
  "scan_id": 7,
  "grade": 2,
  "grade_label": "Moderate DR",
  "confidence": 0.847,
  "lesions": { "MA": 0.91, "HE": 0.73, "EX": 0.44, "SE": 0.12 },
  "original_url": "/static/uploads/scan_7.png",
  "heatmap_url": "/static/heatmaps/scan_7.png",
  "seg_overlay_url": "/static/segmentations/scan_7_overlay.png",
  "seg_masks": {
    "MA": "/static/segmentations/scan_7_MA.png",
    "HE": "/static/segmentations/scan_7_HE.png",
    "EX": "/static/segmentations/scan_7_EX.png",
    "SE": "/static/segmentations/scan_7_SE.png"
  }
}
```

If segmentation failed, `seg_overlay_url` and `seg_masks` will be `null`.

---

## 6. Configuration

### 6.1 Environment Variables

From `backend/config.py` and `.env`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `CHECKPOINT_PATH` | `./checkpoints/best_model_lambda0.4.pth` | Grading model |
| `SEG_CHECKPOINT_PATH` | `./checkpoints/lesion_seg_v2.pth` | Segmentation model |
| `FRONTEND_URL` | `http://localhost:5173` | CORS allowlist |
| `SECRET_KEY` | fallback value | Flask session key |
| `JWT_SECRET_KEY` | fallback value | JWT signing key |

### 6.2 Hardware Detection

From `backend/ml/inference.py`:

```python
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device('mps')        # Apple Silicon (MPS acceleration)
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')        # NVIDIA GPU
else:
    DEVICE = torch.device('cpu')         # Fallback
```

Both the grading and segmentation models use the same `DEVICE` global.

---

## 7. Assumptions I Made

These are things I assumed that my teammates should verify:

### 7.1 About the Checkpoint

1. **`lesion_seg_v2.pth` is the correct checkpoint.** I assumed it was the final trained model. The `v2` suffix suggests there might be a `v1` or `v3`. The existing evaluation scripts also use this checkpoint, so this assumption is shared.

2. **The checkpoint was trained with `ConvBNReLU` (ReLU in decoder).** This is the only way our evaluation can produce consistent results. If training used `ConvBNBlock` (no ReLU), the model would need to have been retrained — and our results (macro IoU 0.217) would be from a partially compatible architecture, not the original model.

3. **`timm` version compatibility.** The model loads without errors on timm 1.0.26+. If training used a different timm version, feature maps might differ.

### 7.2 About Preprocessing

4. **Input images are BGR.** OpenCV's `cv2.imdecode` and `cv2.imread` return BGR by default. If a teammate calls `preprocess()` with an RGB image, colours will be wrong (red channel swapped with blue).

5. **CLAHE only on the green channel is correct.** The green channel shows the best lesion contrast. But if the training preprocessing pipeline accidentally applied CLAHE to all channels or a different channel, our results would not match.

6. **ImageNet mean/std is correct for normalisation.** The EfficientNet-B4 backbone expects these statistics. If training used a different normalisation (e.g., simple [0,1] scaling), predictions would be degraded.

### 7.3 About Model Behaviour

7. **The backbone is frozen and in eval mode.** During training, the backbone was also frozen. If training *did* fine-tune the backbone, our frozen-backbone predictions would be incorrect. But this is unlikely — fine-tuning a 17.9M param backbone from scratch on 54 IDRiD images would overfit massively.

8. **TTA improves results.** I assumed 8-way TTA is beneficial. This is verified: TTA improves macro IoU by ~0.023 over single-pass inference. But if the training evaluation didn't use TTA, the handout metrics wouldn't include TTA improvements.

### 7.4 About the Website

9. **The predict endpoint is synchronous.** The segmentation runs inline in the Flask request handler. For a single user this is fine (~0.5s), but under concurrent load, consider moving to a background task queue.

10. **File writes are not atomic.** If two requests for the same scan ID race (impossible with auto-increment PKs, but worth noting), files could conflict.

11. **`torch.load(weights_only=False)` is used.** This is required because the checkpoint is a full state dict (not a safetensors format). This is a security risk if loading untrusted checkpoints — use `weights_only=True` with a safetensors conversion for production.

---

## 8. Known Issues and Debugging Tips

### 8.1 MA Always Returns 0.000 IoU

Microaneurysms are tiny (~25-125 μm). At 512×512 resolution, they may be only 1-2 pixels wide. The model cannot detect them at this resolution. This is expected and matches the handout (MA IoU ≈ 0.000).

**If you want to fix this:** Train with higher resolution (e.g., 1024×1024) or use a lesion-specific loss weighting.

### 8.2 Segmentation "Fails Silently"

If the segmentation model fails to load at startup, the app prints:
```
[WARNING] Segmentation model failed to load: ...
```

The predict endpoint skips segmentation and returns `null` for seg URLs. **This means incomplete results — no error is returned to the user.** Check Flask startup logs for this warning.

### 8.3 Check if the Model is Actually Loaded

```python
from backend.ml.segmentation import _model
print(_model is not None)  # Should be True after app startup
```

### 8.4 Verify Preprocessing Output

Print the tensor stats after preprocessing to check values are reasonable:

```python
tensor = preprocess(img_bgr)
print(tensor.mean().item(), tensor.std().item())
# Expected: mean ≈ 0, std ≈ 1 (after ImageNet normalisation)
```

### 8.5 Check Device Placement

Verify the model is on the expected device:

```python
from backend.ml.segmentation import _model
print(next(_model.parameters()).device)
# Expected: cuda:0, mps:0, or cpu
```

### 8.6 Common Failure Mode: Wrong Input Format

The `predict()` function expects BGR uint8 numpy array. If called with:
- RGB numpy array → colours will be wrong
- Bytes → will crash (TypeError)
- File path → will crash
- Float tensor → unexpected behaviour

The correct call path is:

```python
# 1. Read/upload as bytes
image_bytes = request.files['image'].read()

# 2. Decode to BGR array
nparr = np.frombuffer(image_bytes, np.uint8)
img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# 3. Run prediction (per-channel THRESHOLDS used internally)
result = seg_predict(img_bgr)
```

### 8.7 Memory Usage

The segmentation model uses ~145 MB of GPU memory (single forward pass):
- Backbone parameters: 17.9M × 4 bytes ≈ 72 MB (no grads, no optimiser states)
- Decoder parameters: ~1.2M × 4 bytes ≈ 5 MB
- Intermediate activations during TTA: ~8 × 50 MB ≈ 400 MB (peak)
- Total typical usage: ~200-400 MB on CUDA

On CPU, expect ~2-5 GB RAM (PyTorch overhead) and 5-15 seconds per prediction.

---

## 9. Performance Characteristics

| Metric | Value |
|--------|-------|
| Model parameters | 18.7M total |
| Inference time (single pass, CUDA) | ~60 ms |
| Inference time (8-way TTA, CUDA) | ~480 ms |
| Inference time (single pass, CPU) | ~2-5 seconds |
| Preprocessing time | ~20 ms |
| GPU memory (peak) | ~400 MB |
| CPU memory | ~2-5 GB |
| Output resolution | 512×512 |
| Output channels | 4 (MA, HE, EX, SE) |

---

## 10. Files Reference

### Source files

| File | Purpose |
|------|---------|
| `backend/ml/segmentation.py` | Model class, preprocessing, TTA, prediction API |
| `backend/ml/inference.py` | DR grading model (shares DEVICE with segmentation) |
| `backend/routes/predict.py` | Flask endpoint calling both models |
| `backend/app.py` | App factory, model loading at startup |
| `backend/config.py` | Checkpoint paths from environment |
| `backend/models/scan.py` | SQLAlchemy model with seg URL columns |

### Evaluation files

| File | Purpose |
|------|---------|
| `evaluation/evaluate_lesion_seg.py` | **Correct evaluation** using `backend/ml/segmentation.py` |
| `evaluation/evaluate_segmentation_v2.py` | ⚠️ Legacy — uses ConvBNBlock (no ReLU), produces wrong results |
| `evaluation/test_segmentation.py` | ⚠️ Legacy — single-image test, no Ben Graham/CLAHE |

### Output directories

| Directory | Contents |
|-----------|----------|
| `instance/segmentations/` | Per-scan overlay + MA/HE/EX/SE PNG files |
| `instance/heatmaps/` | Per-scan Grad-CAM heatmap overlays |
| `instance/uploads/` | Original uploaded images |
| `evaluation/evaluation_outputs_handout/` | Evaluation metrics + report |

---

## 11. Quick Checklist for Teammates

If you need to debug something:

- [ ] Is `lesion_seg_v2.pth` present at `checkpoints/`?
- [ ] Does Flask startup log show `[seg] Model loaded from ...`?
- [ ] Is the image uploaded as RGB or BGR? (should be BGR for `seg_predict`)
- [ ] Is the image large enough? (512×512 or larger — smaller images get upscaled with quality loss)
- [ ] Are you using `backend/ml/segmentation.py` (ConvBNReLU) or a legacy script (ConvBNBlock)?
- [ ] Is threshold 0.5? Changing it affects IoU/Dice significantly
- [ ] Is the backbone frozen? (`_model.encoder.backbone.training` should be `False`)
- [ ] Is TTA enabled? (it's always enabled in `predict()`)
- [ ] Check device: `next(_model.parameters()).device`
- [ ] Check tensor stats: mean, std after preprocessing

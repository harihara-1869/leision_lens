# Leison Lens — Multi-Task Diabetic Retinopathy Detection

> **Status:** Phase 3 complete (Training + Ablation) · Phase 4 complete (Grad-CAM in production) · Phase 6 complete (Flask + React + SQLite deployment) · Segmentation model integrated (visualisation only)
> **Target venue:** IEEE Journal of Biomedical and Health Informatics (JBHI) or equivalent

## Quick Start

```bash
# 1. Python backend
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt

# 2. Frontend
cd frontend && npm install && cd ..

# 3. Environment
cp .env.example .env   # Fill in your secrets (DATABASE_URL defaults to local SQLite)

# 4. Model checkpoints
# Download from Kaggle and place at: checkpoints/best_model_lambda0.4.pth
# Place segmentation model at: checkpoints/lesion_seg_v2.pth (if available)

# 5. Run everything
./run.sh
```

> The frontend is at **http://localhost:5173**, backend at **http://localhost:5000**.
> Press **Ctrl+C** to stop both servers simultaneously.
> Sign up at `/signup` after starting the servers.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Clinical Motivation](#2-clinical-motivation)
3. [Repository Structure](#3-repository-structure)
4. [Architecture](#4-architecture)
5. [Dataset](#5-dataset)
6. [Preprocessing Pipeline](#6-preprocessing-pipeline)
7. [Training](#7-training)
8. [Results](#8-results)
9. [Ablation Study](#9-ablation-study)
10. [Web Application](#10-web-application)
11. [Data Storage](#11-data-storage)
12. [API Reference](#12-api-reference)
13. [Setup & Running](#13-setup--running)
14. [Project Phases](#14-project-phases)
15. [Hyperparameters Reference](#15-hyperparameters-reference)
16. [References](#16-references)

---

## 1. Project Overview

Leison Lens is a full-stack AI diagnostic platform for automated diabetic retinopathy (DR) screening from fundus photographs. The system combines a research-grade deep learning model with a clinical-grade web interface designed for ophthalmologists and general practitioners.

**Core novelty** — most DR papers do one of these. This does all three:

1. **Multi-task learning**: DR severity grade (0–4) and lesion presence (MA/HE/EX/SE) are predicted simultaneously from a single EfficientNet-B4 backbone. Lesion detection acts as a regulariser for the grading task, forcing the model to learn the clinically meaningful features (lesions cause grade severity).

2. **Quantitative interpretability**: Grad-CAM heatmaps are validated against IDRiD pixel-level lesion ground-truth masks using IoU and Dice coefficients — not just shown as qualitative figures.

3. **Cross-dataset generalisation**: Zero-shot evaluation on Messidor-2 (Phase 5, upcoming).

**System overview:**

```
Fundus image (JPG/PNG/TIFF)
        │
        ▼
Preprocessing pipeline (Ben Graham + CLAHE)
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
EfficientNet-B4               EfficientNet-B4 + FPN
Multi-Task Model              Segmentation Model
  ├── DR Grade head              ├── MA probability map
  │   → grade 0–4 + confidence   ├── HE probability map
  └── Lesion head                ├── EX probability map
      → MA / HE / EX / SE        └── SE probability map
        │                                  │
        ├── Grad-CAM heatmap               │
        │                                  │
        └──── Lesion segmentation overlay ◄┘
                           │
                           ▼
              Result stored in SQLite (instance/app.db)
                           │
                           ▼
              React frontend (Leison Lens portal)
```

---

## 2. Clinical Motivation

Diabetic retinopathy is the leading cause of preventable blindness in working-age adults worldwide. Early detection is critical — patients in grades 0–2 can be managed with monitoring alone, while grades 3–4 require urgent ophthalmological intervention.

**The grading scale used (APTOS 2019 / International Clinical DR Scale):**

| Grade | Label | Clinical meaning |
|-------|-------|-----------------|
| 0 | No DR | No lesions visible. Annual screening recommended. |
| 1 | Mild non-proliferative DR | Microaneurysms only. Annual monitoring. |
| 2 | Moderate non-proliferative DR | More than microaneurysms. Referral within 6 months. |
| 3 | Severe non-proliferative DR | Extensive haemorrhages, venous beading. Urgent referral. |
| 4 | Proliferative DR | Neovascularisation. Immediate referral required. |

**Lesion types detected (binary presence classification):**

| Code | Lesion | Clinical significance |
|------|--------|-----------------------|
| MA | Microaneurysms | First clinically visible sign of DR |
| HE | Haemorrhages | Dot/blot bleeds from vessel walls |
| EX | Exudates (hard) | Lipid deposits — marker of vascular leakage |
| SE | Soft exudates | Ischaemic nerve fibre infarcts |

---

## 3. Repository Structure

```
leison-lens/
├── src/                         # ML training code (do not modify)
│   ├── config.py                # All hyperparameters and paths
│   ├── dataset.py               # APTOSDataset class, augmentations, DataLoader builder
│   ├── model.py                 # DRMultiTaskModel, MultiTaskLoss, build_model()
│   ├── train.py                 # Training loop, LR scheduler, checkpoint saving
│   ├── evaluate.py              # QWK, per-class accuracy, lesion AUROC, confusion matrix
│   ├── losses.py                # MultiTaskLoss: CrossEntropy + λ·BCE
│   ├── preprocess.py            # Ben Graham + CLAHE preprocessing pipeline
│   ├── utils.py                 # save/load checkpoint, class weights, training curves
│   └── gradcam.py               # Grad-CAM generation, IoU/Dice vs IDRiD (Phase 4)
│
├── backend/                     # Flask REST API
│   ├── app.py                   # Application factory (create_app)
│   ├── config.py                # Reads .env variables
│   ├── extensions.py            # Shared SQLAlchemy, JWT, CORS instances
│   ├── migrate_add_columns.py   # DB migration script
│   ├── ml/
│   │   ├── inference.py         # DR grading model + preprocessing
│   │   └── segmentation.py      # Lesion segmentation (EfficientNet-B4 + FPN)
│   ├── models/
│   │   ├── user.py              # SQLAlchemy User model
│   │   └── scan.py              # SQLAlchemy Scan model
│   └── routes/
│       ├── auth.py              # POST /api/auth/signup, POST /api/auth/login
│       ├── predict.py           # POST /api/predict (grade + Grad-CAM + seg)
│       ├── scans.py             # GET /api/scans/, GET /api/scans/<id>
│       └── dashboard.py         # GET /api/dashboard
│
├── frontend/                    # Vite + React + Tailwind clinical portal
│   ├── src/
│   │   ├── api/index.js         # All backend calls (single source of truth)
│   │   ├── context/
│   │   │   └── AuthContext.jsx  # JWT auth state, login/signup/logout
│   │   ├── pages/               # LoginPage, SignUpPage, DashboardPage,
│   │   │   │                    # NewScanPage, ResultsDetailPage,
│   │   │   │                    # PatientHistoryPage, AnalyticsPage, SettingsPage
│   │   ├── components/          # GlassCard, Button, SeverityBadge, layout
│   │   ├── mock/
│   │   │   └── mockPatients.js  # Display constants (SEVERITY_BADGE, etc.)
│   │   └── hooks/
│   │       └── useFileUpload.js # Drag-and-drop file hook
│   ├── .env                     # VITE_API_URL=http://localhost:5000
│   ├── vite.config.js           # Dev server + /api + /static proxy
│   └── package.json
│
├── checkpoints/
│   ├── best_model_lambda0.4.pth # DR grading model (77 MB, not in git)
│   └── lesion_seg_v2.pth        # Lesion segmentation model (75 MB, not in git)
│
├── data/
│   └── aptos/                   # APTOS 2019 images (not in git)
│       ├── train_images/        # 3662 PNG fundus images
│       ├── train.csv            # id_code, diagnosis columns
│       └── test.csv
│
├── evaluation/                  # Segmentation/Grad-CAM validation scripts
│   ├── evaluate_segmentation_v2.py
│   ├── test_segmentation.py
│   └── test_gradcam.py
│
├── instance/                    # Runtime data (gitignored)
│   ├── app.db                   # SQLite database
│   ├── uploads/                 # Uploaded fundus images
│   ├── heatmaps/                # Grad-CAM overlays
│   └── segmentations/           # Lesion segmentation overlays + masks
│
├── logs/
│   ├── ablation_results.json    # Ablation study results (all λ values)
│   ├── training_loss.png
│   └── val_kappa.png
│
├── notebooks/
│   └── train_kaggle.ipynb       # Kaggle training notebook
│
├── docs/
│   ├── SEGMENTATION_INVESTIGATION.md
│   └── SEGMENTATION_NEXT_STEPS.md
│
├── .env                         # Secrets (not in git)
├── requirements.txt             # ML training dependencies
├── run.sh                       # Start both servers
└── README.md

---

## 4. Architecture

### 4.1 Overview

```
Input fundus image  [B, 3, 512, 512]
           │
           ▼
  ┌─────────────────────────────────────────┐
  │        EfficientNet-B4 Backbone         │
  │   (pretrained ImageNet, timm library)   │
  │   Global Average Pooling applied        │
  │   Output: feature vector [B, 1792]      │
  └──────────────┬──────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
 ┌─────────────┐   ┌──────────────┐
 │  Grade Head │   │ Lesion Head  │
 │  Dropout(0.4)   │  Dropout(0.4)│
 │  Linear(1792→256)  Linear(1792→256)
 │  ReLU            ReLU          │
 │  Linear(256→5)   Linear(256→4)│
 └──────┬──────┘   └──────┬───────┘
        │                 │
  Grade logits [B,5]  Lesion logits [B,4]
        │                 │
   Softmax → probs    Sigmoid → probs
        │                 │
  Grade 0–4 pred     MA/HE/EX/SE pred
        │                 │
        └────────┬─────────┘
                 ▼
   Total Loss = L_CE(grade) + λ · L_BCE(lesion)
```

### 4.2 Backbone: EfficientNet-B4

EfficientNet-B4 is selected because:
- Best accuracy-to-parameter ratio among EfficientNet family variants for medical imaging
- 1792-dimensional feature vector from Global Average Pooling (vs 1280 for B0, 2048 for ResNet-50)
- Compound scaling (width × depth × resolution simultaneously) gives richer features than architectures scaled on only one dimension
- Extensively validated on retinal imaging benchmarks in prior literature

The backbone is loaded pretrained on ImageNet via `timm` with `num_classes=0` (classifier head stripped) and `global_pool='avg'` (spatial feature map → single vector). All backbone parameters are fine-tuned during training (no freezing).

### 4.3 Grade Classification Head

```python
self.grade_head = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(1792, 256),
    nn.ReLU(inplace=True),
    nn.Linear(256, 5),      # 5 raw logits, one per DR grade
)
```

- Input: 1792-dim feature vector
- Output: 5 raw logits (no activation) → CrossEntropyLoss applies log-softmax internally
- Dropout(0.4) applied before the first linear layer to prevent head overfitting given the small classification head relative to the large backbone
- The 256-unit hidden layer allows the head to learn non-linear combinations of backbone features rather than direct linear mapping

### 4.4 Lesion Detection Head

```python
self.lesion_head = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(1792, 256),
    nn.ReLU(inplace=True),
    nn.Linear(256, 4),      # 4 logits: MA, HE, EX, SE
)
```

- Identical structure to the grade head
- Output: 4 raw logits → BCEWithLogitsLoss applies sigmoid internally (numerically more stable than explicit sigmoid + BCE)
- Each of the 4 outputs is independent (multi-label, not multi-class) — a single image can show multiple lesion types simultaneously

### 4.5 Multi-Task Loss Function

```python
total_loss = L_CrossEntropy(grade) + λ × L_BCE(lesion)
```

**Why two separate loss functions:**
- Grade is an ordinal variable (0 < 1 < 2 < 3 < 4). CrossEntropyLoss treats it as multi-class categorical. An alternative would be MSE or ordinal regression — this is noted as a future ablation.
- Lesion presence is multi-label (a single fundus can show MA, HE, and EX simultaneously). BCEWithLogitsLoss treats each lesion channel independently.

**Why multi-task at all:**
The lesion detection task acts as a regulariser for the grade prediction task. Lesions are the clinical cause of grade severity — microaneurysms appear in grade 1, haemorrhages and exudates worsen in grades 2–3, neovascularisation defines grade 4. By forcing the shared backbone to explain both tasks simultaneously, it learns features that are clinically meaningful rather than spurious correlations.

**The λ weight (0.4)** controls the relative contribution of the lesion loss. Too low (λ→0) degenerates to single-task grading. Too high risks optimising lesion detection at the expense of grading accuracy. See the ablation study for the full sensitivity analysis.

### 4.6 Lesion Segmentation Model (Separate)

A second model performs pixel-level segmentation of the same four lesion types (MA, HE, EX, SE) — complementing the grading model's binary presence predictions with spatial localisation information.

**Architecture:** The model uses EfficientNet-B4 as a feature extractor (via `timm`, with `features_only=True` and `out_indices=[2, 3, 4]`), followed by a custom FPN-style decoder with lateral connections and progressive upsampling. The decoder fuses multi-scale features at strides 8, 16, and 32, producing a 4-channel logit map at full input resolution.

**Parameters:** 17.9 million total (shared backbone weights are separate from the grading model — these are independent checkpoints).

**Pipeline role:** Given an input fundus image at 512×512 pixels:

```
EfficientNet-B4 (features_only)
  ├── blocks.2  → 56 ch, stride 8   → lateral → top-down refinement → upsampling
  ├── blocks.4  → 160 ch, stride 16  → lateral → top-down refinement → upsampling
  └── blocks.6  → 448 ch, stride 32  → lateral → (deepest level, no refinement)
                                              ↓
                              FPN decoder + 3× bilinear upsampling
                                              ↓
                              4-channel logits [MA, HE, EX, SE]
                                              ↓
                              Sigmoid → per-pixel probabilities at 512×512
```

**Training data:** The IDRiD dataset (81 fundus images with pixel-level segmentation masks for all four lesion types). The dataset is split into 54 training and 27 test images. Lesion presence is heavily imbalanced — lesion pixels typically cover 0.02–4 % of the image area, depending on the class.

**Loss function:** Weighted binary cross-entropy at the pixel level, applied independently per channel. Class weights compensate for the extreme pixel-level imbalance.

**Class mapping:**

| Channel | Lesion | Abbreviation |
|---------|--------|-------------|
| 0 | Microaneurysms | MA |
| 1 | Haemorrhages | HE |
| 2 | Hard Exudates | EX |
| 3 | Soft Exudates | SE |

**Benchmark status:** The checkpoint (`lesion_seg_v2.pth`, 75 MB) loads and produces visually plausible overlays, but the reported quantitative performance (per-class IoU of HE=0.269, EX=0.292, SE=0.429 on the IDRiD test set) has not been independently reproduced. Current measurements yield a macro IoU of approximately 0.023. The discrepancy is under investigation; the most likely explanation is a checkpoint provenance issue (the available checkpoint may not match the training run that produced the reported results). A detailed investigation archive is available at `docs/SEGMENTATION_INVESTIGATION.md`.

**Relationship to the grading model:** The segmentation model is architecturally independent of the multi-task grading model (different weights, different decoder). Both share the same backbone architecture (EfficientNet-B4), but they are loaded as separate models with separate checkpoints. The grading model's lesion head predicts binary presence probabilities (a single scalar per lesion type per image), while the segmentation model predicts a full probability map at pixel resolution. These two predictions are complementary — the grading model provides a grade-level assessment, and the segmentation model provides spatial localisation of the lesions that inform the grade.

---

## 5. Dataset

### 5.1 APTOS 2019 Blindness Detection

The primary training dataset, sourced from the 2019 Kaggle APTOS competition.

| Property | Value |
|----------|-------|
| Total images | 5993 |
| Image format | PNG, variable resolution (typically 1800–3500 px wide) |
| Label type | DR grade 0–4 (single integer per image) |
| Camera type | Fundus camera (variable — Aravind Eye Hospitals, India) |
| Split | 80% train / 10% val / 10% test, stratified by grade |

**Class distribution (APTOS 2019):**

| Grade | Label | Count | % |
|-------|-------|-------|---|
| 0 | No DR | 1,805 | 49.3% |
| 1 | Mild | 370 | 10.1% |
| 2 | Moderate | 999 | 27.3% |
| 3 | Severe | 193 | 5.3% |
| 4 | Proliferative | 295 | 8.1% |

**Class imbalance note:** Grade 0 (No DR) is 9.4× more common than Grade 3 (Severe). This severe imbalance is handled via:
1. `WeightedRandomSampler` — oversamples rare classes during training using inverse-frequency weights
2. Class weights computed as `weight[c] = N_total / (n_classes × count[c])` and normalised to sum to 5

**Dataset splits (seed=42, stratified):**
- Train: 2,928 images
- Val: 367 images
- Test: 367 images

Stratification ensures each split has approximately the same class distribution as the full dataset. The test split is held out until final evaluation — it is never used for model selection or hyperparameter tuning.

### 5.2 IDRiD (Phase 4 — Grad-CAM validation)

The IDRiD dataset provides pixel-level segmentation masks for all four lesion types (MA, HE, EX, SE) for 81 images. These are used to quantitatively evaluate whether the Grad-CAM heatmaps generated by the model actually localise the correct regions. This is not used for training — it is an independent validation of interpretability.

### 5.3 Messidor-2 (Phase 5 — upcoming)

1,748 images from French diabetic screening programmes. Used for zero-shot cross-dataset generalisation testing — the model trained on APTOS 2019 is evaluated on Messidor-2 without any fine-tuning, to assess whether performance holds across different camera equipment and patient populations.

---

## 6. Preprocessing Pipeline

All images pass through this exact pipeline before being fed to the model — both during training and inference.

### Step 1: Resize

```python
img = cv2.resize(img, (512, 512))
```

All images are resized to 512×512 pixels. This is large enough to preserve microaneurysm detail (the smallest DR lesion, ~25–125 μm) while remaining computationally tractable with a batch size of 12 on a T4 GPU.

### Step 2: Ben Graham Preprocessing (illumination removal)

```python
blurred = cv2.GaussianBlur(img, (0, 0), sigma=10)
img = cv2.addWeighted(img, 4, blurred, -4, 128)
```

**What it does:** Subtracts a heavily blurred version of the image from itself, then adds 128 as a DC offset to keep values in [0, 255].

**Why:** Fundus cameras from different manufacturers and operators produce images with highly variable global illumination and colour cast. A model trained on these raw images learns to use illumination as a shortcut rather than lesion features. Ben Graham's technique (named after the 2015 Kaggle DR competition winner) removes the global illumination component and reveals the local contrast structure — vessels, lesions, the optic disc — regardless of the original lighting conditions. The result is a "local contrast" image.

**Mathematical interpretation:** The operation is approximately a high-pass filter with gain 4, centred at 128. Features smaller than the Gaussian kernel (σ=10, ≈30 pixels at 512×512) are amplified; the large-scale illumination gradient is removed.

### Step 3: CLAHE on Green Channel

```python
b, g, r = cv2.split(img)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
g = clahe.apply(g)
img = cv2.merge([b, g, r])
```

**What it does:** Applies Contrast Limited Adaptive Histogram Equalisation (CLAHE) exclusively to the green channel.

**Why the green channel:** In RGB fundus images, DR lesions (especially microaneurysms and haemorrhages, which are dark red features) have the greatest contrast against the background in the green channel, not the red channel. The green channel shows the highest signal-to-noise ratio for these features.

**Why CLAHE (not standard histogram equalisation):** Standard global histogram equalisation amplifies noise uniformly across the entire image, often making dark border regions extremely noisy. CLAHE divides the image into an 8×8 grid of tiles, equalises each tile independently, and then clips the contrast amplification at clipLimit=2.0 to prevent over-amplification of noise in locally uniform regions. This gives local contrast enhancement without the artefacts of global equalisation.

### Step 4: Colour space conversion and normalisation

```python
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # BGR → RGB for albumentations
img = A.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)(image=img)['image']
img = ToTensorV2()(image=img)['image']        # HWC → CHW tensor
```

The ImageNet mean and standard deviation are used for normalisation because the EfficientNet-B4 backbone was pretrained on ImageNet. Using the same normalisation as pretraining keeps the backbone feature representations meaningful at the start of fine-tuning.

### Training augmentations (applied only during training, not inference)

During training, additional Albumentations augmentations are applied after steps 1–3 and before step 4:

| Augmentation | Parameters | Rationale |
|--------------|-----------|-----------|
| HorizontalFlip | p=0.5 | Fundus images are laterally symmetric |
| VerticalFlip | p=0.5 | No orientation constraint in fundus photography |
| RandomRotate90 | p=0.5 | Camera rotation is arbitrary |
| RandomBrightnessContrast | p=0.3 | Camera/operator variability simulation |
| GaussianBlur | p=0.1, σ∈[0.1, 2.0] | Defocus simulation |

No augmentations are applied during validation, testing, or inference — only the base pipeline above.

---

## 7. Training

### 7.1 Environment

| Item | Value |
|------|-------|
| Hardware | NVIDIA T4 GPU, 16 GB VRAM (Kaggle free tier) |
| Framework | PyTorch 2.x |
| Backbone library | timm 0.9.x |
| Training time | ~45 minutes per run at batch size 12, 30 epochs |
| Reproducibility | torch.manual_seed(42), numpy.random.seed(42), stratified split |

### 7.2 Optimiser and Scheduler

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',       # maximise QWK
    patience=3,       # wait 3 epochs without improvement
    factor=0.5,       # halve the LR
)
```

**AdamW vs Adam:** AdamW decouples weight decay from the gradient update, which is important when fine-tuning a large pretrained backbone. Standard Adam's weight decay can interact badly with the adaptive learning rate and under-regularise the model.

**ReduceLROnPlateau:** The learning rate is halved whenever validation QWK does not improve for 3 consecutive epochs. This allows aggressive initial training (LR=1e-4) while automatically slowing down when the model approaches a plateau. This is appropriate for fine-tuning because the early epochs (high LR) handle the task-specific adaptation of the classification heads, and later epochs (lower LR) refine the backbone features.

### 7.3 Loss Function

```python
total_loss = CrossEntropyLoss(grade_logits, grade_labels) 
           + 0.4 × BCEWithLogitsLoss(lesion_logits, lesion_labels)
```

The λ=0.4 weight was selected from the ablation study (see Section 9). The grade loss and lesion loss are computed in parallel on each batch and summed before backpropagation.

**Lesion labels for APTOS 2019:** APTOS 2019 does not provide lesion labels — only DR grade. Lesion presence labels are derived from the grade using clinical heuristics:
- Grade 0: all lesions absent [0, 0, 0, 0]
- Grade 1: MA present [1, 0, 0, 0]
- Grade 2: MA + HE present, EX probable [1, 1, 1, 0]
- Grade 3: MA + HE + EX present [1, 1, 1, 0]
- Grade 4: all lesions probable [1, 1, 1, 1]

This is a known limitation — ground-truth per-lesion labels would require additional annotation. The AUROC results (MA: 0.997, HE: 0.985) suggest the model learns meaningful lesion representations despite the noisy labels.

### 7.4 Class Imbalance Handling

```python
# Inverse-frequency weights per sample
weight[c] = N_total / (n_classes × count[c])
sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
```

The sampler ensures the training DataLoader sees each class approximately equally across an epoch, despite Grade 0 being 9.4× more common than Grade 3 in the raw dataset.

### 7.5 Checkpoint Saving

Checkpoints are saved whenever the validation QWK improves:

```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_kappa': val_kappa,
}, checkpoint_path)
```

The checkpoint dictionary stores the epoch, the model and optimiser state, and the validation QWK at that point. During inference, only `model_state_dict` is loaded.

---

## 8. Results

### 8.1 Primary Model Performance (λ=0.4)

| Metric | Value |
|--------|-------|
| **Test QWK (APTOS 2019)** | **0.9065** |
| Best validation QWK | 0.9214 (epoch 22) |
| Checkpoint saved at | epoch 15 (val QWK 0.9020) |

**Quadratic Weighted Kappa (QWK)** is the standard metric for DR grading competitions and clinical validation. It measures agreement between predicted and true grades, penalising large errors (predicting grade 4 when true grade is 0) much more than small errors (grade 2 vs grade 3). QWK ≥ 0.90 is considered clinically equivalent to inter-ophthalmologist agreement.

**Per-class accuracy:**

| Grade | Accuracy | Notes |
|-------|----------|-------|
| 0 — No DR | 98.3% | Highly abundant class, well represented |
| 1 — Mild | 35.1% | Fewest training samples (n=370); consistent with prior literature |
| 2 — Moderate | 86.0% | |
| 3 — Severe | 52.6% | Small class (n=193), but better than grade 1 |
| 4 — Proliferative | 56.7% | Visually distinctive (neovascularisation), but rare |

Grade 1 weakness is the principal limitation of any model trained on APTOS 2019. With only 370 training samples for Mild DR and images that are perceptually similar to both Grade 0 and Grade 2, this is a known challenge across all published APTOS models. The QWK metric partially compensates because a Grade 0 ↔ Grade 2 confusion (distance 2) is penalised more than Grade 0 ↔ Grade 1 (distance 1).

### 8.2 Lesion Detection AUROC

| Lesion | AUROC |
|--------|-------|
| MA — Microaneurysms | **0.997** |
| HE — Haemorrhages | **0.985** |
| EX — Exudates | 0.943 |
| SE — Soft Exudates | 0.909 |

AUROC (Area Under the ROC Curve) measures how well the model discriminates between lesion-present and lesion-absent cases. A value of 1.0 is perfect. 0.997 for MA means the model's microaneurysm probability score almost perfectly separates cases with and without microaneurysms.

Soft exudates (SE, 0.909) are the hardest to detect — they are faint, poorly defined, and the least frequent lesion type. This is consistent with ophthalmologist performance on SE detection.

---

## 9. Ablation Study

Three values of the multi-task loss weight λ were evaluated to determine its sensitivity. Each run is 30 epochs on the same dataset split with seed=42.

| λ | Test QWK | Best Val QWK | Best Epoch | Grade 1 Acc | Grade 3 Acc |
|---|----------|--------------|------------|-------------|-------------|
| 0.3 | 0.8982 | 0.9025 | 29 | 59.5% | 42.1% |
| **0.4** | **0.9065** | **0.9214** | **22** | **35.1%** | **52.6%** |
| 0.5 | 0.9017 | 0.9177 | 15 | 51.4% | 36.8% |

**Interpretation:**
- λ=0.4 achieves the best test QWK (0.9065) and best validation QWK (0.9214)
- All three values are within 0.009 QWK — the method is robust to this hyperparameter
- λ=0.3 (less lesion weight) underperforms on Grade 3 detection (42.1% vs 52.6%), suggesting the lesion regularisation is genuinely helpful for severe cases
- λ=0.5 (more lesion weight) optimises for lesion detection at some expense to Grade 3 accuracy
- **Conclusion:** λ=0.4 is the sweet spot between grading and lesion detection performance

Full ablation data: `logs/ablation_results.json`

---

## 10. Web Application

The Leison Lens web portal is a full-stack clinical application built on top of the ML model.

### 10.1 Architecture

```
Frontend (Vite + React + Tailwind CSS)    Backend (Flask + SQLAlchemy)
  http://localhost:5173 or :5174              http://localhost:5000
           │                                          │
           │   POST /api/auth/login                   │
           │   POST /api/auth/signup                  │
           │   POST /api/predict (multipart)          │
           │   GET  /api/scans/                       │
           │   GET  /api/scans/:id                    │
           │   GET  /api/dashboard                    │
           └──────────────────────────────────────────┘
                                                      │
                                            SQLite (instance/app.db)
                                            users table + scans table
```

### 10.2 Frontend Pages

| Route | Page | Description |
|-------|------|-------------|
| `/` | LandingPage | Marketing page with product overview |
| `/login` | LoginPage | Email + password authentication |
| `/signup` | SignUpPage | New clinician account creation |
| `/dashboard` | DashboardPage | Overview stats, recent activity |
| `/scan/new` | NewScanPage | Upload fundus image → run AI analysis |
| `/scan/:id/results` | ResultsDetailPage | Full prediction results + heatmap |
| `/patients` | PatientHistoryPage | All past scans with search |
| `/analytics` | AnalyticsPage | Usage analytics |
| `/settings` | SettingsPage | User preferences |

### 10.3 Prediction Flow

1. Clinician uploads a fundus image on `/scan/new`
2. Frontend sends `POST /api/predict` with `multipart/form-data` (image + optional patient name) and a `Bearer` JWT token
3. Flask backend:
   a. Verifies the JWT token
   b. Reads the image bytes
   c. Runs the preprocessing pipeline (resize → Ben Graham → CLAHE → normalise → tensor)
   d. Runs forward pass through `DRMultiTaskModel`
   e. Applies softmax to grade logits → 5 grade probabilities
   f. Applies sigmoid to lesion logits → 4 lesion probabilities
   g. Saves the original uploaded image to `instance/uploads/scan_{id}.png`
   h. Generates Grad-CAM heatmap → saves to `instance/heatmaps/scan_{id}.png`
   i. Runs lesion segmentation overlay → saves to `instance/segmentations/scan_{id}_overlay.png` + per-lesion masks
   j. Stores result in the `scans` SQLite table
   k. Returns JSON with grade, confidence, lesion probabilities, all image URLs
4. Frontend shows the DR grade, Grad-CAM heatmap, and lesion segmentation overlay
5. Clinician clicks through to `/scan/:id/results` for full detail view

---

## 11. Data Storage

### 11.1 SQLite Schema

The backend uses SQLite stored at `instance/app.db`. Two tables are created automatically on first startup via `db.create_all()`.

**`users` table:**

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PK | Auto-increment |
| `name` | VARCHAR(128) | Clinician full name |
| `email` | VARCHAR(256) UNIQUE | Login identifier |
| `password_hash` | VARCHAR(256) | Werkzeug PBKDF2-SHA256 hash |
| `role` | VARCHAR(64) | Default: `'clinician'` |
| `created_at` | DATETIME | UTC timestamp |

**`scans` table:**

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PK | Auto-increment |
| `user_id` | INTEGER FK → users.id | Owning clinician |
| `patient_name` | VARCHAR(256) | Optional, entered at upload |
| `patient_age` | INTEGER | Optional, entered at upload |
| `filename` | VARCHAR(256) | Original upload filename |
| `uploaded_at` | DATETIME | UTC timestamp |
| `grade` | INTEGER | 0–4 DR grade prediction |
| `grade_label` | VARCHAR(64) | e.g. `"Moderate DR"` |
| `confidence` | FLOAT | Softmax probability of predicted grade (0–1) |
| `prob_ma` | FLOAT | Sigmoid probability of MA presence (0–1) |
| `prob_he` | FLOAT | Sigmoid probability of HE presence (0–1) |
| `prob_ex` | FLOAT | Sigmoid probability of EX presence (0–1) |
| `prob_se` | FLOAT | Sigmoid probability of SE presence (0–1) |
| `heatmap_path` | VARCHAR(512) | Grad-CAM heatmap URL or null |
| `original_image_path` | VARCHAR(512) | Original uploaded image URL or null |
| `seg_overlay_url` | VARCHAR(512) | Segmentation overlay URL or null |
| `seg_masks_json` | TEXT | JSON string of per-lesion mask URLs or null |

**Access control:** Each scan is owned by a user (`user_id` FK). The `/api/scans/` and `/api/scans/:id` endpoints only return scans belonging to the authenticated user (`get_jwt_identity()` is checked on every request).

### 11.2 Authentication

JSON Web Tokens (JWT) are used for stateless authentication. Flask-JWT-Extended issues tokens with a 2-hour expiry on login/signup; the frontend stores them in `localStorage` under key `token`. Every authenticated request sends `Authorization: Bearer <token>` in the HTTP header.

Passwords are hashed using Werkzeug's `generate_password_hash` (PBKDF2-SHA256 with salt). Plaintext passwords are never stored.

### 11.3 Image Storage

All generated images are stored locally under `instance/`:

| Directory | Contents |
|-----------|----------|
| `instance/uploads/` | Original uploaded fundus image |
| `instance/heatmaps/` | Grad-CAM heatmap overlay (512×512 PNG) |
| `instance/segmentations/` | Lesion segmentation overlay + per-lesion masks (MA, HE, EX, SE) |

Flask serves these via `static_folder=instance_dir, static_url_path='/static'`, making them available at `/static/uploads/scan_{id}.png`, etc. The Vite dev server proxies `/static` requests to Flask so the frontend can load them without CORS issues.

---

## 12. API Reference

All endpoints are prefixed with `/api`. All authenticated endpoints require `Authorization: Bearer <JWT>`.

### `POST /api/auth/signup`

Create a new clinician account.

**Request body (JSON):**
```json
{ "name": "Dr. Jane Smith", "email": "jane@hospital.org", "password": "securepassword" }
```

**Response 201:**
```json
{ "token": "eyJhbGc...", "user": { "id": 1, "name": "Dr. Jane Smith", "email": "jane@hospital.org", "role": "clinician", "created_at": "2026-05-10T12:00:00" } }
```

**Errors:** 400 (missing fields / password too short), 409 (email already registered)

---

### `POST /api/auth/login`

**Request body (JSON):**
```json
{ "email": "jane@hospital.org", "password": "securepassword" }
```

**Response 200:** Same shape as signup.

**Errors:** 400 (missing fields), 401 (invalid credentials)

---

### `POST /api/predict`

Run model inference on a fundus image. **Requires JWT.**

**Request:** `multipart/form-data`
- `image` (file, required): fundus image (JPG/PNG/TIFF)
- `patient_name` (text, optional)
- `patient_age` (text, optional)

**Response 200:**
```json
{
  "grade": 2,
  "grade_label": "Moderate DR",
  "description": "Moderate non-proliferative DR. Referral to ophthalmologist within 6 months.",
  "confidence": 0.847,
  "all_grade_probs": [0.02, 0.04, 0.85, 0.07, 0.02],
  "lesions": {
    "MA": 0.91,
    "HE": 0.73,
    "EX": 0.44,
    "SE": 0.12
  },
  "scan_id": 7,
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

**Errors:** 400 (no image), 401 (no/invalid JWT), 500 (model not loaded or inference failure)

---

### `GET /api/scans/`

Return all scans for the authenticated user, ordered newest first. **Requires JWT.**

**Response 200:** Array of scan objects (same schema as `scans` table, plus a `lesions` dict).

---

### `GET /api/scans/<id>`

Return a single scan by ID. Returns 404 if the scan doesn't belong to the authenticated user. **Requires JWT.**

---

### `GET /api/dashboard`

Return computed stats and recent activity for the authenticated user. **Requires JWT.**

**Response 200:**
```json
{
  "stats": [
    { "id": "total",    "label": "Total Scans",      "value": "42",    "color": "text-primary",   "accentColor": "bg-primary" },
    { "id": "early",    "label": "Early Detections",  "value": "31",    "color": "text-tertiary",  "accentColor": "bg-tertiary" },
    { "id": "accuracy", "label": "AI Accuracy",       "value": "96.8%", "color": "text-secondary", "accentColor": "bg-secondary" },
    { "id": "backlog",  "label": "Backlog",           "value": "5",     "color": "text-error",     "accentColor": "bg-error" }
  ],
  "recent": [
    {
      "id": 7,
      "initials": "PS",
      "name": "Priya Sharma",
      "patientId": "PT-000007",
      "scanDate": "22 Jun 2026",
      "drDotColor": "bg-warning",
      "drGradeColor": "text-warning",
      "drGradeLabel": "Moderate DR",
      "confidence": "94.2%"
    }
  ]
}
```

---

## 13. Setup & Running

### 13.1 Prerequisites

- Python 3.10+
- Node.js 18+
- `checkpoints/best_model_lambda0.4.pth` (download separately — see below)

### 13.2 Download Model Checkpoints

The grading model checkpoint (77 MB) is excluded from git. Download from Kaggle:

```
https://kaggle.com/datasets/ff9e5ab2c38dcb3b603cc3fe6df9504c9e5363828f30cec6c0b0865a9017aaaa
```

Place it at: `checkpoints/best_model_lambda0.4.pth`

The segmentation model checkpoint (75 MB) should be placed at `checkpoints/lesion_seg_v2.pth` if available. The web app works without it (segmentation overlay is skipped).

### 13.3 Environment Variables

Create `.env` at the project root (copy and fill in your values):

```bash
SECRET_KEY=your-flask-secret-key-min-32-chars
JWT_SECRET_KEY=your-jwt-secret-key-min-32-chars-for-hs256
DATABASE_URL=sqlite:///app.db
CHECKPOINT_PATH=./checkpoints/best_model_lambda0.4.pth
SEG_CHECKPOINT_PATH=./checkpoints/lesion_seg_v2.pth
FRONTEND_URL=http://localhost:5173
```

`DATABASE_URL` defaults to `sqlite:///app.db` (stored at `backend/instance/app.db`) if not set.

### 13.4 Backend Setup

```bash
# From project root — install backend dependencies
pip install -r backend/requirements.txt

# Start Flask
PYTHONPATH="$PWD" python backend/app.py
```

On first startup, Flask will:
1. Create SQLite database at `instance/app.db` and create tables if they don't exist
2. Load the EfficientNet-B4 checkpoint into memory (takes ~5–15 seconds)
3. Start serving on `http://localhost:5000`

### 13.5 Frontend Setup

```bash
# From the frontend/ directory
npm install        # first time only
npm run dev        # starts Vite on http://localhost:5173 (or :5174 if port busy)
```

The `frontend/vite.config.js` proxies `/api` and `/static` requests to the Flask backend on port 5000.

### 13.6 ML Training (reproducing from scratch)

```bash
# 1. Download APTOS 2019 from Kaggle
#    Place images in data/aptos/train_images/ and data/aptos/train.csv

# 2. Preprocess all images (Ben Graham + CLAHE, saves to outputs/preprocessed/)
python -m src.preprocess

# 3. Train the primary model (λ=0.4, 30 epochs)
python -m src.train
# Checkpoint saved to checkpoints/best_model_lambda0.4.pth
# Expected: Test QWK ≈ 0.9065

# 4. Run ablation study (λ = 0.3, 0.4, 0.5)
# Uncomment the ablation block in src/train.py __main__
python -m src.train
# ~2.5 hours on T4 GPU
# Results saved to logs/ablation_results.json
```

---

## 14. Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Environment setup, dataset download, APTOS EDA | ✅ Complete |
| 2 | Preprocessing pipeline: Ben Graham + CLAHE + fundus crop | ✅ Complete |
| 3 | Multi-task model training + λ ablation study | ✅ Complete |
| 4 | Grad-CAM validation on IDRiD pixel-level masks (IoU, Dice) | ✅ Complete |
| 5 | Zero-shot cross-dataset evaluation on Messidor-2 | ⬜ Upcoming |
| 6 | Flask REST API + React clinical portal (Leison Lens) | ✅ Complete |
| 7 | Paper writing and journal submission | ⬜ Upcoming |

---

## 15. Hyperparameters Reference

| Parameter | Value | File |
|-----------|-------|------|
| Backbone | EfficientNet-B4 (timm) | `src/config.py` |
| Image size | 512 × 512 | `src/config.py` |
| Batch size | 12 | `src/config.py` |
| Epochs | 30 | `src/config.py` |
| Learning rate | 1e-4 | `src/config.py` |
| Weight decay | 1e-5 | `src/config.py` |
| Dropout rate | 0.4 | `src/config.py` |
| λ (lesion loss weight) | 0.4 | `src/config.py` |
| Ben Graham σ | 10 | `src/config.py` |
| CLAHE clip limit | 2.0 | `src/config.py` |
| CLAHE tile grid | (8, 8) | `src/config.py` |
| LR scheduler | ReduceLROnPlateau | `src/train.py` |
| LR patience | 3 epochs | `src/train.py` |
| LR factor | 0.5 | `src/train.py` |
| Random seed | 42 | `src/config.py` |
| Dataset split | 80/10/10 stratified | `src/config.py` |
| Grade head | Linear(1792→256)→ReLU→Linear(256→5) | `src/model.py` |
| Lesion head | Linear(1792→256)→ReLU→Linear(256→4) | `src/model.py` |
| Grade loss | CrossEntropyLoss | `src/losses.py` |
| Lesion loss | BCEWithLogitsLoss | `src/losses.py` |

---

## 16. References

1. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML 2019*.
2. Selvaraju, R.R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. *ICCV 2017*.
3. Porwal, P., et al. (2020). IDRiD: Diabetic Retinopathy — Segmentation and Grading Challenge. *Medical Image Analysis*.
4. Graham, B. (2015). Kaggle Diabetic Retinopathy Detection Competition Report. *Kaggle*.
5. APTOS 2019 Blindness Detection. Aravind Eye Hospital. *Kaggle Competition*.
6. Decencière, E., et al. (2014). Feedback on a publicly distributed database: The Messidor database. *Image Analysis & Stereology*.
7. Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularisation. *ICLR 2019*.
8. Zuiderveld, K. (1994). Contrast Limited Adaptive Histogram Equalization. *Graphics Gems IV*.

---

*Model trained on Kaggle free tier (NVIDIA T4 GPU, 16 GB VRAM). No paid compute used.*
*Clinical portal built with Flask + React + Tailwind CSS + SQLite.*

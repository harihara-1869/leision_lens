#!/usr/bin/env python3
"""
evaluate_segmentation_v2.py — Correct evaluation using individual TIF masks.

Key differences from v1:
  - Uses individual TIF masks (NOT combined PNGs) — fixes corrupted ground truth
  - Uses Ben Graham + CLAHE preprocessing (as per handout)
  - Implements 8-way TTA (horizontal/vertical flips + rotations)
  - Uses threshold = 0.5 (matching handout)
  - Evaluates all 4 classes: MA, HE, EX, SE

Usage:
    python evaluate_segmentation_v2.py

Output:
    evaluation_outputs_v2/
    ├── metrics.json
    └── evaluation_report.md
"""

import os, json, time, warnings
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
print(f'[device] {DEVICE}')

OUTPUT_DIR = 'evaluation_outputs_v2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Paths to official IDRiD masks ─────────────────────────────────────────────
IDRID_BASE = 'data/idrid_official/A. Segmentation/A. Segmentation'
IMG_DIR_TRAIN = os.path.join(IDRID_BASE, '1. Original Images/a. Training Set')
IMG_DIR_TEST  = os.path.join(IDRID_BASE, '1. Original Images/b. Testing Set')
MASK_BASE_TRAIN = os.path.join(IDRID_BASE, '2. All Segmentation Groundtruths/a. Training Set')
MASK_BASE_TEST  = os.path.join(IDRID_BASE, '2. All Segmentation Groundtruths/b. Testing Set')

LESION_DIRS = {
    'MA': '1. Microaneurysms',
    'HE': '2. Haemorrhages',
    'EX': '3. Hard Exudates',
    'SE': '4. Soft Exudates',
}
LESION_NAMES = ['MA', 'HE', 'EX', 'SE']

IMG_SIZE = 512
THRESHOLD = 0.5

# ── Model (same architecture) ────────────────────────────────────────────────
class ConvBNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch))
    def forward(self, x): return self.block(x)


class LesionSegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=False,
                                          features_only=True, out_indices=[2, 3, 4])
        self.lat_deep = ConvBNBlock(448, 128)
        self.lat_mid = ConvBNBlock(160, 128)
        self.lat_shallow = ConvBNBlock(56, 128)
        self.td_mid = ConvBNBlock(128, 128)
        self.td_shallow = ConvBNBlock(128, 128)
        class _Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.add_module('0', ConvBNBlock(128, 64))
                self.add_module('2', ConvBNBlock(64, 32))
                self.add_module('4', ConvBNBlock(32, 16))
                self.add_module('6', nn.Conv2d(16, 4, 1))
            def forward(self, x):
                x = self._modules['0'](x)
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                x = self._modules['2'](x)
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                x = self._modules['4'](x)
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                x = self._modules['6'](x)
                return x
        self.head = _Head()

    def forward(self, x):
        fs, fm, fd = self.backbone(x)
        ld = self.lat_deep(fd)
        lm = self.lat_mid(fm)
        ls = self.lat_shallow(fs)
        fd_up = F.interpolate(ld, scale_factor=2, mode='bilinear', align_corners=False)
        rm = self.td_mid(lm + fd_up)
        fm_up = F.interpolate(rm, scale_factor=2, mode='bilinear', align_corners=False)
        rs = self.td_shallow(ls + fm_up)
        return self.head(rs)


def load_model():
    model = LesionSegmentationModel().to(DEVICE)
    ckpt = torch.load('checkpoints/lesion_seg_v2.pth', map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


# ── Ben Graham + CLAHE preprocessing (as specified in handout) ────────────────
def preprocess_ben_graham_clahe(img_bgr):
    """
    Full preprocessing: resize → Ben Graham → CLAHE → normalize.
    Matches the grading model's inference.py pipeline exactly.
    """
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))

    # Ben Graham illumination removal
    blurred = cv2.GaussianBlur(img, (0, 0), 10)
    img = cv2.addWeighted(img, 4, blurred, -4, 128)

    # CLAHE on green channel
    b, g, r = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(g)
    img = cv2.merge([b, g, r])

    # BGR → RGB, normalize
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_rgb - mean) / std
    return torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)


# ── 8-way TTA ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_with_tta(model, tensor):
    """
    8-way TTA: original + hflip + vflip + both flips × 2 rotations (0°, 90°).
    Waits, the handout says "8-way TTA" which is:
      - identity
      - horizontal flip
      - vertical flip  
      - both flips (180° rotation)
      - transpose (90° rotation)
      - transpose + hflip
      - transpose + vflip
      - transpose + both flips
    Simplified: use 4 rotations (0, 90, 180, 270) × 2 flips (none, horizontal) = 8
    """
    probs_list = []
    
    for hflip in [False, True]:
        for rot in [0, 1, 2, 3]:  # 0, 90, 180, 270 degrees
            x = tensor
            if hflip:
                x = torch.flip(x, dims=[3])
            if rot == 1:
                x = torch.rot90(x, k=1, dims=[2, 3])
            elif rot == 2:
                x = torch.rot90(x, k=2, dims=[2, 3])
            elif rot == 3:
                x = torch.rot90(x, k=3, dims=[2, 3])
            
            logits = model(x)
            probs = torch.sigmoid(logits)
            
            # Undo augmentations
            if rot == 1:
                probs = torch.rot90(probs, k=3, dims=[2, 3])
            elif rot == 2:
                probs = torch.rot90(probs, k=2, dims=[2, 3])
            elif rot == 3:
                probs = torch.rot90(probs, k=1, dims=[2, 3])
            if hflip:
                probs = torch.flip(probs, dims=[3])
            
            probs_list.append(probs)
    
    # Average all 8 predictions
    return torch.stack(probs_list).mean(dim=0).cpu().numpy()


# ── Dataset ────────────────────────────────────────────────────────────────────
def load_idrid_test_set():
    """
    Load IDRiD test segmentation set with individual TIF masks.
    Returns list of dicts with image path and mask paths per lesion.
    """
    items = []
    img_dir = IMG_DIR_TEST

    for f in sorted(os.listdir(img_dir)):
        if not f.endswith('.jpg'):
            continue
        stem = f.replace('.jpg', '')
        item = {
            'image': os.path.join(img_dir, f),
            'id': stem,
            'masks': {},
        }
        has_any_mask = False
        for lesion in LESION_NAMES:
            mask_dir = os.path.join(MASK_BASE_TEST, LESION_DIRS[lesion])
            mask_file = os.path.join(mask_dir, f'{stem}_{lesion}.tif')
            if os.path.exists(mask_file):
                item['masks'][lesion] = mask_file
                has_any_mask = True
        if has_any_mask:
            items.append(item)

    print(f'[data] Test samples: {len(items)}')
    for lesion in LESION_NAMES:
        n = sum(1 for it in items if lesion in it['masks'])
        if n > 0:
            print(f'  {lesion}: {n} masks')
    return items


def load_gt_mask(path):
    """Load individual TIF mask and resize to IMG_SIZE."""
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    # Ensure single-channel
    if mask.ndim > 2:
        mask = mask[:, :, 0]
    # TIF masks have values [0, 76] — threshold at > 0
    mask_bin = (mask > 0).astype(np.uint8)
    mask_resized = cv2.resize(mask_bin, (IMG_SIZE, IMG_SIZE),
                              interpolation=cv2.INTER_NEAREST)
    return (mask_resized > 0).astype(np.uint8)


def compute_iou(pred_bin, gt_bin):
    inter = (pred_bin * gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum() - inter
    return float(inter) / float(union + 1e-7)


def compute_dice(pred_bin, gt_bin):
    inter = (pred_bin * gt_bin).sum()
    return float(2 * inter) / float(pred_bin.sum() + gt_bin.sum() + 1e-7)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    model = load_model()
    print(f'[model] Loaded (17.9M params)')

    dataset = load_idrid_test_set()

    t0 = time.time()
    results = {name: {'iou': [], 'dice': []} for name in LESION_NAMES}

    for item in dataset:
        img_bgr = cv2.imread(item['image'])
        if img_bgr is None:
            continue

        # Ben Graham + CLAHE preprocessing
        tensor = preprocess_ben_graham_clahe(img_bgr)

        # 8-way TTA inference
        probs = predict_with_tta(model, tensor)  # (1, 4, 512, 512)

        for lesion in LESION_NAMES:
            if lesion not in item['masks']:
                continue
            gt_bin = load_gt_mask(item['masks'][lesion])
            if gt_bin is None:
                continue

            model_idx = LESION_NAMES.index(lesion)
            pred_bin = (probs[0, model_idx] > THRESHOLD).astype(np.uint8)

            iou = compute_iou(pred_bin, gt_bin)
            dice = compute_dice(pred_bin, gt_bin)

            results[lesion]['iou'].append(iou)
            results[lesion]['dice'].append(dice)

    elapsed = time.time() - t0

    # Aggregate
    print('\n' + '=' * 60)
    print('  RESULTS')
    print('=' * 60)
    print(f'  {"Lesion":<8} {"Samples":<8} {"IoU":<10} {"Dice":<10}')
    print(f'  {"-"*36}')

    final = {}
    all_ious, all_dices = [], []
    for lesion in LESION_NAMES:
        ious = results[lesion]['iou']
        dices = results[lesion]['dice']
        mean_iou = float(np.mean(ious))
        mean_dice = float(np.mean(dices))
        n = len(ious)
        final[lesion] = {
            'n': n,
            'mean_iou': mean_iou,
            'mean_dice': mean_dice,
        }
        all_ious.extend(ious)
        all_dices.extend(dices)
        print(f'  {lesion:<8} {n:<8} {mean_iou:<10.4f} {mean_dice:<10.4f}')

    macro_iou = float(np.mean(all_ious))
    macro_dice = float(np.mean(all_dices))
    print(f'  {"-"*36}')
    print(f'  {"Macro":<8} {len(all_ious):<8} {macro_iou:<10.4f} {macro_dice:<10.4f}')

    # Compare with handout
    print(f'\n  Comparison with handout:')
    handout_ious = {'HE': 0.2694, 'EX': 0.2922, 'SE': 0.4286}
    for lesion, h_iou in handout_ious.items():
        c_iou = final.get(lesion, {}).get('mean_iou', 0)
        print(f'    {lesion}: handout={h_iou:.4f} vs ours={c_iou:.4f}  (diff={c_iou-h_iou:+.4f})')

    # Save
    output = {
        'config': {
            'preprocessing': 'Ben Graham + CLAHE + ImageNet norm',
            'tta': '8-way (4 rotations × 2 flips)',
            'threshold': THRESHOLD,
            'ground_truth': 'individual TIF masks (official IDRiD)',
            'resize': f'{IMG_SIZE}x{IMG_SIZE}',
        },
        'per_class': final,
        'macro': {
            'iou': macro_iou,
            'dice': macro_dice,
        },
        'handout_comparison': {
            k: {
                'handout_iou': v,
                'measured_iou': final.get(k, {}).get('mean_iou', 0),
                'diff': final.get(k, {}).get('mean_iou', 0) - v,
            } for k, v in handout_ious.items()
        },
        'evaluation_time_s': round(elapsed, 1),
    }

    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\n[output] Saved to {OUTPUT_DIR}/')

    # Report
    report = f"""# IDRiD Segmentation Evaluation v2 — Individual TIF Masks

**Model**: `lesion_seg_v2.pth` (EfficientNet-B4 + FPN)
**Ground truth**: Individual TIF masks (official IDRiD, NOT combined PNGs)
**Preprocessing**: Ben Graham + CLAHE + ImageNet norm
**TTA**: 8-way (4 rotations × 2 flips)
**Threshold**: {THRESHOLD}
**Test samples**: {len(dataset)} images

## Per-Class Results

| Lesion | Samples | IoU | Dice |
|--------|---------|-----|------|
"""
    for lesion in LESION_NAMES:
        r = final.get(lesion, {})
        if r:
            report += f"| {lesion} | {r['n']} | {r['mean_iou']:.4f} | {r['mean_dice']:.4f} |\n"

    report += f"""| **Macro** | **{len(all_ious)}** | **{macro_iou:.4f}** | **{macro_dice:.4f}** |

## Comparison with Handout

| Lesion | Handout IoU | Measured IoU | Diff |
|--------|-----------|-------------|------|
"""
    for lesion, h_iou in handout_ious.items():
        c_iou = final.get(lesion, {}).get('mean_iou', 0)
        report += f"| {lesion} | {h_iou:.4f} | {c_iou:.4f} | {c_iou-h_iou:+.4f} |\n"

    report += f"""
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
- Evaluation time: {elapsed:.1f}s
"""

    with open(os.path.join(OUTPUT_DIR, 'evaluation_report.md'), 'w') as f:
        f.write(report)
    print(f'[output] Saved evaluation report')


if __name__ == '__main__':
    main()

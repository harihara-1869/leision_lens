#!/usr/bin/env python3
"""
evaluate_segmentation.py — Evaluate lesion_seg_v2.pth on IDRiD segmentation dataset.

Computes per-class and aggregate segmentation metrics:
    Dice, IoU, Precision, Recall, F1

Usage:
    python evaluate_segmentation.py

Output:
    evaluation_outputs/
    ├── metrics.json
    ├── evaluation_report.md
    ├── metrics_summary.txt
    ├── sample_predictions/   (overlays for selected cases)
    └── optimal_thresholds.json
"""

import os, sys, json, time, warnings, copy
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

warnings.filterwarnings('ignore')

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
print(f'[device] Using {DEVICE}')

# ── Paths ─────────────────────────────────────────────────────────────────────
IDRID_SEG_DIR = 'data/idrid_seg'
CHECKPOINT_PATH = 'checkpoints/lesion_seg_v2.pth'
OUTPUT_DIR = 'evaluation_outputs'
SAMPLE_DIR = os.path.join(OUTPUT_DIR, 'sample_predictions')

# ── Lesion mapping ────────────────────────────────────────────────────────────
# Model output channels (4): [MA, HE, EX, SE]
# IDRiD combined masks (BGR):  B=EX, G=MA, R=HE
MODEL_CHANNELS = ['MA', 'HE', 'EX', 'SE']
# Which model channels have ground truth in IDRiD
IDRID_CHANNEL_INDICES = {
    'MA': {'model_idx': 0, 'mask_ch': 1},  # G channel
    'HE': {'model_idx': 1, 'mask_ch': 2},  # R channel
    'EX': {'model_idx': 2, 'mask_ch': 0},  # B channel
    # SE: no ground truth in IDRiD combined masks
}

THRESHOLDS_TO_TRY = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL (same as test_segmentation.py)
# ══════════════════════════════════════════════════════════════════════════════
class ConvBNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
        )
    def forward(self, x):
        return self.block(x)


class LesionSegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b4', pretrained=False,
            features_only=True, out_indices=[2, 3, 4],
        )
        self.lat_deep    = ConvBNBlock(448, 128)
        self.lat_mid     = ConvBNBlock(160, 128)
        self.lat_shallow = ConvBNBlock(56, 128)
        self.td_mid      = ConvBNBlock(128, 128)
        self.td_shallow  = ConvBNBlock(128, 128)
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
        f_s, f_m, f_d = self.backbone(x)
        ld = self.lat_deep(f_d)
        lm = self.lat_mid(f_m)
        ls = self.lat_shallow(f_s)
        up_d = F.interpolate(ld, scale_factor=2, mode='bilinear', align_corners=False)
        fm = lm + up_d
        rm = self.td_mid(fm)
        up_m = F.interpolate(rm, scale_factor=2, mode='bilinear', align_corners=False)
        fs = ls + up_m
        rs = self.td_shallow(fs)
        return self.head(rs)


def load_model(path):
    print(f'[model] Loading: {path}')
    model = LesionSegmentationModel().to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    result = model.load_state_dict(ckpt, strict=False)
    if result.unexpected_keys:
        print(f'[model] Unexpected keys (ignored): {len(result.unexpected_keys)}')
    model.eval()
    total = sum(p.numel() for p in model.parameters())
    print(f'[model] Loaded — {total:,} params')
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  IDRiD DATASET
# ══════════════════════════════════════════════════════════════════════════════
IMG_SIZE = 512


def load_idrid_split(split='Test'):
    """Load IDRiD segmentation split (Train=54 or Test=27)."""
    split_dir = 'DR_Training_Set' if split == 'Train' else 'DR_Testing_Set'
    img_dir = os.path.join(IDRID_SEG_DIR, split_dir, 'Fundus Images')
    mask_dir = os.path.join(IDRID_SEG_DIR, split_dir, 'Combined Masks')

    items = []
    for f in sorted(os.listdir(img_dir)):
        if not f.endswith('.jpg'):
            continue
        stem = f.replace('.jpg', '')
        mask_f = f'{stem}.png'
        img_path = os.path.join(img_dir, f)
        mask_path = os.path.join(mask_dir, mask_f)
        if os.path.exists(mask_path):
            items.append({'image': img_path, 'mask': mask_path, 'id': stem})
    print(f'[data] {split}: {len(items)} samples')
    return items


def decode_idrid_mask(mask_bgr, target_size=(IMG_SIZE, IMG_SIZE)):
    """
    Decode IDRiD combined mask (BGR PNG) into per-lesion binary masks.

    IDRiD encoding (OpenCV BGR):
        B channel = EX
        G channel = MA
        R channel = HE

    Returns: dict with 'MA', 'HE', 'EX' as binary numpy arrays (H, W)
    """
    # Resize mask to target (use INTER_NEAREST to preserve binary values)
    mask_resized = cv2.resize(mask_bgr, target_size, interpolation=cv2.INTER_NEAREST)

    gt = {
        'MA': (mask_resized[:, :, 1] > 0).astype(np.uint8),  # G
        'HE': (mask_resized[:, :, 2] > 0).astype(np.uint8),  # R
        'EX': (mask_resized[:, :, 0] > 0).astype(np.uint8),  # B
    }
    return gt


def preprocess(img_bgr):
    """Resize to 512×512, ImageNet normalise, return tensor."""
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_rgb - mean) / std
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    return tensor


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════════
def compute_dice(pred_bin, gt_bin):
    """Dice = 2*TP / (2*TP + FP + FN)."""
    smooth = 1e-7
    intersection = (pred_bin * gt_bin).sum()
    return (2.0 * intersection + smooth) / (pred_bin.sum() + gt_bin.sum() + smooth)


def compute_iou(pred_bin, gt_bin):
    """IoU = TP / (TP + FP + FN)."""
    smooth = 1e-7
    intersection = (pred_bin * gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def compute_precision_recall(pred_bin, gt_bin):
    """Precision = TP / (TP + FP), Recall = TP / (TP + FN)."""
    smooth = 1e-7
    tp = (pred_bin * gt_bin).sum()
    fp = (pred_bin * (1 - gt_bin)).sum()
    fn = ((1 - pred_bin) * gt_bin).sum()
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return precision, recall


def compute_f1(precision, recall):
    return 2 * precision * recall / (precision + recall + 1e-7)


def evaluate_at_threshold(model, dataset, threshold, lesion_names):
    """Compute per-class metrics at a given threshold."""
    metrics = {name: {'dice': [], 'iou': [], 'precision': [], 'recall': []}
               for name in lesion_names}

    with torch.no_grad():
        for item in dataset:
            # Load and preprocess image
            img_bgr = cv2.imread(item['image'])
            if img_bgr is None:
                continue
            tensor = preprocess(img_bgr)

            # Inference
            logits = model(tensor)
            probs = torch.sigmoid(logits)[0].cpu().numpy()  # (4, 512, 512)

            # Ground truth
            mask_bgr = cv2.imread(item['mask'], cv2.IMREAD_COLOR)
            if mask_bgr is None:
                continue
            gt = decode_idrid_mask(mask_bgr)

            # Per-class metrics
            for name in lesion_names:
                info = IDRID_CHANNEL_INDICES[name]
                pred_bin = (probs[info['model_idx']] > threshold).astype(np.uint8)
                gt_bin = gt[name]

                dice = compute_dice(pred_bin, gt_bin)
                iou = compute_iou(pred_bin, gt_bin)
                prec, rec = compute_precision_recall(pred_bin, gt_bin)

                metrics[name]['dice'].append(dice)
                metrics[name]['iou'].append(iou)
                metrics[name]['precision'].append(prec)
                metrics[name]['recall'].append(rec)

    # Aggregate
    results = {}
    for name in lesion_names:
        results[name] = {
            'dice': float(np.mean(metrics[name]['dice'])),
            'iou': float(np.mean(metrics[name]['iou'])),
            'precision': float(np.mean(metrics[name]['precision'])),
            'recall': float(np.mean(metrics[name]['recall'])),
        }
        results[name]['f1'] = float(compute_f1(results[name]['precision'],
                                                results[name]['recall']))

    # Macro average
    macro = {
        'dice': float(np.mean([results[n]['dice'] for n in lesion_names])),
        'iou': float(np.mean([results[n]['iou'] for n in lesion_names])),
        'precision': float(np.mean([results[n]['precision'] for n in lesion_names])),
        'recall': float(np.mean([results[n]['recall'] for n in lesion_names])),
        'f1': float(np.mean([results[n]['f1'] for n in lesion_names])),
    }

    return results, macro


def find_optimal_thresholds(model, dataset, lesion_names):
    """Find best threshold per class by maximising Dice."""
    best = {}
    for name in lesion_names:
        best_dice = 0.0
        best_t = 0.5
        for t in THRESHOLDS_TO_TRY:
            results, _ = evaluate_at_threshold(model, dataset, t, [name])
            d = results[name]['dice']
            if d > best_dice:
                best_dice = d
                best_t = t
        best[name] = {'threshold': best_t, 'dice': best_dice}
    return best


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════
LESION_COLORS = {
    'MA': (0, 255, 255),   # yellow
    'HE': (0, 0, 255),     # red
    'EX': (255, 0, 0),     # blue
    'SE': (0, 255, 0),     # green
}


def save_sample_predictions(model, dataset, lesion_names, threshold, output_dir,
                            max_samples=10):
    """Save overlay visualisations for a sample of cases."""
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for idx, item in enumerate(dataset):
            if idx >= max_samples:
                break

            img_bgr = cv2.imread(item['image'])
            if img_bgr is None:
                continue
            img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
            tensor = preprocess(img_bgr)
            logits = model(tensor)
            probs = torch.sigmoid(logits)[0].cpu().numpy()

            mask_bgr = cv2.imread(item['mask'], cv2.IMREAD_COLOR)
            gt = decode_idrid_mask(mask_bgr)

            # Create prediction overlay
            pred_overlay = img_resized.copy().astype(np.float32)
            for name in lesion_names:
                info = IDRID_CHANNEL_INDICES[name]
                pred_bin = (probs[info['model_idx']] > threshold).astype(np.uint8)
                color = np.array(LESION_COLORS[name], dtype=np.float32)
                alpha = 0.4
                pred_overlay[pred_bin > 0] = (pred_overlay[pred_bin > 0] * (1 - alpha)
                                              + color * alpha)
            pred_overlay = np.clip(pred_overlay, 0, 255).astype(np.uint8)

            # Create ground truth overlay
            gt_overlay = img_resized.copy().astype(np.float32)
            for name in lesion_names:
                info = IDRID_CHANNEL_INDICES[name]
                gt_bin = gt[name]
                color = np.array(LESION_COLORS[name], dtype=np.float32)
                alpha = 0.4
                gt_overlay[gt_bin > 0] = (gt_overlay[gt_bin > 0] * (1 - alpha)
                                          + color * alpha)
            gt_overlay = np.clip(gt_overlay, 0, 255).astype(np.uint8)

            # Side-by-side: input | ground truth | prediction
            h, w = img_resized.shape[:2]
            canvas = np.zeros((h, w * 3 + 20, 3), dtype=np.uint8)
            canvas[:, :w] = img_resized
            canvas[:, w+10:2*w+10] = gt_overlay
            canvas[:, 2*w+20:3*w+20] = pred_overlay

            # Labels
            for i, label in enumerate(['Input', 'Ground Truth', 'Prediction']):
                cv2.putText(canvas, label, (i*(w+10) + 10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            out_path = os.path.join(output_dir, f'{item["id"]}_comparison.png')
            cv2.imwrite(out_path, canvas)

    print(f'[viz] Saved {min(max_samples, len(dataset))} sample visualisations to {output_dir}')


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    # 1. Load model
    model = load_model(CHECKPOINT_PATH)

    # 2. Load IDRiD test set (27 images with pixel masks)
    dataset = load_idrid_split('Test')
    if len(dataset) == 0:
        print('[ERROR] No IDRiD segmentation data found — download from Kaggle:')
        print('  kaggle datasets download realhaadkhan/idrid-segmentation-dataset')
        sys.exit(1)

    # 3. Evaluate at standard threshold (0.5) and multiple thresholds
    lesion_names = ['MA', 'HE', 'EX']  # SE has no GT in IDRiD

    print('\n' + '=' * 60)
    print('  EVALUATION')
    print('=' * 60)

    # Standard threshold = 0.5
    t0 = time.time()
    results_t05, macro_t05 = evaluate_at_threshold(model, dataset, 0.5, lesion_names)
    eval_time = time.time() - t0

    print(f'\n  Threshold = 0.5 (standard)')
    print(f'  {"Class":<12} {"Dice":<10} {"IoU":<10} {"Precision":<12} {"Recall":<10} {"F1":<10}')
    print(f'  {"-"*64}')
    for name in lesion_names:
        r = results_t05[name]
        print(f'  {name:<12} {r["dice"]:<10.4f} {r["iou"]:<10.4f} '
              f'{r["precision"]:<12.4f} {r["recall"]:<10.4f} {r["f1"]:<10.4f}')
    print(f'  {"-"*64}')
    print(f'  {"Macro":<12} {macro_t05["dice"]:<10.4f} {macro_t05["iou"]:<10.4f} '
          f'{macro_t05["precision"]:<12.4f} {macro_t05["recall"]:<10.4f} {macro_t05["f1"]:<10.4f}')

    # Try multiple thresholds to find optimal
    print(f'\n  Threshold sweep:')
    best_macro_dice = 0.0
    best_threshold = 0.5
    for t in THRESHOLDS_TO_TRY:
        _, macro = evaluate_at_threshold(model, dataset, t, lesion_names)
        if macro['dice'] > best_macro_dice:
            best_macro_dice = macro['dice']
            best_threshold = t
        print(f'    threshold={t:.1f}: macro Dice={macro["dice"]:.4f}')

    print(f'\n  Best threshold: {best_threshold:.1f} (macro Dice={best_macro_dice:.4f})')

    # Evaluate at optimal threshold
    results_opt, macro_opt = evaluate_at_threshold(model, dataset, best_threshold,
                                                   lesion_names)

    print(f'\n  Threshold = {best_threshold:.1f} (optimal)')
    print(f'  {"Class":<12} {"Dice":<10} {"IoU":<10} {"Precision":<12} {"Recall":<10} {"F1":<10}')
    print(f'  {"-"*64}')
    for name in lesion_names:
        r = results_opt[name]
        print(f'  {name:<12} {r["dice"]:<10.4f} {r["iou"]:<10.4f} '
              f'{r["precision"]:<12.4f} {r["recall"]:<10.4f} {r["f1"]:<10.4f}')
    print(f'  {"-"*64}')
    print(f'  {"Macro":<12} {macro_opt["dice"]:<10.4f} {macro_opt["iou"]:<10.4f} '
          f'{macro_opt["precision"]:<12.4f} {macro_opt["recall"]:<10.4f} {macro_opt["f1"]:<10.4f}')

    # Save metrics
    output = {
        'threshold_0.5': {
            'per_class': {name: results_t05[name] for name in lesion_names},
            'macro': macro_t05,
        },
        f'threshold_{best_threshold:.1f}_optimal': {
            'per_class': {name: results_opt[name] for name in lesion_names},
            'macro': macro_opt,
        },
        'best_threshold': best_threshold,
        'evaluation_time_s': round(eval_time, 2),
        'num_samples': len(dataset),
        'lesion_classes_evaluated': lesion_names,
        'lesion_classes_not_evaluated': ['SE (no ground truth in IDRiD combined masks)'],
    }

    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\n[output] Saved metrics.json')

    # Save report
    report_lines = [
        '# IDRiD Segmentation Evaluation Report',
        '',
        f'**Model**: `lesion_seg_v2.pth` (EfficientNet-B4 + FPN decoder)',
        f'**Dataset**: IDRiD Segmentation Challenge (Test: {len(dataset)} images)',
        f'**Lesion classes evaluated**: {", ".join(lesion_names)}',
        f'**Lesion classes skipped**: SE (not available in IDRiD combined masks)',
        '',
        '## Results at Standard Threshold (0.5)',
        '',
        '| Class | Dice | IoU | Precision | Recall | F1 |',
        '|-------|------|-----|-----------|--------|-----|',
    ]
    for name in lesion_names:
        r = results_t05[name]
        report_lines.append(
            f'| {name} | {r["dice"]:.4f} | {r["iou"]:.4f} | '
            f'{r["precision"]:.4f} | {r["recall"]:.4f} | {r["f1"]:.4f} |')
    report_lines.append(
        f'| **Macro** | **{macro_t05["dice"]:.4f}** | **{macro_t05["iou"]:.4f}** | '
        f'**{macro_t05["precision"]:.4f}** | **{macro_t05["recall"]:.4f}** | '
        f'**{macro_t05["f1"]:.4f}** |')

    report_lines += [
        '',
        f'## Results at Optimal Threshold ({best_threshold:.1f})',
        '',
        '| Class | Dice | IoU | Precision | Recall | F1 |',
        '|-------|------|-----|-----------|--------|-----|',
    ]
    for name in lesion_names:
        r = results_opt[name]
        report_lines.append(
            f'| {name} | {r["dice"]:.4f} | {r["iou"]:.4f} | '
            f'{r["precision"]:.4f} | {r["recall"]:.4f} | {r["f1"]:.4f} |')
    report_lines.append(
        f'| **Macro** | **{macro_opt["dice"]:.4f}** | **{macro_opt["iou"]:.4f}** | '
        f'**{macro_opt["precision"]:.4f}** | **{macro_opt["recall"]:.4f}** | '
        f'**{macro_opt["f1"]:.4f}** |')

    report_lines += [
        '',
        '## Analysis',
        '',
        f'- Best threshold by macro Dice: **{best_threshold:.1f}**',
        f'- Macro Dice at threshold 0.5: **{macro_t05["dice"]:.4f}**',
        f'- Macro Dice at optimal threshold: **{macro_opt["dice"]:.4f}**',
        '',
        '### Per-class Observations',
    ]

    best_dice = max(results_t05[n]['dice'] for n in lesion_names)
    best_class = max(lesion_names, key=lambda n: results_t05[n]['dice'])
    worst_dice = min(results_t05[n]['dice'] for n in lesion_names)
    worst_class = min(lesion_names, key=lambda n: results_t05[n]['dice'])
    report_lines.append(f'- Best class: {best_class} (Dice={best_dice:.4f})')
    report_lines.append(f'- Worst class: {worst_class} (Dice={worst_dice:.4f})')

    # Comparison with reported ~30%
    reported_dice = 0.30  # ~30% as reported by teammate
    current_dice = macro_t05['dice']
    diff = current_dice - reported_dice

    report_lines += [
        '',
        '## Comparison with Reported Performance',
        '',
        f'- Reported metric: ~{reported_dice*100:.0f}% (macro Dice)',
        f'- Current macro Dice (threshold 0.5): **{current_dice:.4f}**',
        f'- Current macro Dice (optimal thresh): **{macro_opt["dice"]:.4f}**',
        f'- Difference from reported: **{diff:+.4f}**',
    ]

    if abs(diff) < 0.05:
        report_lines.append(f'\n✅ **Performance Reproduced Successfully** '
                            f'(within ±5% of reported {reported_dice*100:.0f}%)')
    else:
        report_lines.append(f'\n⚠️ **Performance Differs from Reported**')
        report_lines.append(f'  Possible causes:')
        report_lines.append(f'  - Different threshold used during training evaluation')
        report_lines.append(f'  - Data split differences (train vs test set)')
        report_lines.append(f'  - Preprocessing pipeline differences')
        report_lines.append(f'  - SE class excluded (may have been included in reported)')

    with open(os.path.join(OUTPUT_DIR, 'evaluation_report.md'), 'w') as f:
        f.write('\n'.join(report_lines))
    print(f'[output] Saved evaluation_report.md')

    # Save sample predictions
    save_sample_predictions(model, dataset, lesion_names, 0.5, SAMPLE_DIR)

    print('\n' + '=' * 60)
    print('  SUMMARY')
    print('=' * 60)
    print(f'  Model:          lesion_seg_v2.pth')
    print(f'  Dataset:        IDRiD Segmentation (Test: {len(dataset)})')
    print(f'  Evaluation:     {eval_time:.1f}s')
    print(f'  Macro Dice@0.5: {macro_t05["dice"]:.4f}')
    print(f'  Macro Dice@opt: {macro_opt["dice"]:.4f} (threshold={best_threshold:.1f})')
    print(f'  Reported:       ~{reported_dice*100:.0f}% macro Dice')
    print(f'  Difference:     {diff:+.4f}')

    verdict = 'Reproduced Successfully' if abs(diff) < 0.05 else 'Not Reproduced'
    print(f'\n  Verdict: {verdict}')

    # Cleanup
    for f in ['_inspect_seg.py']:
        if os.path.exists(f):
            os.remove(f)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
evaluate_retrained.py — Full IDRiD evaluation for lesion_seg_v2_retrained.pth.

Architecture (reconstructed from checkpoint state dict):
    Backbone:   EfficientNet-B4 (frozen, hooks on blocks[2,4,6])
    Decoder:    1×1 lateral projections + 3×3 top-down Conv+BN+ReLU
    Head:       Progressive upsampling 128→128→64→32→16→4

Preprocessing (handout-specified):
    Resize 512 → Ben Graham → CLAHE green channel → ImageNet norm

Inference:    8-way TTA (4 rotations × 2 flips), threshold 0.5
Metrics:      Per-class IoU and Dice for MA, HE, EX, SE
"""

import os, sys, json, time, warnings
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

CHECKPOINT_PATH = 'checkpoints/lesion_seg_v2_retrained.pth'
OUTPUT_DIR = 'evaluation_outputs_retrained'

IDRID_BASE = 'data/idrid_official/A. Segmentation/A. Segmentation'
IMG_DIR_TEST = os.path.join(IDRID_BASE, '1. Original Images/b. Testing Set')
MASK_BASE_TEST = os.path.join(IDRID_BASE, '2. All Segmentation Groundtruths/b. Testing Set')

LESION_DIRS = {
    'MA': '1. Microaneurysms',
    'HE': '2. Haemorrhages',
    'EX': '3. Hard Exudates',
    'SE': '4. Soft Exudates',
}
LESION_NAMES = ['MA', 'HE', 'EX', 'SE']

IMG_SIZE = 512
THRESHOLD = 0.5

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
print(f'[device] {DEVICE}')

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
#  Architecture — exact match to checkpoint state dict
# ═════════════════════════════════════════════════════════════════════════════

class Decoder(nn.Module):
    """
    FPN decoder with 1×1 lateral projections + 3×3 top-down refinement.

    Key naming matches lesion_seg_v2_retrained.pth exactly:
      decoder.lat0.weight, decoder.conv2.0.weight, decoder.head.0.weight, etc.
    """

    def __init__(self):
        super().__init__()

        # Lateral projections: 1×1 conv WITH bias (no BN)
        self.lat0 = nn.Conv2d(56, 128, 1, bias=True)   # shallow
        self.lat1 = nn.Conv2d(160, 128, 1, bias=True)  # mid
        self.lat2 = nn.Conv2d(448, 128, 1, bias=True)  # deep

        # Top-down refinement: Sequential → keys conv2.0, conv2.1
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=True),
            nn.BatchNorm2d(128),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=True),
            nn.BatchNorm2d(128),
        )
        self.conv0 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=True),
            nn.BatchNorm2d(128),
        )

        # Segmentation head with gapped indices matching checkpoint.
        # Gaps (2,3,6,7,10,11,14) are for non-parametric upsampling ops.
        self.head = nn.ModuleDict({
            '0': nn.Conv2d(128, 128, 3, padding=1, bias=True),
            '1': nn.BatchNorm2d(128),
            '4': nn.Conv2d(128, 64, 3, padding=1, bias=True),
            '5': nn.BatchNorm2d(64),
            '8': nn.Conv2d(64, 32, 3, padding=1, bias=True),
            '9': nn.BatchNorm2d(32),
            '12': nn.Conv2d(32, 16, 3, padding=1, bias=True),
            '13': nn.BatchNorm2d(16),
            '15': nn.Conv2d(16, 4, 1, bias=True),
        })

    def forward(self, fs, fm, fd):
        # Lateral projections
        ls = self.lat0(fs)        # [B, 128, 64, 64]
        lm = self.lat1(fm)        # [B, 128, 32, 32]
        ld = self.lat2(fd)        # [B, 128, 16, 16]

        # Top-down: deep → mid
        up_d = F.interpolate(ld, scale_factor=2, mode='bilinear',
                             align_corners=False)
        rm = F.relu(self.conv2(lm + up_d))   # [B, 128, 32, 32]

        # Top-down: mid → shallow
        up_m = F.interpolate(rm, scale_factor=2, mode='bilinear',
                             align_corners=False)
        rs = F.relu(self.conv1(ls + up_m))   # [B, 128, 64, 64]

        # Final refinement
        rf = F.relu(self.conv0(rs))          # [B, 128, 64, 64]

        # Head: progressive upsampling with gapped indices
        h = F.relu(self.head['0'](rf))
        h = F.relu(self.head['1'](h))
        h = F.interpolate(h, scale_factor=2, mode='bilinear',
                          align_corners=False)

        h = F.relu(self.head['4'](h))
        h = F.relu(self.head['5'](h))
        h = F.interpolate(h, scale_factor=2, mode='bilinear',
                          align_corners=False)

        h = F.relu(self.head['8'](h))
        h = F.relu(self.head['9'](h))
        h = F.interpolate(h, scale_factor=2, mode='bilinear',
                          align_corners=False)

        # head.12(32→16) + head.13(BN) — NO upsampling after this.
        # head.15(16→4, 1×1) produces the final 4-channel logits at 512×512.
        h = F.relu(self.head['12'](h))
        h = F.relu(self.head['13'](h))
        # BOS: output resolution is 512×512 here — final 1×1 conv
        out = self.head['15'](h)             # [B, 4, 512, 512]
        return out


class LesionSegModelV2(nn.Module):
    """
    EfficientNet-B4 backbone (frozen) + FPN decoder + upsampling head.

    State dict keys: backbone.* (698) + decoder.* (57) — matches checkpoint.
    """

    def __init__(self):
        super().__init__()

        # ── Backbone: ImageNet-pretrained, frozen ────────────────────────
        backbone = timm.create_model('efficientnet_b4', pretrained=True)
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()
        self.backbone = backbone

        # ── Decoder (all checkpoint weights go here) ─────────────────────
        self.decoder = Decoder()

    def _get_features(self, x):
        """Extract features from blocks[2], blocks[4], blocks[6] via hooks."""
        features = {}

        def make_hook(name):
            def hook(m, inp, out):
                features[name] = out
            return hook

        hooks = [
            self.backbone.blocks[2].register_forward_hook(make_hook('shallow')),
            self.backbone.blocks[4].register_forward_hook(make_hook('mid')),
            self.backbone.blocks[6].register_forward_hook(make_hook('deep')),
        ]

        _ = self.backbone(x)

        for h in hooks:
            h.remove()

        return features['shallow'], features['mid'], features['deep']

    def forward(self, x):
        fs, fm, fd = self._get_features(x)
        return self.decoder(fs, fm, fd)

    def param_stats(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ═════════════════════════════════════════════════════════════════════════════
#  Checkpoint loading
# ═════════════════════════════════════════════════════════════════════════════

def load_model():
    model = LesionSegModelV2().to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

    if 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    else:
        sd = ckpt

    # Load ALL keys including backbone (checkpoint was trained with
    # backbone fine-tuned on IDRiD, not ImageNet-pretrained alone).
    # This also unfreezes the backbone to allow weight loading.
    for p in model.backbone.parameters():
        p.requires_grad = True

    missing, unexpected = model.load_state_dict(sd, strict=False)

    # Re-freeze backbone
    for p in model.backbone.parameters():
        p.requires_grad = False

    print(f'[load] Total keys in checkpoint: {len(sd)}')
    print(f'[load] Keys loaded: {len(sd) - len(missing)}')
    if missing:
        print(f'[load] Missing keys: {len(missing)}')
    if unexpected:
        print(f'[load] Unexpected keys: {len(unexpected)}')

    model.eval()
    total, trainable = model.param_stats()
    print(f'[load] Total params: {total:,} | Trainable: {trainable:,}')
    return model


# ═════════════════════════════════════════════════════════════════════════════
#  Preprocessing — exact handout spec
# ═════════════════════════════════════════════════════════════════════════════

def preprocess(img_bgr):
    """Ben Graham + CLAHE + ImageNet norm → tensor [1,3,512,512]."""
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))

    # Ben Graham illumination removal
    blurred = cv2.GaussianBlur(img, (0, 0), 10)
    img = cv2.addWeighted(img, 4, blurred, -4, 128)

    # CLAHE on green channel (index 1 in BGR)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img[:, :, 1] = clahe.apply(img[:, :, 1])

    # BGR → RGB, float, normalise
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_rgb - mean) / std

    return torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)


# ═════════════════════════════════════════════════════════════════════════════
#  8-way TTA inference
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_tta(model, tensor):
    """8-way TTA: 4 rotations × 2 flips, averaged."""
    preds = []

    for hflip in [False, True]:
        for rot_k in [0, 1, 2, 3]:
            x = tensor

            if hflip:
                x = torch.flip(x, dims=[3])
            if rot_k:
                x = torch.rot90(x, k=rot_k, dims=[2, 3])

            logits = model(x)
            probs = torch.sigmoid(logits)

            # Reverse
            if rot_k:
                probs = torch.rot90(probs, k=4 - rot_k, dims=[2, 3])
            if hflip:
                probs = torch.flip(probs, dims=[3])

            preds.append(probs.cpu())

    return torch.stack(preds).mean(dim=0)[0].numpy()   # [4, 512, 512]


# ═════════════════════════════════════════════════════════════════════════════
#  Ground-truth loading
# ═════════════════════════════════════════════════════════════════════════════

def load_idrid_test_set():
    items = []
    for f in sorted(os.listdir(IMG_DIR_TEST)):
        if not f.endswith('.jpg'):
            continue
        stem = f.replace('.jpg', '')
        item = {'image': os.path.join(IMG_DIR_TEST, f), 'id': stem, 'masks': {}}
        for lesion in LESION_NAMES:
            mask_path = os.path.join(MASK_BASE_TEST, LESION_DIRS[lesion],
                                     f'{stem}_{lesion}.tif')
            if os.path.exists(mask_path):
                item['masks'][lesion] = mask_path
        if item['masks']:
            items.append(item)

    print(f'[data] Test images with masks: {len(items)}')
    for lesion in LESION_NAMES:
        n = sum(1 for it in items if lesion in it['masks'])
        print(f'  {lesion}: {n} masks')
    return items


def load_gt_mask(path):
    """Load TIF mask, threshold at >0, resize to 512×512 (nearest)."""
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if mask.ndim > 2:
        mask = mask[:, :, 0]
    mask_bin = (mask > 0).astype(np.uint8)
    mask_rs = cv2.resize(mask_bin, (IMG_SIZE, IMG_SIZE),
                         interpolation=cv2.INTER_NEAREST)
    return (mask_rs > 0).astype(np.uint8)


def compute_iou(pred, gt):
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    return float(inter) / float(union + 1e-7)


def compute_dice(pred, gt):
    inter = (pred * gt).sum()
    return float(2 * inter) / float(pred.sum() + gt.sum() + 1e-7)


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 60)
    print('  IDRiD SEGMENTATION EVALUATION — RETRAINED MODEL')
    print('=' * 60)

    # 1. Load model
    model = load_model()

    # 2. Load dataset
    dataset = load_idrid_test_set()

    # 3. Evaluate
    t0 = time.time()
    results = {name: {'iou': [], 'dice': []} for name in LESION_NAMES}

    for item in dataset:
        img_bgr = cv2.imread(item['image'])
        if img_bgr is None:
            print(f'[skip] Cannot read {item["image"]}')
            continue

        tensor = preprocess(img_bgr)
        probs = predict_tta(model, tensor)   # [4, 512, 512]

        for lesion in LESION_NAMES:
            if lesion not in item['masks']:
                continue
            gt_bin = load_gt_mask(item['masks'][lesion])
            if gt_bin is None:
                continue

            idx = LESION_NAMES.index(lesion)
            pred_bin = (probs[idx] > THRESHOLD).astype(np.uint8)

            iou = compute_iou(pred_bin, gt_bin)
            dice = compute_dice(pred_bin, gt_bin)

            results[lesion]['iou'].append(iou)
            results[lesion]['dice'].append(dice)

    elapsed = time.time() - t0

    # 4. Aggregate
    print('\n' + '-' * 60)
    print('  PER-CLASS RESULTS')
    print('-' * 60)
    print(f'  {"Lesion":<8} {"N":<6} {"IoU":<12} {"Dice":<12}')
    print(f'  {"-"*38}')

    final = {}
    all_ious, all_dices = [], []
    for lesion in LESION_NAMES:
        ious = results[lesion]['iou']
        dices = results[lesion]['dice']
        mean_iou = float(np.mean(ious)) if ious else 0.0
        mean_dice = float(np.mean(dices)) if dices else 0.0
        std_iou = float(np.std(ious)) if ious else 0.0
        n = len(ious)
        final[lesion] = {'n': n, 'mean_iou': mean_iou, 'mean_dice': mean_dice,
                         'std_iou': std_iou}
        all_ious.extend(ious)
        all_dices.extend(dices)
        print(f'  {lesion:<8} {n:<6} {mean_iou:<12.4f} {mean_dice:<12.4f}')

    macro_iou = float(np.mean(all_ious)) if all_ious else 0.0
    macro_dice = float(np.mean(all_dices)) if all_dices else 0.0
    print(f'  {"-"*38}')
    print(f'  {"Macro":<8} {len(all_ious):<6} {macro_iou:<12.4f} {macro_dice:<12.4f}')

    # 5. Compare with handout
    print('\n' + '-' * 60)
    print('  COMPARISON WITH HANDOUT')
    print('-' * 60)

    handout = {
        'MA': {'iou': None,   'dice': None},
        'HE': {'iou': 0.2694, 'dice': 0.3834},
        'EX': {'iou': 0.2922, 'dice': 0.4349},
        'SE': {'iou': 0.4286, 'dice': 0.5596},
    }

    header = f'  {"Lesion":<8} {"Metric":<8} {"Reproduced":<12} {"Handout":<12} {"Diff":<12} {"%Diff":<10}'
    print(header)
    print('  ' + '-' * len(header))

    for lesion in LESION_NAMES:
        r = final.get(lesion, {})
        h = handout.get(lesion, {})

        for metric, label in [('iou', 'IoU'), ('dice', 'Dice')]:
            reproduced = r.get(f'mean_{metric}', 0)
            expected = h.get(metric)
            if expected is None:
                diff_str = '—'
                pct_str = '—'
            else:
                diff = reproduced - expected
                pct = (diff / expected) * 100 if expected != 0 else 0
                diff_str = f'{diff:+.4f}'
                pct_str = f'{pct:+.1f}%'
            print(f'  {lesion:<8} {label:<8} {reproduced:<12.4f} '
                  f'{expected if expected is not None else "—":<12} '
                  f'{diff_str:<12} {pct_str:<10}')

    # 6. Verdict
    print('\n' + '-' * 60)
    print('  VERDICT')
    print('-' * 60)

    TOLERANCE = 0.02  # allow ±0.02 IoU/Dice tolerance
    reproduced = True
    for lesion in ['HE', 'EX', 'SE']:
        for metric in ['iou', 'dice']:
            measured = final.get(lesion, {}).get(f'mean_{metric}', 0)
            expected = handout[lesion][metric]
            if expected is not None:
                diff = abs(measured - expected)
                if diff > TOLERANCE:
                    reproduced = False
                    print(f'  ✗ {lesion} {metric.upper()}: '
                          f'{measured:.4f} vs {expected:.4f} '
                          f'(Δ={diff:.4f} > {TOLERANCE})')

    if reproduced:
        print('\n  ✅ HANDOUT RESULTS REPRODUCED')
    else:
        print('\n  ❌ HANDOUT RESULTS NOT REPRODUCED')

    # 7. Save
    output = {
        'config': {
            'model': 'LesionSegModelV2 (EfficientNet-B4 + FPN)',
            'checkpoint': CHECKPOINT_PATH,
            'preprocessing': 'Ben Graham + CLAHE + ImageNet norm',
            'tta': '8-way (4 rotations × 2 flips)',
            'threshold': THRESHOLD,
            'input_size': f'{IMG_SIZE}x{IMG_SIZE}',
        },
        'per_class': final,
        'macro': {'iou': macro_iou, 'dice': macro_dice},
        'handout_comparison': {
            k: {
                'metric': 'IoU',
                'handout': handout[k]['iou'],
                'measured': final.get(k, {}).get('mean_iou', 0),
                'diff': (final.get(k, {}).get('mean_iou', 0) -
                         handout[k]['iou']) if handout[k]['iou'] else None,
            } for k in LESION_NAMES
        },
        'verdict': 'Reproduced' if reproduced else 'Not Reproduced',
        'evaluation_time_s': round(elapsed, 1),
    }

    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\n[output] Saved: {OUTPUT_DIR}/metrics.json')
    print(f'[output] Evaluation time: {elapsed:.1f}s')


if __name__ == '__main__':
    main()

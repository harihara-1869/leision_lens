#!/usr/bin/env python3
"""
Debug evaluation — test ConvBNBlock (no ReLU) vs ConvBNReLU hypothesis.
"""
import os, json, time, warnings
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDRID_BASE = os.path.join(PROJECT_ROOT, 'data/idrid_official/A. Segmentation/A. Segmentation')
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'checkpoints/lesion_seg_v2.pth')

IMG_DIR_TEST = os.path.join(IDRID_BASE, '1. Original Images/b. Testing Set')
MASK_BASE_TEST = os.path.join(IDRID_BASE, '2. All Segmentation Groundtruths/b. Testing Set')
LESION_DIRS = {'MA': '1. Microaneurysms', 'HE': '2. Haemorrhages', 'EX': '3. Hard Exudates', 'SE': '4. Soft Exudates'}
LESION_NAMES = ['MA', 'HE', 'EX', 'SE']

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
print(f'[device] {DEVICE}')

HANDOUT = {
    'MA': {'iou': 0.0000, 'dice': 0.0000},
    'HE': {'iou': 0.2694, 'dice': 0.3834},
    'EX': {'iou': 0.2922, 'dice': 0.4349},
    'SE': {'iou': 0.4286, 'dice': 0.5596},
}

# ── ConvBNBlock (no ReLU — matches training) ─────────────────────────────────
class ConvBNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch))
    def forward(self, x): return self.block(x)


class LesionSegModel_NoReLU(nn.Module):
    """Matches the training architecture EXACTLY (ConvBNBlock, features_only)."""
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


# ── Preprocessing ────────────────────────────────────────────────────────────
def preprocess_ben_graham_clahe(img_bgr):
    img = cv2.resize(img_bgr, (512, 512))
    blurred = cv2.GaussianBlur(img, (0, 0), 10)
    img = cv2.addWeighted(img, 4, blurred, -4, 128)
    b, g, r = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(g)
    img = cv2.merge([b, g, r])
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_rgb - mean) / std
    return torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)


# ── 8-way TTA ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_with_tta(model, tensor):
    predictions = []
    for k in range(4):
        x_rot = torch.rot90(tensor, k, dims=(2, 3))
        logits = model(x_rot)
        probs = torch.sigmoid(logits)
        probs = torch.rot90(probs, -k, dims=(2, 3))
        predictions.append(probs)
        x_flip = torch.flip(x_rot, dims=(3,))
        logits = model(x_flip)
        probs = torch.sigmoid(logits)
        probs = torch.flip(probs, dims=(3,))
        probs = torch.rot90(probs, -k, dims=(2, 3))
        predictions.append(probs)
    return torch.stack(predictions).mean(dim=0).cpu().numpy()


# ── GT / metrics ──────────────────────────────────────────────────────────────
def load_gt_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is None: return None
    if mask.ndim > 2: mask = mask[:, :, 0]
    mask_resized = cv2.resize((mask > 0).astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)
    return (mask_resized > 0).astype(np.uint8)

def compute_iou(p, g):
    i = (p & g).sum()
    u = p.sum() + g.sum() - i
    return float(i) / float(u + 1e-7)

def compute_dice(p, g):
    i = (p & g).sum()
    return 2.0 * float(i) / float(p.sum() + g.sum() + 1e-7)


# ── Load dataset ──────────────────────────────────────────────────────────────
def load_dataset():
    items = []
    for f in sorted(os.listdir(IMG_DIR_TEST)):
        if not f.endswith('.jpg'): continue
        stem = f.replace('.jpg', '')
        item = {'image': os.path.join(IMG_DIR_TEST, f), 'id': stem, 'masks': {}}
        for lesion in LESION_NAMES:
            mask_file = os.path.join(MASK_BASE_TEST, LESION_DIRS[lesion], f'{stem}_{lesion}.tif')
            if os.path.exists(mask_file):
                item['masks'][lesion] = mask_file
        if item['masks']:
            items.append(item)
    return items


# ═════════════════════════════════════════════════════════════════════════════
def main():
    dataset = load_dataset()
    print(f'[data] {len(dataset)} images')

    model = LesionSegModel_NoReLU().to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        ckpt = ckpt['model_state_dict']
    # No key remapping needed — old model matches ckpt directly
    result = model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f'[model] Loaded: missing={len(result.missing_keys)} extra={len(result.unexpected_keys)}')

    results_no_tta = {name: {'iou': [], 'dice': []} for name in LESION_NAMES}
    results_tta = {name: {'iou': [], 'dice': []} for name in LESION_NAMES}

    # ── Also test with v2 TTA ordering (flip first, then rotate) ────────────
    @torch.no_grad()
    def tta_v2_style(model, tensor):
        preds = []
        for hflip in [False, True]:
            for rot in [0, 1, 2, 3]:
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

                # Undo: reverse rotations first, then flips
                if rot == 1:
                    probs = torch.rot90(probs, k=3, dims=[2, 3])
                elif rot == 2:
                    probs = torch.rot90(probs, k=2, dims=[2, 3])
                elif rot == 3:
                    probs = torch.rot90(probs, k=1, dims=[2, 3])
                if hflip:
                    probs = torch.flip(probs, dims=[3])
                preds.append(probs)
        return torch.stack(preds).mean(dim=0).cpu().numpy()

    results_tta_v2 = {name: {'iou': [], 'dice': []} for name in LESION_NAMES}

    # Also generate per-image EX probabilities to debug
    ex_probs_debug = []

    for item in dataset:
        img_bgr = cv2.imread(item['image'])
        if img_bgr is None: continue

        tensor = preprocess_ben_graham_clahe(img_bgr)

        # Without TTA
        with torch.no_grad():
            logits = model(tensor)
            probs_no_tta = torch.sigmoid(logits)[0].cpu().numpy()

        # With TTA (my ordering)
        probs_tta = predict_with_tta(model, tensor)[0]

        # With TTA (v2 ordering)
        probs_tta_v2 = tta_v2_style(model, tensor)
        if probs_tta_v2.ndim == 4:
            probs_tta_v2 = probs_tta_v2[0]

        # Debug: check EX probs distribution
        ex_probs_debug.append({
            'id': item['id'],
            'ex_mean': float(probs_tta[2].mean()),
            'ex_max': float(probs_tta[2].max()),
            'gt_exists': 'EX' in item['masks'],
        })

        for lesion in LESION_NAMES:
            if lesion not in item['masks']:
                continue
            gt_bin = load_gt_mask(item['masks'][lesion])
            if gt_bin is None: continue

            ch = LESION_NAMES.index(lesion)

            pred_no_tta = (probs_no_tta[ch] > 0.5).astype(np.uint8)
            results_no_tta[lesion]['iou'].append(compute_iou(pred_no_tta, gt_bin))
            results_no_tta[lesion]['dice'].append(compute_dice(pred_no_tta, gt_bin))

            pred_tta = (probs_tta[ch] > 0.5).astype(np.uint8)
            results_tta[lesion]['iou'].append(compute_iou(pred_tta, gt_bin))
            results_tta[lesion]['dice'].append(compute_dice(pred_tta, gt_bin))

            pred_tta_v2 = (probs_tta_v2[ch] > 0.5).astype(np.uint8)
            results_tta_v2[lesion]['iou'].append(compute_iou(pred_tta_v2, gt_bin))
            results_tta_v2[lesion]['dice'].append(compute_dice(pred_tta_v2, gt_bin))

    # Print EX debug
    print(f'\n===== EX Probability Debug =====')
    for d in ex_probs_debug:
        print(f'  {d["id"]}: EX mean={d["ex_mean"]:.6f} max={d["ex_max"]:.4f} gt={d["gt_exists"]}')

    # Print results
    for mode, results in [('No TTA', results_no_tta), ('8-way TTA (my order)', results_tta), ('8-way TTA (v2 order)', results_tta_v2)]:
        print(f'\n===== {mode} =====')
        print(f'  {"Lesion":<8} {"N":<5} {"IoU":<10} {"Dice":<10} {"H-IoU":<10} {"H-Dice":<10} {"ΔIoU":<10} {"ΔDice":<10}')
        print(f'  {"-"*68}')
        all_ious, all_dices = [], []
        for lesion in LESION_NAMES:
            r = results[lesion]
            n = len(r['iou'])
            if n == 0: continue
            mi = float(np.mean(r['iou']))
            md = float(np.mean(r['dice']))
            hi = HANDOUT[lesion]['iou']
            hd = HANDOUT[lesion]['dice']
            all_ious.extend(r['iou'])
            all_dices.extend(r['dice'])
            print(f'  {lesion:<8} {n:<5} {mi:<10.4f} {md:<10.4f} {hi:<10.4f} {hd:<10.4f} {mi-hi:<+10.4f} {md-hd:<+10.4f}')
        print(f'  {"Macro":<8} {len(all_ious):<5} {float(np.mean(all_ious)):<10.4f} {float(np.mean(all_dices)):<10.4f}')


if __name__ == '__main__':
    main()

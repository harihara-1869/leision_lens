#!/usr/bin/env python3
"""Debug EX discrepancy — test ConvBNReLU model with different TTA orderings."""
import os, json, time, warnings
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, PROJECT_ROOT)

from backend.ml.segmentation import LesionSegmentationModel, preprocess
from backend.ml.segmentation import _tta_inference as tta_my_order

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

HEAD_MAP = {'head.0.': 'head_1.', 'head.2.': 'head_2.', 'head.4.': 'head_3.', 'head.6.': 'head_out.'}

def load_model():
    model = LesionSegmentationModel().to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        ckpt = ckpt['model_state_dict']
    remapped = {}
    for key, val in ckpt.items():
        if key.startswith('backbone.'):
            remapped['encoder.' + key] = val
        elif any(key.startswith(p) for p in HEAD_MAP):
            for op, np_ in HEAD_MAP.items():
                if key.startswith(op):
                    remapped[key.replace(op, np_, 1)] = val
                    break
        else:
            remapped[key] = val
    model.load_state_dict(remapped, strict=False)
    model.eval()
    return model


@torch.no_grad()
def tta_v1(model, tensor):
    """My TTA: rotate first, then flip."""
    return tta_my_order(model, tensor)


@torch.no_grad()
def tta_v2(model, tensor):
    """v2 TTA: flip first, then rotate."""
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
            if rot == 1:
                probs = torch.rot90(probs, k=3, dims=[2, 3])
            elif rot == 2:
                probs = torch.rot90(probs, k=2, dims=[2, 3])
            elif rot == 3:
                probs = torch.rot90(probs, k=1, dims=[2, 3])
            if hflip:
                probs = torch.flip(probs, dims=[3])
            preds.append(probs)
    return torch.stack(preds).mean(dim=0)


@torch.no_grad()
def tta_v3(model, tensor):
    """Full D4 group (8 transforms): identity, rot90, rot180, rot270,
       plus each with hflip. Rotate AFTER flip."""
    preds = []
    for rot in [0, 1, 2, 3]:
        for hflip in [False, True]:
            x = tensor
            if hflip:
                x = torch.flip(x, dims=[3])
            if rot:
                x = torch.rot90(x, k=rot, dims=[2, 3])
            logits = model(x)
            probs = torch.sigmoid(logits)
            if rot:
                probs = torch.rot90(probs, k=4-rot, dims=[2, 3])
            if hflip:
                probs = torch.flip(probs, dims=[3])
            preds.append(probs)
    return torch.stack(preds).mean(dim=0)


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


def evaluate(model, dataset, tta_fn, name):
    results = {name: {'iou': [], 'dice': [], 'ids': []} for name in LESION_NAMES}
    for item in dataset:
        img_bgr = cv2.imread(item['image'])
        if img_bgr is None: continue
        tensor = preprocess(img_bgr)
        probs_4d = tta_fn(model, tensor)
        probs = probs_4d[0].cpu().numpy() if probs_4d.ndim == 4 else probs_4d
        for lesion in LESION_NAMES:
            if lesion not in item['masks']: continue
            gt_bin = load_gt_mask(item['masks'][lesion])
            if gt_bin is None: continue
            ch = LESION_NAMES.index(lesion)
            pred = (probs[ch] > 0.5).astype(np.uint8)
            results[lesion]['iou'].append(compute_iou(pred, gt_bin))
            results[lesion]['dice'].append(compute_dice(pred, gt_bin))
            results[lesion]['ids'].append(item['id'])

    print(f'\n===== {name} =====')
    print(f'  {"Lesion":<8} {"N":<5} {"IoU":<10} {"Dice":<10} {"H-IoU":<10} {"H-Dice":<10} {"ΔIoU":<10} {"ΔDice":<10}')
    print(f'  {"-"*68}')
    all_i, all_d = [], []
    for lesion in LESION_NAMES:
        r = results[lesion]
        n = len(r['iou'])
        if n == 0: continue
        mi = float(np.mean(r['iou']))
        md = float(np.mean(r['dice']))
        hi = HANDOUT[lesion]['iou']
        hd = HANDOUT[lesion]['dice']
        all_i.extend(r['iou'])
        all_d.extend(r['dice'])
        print(f'  {lesion:<8} {n:<5} {mi:<10.4f} {md:<10.4f} {hi:<10.4f} {hd:<10.4f} {mi-hi:<+10.4f} {md-hd:<+10.4f}')
    print(f'  {"Macro":<8} {len(all_i):<5} {float(np.mean(all_i)):<10.4f} {float(np.mean(all_d)):<10.4f}')
    return results


def main():
    model = load_model()
    print('[model] ConvBNReLU model loaded')
    dataset = load_dataset()

    # Test all TTA variants
    results_v1 = evaluate(model, dataset, tta_v1, 'TTA v1 (rotate→flip)')
    results_v2 = evaluate(model, dataset, tta_v2, 'TTA v2 (flip→rotate)')
    results_v3 = evaluate(model, dataset, tta_v3, 'TTA v3 (D4 group, flip→rotate)')

    # Compare EX per-image for v1 vs v2
    print('\n===== EX Per-Image Comparison =====')
    print(f'  {"Image":<16} {"v1 IoU":<10} {"v2 IoU":<10} {"v3 IoU":<10} {"v1 Dice":<10} {"v2 Dice":<10} {"v3 Dice":<10}')
    print(f'  {"-"*76}')
    for i, item in enumerate(dataset):
        if 'EX' not in item['masks']: continue
        img_id = item['id']
        v1_i = results_v1['EX']['iou'][[k for k, id_ in enumerate(results_v1['EX']['ids']) if id_ == img_id][0]] if img_id in results_v1['EX']['ids'] else -1
        v2_i = results_v2['EX']['iou'][[k for k, id_ in enumerate(results_v2['EX']['ids']) if id_ == img_id][0]] if img_id in results_v2['EX']['ids'] else -1
        v3_i = results_v3['EX']['iou'][[k for k, id_ in enumerate(results_v3['EX']['ids']) if id_ == img_id][0]] if img_id in results_v3['EX']['ids'] else -1
        v1_d = results_v1['EX']['dice'][[k for k, id_ in enumerate(results_v1['EX']['ids']) if id_ == img_id][0]] if img_id in results_v1['EX']['ids'] else -1
        v2_d = results_v2['EX']['dice'][[k for k, id_ in enumerate(results_v2['EX']['ids']) if id_ == img_id][0]] if img_id in results_v2['EX']['ids'] else -1
        v3_d = results_v3['EX']['dice'][[k for k, id_ in enumerate(results_v3['EX']['ids']) if id_ == img_id][0]] if img_id in results_v3['EX']['ids'] else -1
        print(f'  {img_id:<16} {v1_i:<10.4f} {v2_i:<10.4f} {v3_i:<10.4f} {v1_d:<10.4f} {v2_d:<10.4f} {v3_d:<10.4f}')


if __name__ == '__main__':
    main()

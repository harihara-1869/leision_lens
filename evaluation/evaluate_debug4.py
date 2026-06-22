#!/usr/bin/env python3
"""Debug: check GT mask pixel values and resize behavior."""
import os, cv2, numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASK_BASE = os.path.join(PROJECT_ROOT, 'data/idrid_official/A. Segmentation/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set')
LDIRS = {'MA': '1. Microaneurysms', 'HE': '2. Haemorrhages', 'EX': '3. Hard Exudates', 'SE': '4. Soft Exudates'}

# Check a few masks
test_ids = ['IDRiD_55', 'IDRiD_65', 'IDRiD_75']
for lesion in ['MA', 'HE', 'EX', 'SE']:
    for img_id in test_ids:
        path = os.path.join(MASK_BASE, LDIRS[lesion], f'{img_id}_{lesion}.tif')
        if not os.path.exists(path):
            continue
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f'{img_id} {lesion}: FAILED to load')
            continue
        orig_h, orig_w = mask.shape[:2]
        uniq = np.unique(mask)
        pct = (mask > 0).mean() * 100
        # Resize check
        mask_bin = (mask > 0).astype(np.uint8)
        resized = cv2.resize(mask_bin, (512, 512), interpolation=cv2.INTER_NEAREST)
        pct_after = (resized > 0).mean() * 100
        print(f'{img_id} {lesion}: shape={orig_h}x{orig_w} unique_vals={uniq} '
              f'pct_pos_before={pct:.2f}% after_resize={pct_after:.2f}% '
              f'dim={mask.ndim}')

# Also check: does cv2.IMREAD_UNCHANGED work for EX?
print(f'\nChecking specific pixel values:')
for lesion in ['EX', 'HE', 'MA', 'SE']:
    path = os.path.join(MASK_BASE, LDIRS[lesion], 'IDRiD_55_' + lesion + '.tif')
    if not os.path.exists(path): continue
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    print(f'{lesion}: dtype={mask.dtype} min={mask.min()} max={mask.max()} '
          f'pos_pixels={(mask>0).sum()} unique={np.unique(mask)}')
    # Check a specific region
    print(f'  Sample values: {mask[1000:1010, 1000:1010]}')

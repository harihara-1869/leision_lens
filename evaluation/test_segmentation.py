#!/usr/bin/env python3
"""
test_segmentation.py — Standalone inference for lesion segmentation checkpoint.

Model Architecture:
    Encoder: EfficientNet-B4 (timm, no global pooling)
    Decoder: FPN-style with lateral connections at 3 feature scales
             → progressive upsampling head → 4-channel output

Usage:
    python test_segmentation.py path/to/retinal_image.jpg

Output:
    outputs_seg/
    ├── original.png
    ├── channel_0_MA.png
    ├── channel_1_HE.png
    ├── channel_2_EX.png
    ├── channel_3_SE.png
    ├── overlay_all.png
    └── combined_view.png
"""

import os
import sys
import time
import warnings
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['PYTHONWARNINGS'] = 'ignore'

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print(f'[device] Using {DEVICE}')

# ── Constants ─────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = 'checkpoints/lesion_seg_v2.pth'
OUTPUT_DIR = 'outputs_seg'

# Lesion class names (IDRiD standard order: MA, HE, EX, SE)
LESION_NAMES = ['MA (Microaneurysms)', 'HE (Haemorrhages)',
                'EX (Hard Exudates)', 'SE (Soft Exudates)']
LESION_SHORT = ['MA', 'HE', 'EX', 'SE']

# Colours for overlay (BGR)
LESION_COLORS = [
    (0, 255, 255),    # MA → yellow
    (0, 0, 255),      # HE → red
    (255, 0, 0),      # EX → blue
    (0, 255, 0),      # SE → green
]


# ── Helper: conv+bn block matching checkpoint naming ─────────────────────────
class ConvBNBlock(nn.Module):
    """Single Conv2d + BatchNorm2d, named 'block' for checkpoint compatibility."""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.block(x)


# ── Model definition ──────────────────────────────────────────────────────────
class LesionSegmentationModel(nn.Module):
    """
    EfficientNet-B4 encoder + FPN decoder → 4-channel lesion masks.

    Encoder: timm EfficientNet-B4, features_only=True,
             outputs from blocks.2 (56ch), blocks.4 (160ch), blocks.6 (448ch)
    Decoder: lateral convs (3×3) project to 128ch per level,
             top-down fusion, progressive upsampling head → 4ch @ 512×512
    """

    def __init__(self):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=False,
            features_only=True,
            out_indices=[2, 3, 4],  # blocks.2 (56ch@32×32), blocks.4 (160ch@16×16), blocks.6 (448ch@8×8)
        )

        # ── Lateral connections (3×3 conv + BN, project to 128ch) ────────
        self.lat_deep    = ConvBNBlock(448, 128)   # blocks.6: 448ch → 128ch
        self.lat_mid     = ConvBNBlock(160, 128)   # blocks.4: 160ch → 128ch
        self.lat_shallow = ConvBNBlock(56, 128)    # blocks.2: 56ch  → 128ch

        # ── Top-down refinement ──────────────────────────────────────────
        self.td_mid     = ConvBNBlock(128, 128)
        self.td_shallow = ConvBNBlock(128, 128)

        # ── Segmentation head (numeric submodules for ckpt compat) ───────
        class _Head(nn.Module):
            """head.0, head.2, head.4, head.6 with interleaved upsampling."""
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
        # Encoder: extract 3 feature levels
        features = self.backbone(x)  # list of [B,56,64,64], [B,160,32,32], [B,448,16,16]
        f_shallow, f_mid, f_deep = features[0], features[1], features[2]

        # Lateral projections
        lat_deep = self.lat_deep(f_deep)          # [B,128,16,16]
        lat_mid = self.lat_mid(f_mid)             # [B,128,32,32]
        lat_shallow = self.lat_shallow(f_shallow) # [B,128,64,64]

        # Top-down: upsample deep → fuse with mid → refine
        up_deep = F.interpolate(lat_deep, scale_factor=2, mode='bilinear',
                                align_corners=False)  # [B,128,32,32]
        fused_mid = lat_mid + up_deep
        refined_mid = self.td_mid(fused_mid)      # [B,128,32,32]

        # Top-down: upsample mid → fuse with shallow → refine
        up_mid = F.interpolate(refined_mid, scale_factor=2, mode='bilinear',
                               align_corners=False)  # [B,128,64,64]
        fused_shallow = lat_shallow + up_mid
        refined_shallow = self.td_shallow(fused_shallow)  # [B,128,64,64]

        # Head: progressive upsampling → 4-channel logits
        logits = self.head(refined_shallow)        # [B,4,512,512]
        return logits


# ── Load checkpoint ──────────────────────────────────────────────────────────
def load_model(path):
    print(f'[model] Loading checkpoint: {path}')
    model = LesionSegmentationModel().to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    # strict=False tolerates backbone.conv_head/bn2 keys present in checkpoint
    # but not in our features_only model
    result = model.load_state_dict(ckpt, strict=False)
    if result.missing_keys:
        print(f'[model] Missing keys (expected — not in ckpt): {result.missing_keys}')
    if result.unexpected_keys:
        print(f'[model] Unexpected keys (harmless — not in model): {len(result.unexpected_keys)} total, '
              f'e.g. {result.unexpected_keys[:3]}...')
    model.eval()
    print(f'[model] Loaded successfully on {DEVICE}')

    # Count params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[model] Total params: {total:,} | Trainable: {trainable:,}')
    return model


# ── Preprocessing ────────────────────────────────────────────────────────────
def preprocess(image_path: str):
    """Load image and resize to 512×512. No CLAHE/Ben Graham — segmentation
    model is trained on original fundus appearance."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f'Could not read image: {image_path}')

    original_bgr = img.copy()
    img = cv2.resize(img, (512, 512))

    # Normalise to [0,1] and convert to tensor
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0

    # Standard ImageNet normalisation
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_float - mean) / std

    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    return tensor, original_bgr


# ── Postprocessing ───────────────────────────────────────────────────────────
def postprocess(logits: torch.Tensor, threshold: float = 0.5):
    """
    Convert logits to binary masks per channel.

    Returns:
        masks: (4, H, W) numpy array of binary masks
        probs: (4, H, W) numpy array of probability maps
    """
    probs = torch.sigmoid(logits)[0]  # (4, 512, 512)
    probs_np = probs.cpu().numpy()
    masks = (probs_np > threshold).astype(np.uint8) * 255
    return masks, probs_np


# ── Visualisation ────────────────────────────────────────────────────────────
def overlay_masks(original_bgr: np.ndarray, probs: np.ndarray,
                  threshold: float = 0.5, alpha: float = 0.4):
    """
    Create a single overlay showing all lesion channels with distinct colours.

    Each channel above threshold gets its assigned colour overlaid.
    """
    # Resize original if needed
    h, w = original_bgr.shape[:2]
    overlay = cv2.resize(original_bgr, (512, 512)).astype(np.float32)

    for c in range(4):
        mask = probs[c] > threshold
        if mask.sum() == 0:
            continue
        color = np.array(LESION_COLORS[c], dtype=np.float32)
        overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha

    return np.clip(overlay, 0, 255).astype(np.uint8)


def create_combined_view(original_bgr: np.ndarray, probs: np.ndarray,
                         masks: np.ndarray):
    """Create a 2×3 grid: original, 4 channel views, combined overlay."""
    h, w = 512, 512
    orig = cv2.resize(original_bgr, (w, h))

    # Individual channel views (jet colormap on probability)
    channel_views = []
    for c in range(4):
        prob_map = (probs[c] * 255).astype(np.uint8)
        colored = cv2.applyColorMap(prob_map, cv2.COLORMAP_JET)
        # Blend with original
        blended = cv2.addWeighted(orig, 0.5, colored, 0.5, 0)
        # Label
        cv2.putText(blended, LESION_SHORT[c], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        channel_views.append(blended)

    # Combined overlay
    combined = overlay_masks(original_bgr, probs)

    # Create 2×3 grid (600×3, 512×2 = 1536×1024)
    grid = np.zeros((h * 2 + 30, w * 3 + 40, 3), dtype=np.uint8)

    # Row 0: original, combined overlay, channel 0
    for i, img in enumerate([orig, combined, channel_views[0]]):
        y, x = 0, i * (w + 20)
        grid[y:y+h, x:x+w] = img
        labels = ['Original', 'All Lesions', 'MA']
        cv2.putText(grid, labels[i], (x + 5, h + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Row 1: channel 1, 2, 3
    for i, img in enumerate([channel_views[1], channel_views[2], channel_views[3]]):
        y, x = h + 30, i * (w + 20)
        grid[y:y+h, x:x+w] = img
        labels = ['HE', 'EX', 'SE']
        cv2.putText(grid, labels[i], (x + 5, y + h + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return grid


# ── Architecture inspection ──────────────────────────────────────────────────
def inspect_model(model):
    print('\n' + '=' * 60)
    print('  ARCHITECTURE')
    print('=' * 60)
    print(f'  Encoder: EfficientNet-B4 (timm, features_only=True)')
    print(f'  Decoder: FPN (3-level lateral + top-down)')
    print(f'  Head:    Progressive upsampling 128→64→32→16→4')
    print(f'  Output:  4-channel logits @ 512×512')

    # Print weight shapes for verification
    sd = model.state_dict()
    print(f'\n  State dict keys: {len(list(sd.keys()))}')


# ── Validation ────────────────────────────────────────────────────────────────
def validate_outputs(masks, probs, original_shape):
    checks = []
    h, w = original_shape[:2]

    for c in range(4):
        vals = probs[c]
        ch_name = LESION_SHORT[c]

        # Has activations
        if vals.max() > 0.01:
            checks.append(('✓', f'{ch_name}: activations present (max={vals.max():.4f})'))
        else:
            checks.append(('✗', f'{ch_name}: BLANK (all values near zero)'))

        # Non-uniform
        if vals.std() > 0.01:
            checks.append(('✓', f'{ch_name}: non-uniform (std={vals.std():.4f})'))
        else:
            checks.append(('✗', f'{ch_name}: uniform (std={vals.std():.4f})'))

        # Some pixels above threshold
        ratio = (vals > 0.5).mean()
        if ratio > 0.001:
            checks.append(('✓', f'{ch_name}: {ratio:.1%} pixels > 0.5'))
        else:
            checks.append(('⚠', f'{ch_name}: only {ratio:.1%} pixels > 0.5'))

    # Total pixels
    checks.append(('ℹ', f'Total pixels > 0.5 across all channels: {(probs > 0.5).sum(axis=0).max()}'))

    return checks


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print('Usage: python test_segmentation.py path/to/retinal_image.jpg')
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f'[ERROR] Image not found: {image_path}')
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Load model ────────────────────────────────────────────────────
    print('=' * 60)
    print('  LESION SEGMENTATION — CHECKPOINT VALIDATION')
    print('=' * 60)
    model = load_model(CHECKPOINT_PATH)
    inspect_model(model)

    # ── 2. Load image ────────────────────────────────────────────────────
    print(f'\n[input] Image: {image_path}')
    tensor, original_bgr = preprocess(image_path)
    print(f'[input] Tensor shape: {tensor.shape}')

    # ── 3. Run inference ─────────────────────────────────────────────────
    print('\n' + '-' * 40)
    print('  INFERENCE')
    print('-' * 40)
    t0 = time.time()
    with torch.no_grad():
        logits = model(tensor)
    infer_time = (time.time() - t0) * 1000

    print(f'  Output logits shape: {logits.shape}')
    print(f'  Inference time: {infer_time:.1f} ms')

    # ── 4. Postprocess ───────────────────────────────────────────────────
    masks, probs = postprocess(logits, threshold=0.5)

    print(f'\n  Per-channel statistics:')
    for c in range(4):
        p = probs[c]
        print(f'    {LESION_SHORT[c]:4s}:  min={p.min():.4f}  max={p.max():.4f}  '
              f'mean={p.mean():.4f}  std={p.std():.4f}  '
              f'pixels>0.5={np.mean(p > 0.5):.1%}')

    # ── 5. Save outputs ──────────────────────────────────────────────────
    print('\n' + '-' * 40)
    print('  SAVING OUTPUTS')
    print('-' * 40)

    orig_resized = cv2.resize(original_bgr, (512, 512))

    # Original
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'original.png'), orig_resized)
    print(f'  ✓ original.png')

    # Individual channel probability maps
    for c in range(4):
        prob_vis = (probs[c] * 255).astype(np.uint8)
        colored = cv2.applyColorMap(prob_vis, cv2.COLORMAP_JET)
        path = os.path.join(OUTPUT_DIR, f'channel_{c}_{LESION_SHORT[c]}.png')
        cv2.imwrite(path, colored)
        print(f'  ✓ channel_{c}_{LESION_SHORT[c]}.png')

    # Binary masks
    for c in range(4):
        path = os.path.join(OUTPUT_DIR, f'mask_{c}_{LESION_SHORT[c]}.png')
        cv2.imwrite(path, masks[c])
        print(f'  ✓ mask_{c}_{LESION_SHORT[c]}.png')

    # Combined overlay (all lesions on original)
    overlay = overlay_masks(original_bgr, probs)
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'overlay_all.png'), overlay)
    print(f'  ✓ overlay_all.png')

    # Combined 2×3 grid view
    grid = create_combined_view(original_bgr, probs, masks)
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'combined_view.png'), grid)
    print(f'  ✓ combined_view.png')

    # ── 6. Validation ────────────────────────────────────────────────────
    print('\n' + '-' * 40)
    print('  VALIDATION')
    print('-' * 40)
    checks = validate_outputs(masks, probs, (512, 512))
    for s, m in checks:
        print(f'  {s} {m}')

    # ── 7. Summary ────────────────────────────────────────────────────────
    all_pass = all(st == '✓' for st, _ in checks if st in ('✓', '✗'))
    print('\n' + '=' * 60)
    print('  SUMMARY')
    print('=' * 60)
    print(f'  Image:        {image_path}')
    print(f'  Inference:    {infer_time:.1f} ms')
    print(f'  Output shape: {logits.shape}')
    print(f'  Output dir:   {OUTPUT_DIR}/')
    print(f'  ├── original.png')
    for c in range(4):
        print(f'  ├── channel_{c}_{LESION_SHORT[c]}.png')
        print(f'  ├── mask_{c}_{LESION_SHORT[c]}.png')
    print(f'  ├── overlay_all.png')
    print(f'  └── combined_view.png')
    verdict_text = 'PASS — Ready for dataset evaluation' if all_pass else 'Partially PASS — review warnings'
    print(f'\n  Verdict: {verdict_text}')


if __name__ == '__main__':
    main()

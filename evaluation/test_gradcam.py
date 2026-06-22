#!/usr/bin/env python3
"""
test_gradcam.py — Standalone Grad-CAM validation for DRMultiTaskModel.

Usage:
    python test_gradcam.py path/to/retinal_image.jpg
    python test_gradcam.py data/aptos/test_images/0005cfc8afb6.png

Output:
    outputs/original.png    — original (or preprocessed) image
    outputs/heatmap.png     — raw Grad-CAM heatmap (colormap)
    outputs/overlay.png     — heatmap overlaid on original
"""

import io
import os
import sys
import time
import warnings
import numpy as np
import cv2
import timm
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ──────────────────────────────────────────────────────────────────────────────
# Ignore noisy deprecation / future warnings
# ──────────────────────────────────────────────────────────────────────────────
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['PYTHONWARNINGS'] = 'ignore'

# ──────────────────────────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device('mps')
    import os as _os
    _os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print(f'[device] Using {DEVICE}')

# ──────────────────────────────────────────────────────────────────────────────
# Constants (mirror inference.py exactly)
# ──────────────────────────────────────────────────────────────────────────────
GRADE_LABELS = {
    0: 'No DR',
    1: 'Mild DR',
    2: 'Moderate DR',
    3: 'Severe DR',
    4: 'Proliferative DR',
}
LESION_NAMES = ['MA', 'HE', 'EX', 'SE']
CHECKPOINT_PATH = 'checkpoints/best_model_lambda0.4.pth'
OUTPUT_DIR = 'outputs'

# Albumentations pipeline (normalise → tensor)
_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# ──────────────────────────────────────────────────────────────────────────────
# Model definition (identical to backend/ml/inference.py)
# ──────────────────────────────────────────────────────────────────────────────
class DRMultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=False,
            num_classes=0,
            global_pool='avg',
        )
        feat_dim = 1792
        self.grade_head = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 5),
        )
        self.lesion_head = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.grade_head(features), self.lesion_head(features)


# ──────────────────────────────────────────────────────────────────────────────
# Thin wrapper that only exposes grade_logits for Grad-CAM
# ──────────────────────────────────────────────────────────────────────────────
class GradeOnlyModel(nn.Module):
    """Wraps DRMultiTaskModel so forward() returns only grade_logits."""
    def __init__(self, full_model):
        super().__init__()
        self.backbone = full_model.backbone
        self.grade_head = full_model.grade_head

    def forward(self, x):
        features = self.backbone(x)
        return self.grade_head(features)


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing (identical to inference.py)
# ──────────────────────────────────────────────────────────────────────────────
def preprocess(image_bytes: bytes) -> tuple:
    """
    Apply the exact same preprocessing pipeline used during inference.

    Returns:
        tensor: (1, 3, 512, 512) on DEVICE
        original_bgr: original decoded image (BGR, uint8) for overlay
        preprocessed_rgb: preprocessed image (RGB, uint8) for overlay
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('Could not decode image — is this a valid image file?')

    original_bgr = img.copy()

    # 1. Resize to 512×512
    img = cv2.resize(img, (512, 512))

    # 2. Ben Graham illumination removal
    blurred = cv2.GaussianBlur(img, (0, 0), 10)
    img = cv2.addWeighted(img, 4, blurred, -4, 128)

    # 3. CLAHE on green channel
    b, g, r = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(g)
    img = cv2.merge([b, g, r])

    # 4. BGR → RGB for albumentations
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 5. Normalise + to tensor
    tensor = _transform(image=img_rgb)['image']          # (C, H, W) float32
    tensor = tensor.unsqueeze(0).to(DEVICE)               # (1, C, H, W)

    return tensor, original_bgr, img_rgb


# ──────────────────────────────────────────────────────────────────────────────
# Model loader
# ──────────────────────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str) -> DRMultiTaskModel:
    print(f'[model] Loading checkpoint: {checkpoint_path}')
    model = DRMultiTaskModel().to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        sd = ckpt.state_dict() if hasattr(ckpt, 'state_dict') else {}

    model.load_state_dict(sd)
    model.eval()
    print(f'[model] Loaded successfully on {DEVICE}')
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────
def run_inference(model: DRMultiTaskModel, tensor: torch.Tensor) -> dict:
    with torch.no_grad():
        grade_logits, lesion_logits = model(tensor)

    grade_probs = torch.softmax(grade_logits, dim=1)[0].cpu().numpy()
    lesion_probs = torch.sigmoid(lesion_logits)[0].cpu().numpy()

    grade = int(np.argmax(grade_probs))
    confidence = float(grade_probs[grade])

    return {
        'grade': grade,
        'grade_label': GRADE_LABELS[grade],
        'confidence': confidence,
        'all_grade_probs': grade_probs.tolist(),
        'lesions': {name: float(lesion_probs[i]) for i, name in enumerate(LESION_NAMES)},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Grad-CAM generation
# ──────────────────────────────────────────────────────────────────────────────
def generate_gradcam(model: nn.Module, tensor: torch.Tensor,
                     target_class: int) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for the given target class.

    Args:
        model: DRMultiTaskModel (or wrapper returning grade_logits)
        tensor: preprocessed input (1, 3, 512, 512)
        target_class: DR grade index to backpropagate from (0–4)

    Returns:
        heatmap: (512, 512) float32 in [0, 1]
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    # Wrap model to only return grade_logits
    grade_model = GradeOnlyModel(model).to(DEVICE)
    grade_model.eval()

    # Target layer: last conv before global pooling
    target_layers = [grade_model.backbone.conv_head]
    print(f'[gradcam] Target layer: backbone.conv_head '
          f'({grade_model.backbone.conv_head})')

    # Build CAM
    cam = GradCAM(model=grade_model, target_layers=target_layers)

    # Target: the predicted grade
    targets = [ClassifierOutputTarget(target_class)]

    # Generate heatmap
    # Input tensor shape: (1, 3, 512, 512)
    # Returns: (1, 512, 512)
    grayscale_cam = cam(input_tensor=tensor, targets=targets)
    heatmap = grayscale_cam[0]  # (512, 512)

    return heatmap


# ──────────────────────────────────────────────────────────────────────────────
# Overlay generation
# ──────────────────────────────────────────────────────────────────────────────
def overlay_heatmap(original_bgr: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.5) -> np.ndarray:
    """
    Blend heatmap (colormap) over the original image.

    Args:
        original_bgr: BGR uint8 image (512, 512, 3)
        heatmap: float32 array (512, 512) in [0, 1]
        alpha: blend factor

    Returns:
        overlay: BGR uint8 image
    """
    # Resize heatmap to match the original
    heatmap_resized = cv2.resize(heatmap, (original_bgr.shape[1], original_bgr.shape[0]))

    # Apply colormap: grayscale → BGR (jet)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Blend: overlay = alpha * colored + (1 - alpha) * original
    overlay = cv2.addWeighted(heatmap_colored, alpha, original_bgr, 1.0 - alpha, 0)

    return overlay


# ──────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────────────────────────────────────
def validate_heatmap(heatmap: np.ndarray, original_shape) -> list:
    checks = []

    # Not blank (all zeros)
    if np.max(heatmap) > 0:
        checks.append(('✓', 'Heatmap generated (non-zero values present)'))
    else:
        checks.append(('✗', 'Heatmap is BLANK (all zeros)'))

    # Not uniform
    if np.std(heatmap) > 0.01:
        checks.append(('✓', 'Heatmap contains non-uniform activations'))
    else:
        checks.append(('✗', 'Heatmap is UNIFORM (std < 0.01)'))

    # Dimensions match
    h, w = heatmap.shape[:2]
    expected_h, expected_w = original_shape[:2]
    if h == expected_h and w == expected_w:
        checks.append(('✓', f'Heatmap dimensions match input ({h}×{w})'))
    else:
        checks.append(('✗', f'Heatmap dimensions ({h}×{w}) do not match input ({expected_h}×{expected_w})'))

    # Activation statistics
    mean_act = np.mean(heatmap)
    max_act = np.max(heatmap)
    checks.append(('ℹ', f'Heatmap stats — mean: {mean_act:.4f}, max: {max_act:.4f}, std: {np.std(heatmap):.4f}'))

    # Activation spread: how many pixels have > 50% activation
    high_act_ratio = np.mean(heatmap > 0.5)
    if high_act_ratio > 0.001:
        checks.append(('✓', f'{high_act_ratio:.1%} of pixels have > 50% activation'))
    else:
        checks.append(('⚠', f'Only {high_act_ratio:.1%} of pixels have > 50% activation (very focused or dim)'))

    return checks


# ──────────────────────────────────────────────────────────────────────────────
# Architecture inspection
# ──────────────────────────────────────────────────────────────────────────────
def inspect_architecture(model: DRMultiTaskModel):
    print('\n' + '=' * 60)
    print('  ARCHITECTURE INSPECTION')
    print('=' * 60)

    # Backbone type
    print(f'\n  Backbone: EfficientNet-B4 (timm)')
    print(f'  Feature dim: {model.backbone.num_features}')

    # Conv head
    if hasattr(model.backbone, 'conv_head'):
        ch = model.backbone.conv_head
        print(f'  conv_head: {ch}')
    if hasattr(model.backbone, 'bn2'):
        print(f'  bn2: {model.backbone.bn2}')

    # Blocks
    print(f'  Number of blocks: {len(model.backbone.blocks)}')

    # Grade head
    print(f'  Grade head: {model.grade_head}')

    # Lesion head
    print(f'  Lesion head: {model.lesion_head}')

    # Parameter count
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n  Total params: {total:,}')
    print(f'  Trainable params: {trainable:,}')


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print('Usage: python test_gradcam.py path/to/retinal_image.jpg')
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f'[ERROR] Image not found: {image_path}')
        sys.exit(1)

    # Ensure output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Load model ────────────────────────────────────────────────────
    print('=' * 60)
    print('  GRAD-CAM VALIDATION')
    print('=' * 60)
    model = load_model(CHECKPOINT_PATH)
    inspect_architecture(model)

    # ── 2. Load and preprocess image ─────────────────────────────────────
    print(f'\n[input] Image: {image_path}')
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    tensor, original_bgr, preprocessed_rgb = preprocess(image_bytes)
    print(f'[input] Tensor shape: {tensor.shape}')

    # ── 3. Run inference ─────────────────────────────────────────────────
    print('\n' + '-' * 40)
    print('  INFERENCE')
    print('-' * 40)
    t0 = time.time()
    result = run_inference(model, tensor)
    infer_time = (time.time() - t0) * 1000

    print(f'  Predicted Grade: {result["grade"]} ({result["grade_label"]})')
    print(f'  Confidence: {result["confidence"]:.4f}')
    print(f'  All grade probs: {[f"{p:.3f}" for p in result["all_grade_probs"]]}')
    print(f'  Lesions: {result["lesions"]}')
    print(f'  Inference Time: {infer_time:.1f} ms')

    # ── 4. Generate Grad-CAM ─────────────────────────────────────────────
    print('\n' + '-' * 40)
    print('  GRAD-CAM GENERATION')
    print('-' * 40)
    t1 = time.time()
    heatmap = generate_gradcam(model, tensor, result['grade'])
    gradcam_time = (time.time() - t1) * 1000
    print(f'  Grad-CAM Time: {gradcam_time:.1f} ms')
    print(f'  Heatmap shape: {heatmap.shape}')
    print(f'  Heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]')

    # ── 5. Generate overlay ──────────────────────────────────────────────
    overlay = overlay_heatmap(original_bgr, heatmap, alpha=0.5)

    # ── 6. Save outputs ──────────────────────────────────────────────────
    print('\n' + '-' * 40)
    print('  SAVING OUTPUTS')
    print('-' * 40)

    # Original image (resized to 512x512 as the model sees it)
    orig_resized = cv2.resize(original_bgr, (512, 512))
    orig_path = os.path.join(OUTPUT_DIR, 'original.png')
    cv2.imwrite(orig_path, orig_resized)
    print(f'  ✓ Saved: {orig_path}')

    # Raw heatmap (jet colormap)
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_path = os.path.join(OUTPUT_DIR, 'heatmap.png')
    cv2.imwrite(heatmap_path, heatmap_colored)
    print(f'  ✓ Saved: {heatmap_path}')

    # Overlay
    overlay_path = os.path.join(OUTPUT_DIR, 'overlay.png')
    cv2.imwrite(overlay_path, overlay)
    print(f'  ✓ Saved: {overlay_path}')

    # ── 7. Validation checks ──────────────────────────────────────────────
    print('\n' + '-' * 40)
    print('  VALIDATION')
    print('-' * 40)
    checks = validate_heatmap(heatmap, (512, 512))
    for symbol, msg in checks:
        print(f'  {symbol} {msg}')

    # ── 8. Summary ────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('  SUMMARY')
    print('=' * 60)
    print(f'  Image:          {image_path}')
    print(f'  Predicted:      Grade {result["grade"]} ({result["grade_label"]})')
    print(f'  Confidence:     {result["confidence"]:.2%}')
    print(f'  Inference:      {infer_time:.1f} ms')
    print(f'  Grad-CAM:       {gradcam_time:.1f} ms')
    print(f'  Total:          {infer_time + gradcam_time:.1f} ms')
    print(f'  Grad-CAM OH:    {gradcam_time / infer_time:.1f}× inference')
    print(f'  Outputs:        {OUTPUT_DIR}/')
    print(f'    ├── original.png')
    print(f'    ├── heatmap.png')
    print(f'    └── overlay.png')

    all_pass = all(s == '✓' for s, _ in checks if s in ('✓', '✗'))
    verdict = 'PASS — Ready for integration' if all_pass else 'Partially PASS — review warnings'
    print(f'\n  Verdict: {verdict}')

    # Clean up temp files
    for f in ['_inspect_ckpt.py', '_inspect_model.py']:
        if os.path.exists(f):
            os.remove(f)


if __name__ == '__main__':
    main()

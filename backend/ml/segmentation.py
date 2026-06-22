"""
Lesion segmentation model — EfficientNet-B4 (frozen) + FPN decoder + 8-way TTA.

Produces 4-channel lesion probability maps at 512×512 resolution:

    Channel 0 — MA  (Microaneurysms)   Yellow
    Channel 1 — HE  (Haemorrhages)     Red
    Channel 2 — EX  (Hard Exudates)    Blue
    Channel 3 — SE  (Soft Exudates)    Green

Inference pipeline:
    Fundus image → Resize→Ben Graham→CLAHE→Normalise→Tensor
                → 8-way TTA (4 rotations × 2 flips)
                → Sigmoid → Probability maps + binary masks + overlay
"""
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from backend.ml.inference import DEVICE

# ── Lesion metadata ───────────────────────────────────────────────────────────
LESION_NAMES = ['MA', 'HE', 'EX', 'SE']
# BGR colours for overlay (OpenCV convention)
LESION_COLORS = {
    'MA': (0, 255, 255),   # yellow
    'HE': (0, 0, 255),     # red
    'EX': (255, 0, 0),     # blue
    'SE': (0, 255, 0),     # green
}

OVERLAY_ALPHA = 0.4


# ═════════════════════════════════════════════════════════════════════════════
# Model architecture
# ═════════════════════════════════════════════════════════════════════════════

class ConvBNReLU(nn.Module):
    """Conv2d(3×3) → BatchNorm → ReLU. No bias in Conv (BN follows)."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _FeatureExtractor(nn.Module):
    """EfficientNet-B4 with forward hooks on blocks[2], blocks[4], blocks[6].

    Returns (shallow, mid, deep) feature maps at strides 8, 16, 32.
    Backbone is frozen and stays in eval mode.
    """
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b4', pretrained=False, num_classes=0, global_pool='')

        # Freeze all backbone parameters
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        # Storage for hook outputs
        self._features: dict = {}

        # Register forward hooks on the three target blocks
        self._hook_handles = [
            self.backbone.blocks[2].register_forward_hook(self._make_hook('shallow')),
            self.backbone.blocks[4].register_forward_hook(self._make_hook('mid')),
            self.backbone.blocks[6].register_forward_hook(self._make_hook('deep')),
        ]

    def _make_hook(self, name: str):
        def hook(module, _input, output):
            self._features[name] = output
        return hook

    def forward(self, x: torch.Tensor):
        self._features.clear()
        self.backbone(x)  # hooks populate self._features during forward pass
        return (self._features['shallow'], self._features['mid'], self._features['deep'])

    def remove_hooks(self):
        """Clean up hook references (call if model is discarded)."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()


class LesionSegmentationModel(nn.Module):
    """EfficientNet-B4 (frozen) + FPN decoder → 4-channel lesion masks.

    Feature levels from backbone:
        Shallow — blocks[2]:   56 ch, stride 8  (64×64  at 512 input)
        Mid     — blocks[4]:  160 ch, stride 16 (32×32)
        Deep    — blocks[6]:  448 ch, stride 32 (16×16)

    Decoder:
        Lateral ConvBNReLU(•→128) on each level
        Top-down: deep → mid → shallow via bilinear upsampling + addition
        Segmentation head: ConvBNReLU → upsample×3 → Conv2d(16→4)
    """
    def __init__(self):
        super().__init__()
        self.encoder = _FeatureExtractor()

        # ── Lateral projections (all → 128 channels) ───────────────────────
        self.lat_shallow = ConvBNReLU(56, 128)
        self.lat_mid     = ConvBNReLU(160, 128)
        self.lat_deep    = ConvBNReLU(448, 128)

        # ── Top-down refinement ─────────────────────────────────────────────
        self.td_mid     = ConvBNReLU(128, 128)
        self.td_shallow = ConvBNReLU(128, 128)

        # ── Segmentation head (128ch, 64×64 → 4ch, 512×512) ────────────────
        self.head_1    = ConvBNReLU(128, 64)
        self.head_2    = ConvBNReLU(64, 32)
        self.head_3    = ConvBNReLU(32, 16)
        self.head_out  = nn.Conv2d(16, 4, 1)   # 1×1 pointwise → 4 logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hook-based feature extraction (backbone frozen, no gradients)
        fs, fm, fd = self.encoder(x)

        # Lateral projections
        ld = self.lat_deep(fd)        # 448→128, stride 32 (16×16)
        lm = self.lat_mid(fm)         # 160→128, stride 16 (32×32)
        ls = self.lat_shallow(fs)     #  56→128, stride 8  (64×64)

        # Top-down pathway: deep → mid → shallow
        fd_up = F.interpolate(ld, scale_factor=2, mode='bilinear', align_corners=False)
        rm = self.td_mid(lm + fd_up)  # fused mid: 128ch, stride 16

        fm_up = F.interpolate(rm, scale_factor=2, mode='bilinear', align_corners=False)
        rs = self.td_shallow(ls + fm_up)  # fused shallow: 128ch, stride 8

        # Segmentation head: upsample ×3 (stride 8 → stride 1)
        x = self.head_1(rs)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.head_2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.head_3(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.head_out(x)          # [B, 4, 512, 512]

        return x


# ═════════════════════════════════════════════════════════════════════════════
# Preprocessing pipeline
# ═════════════════════════════════════════════════════════════════════════════

def preprocess(image_bgr: np.ndarray) -> torch.Tensor:
    """Full preprocessing: Resize → Ben Graham → CLAHE → Normalise → Tensor.

    Args:
        image_bgr: BGR uint8 numpy array (any size fundus image).

    Returns:
        Tensor [1, 3, 512, 512] on DEVICE, ready for model inference.
    """
    # 1. Resize to 512×512
    img = cv2.resize(image_bgr, (512, 512), interpolation=cv2.INTER_LINEAR)

    # 2. Ben Graham illumination removal (local contrast enhancement)
    blurred = cv2.GaussianBlur(img, (0, 0), 10)
    img = cv2.addWeighted(img, 4, blurred, -4, 128)

    # 3. CLAHE on green channel only (best lesion contrast)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img[:, :, 1] = clahe.apply(img[:, :, 1])

    # 4. BGR → RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    # 5. ImageNet normalisation
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_rgb / 255.0 - mean) / std

    # 6. HWC → CHW → batch dimension
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    return tensor


# ═════════════════════════════════════════════════════════════════════════════
# 8-way test-time augmentation
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _tta_inference(model: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    """8-way TTA: 4 rotations × 2 flips, averaged.

    Rotations: 0°, 90°, 180°, 270°
    For each rotation: original + horizontal flip

    Each prediction is reverse-transformed before averaging.

    Args:
        model: LesionSegmentationModel in eval mode.
        tensor: Input [1, 3, 512, 512] on DEVICE.

    Returns:
        Averaged probability map [1, 4, 512, 512].
    """
    predictions = []

    for k in range(4):  # k = 0, 1, 2, 3 → rot 0°, 90°, 180°, 270°
        # Rotate input
        x_rot = torch.rot90(tensor, k, dims=(2, 3))

        # ── Variant A: no flip ──
        logits = model(x_rot)
        probs = torch.sigmoid(logits)
        probs = torch.rot90(probs, -k, dims=(2, 3))  # counter-rotate
        predictions.append(probs)

        # ── Variant B: horizontal flip ──
        x_flip = torch.flip(x_rot, dims=(3,))         # flip width
        logits = model(x_flip)
        probs = torch.sigmoid(logits)
        probs = torch.flip(probs, dims=(3,))          # reverse flip
        probs = torch.rot90(probs, -k, dims=(2, 3))   # counter-rotate
        predictions.append(probs)

    return torch.stack(predictions).mean(dim=0)


# ═════════════════════════════════════════════════════════════════════════════
# Global model singleton
# ═════════════════════════════════════════════════════════════════════════════

_model: LesionSegmentationModel | None = None

# Key remapping from old checkpoint format to new model.
# Checkpoint trained with backbone.* and head.N.* keys,
# model uses encoder.backbone.* and head_M.* keys.
_HEAD_REMAP = {
    'head.0.': 'head_1.',
    'head.2.': 'head_2.',
    'head.4.': 'head_3.',
    'head.6.': 'head_out.',
}


def _remap_ckpt_key(key: str) -> str:
    """Remap old checkpoint key to new model key."""
    if key.startswith('backbone.'):
        return 'encoder.' + key
    for old_pfx, new_pfx in _HEAD_REMAP.items():
        if key.startswith(old_pfx):
            return key.replace(old_pfx, new_pfx, 1)
    return key  # lat_*, td_* keys unchanged


def init_model(checkpoint_path: str):
    """Load checkpoint and initialise the segmentation model singleton.

    Handles key remapping from the old checkpoint format
    (``backbone.*``, ``head.N.*``) to the new model's
    (``encoder.backbone.*``, ``head_M.*``).
    """
    global _model
    _model = LesionSegmentationModel().to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    remapped = {_remap_ckpt_key(k): v for k, v in ckpt.items()}
    _model.load_state_dict(remapped, strict=False)
    _model.eval()
    print(f'[seg] Model loaded from {checkpoint_path} on {DEVICE}')


# ═════════════════════════════════════════════════════════════════════════════
# Prediction API
# ═════════════════════════════════════════════════════════════════════════════

def predict(image_bgr: np.ndarray, threshold: float = 0.5) -> dict:
    """Run full segmentation pipeline with 8-way TTA.

    Args:
        image_bgr: BGR uint8 numpy array (fundus image from cv2.imdecode
                   or cv2.imread).
        threshold: Probability threshold for binary masks (default 0.5).

    Returns:
        dict with keys:
            'probabilities': ndarray (4, 512, 512) — sigmoid probabilities
                             for [MA, HE, EX, SE].
            'masks': dict[str, ndarray] — binary masks (uint8 0/1) per lesion.
            'overlay': ndarray (512, 512, 3) BGR uint8 — coloured overlay
                       blended on the resized fundus image.
    """
    if _model is None:
        raise RuntimeError('Segmentation model not loaded')

    # Preprocess
    tensor = preprocess(image_bgr)

    # 8-way TTA inference
    probs_4d = _tta_inference(_model, tensor)   # [1, 4, 512, 512]
    probs = probs_4d[0].cpu().numpy()           # [4, 512, 512]

    # Binary masks
    masks = {}
    for i, name in enumerate(LESION_NAMES):
        masks[name] = (probs[i] > threshold).astype(np.uint8)

    # Coloured overlay
    overlay = _build_overlay(image_bgr, probs, threshold)

    return {
        'probabilities': probs,
        'masks': masks,
        'overlay': overlay,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Overlay / mask utilities (public — used by predict.py)
# ═════════════════════════════════════════════════════════════════════════════

def _build_overlay(image_bgr: np.ndarray, probs: np.ndarray,
                   threshold: float = 0.5) -> np.ndarray:
    """Blend coloured probability regions onto resized fundus image."""
    overlay = cv2.resize(image_bgr, (512, 512)).astype(np.float32)
    for i, name in enumerate(LESION_NAMES):
        mask = probs[i] > threshold
        if not mask.any():
            continue
        colour = np.array(LESION_COLORS[name], dtype=np.float32)
        overlay[mask] = overlay[mask] * (1.0 - OVERLAY_ALPHA) + colour * OVERLAY_ALPHA
    return np.clip(overlay, 0, 255).astype(np.uint8)


def create_overlay(image_bgr: np.ndarray, probs: np.ndarray,
                   threshold: float = 0.5) -> np.ndarray:
    """Public overlay utility (preserved for backward compatibility).

    Args:
        image_bgr: BGR fundus image.
        probs: (4, 512, 512) probability array.
        threshold: Binarisation threshold.

    Returns:
        BGR uint8 overlay image at 512×512.
    """
    return _build_overlay(image_bgr, probs, threshold)


def create_mask_image(probs: np.ndarray, channel: int,
                      threshold: float = 0.5) -> np.ndarray:
    """Create a coloured mask image for a single lesion channel.

    Args:
        probs: (4, 512, 512) probability array.
        channel: Index into LESION_NAMES (0=MA, 1=HE, 2=EX, 3=SE).
        threshold: Binarisation threshold.

    Returns:
        BGR uint8 image at 512×512 with lesion pixels in the channel's
        colour on a black background.
    """
    mask = (probs[channel] > threshold).astype(np.uint8) * 255
    colour = np.array(LESION_COLORS[LESION_NAMES[channel]], dtype=np.uint8)
    result = np.zeros((512, 512, 3), dtype=np.uint8)
    result[mask > 0] = colour
    return result

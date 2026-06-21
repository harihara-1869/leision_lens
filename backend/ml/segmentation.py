"""
Segmentation inference module — loads lesion_seg_v2.pth and generates
4-channel lesion masks (MA, HE, EX, SE) from retinal fundus images.

Inference is validated; benchmark reproduction is pending investigation.
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
# BGR colours for overlay
LESION_COLORS = {
    'MA': (0, 255, 255),   # yellow
    'HE': (0, 0, 255),     # red
    'EX': (255, 0, 0),     # blue
    'SE': (0, 255, 0),     # green
}

# ── Model architecture ────────────────────────────────────────────────────────
class ConvBNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch))
    def forward(self, x):
        return self.block(x)


class LesionSegmentationModel(nn.Module):
    """EfficientNet-B4 encoder + FPN decoder → 4-channel lesion masks."""
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b4', pretrained=False,
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


# ── Global model state ────────────────────────────────────────────────────────
_model = None


def init_model(checkpoint_path: str):
    global _model
    _model = LesionSegmentationModel().to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    _model.load_state_dict(ckpt, strict=False)
    _model.eval()
    print(f'[seg] Model loaded from {checkpoint_path} on {DEVICE}')


def preprocess(image_bgr: np.ndarray) -> torch.Tensor:
    """Resize to 512×512, ImageNet normalise → tensor."""
    img = cv2.resize(image_bgr, (512, 512))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_rgb - mean) / std
    return torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)


def predict(image_bgr: np.ndarray) -> np.ndarray:
    """
    Run segmentation inference.

    Returns:
        probs: (4, 512, 512) numpy array with sigmoid probabilities
               for [MA, HE, EX, SE] respectively.
    """
    if _model is None:
        raise RuntimeError('Segmentation model not loaded')

    tensor = preprocess(image_bgr)
    with torch.no_grad():
        logits = _model(tensor)
    probs = torch.sigmoid(logits)[0].cpu().numpy()
    return probs


def create_overlay(image_bgr: np.ndarray, probs: np.ndarray,
                   threshold: float = 0.5) -> np.ndarray:
    """
    Create combined overlay showing all 4 lesion types with distinct colours.
    Returns BGR uint8 image at 512×512.
    """
    overlay = cv2.resize(image_bgr, (512, 512)).astype(np.float32)
    for i, name in enumerate(LESION_NAMES):
        mask = probs[i] > threshold
        if mask.sum() == 0:
            continue
        color = np.array(LESION_COLORS[name], dtype=np.float32)
        alpha = 0.4
        overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def create_mask_image(probs: np.ndarray, channel: int,
                      threshold: float = 0.5) -> np.ndarray:
    """
    Create a coloured mask image for a single lesion channel.
    Returns BGR uint8 image at 512×512.
    """
    mask = (probs[channel] > threshold).astype(np.uint8) * 255
    color = np.array(LESION_COLORS[LESION_NAMES[channel]], dtype=np.uint8)
    # Create coloured mask on black background
    result = np.zeros((512, 512, 3), dtype=np.uint8)
    result[mask > 0] = color
    return result

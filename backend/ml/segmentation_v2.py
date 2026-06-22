"""
Segmentation v2 — EfficientNet-B4 (frozen) + FPN decoder with hook-based features.
Preprocessing matches the grading model pipeline (Ben Graham + CLAHE).
Inference supports 8-way TTA (4 rotations x 2 flips).
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# ── Constants ─────────────────────────────────────────────────────────────────
LESION_NAMES = ['MA', 'HE', 'EX', 'SE']

LESION_COLORS_BGR = {
    'MA': (0, 255, 255),   # Yellow
    'HE': (0, 0, 255),     # Red
    'EX': (255, 0, 0),     # Blue
    'SE': (0, 255, 0),     # Green
}

IMG_SIZE = 512
THRESHOLD = 0.5
ALPHA = 0.4


# ═════════════════════════════════════════════════════════════════════════════
#  Architecture Components
# ═════════════════════════════════════════════════════════════════════════════

class ConvBNReLU(nn.Module):
    """Conv3x3 + BatchNorm + ReLU (inplace)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class HookFeatureExtractor(nn.Module):
    """
    Extracts feature maps from frozen EfficientNet-B4 via forward hooks
    registered on blocks[2], blocks[4], blocks[6].

    Hook targets and their output dimensions (at 512 px input):
        blocks[2]  ->  56 ch,  64x64  (stride 8)
        blocks[4]  -> 160 ch,  32x32  (stride 16)
        blocks[6]  -> 448 ch,  16x16  (stride 32)
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self._features: dict = {}

        self._hooks = [
            backbone.blocks[2].register_forward_hook(self._make_hook('shallow')),
            backbone.blocks[4].register_forward_hook(self._make_hook('mid')),
            backbone.blocks[6].register_forward_hook(self._make_hook('deep')),
        ]

    def _make_hook(self, name: str):
        def hook(module, inp, out):
            self._features[name] = out
        return hook

    def forward(self, x):
        self._features = {}
        _ = self.backbone(x)
        return (
            self._features['shallow'],   # [B,  56, 64, 64]
            self._features['mid'],       # [B, 160, 32, 32]
            self._features['deep'],      # [B, 448, 16, 16]
        )

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


class FPNDecoder(nn.Module):
    """
    Three-level FPN decoder.

    Lateral projections (all to 128 ch):
        shallow 56 -> 128
        mid    160 -> 128
        deep   448 -> 128

    Top-down pathway (bilinear upsample + add + refine):
        deep -> mid -> shallow
    """
    def __init__(self):
        super().__init__()
        self.lat_shallow = ConvBNReLU(56, 128)
        self.lat_mid     = ConvBNReLU(160, 128)
        self.lat_deep    = ConvBNReLU(448, 128)

        self.td_mid     = ConvBNReLU(128, 128)
        self.td_shallow = ConvBNReLU(128, 128)

    def forward(self, fs, fm, fd):
        # Lateral projections
        ls = self.lat_shallow(fs)   # [B, 128, 64, 64]
        lm = self.lat_mid(fm)       # [B, 128, 32, 32]
        ld = self.lat_deep(fd)      # [B, 128, 16, 16]

        # Deep -> mid
        up_deep = F.interpolate(ld, scale_factor=2, mode='bilinear',
                                align_corners=False)
        rm = self.td_mid(lm + up_deep)   # [B, 128, 32, 32]

        # Mid -> shallow
        up_mid = F.interpolate(rm, scale_factor=2, mode='bilinear',
                               align_corners=False)
        rs = self.td_shallow(ls + up_mid)   # [B, 128, 64, 64]

        return rs


class SegmentationHead(nn.Module):
    """
    Progressive upsampling: 128 -> 64 -> 32 -> 16 -> 4 channels,
    with bilinear x2 between each ConvBNReLU stage.
    Final output: [B, 4, 512, 512].
    """
    def __init__(self):
        super().__init__()
        self.up0 = ConvBNReLU(128, 64)
        self.up1 = ConvBNReLU(64, 32)
        self.up2 = ConvBNReLU(32, 16)
        self.out = nn.Conv2d(16, 4, 1)

    def forward(self, x):
        x = self.up0(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.up1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.up2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        return self.out(x)   # [B, 4, 512, 512]


class LesionSegmentationModel(nn.Module):
    """
    Frozen EfficientNet-B4 + FPN decoder -> 4-channel lesion masks.

    Channel mapping:
        0 = MA (Microaneurysms)
        1 = HE (Haemorrhages)
        2 = EX (Hard Exudates)
        3 = SE (Soft Exudates)
    """
    def __init__(self):
        super().__init__()

        # ImageNet-pretrained backbone, frozen
        backbone = timm.create_model('efficientnet_b4', pretrained=True)

        for param in backbone.parameters():
            param.requires_grad = False
        backbone.eval()

        self.encoder = HookFeatureExtractor(backbone)
        self.decoder = FPNDecoder()
        self.head = SegmentationHead()

    def forward(self, x):
        fs, fm, fd = self.encoder(x)
        features = self.decoder(fs, fm, fd)
        return self.head(features)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═════════════════════════════════════════════════════════════════════════════
#  Checkpoint Loading
# ═════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, device: torch.device = DEVICE
               ) -> LesionSegmentationModel:
    """
    Load segmentation checkpoint onto a frozen-backbone model.

    Backbone weights come from ImageNet pretraining (not the checkpoint).
    Only decoder and head keys are expected from the checkpoint.
    """
    model = LesionSegmentationModel().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        # Load only decoder + head keys; backbone is frozen pretrained
        decoder_state = {k: v for k, v in ckpt.items()
                         if not k.startswith('encoder.backbone.')}
        missing, unexpected = model.load_state_dict(decoder_state, strict=False)

        expected_missing = sum(1 for k in missing
                               if k.startswith('encoder.backbone.'))
        print(f'[seg_v2] Decoder keys loaded: {len(decoder_state)}')
        print(f'[seg_v2] Backbone keys skipped (pretrained): {expected_missing}')
        if unexpected:
            print(f'[seg_v2] Unexpected keys in checkpoint: {len(unexpected)}')

    model.eval()
    total = model.total_params()
    trainable = model.trainable_params()
    print(f'[seg_v2] Total params: {total:,} | Trainable (decoder only): '
          f'{trainable:,}')
    print(f'[seg_v2] Model on {device}')
    return model


# ═════════════════════════════════════════════════════════════════════════════
#  Preprocessing  (Ben Graham + CLAHE, matching grading-model pipeline)
# ═════════════════════════════════════════════════════════════════════════════

def preprocess(image_bgr: np.ndarray) -> torch.Tensor:
    """
    Full preprocessing pipeline used during training:
        1. Resize 512x512
        2. Ben Graham illumination removal
        3. CLAHE on green channel
        4. BGR -> RGB
        5. ImageNet normalise
        6. To tensor [1, 3, 512, 512]
    """
    img = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE))

    # -- 2. Ben Graham --
    blurred = cv2.GaussianBlur(img, (0, 0), 10)
    img = cv2.addWeighted(img, 4, blurred, -4, 128)

    # -- 3. CLAHE on green channel only --
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img[:, :, 1] = clahe.apply(img[:, :, 1])

    # -- 4. BGR -> RGB --
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # -- 5. Normalise --
    img_float = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_float - mean) / std

    # -- 6. To tensor --
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1))
    return tensor.unsqueeze(0).to(DEVICE)   # [1, 3, 512, 512]


# ═════════════════════════════════════════════════════════════════════════════
#  Inference
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_single(model: LesionSegmentationModel, tensor: torch.Tensor
                   ) -> np.ndarray:
    """Single forward pass. Returns sigmoid probabilities [4, 512, 512]."""
    logits = model(tensor)
    return torch.sigmoid(logits)[0].cpu().numpy()


@torch.no_grad()
def predict_tta(model: LesionSegmentationModel, tensor: torch.Tensor
                ) -> np.ndarray:
    """
    8-way test-time augmentation: 4 rotations x 2 flips.

    Each permutation:
      - Apply augmentation
      - Run inference
      - Reverse augmentation on output
      - Average all 8 results

    Returns averaged sigmoid probabilities [4, 512, 512].
    """
    preds = []

    for hflip in [False, True]:
        for rot_k in [0, 1, 2, 3]:
            x = tensor

            # Apply
            if hflip:
                x = torch.flip(x, dims=[3])
            if rot_k:
                x = torch.rot90(x, k=rot_k, dims=[2, 3])

            # Infer
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
#  Postprocessing
# ═════════════════════════════════════════════════════════════════════════════

def threshold_masks(probs: np.ndarray, threshold: float = THRESHOLD) -> dict:
    """
    Threshold probability maps to binary masks.

    Args:
        probs: [4, 512, 512] float32
        threshold: cutoff (default 0.5)

    Returns:
        {lesion_name: (512,512) uint8, values 0 or 255}
    """
    return {name: (probs[i] > threshold).astype(np.uint8) * 255
            for i, name in enumerate(LESION_NAMES)}


def create_overlay(image_bgr: np.ndarray, probs: np.ndarray,
                   threshold: float = THRESHOLD, alpha: float = ALPHA
                   ) -> np.ndarray:
    """
    Blend lesion predictions onto the original image.

    Each lesion above threshold gets its assigned colour overlaid.
    Returns BGR uint8 at 512x512.
    """
    overlay = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE)).astype(np.float32)

    for i, name in enumerate(LESION_NAMES):
        mask = probs[i] > threshold
        if not mask.any():
            continue
        color = np.array(LESION_COLORS_BGR[name], dtype=np.float32)
        overlay[mask] = overlay[mask] * (1.0 - alpha) + color * alpha

    return np.clip(overlay, 0, 255).astype(np.uint8)


# ═════════════════════════════════════════════════════════════════════════════
#  High-Level API
# ═════════════════════════════════════════════════════════════════════════════

def predict(image_path: str, model: LesionSegmentationModel,
            use_tta: bool = True) -> dict:
    """
    Full prediction pipeline on a single image file.

    Args:
        image_path: path to fundus image (JPG / PNG)
        model: loaded LesionSegmentationModel instance
        use_tta: if True, applies 8-way TTA

    Returns:
        {
            'probabilities': ndarray [4, 512, 512] float32,
            'masks':         {MA: ..., HE: ..., EX: ..., SE: ...} uint8,
            'overlay':       ndarray [512, 512, 3] BGR uint8,
        }
    """
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f'Cannot read image: {image_path}')

    tensor = preprocess(image_bgr)

    if use_tta:
        probs = predict_tta(model, tensor)
    else:
        probs = predict_single(model, tensor)

    return {
        'probabilities': probs,
        'masks': threshold_masks(probs),
        'overlay': create_overlay(image_bgr, probs),
    }

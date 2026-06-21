import io
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# ── Labels / descriptions ─────────────────────────────────────────────────────
GRADE_LABELS = {
    0: 'No DR',
    1: 'Mild DR',
    2: 'Moderate DR',
    3: 'Severe DR',
    4: 'Proliferative DR',
}

GRADE_DESCRIPTIONS = {
    0: 'No signs of diabetic retinopathy detected. Routine annual screening recommended.',
    1: 'Mild non-proliferative DR. Microaneurysms present. Annual monitoring advised.',
    2: 'Moderate non-proliferative DR. Referral to ophthalmologist within 6 months.',
    3: 'Severe non-proliferative DR. Urgent referral required.',
    4: 'Proliferative DR. Neovascularisation present. Immediate referral required.',
}

# ── Model architecture (self-contained — matches src/model.py exactly) ────────
class DRMultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=False,
            num_classes=0,
            global_pool='avg',
        )
        feat_dim = 1792  # EfficientNet-B4 feature dimension

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


# ── Albumentations transform (normalise + to tensor) ─────────────────────────
_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# ── Global model state ────────────────────────────────────────────────────────
_model = None


def init_model(checkpoint_path: str):
    global _model
    _model = DRMultiTaskModel().to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        _model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict):
        _model.load_state_dict(checkpoint)
    else:
        _model = checkpoint.to(DEVICE)
    _model.eval()
    print(f'[inference] Model loaded from {checkpoint_path} on {DEVICE}')


def _preprocess(image_bytes: bytes) -> torch.Tensor:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR uint8
    if img is None:
        raise ValueError('Could not decode image')

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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 5. Normalise + convert to tensor
    tensor = _transform(image=img)['image']  # (C, H, W) float32
    return tensor.unsqueeze(0).to(DEVICE)    # (1, C, H, W)


def predict(image_bytes: bytes) -> dict:
    if _model is None:
        raise RuntimeError('Model not loaded — call init_model() first')

    tensor = _preprocess(image_bytes)

    with torch.no_grad():
        grade_logits, lesion_logits = _model(tensor)

    grade_probs  = torch.softmax(grade_logits, dim=1)[0].cpu().tolist()
    lesion_probs = torch.sigmoid(lesion_logits)[0].cpu().tolist()

    grade = int(np.argmax(grade_probs))
    confidence = float(grade_probs[grade])

    return {
        'grade': grade,
        'grade_label': GRADE_LABELS[grade],
        'description': GRADE_DESCRIPTIONS[grade],
        'confidence': confidence,
        'all_grade_probs': grade_probs,
        'lesions': {
            'MA': float(lesion_probs[0]),
            'HE': float(lesion_probs[1]),
            'EX': float(lesion_probs[2]),
            'SE': float(lesion_probs[3]),
        },
    }


# ── Grad-CAM heatmap generation ──────────────────────────────────────────────
class _GradeOnlyModel(nn.Module):
    """Wraps DRMultiTaskModel to return only grade_logits for Grad-CAM."""
    def __init__(self, full_model):
        super().__init__()
        self.backbone = full_model.backbone
        self.grade_head = full_model.grade_head

    def forward(self, x):
        return self.grade_head(self.backbone(x))


def generate_gradcam(image_bytes: bytes, predicted_grade: int,
                     alpha: float = 0.5) -> bytes:
    """
    Generate Grad-CAM heatmap overlay for the predicted grade.

    Returns PNG-encoded bytes of the overlay image (512×512 BGR).
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    if _model is None:
        raise RuntimeError('Model not loaded')

    # Preprocess for Grad-CAM (reuses the same pipeline, but needs gradients)
    nparr = np.frombuffer(image_bytes, np.uint8)
    original_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if original_bgr is None:
        raise ValueError('Could not decode image')
    original_bgr = cv2.resize(original_bgr, (512, 512))

    tensor = _preprocess(image_bytes)

    # Wrap model to expose grade_logits
    grade_model = _GradeOnlyModel(_model).to(DEVICE)
    grade_model.eval()

    # Target: last conv layer before global pooling
    target_layers = [grade_model.backbone.conv_head]

    cam = GradCAM(model=grade_model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(predicted_grade)]
    grayscale_cam = cam(input_tensor=tensor, targets=targets)
    heatmap = grayscale_cam[0]  # (512, 512) in [0, 1]

    # Apply jet colormap
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_colored, alpha, original_bgr, 1.0 - alpha, 0)

    # Encode to PNG bytes
    success, png_bytes = cv2.imencode('.png', overlay)
    if not success:
        raise RuntimeError('Failed to encode Grad-CAM overlay')
    return png_bytes.tobytes()

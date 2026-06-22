#!/usr/bin/env python3
"""
Deep debug: Compare ConvBNReLU vs ConvBNBlock features.
Tests: features_only=True vs hooks; with vs without ReLU.
"""
import os, sys, warnings
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

warnings.filterwarnings('ignore')

# ── Setup ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from backend.ml.segmentation import preprocess, LesionSegmentationModel as NewModel

CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'checkpoints/lesion_seg_v2.pth')

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
print(f'[device] {DEVICE}')

# ── Feature extractors ────────────────────────────────────────────────────────
backbone_old = timm.create_model('efficientnet_b4', pretrained=False,
                                 features_only=True, out_indices=[2, 3, 4])
backbone_old.eval().to(DEVICE)

backbone_new = timm.create_model('efficientnet_b4', pretrained=False, num_classes=0, global_pool='')
backbone_new.eval().to(DEVICE)

feat_store = {}
def make_hook(name):
    def hook(m, i, o):
        feat_store[name] = o.detach()
    return hook
handles = [
    backbone_new.blocks[2].register_forward_hook(make_hook('s')),
    backbone_new.blocks[4].register_forward_hook(make_hook('m')),
    backbone_new.blocks[6].register_forward_hook(make_hook('d')),
]

# Load backbone weights
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    ckpt = ckpt['model_state_dict']

bb_sd = {k.replace('backbone.', ''): v for k, v in ckpt.items() if k.startswith('backbone.')}

r1 = backbone_old.load_state_dict(bb_sd, strict=False)
r2 = backbone_new.load_state_dict(bb_sd, strict=False)
print(f'[old backbone] missing={len(r1.missing_keys)} extra={len(r1.unexpected_keys)}')
print(f'[new backbone] missing={len(r2.missing_keys)} (conv_head/bn2 expected) extra={len(r2.unexpected_keys)}')

# ── Load test image ──────────────────────────────────────────────────────────
test_img_path = os.path.join(PROJECT_ROOT, 'data/idrid_official/A. Segmentation/A. Segmentation/1. Original Images/b. Testing Set', 'IDRiD_55.jpg')
img_bgr = cv2.imread(test_img_path)
tensor = preprocess(img_bgr)

# ── Compare features ─────────────────────────────────────────────────────────
with torch.no_grad():
    # Old: features_only
    f_old = backbone_old(tensor)
    
    # New: hooks
    feat_store.clear()
    backbone_new(tensor)
    f_new = [feat_store['s'], feat_store['m'], feat_store['d']]
    
    # Compare
    print(f'\nFeature comparison:')
    names = ['blocks[2] (shallow)', 'blocks[4] (mid)', 'blocks[6] (deep)']
    for i, name in enumerate(names):
        d = (f_old[i] - f_new[i]).abs().max().item()
        print(f'  {name}: max_diff={d:.10f} old_shape={tuple(f_old[i].shape)} new_shape={tuple(f_new[i].shape)}')

# ── Build decoders with SAME features but different activations ──────────────
class ConvBNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch))
    def forward(self, x): return self.block(x)

class _HeadNoReLU(nn.Module):
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

# Build a model with hook features + ConvBNBlock decoder (no ReLU)
class MixedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = backbone_new
        self.lat_shallow = ConvBNBlock(56, 128)
        self.lat_mid = ConvBNBlock(160, 128)
        self.lat_deep = ConvBNBlock(448, 128)
        self.td_mid = ConvBNBlock(128, 128)
        self.td_shallow = ConvBNBlock(128, 128)
        self.head = _HeadNoReLU()
    def forward(self, x):
        feat_store.clear()
        self.backbone(x)
        fs, fm, fd = feat_store['s'], feat_store['m'], feat_store['d']
        ld = self.lat_deep(fd)
        lm = self.lat_mid(fm)
        ls = self.lat_shallow(fs)
        fd_up = F.interpolate(ld, scale_factor=2, mode='bilinear', align_corners=False)
        rm = self.td_mid(lm + fd_up)
        fm_up = F.interpolate(rm, scale_factor=2, mode='bilinear', align_corners=False)
        rs = self.td_shallow(ls + fm_up)
        return self.head(rs)

mixed = MixedModel().to(DEVICE)
decoder_sd = {k: v for k, v in ckpt.items() if not k.startswith('backbone.')}
mixed.load_state_dict(decoder_sd, strict=False)

# ── Load new model ───────────────────────────────────────────────────────────
new_model = NewModel().to(DEVICE)
HEAD_MAP = {'head.0.': 'head_1.', 'head.2.': 'head_2.', 'head.4.': 'head_3.', 'head.6.': 'head_out.'}
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
new_model.load_state_dict(remapped, strict=False)
new_model.eval()
mixed.eval()

# ── Compare outputs ──────────────────────────────────────────────────────────
with torch.no_grad():
    out_new = new_model(tensor)
    out_mixed = mixed(tensor)

print(f'\nOutput comparison:')
names = ['MA', 'HE', 'EX', 'SE']
for ch in range(4):
    diff = (out_new[0,ch] - out_mixed[0,ch]).abs().max().item()
    p_new = torch.sigmoid(out_new[0,ch]).mean().item()
    p_mix = torch.sigmoid(out_mixed[0,ch]).mean().item()
    print(f'  {names[ch]}: max_logit_diff={diff:.8f}  new_sigmoid_mean={p_new:.6f}  mixed_sigmoid_mean={p_mix:.6f}')

# ── Run full evaluation with both models ──────────────────────────────────────
print(f'\n─── Running full evaluation ───')
IDRID_BASE = os.path.join(PROJECT_ROOT, 'data/idrid_official/A. Segmentation/A. Seg')
IMG_DIR = os.path.join(PROJECT_ROOT, 'data/idrid_official/A. Segmentation/A. Segmentation/1. Original Images/b. Testing Set')
MASK_BASE = os.path.join(PROJECT_ROOT, 'data/idrid_official/A. Segmentation/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set')
LDIRS = {'MA': '1. Microaneurysms', 'HE': '2. Haemorrhages', 'EX': '3. Hard Exudates', 'SE': '4. Soft Exudates'}
LNAMES = ['MA', 'HE', 'EX', 'SE']

def load_gt(p):
    m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if m is None: return None
    if m.ndim > 2: m = m[:,:,0]
    r = cv2.resize((m>0).astype(np.uint8), (512,512), interpolation=cv2.INTER_NEAREST)
    return (r>0).astype(np.uint8)

def iou(p,g):
    i=(p&g).sum(); u=p.sum()+g.sum()-i
    return float(i)/float(u+1e-7)
def dice(p,g):
    i=(p&g).sum()
    return 2.*float(i)/float(p.sum()+g.sum()+1e-7)

dataset = []
for f in sorted(os.listdir(IMG_DIR)):
    if not f.endswith('.jpg'): continue
    stem = f.replace('.jpg','')
    item = {'image': os.path.join(IMG_DIR,f), 'id':stem, 'masks':{}}
    for l in LNAMES:
        mp = os.path.join(MASK_BASE, LDIRS[l], f'{stem}_{l}.tif')
        if os.path.exists(mp): item['masks'][l]=mp
    if item['masks']: dataset.append(item)

print(f'[data] {len(dataset)} images')

for model, tag in [(new_model, 'ConvBNReLU (new model)'), (mixed, 'ConvBNBlock (no ReLU)')]:
    results = {l:{'iou':[],'dice':[]} for l in LNAMES}
    for item in dataset:
        bgr = cv2.imread(item['image'])
        if bgr is None: continue
        t = preprocess(bgr)
        with torch.no_grad():
            logits = model(t)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
        for l in LNAMES:
            if l not in item['masks']: continue
            gt = load_gt(item['masks'][l])
            if gt is None: continue
            ch = LNAMES.index(l)
            pred = (probs[ch] > 0.5).astype(np.uint8)
            results[l]['iou'].append(iou(pred,gt))
            results[l]['dice'].append(dice(pred,gt))
    print(f'\n{tag}:')
    print(f'  {"Lesion":<8} {"N":<5} {"IoU":<10} {"Dice":<10}')
    for l in LNAMES:
        r=results[l]; n=len(r['iou'])
        if n==0: continue
        print(f'  {l:<8} {n:<5} {np.mean(r["iou"]):<10.4f} {np.mean(r["dice"]):<10.4f}')

# Cleanup
for h in handles: h.remove()
print(f'\nDone.')

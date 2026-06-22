"""
Unified prediction endpoint — runs the full pipeline:
    Image → DR Grade → Grad-CAM → Segmentation → Response JSON
"""
import os
import json
import cv2
import numpy as np
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from backend.extensions import db
from backend.models.scan import Scan
from backend.ml.inference import predict, generate_gradcam
from backend.ml.segmentation import predict as seg_predict
from backend.ml.segmentation import create_overlay, create_mask_image, LESION_NAMES

predict_bp = Blueprint('predict', __name__)

# Directories for static outputs (under project instance/)
INSTANCE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            'instance')
HEATMAP_DIR = os.path.join(INSTANCE_DIR, 'heatmaps')
SEG_DIR = os.path.join(INSTANCE_DIR, 'segmentations')
UPLOADS_DIR = os.path.join(INSTANCE_DIR, 'uploads')


def _save_original(image_bytes: bytes, scan_id: int) -> str:
    """Save original uploaded image, return URL path."""
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    filename = f'scan_{scan_id}.png'
    path = os.path.join(UPLOADS_DIR, filename)
    with open(path, 'wb') as f:
        f.write(image_bytes)
    return f'/static/uploads/{filename}'


def _save_heatmap(png_bytes: bytes, scan_id: int) -> str:
    """Save Grad-CAM PNG, return URL path."""
    os.makedirs(HEATMAP_DIR, exist_ok=True)
    filename = f'scan_{scan_id}.png'
    path = os.path.join(HEATMAP_DIR, filename)
    with open(path, 'wb') as f:
        f.write(png_bytes)
    return f'/static/heatmaps/{filename}'


def _save_segmentation(overlay: 'np.ndarray', masks: dict, scan_id: int) -> tuple:
    """
    Save segmentation overlay and individual lesion masks.

    Returns:
        (overlay_url, masks_url_dict)
    """
    os.makedirs(SEG_DIR, exist_ok=True)
    base = os.path.join(SEG_DIR, f'scan_{scan_id}')

    # Combined overlay
    overlay_path = f'{base}_overlay.png'
    cv2_img = overlay if overlay.ndim == 3 else cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(overlay_path, cv2_img)
    overlay_url = f'/static/segmentations/scan_{scan_id}_overlay.png'

    # Individual masks
    mask_urls = {}
    for name, mask_img in masks.items():
        path = f'{base}_{name}.png'
        cv2.imwrite(path, mask_img)
        mask_urls[name] = f'/static/segmentations/scan_{scan_id}_{name}.png'

    return overlay_url, mask_urls


@predict_bp.route('/predict', methods=['POST'])
@jwt_required()
def run_predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    patient_name = (request.form.get('patient_name') or '').strip()
    patient_age_raw = request.form.get('patient_age')
    patient_age = None
    if patient_age_raw:
        try:
            patient_age = int(patient_age_raw)
        except ValueError:
            pass

    image_bytes = image_file.read()

    # ── 1. DR Grade prediction ───────────────────────────────────────────
    try:
        result = predict(image_bytes)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    predicted_grade = result['grade']

    # ── 2. Save Scan to DB (need scan_id for file naming) ────────────────
    user_id = int(get_jwt_identity())
    scan = Scan(
        user_id=user_id,
        patient_name=patient_name or 'Unknown Patient',
        patient_age=patient_age,
        filename=image_file.filename or 'upload',
        grade=result['grade'],
        grade_label=result['grade_label'],
        confidence=result['confidence'],
        prob_ma=result['lesions']['MA'],
        prob_he=result['lesions']['HE'],
        prob_ex=result['lesions']['EX'],
        prob_se=result['lesions']['SE'],
    )
    db.session.add(scan)
    db.session.commit()

    # ── 3. Save original uploaded image ──────────────────────────────────
    original_url = None
    try:
        original_url = _save_original(image_bytes, scan.id)
        scan.original_image_path = original_url
    except Exception as e:
        print(f'[warn] Failed to save original image: {e}')

    # ── 4. Grad-CAM heatmap ──────────────────────────────────────────────
    heatmap_url = None
    try:
        heatmap_png = generate_gradcam(image_bytes, predicted_grade)
        heatmap_url = _save_heatmap(heatmap_png, scan.id)
        scan.heatmap_path = heatmap_url
    except Exception as e:
        print(f'[warn] Grad-CAM failed: {e}')
        # Non-fatal — proceed without heatmap

    # ── 5. Lesion segmentation ───────────────────────────────────────────
    seg_overlay_url = None
    seg_masks = {}
    try:
        # Decode image bytes for segmentation
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            seg_probs = seg_predict(img_bgr)

            # Combined overlay
            overlay_img = create_overlay(img_bgr, seg_probs, threshold=0.5)

            # Individual masks
            mask_imgs = {}
            for i, name in enumerate(LESION_NAMES):
                mask_imgs[name] = create_mask_image(seg_probs, i, threshold=0.5)

            seg_overlay_url, seg_masks = _save_segmentation(overlay_img, mask_imgs, scan.id)
            scan.seg_overlay_url = seg_overlay_url
            scan.seg_masks_json = json.dumps(seg_masks)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f'[warn] Segmentation failed: {e}')
        # Non-fatal — proceed without segmentation

    db.session.commit()

    return jsonify({
        **result,
        'scan_id': scan.id,
        'original_url': original_url,
        'heatmap_url': heatmap_url,
        'seg_overlay_url': seg_overlay_url,
        'seg_masks': seg_masks,
    }), 200

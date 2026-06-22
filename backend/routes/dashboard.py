"""
Dashboard endpoint — computes stats and recent activity from the scans table.
"""
from flask import Blueprint, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from backend.extensions import db
from backend.models.scan import Scan

dashboard_bp = Blueprint('dashboard', __name__)

# Grade → display colour mappings
_DOT_COLOR = {0: 'bg-success', 1: 'bg-info', 2: 'bg-warning', 3: 'bg-error', 4: 'bg-error'}
_TEXT_COLOR = {0: 'text-success', 1: 'text-info', 2: 'text-warning', 3: 'text-error', 4: 'text-error'}


def _scan_to_activity(s: Scan) -> dict:
    """Transform a Scan into the activity-row shape expected by the frontend."""
    name_parts = (s.patient_name or '').split()
    initials = ''.join(w[0].upper() for w in name_parts[:2] if w) or '?'
    return {
        'id': s.id,
        'initials': initials,
        'name': s.patient_name or 'Unknown',
        'patientId': f'PT-{s.id:06d}',
        'scanDate': s.uploaded_at.strftime('%d %b %Y') if s.uploaded_at else '—',
        'drDotColor': _DOT_COLOR.get(s.grade, 'bg-surface-variant'),
        'drGradeColor': _TEXT_COLOR.get(s.grade, 'text-on-surface-variant'),
        'drGradeLabel': s.grade_label,
        'confidence': f'{s.confidence * 100:.1f}%',
    }


@dashboard_bp.route('/dashboard', methods=['GET'])
@jwt_required()
def get_dashboard():
    user_id = int(get_jwt_identity())

    scans = Scan.query.filter_by(user_id=user_id).order_by(Scan.uploaded_at.desc()).all()
    total = len(scans)

    if total == 0:
        return jsonify({
            'stats': [
                {'id': 'total',    'label': 'Total Scans',      'value': '0',    'color': 'text-primary',   'accentColor': 'bg-primary'},
                {'id': 'early',    'label': 'Early Detections',  'value': '0',    'color': 'text-tertiary',  'accentColor': 'bg-tertiary'},
                {'id': 'accuracy', 'label': 'AI Accuracy',       'value': '—',    'color': 'text-secondary', 'accentColor': 'bg-secondary'},
                {'id': 'backlog',  'label': 'Backlog',           'value': '0',    'color': 'text-error',     'accentColor': 'bg-error'},
            ],
            'recent': [],
        }), 200

    # Compute stats
    grade_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    confidences = []
    for s in scans:
        grade_counts[s.grade] = grade_counts.get(s.grade, 0) + 1
        confidences.append(s.confidence)

    avg_conf = (sum(confidences) / len(confidences)) * 100

    return jsonify({
        'stats': [
            {'id': 'total',    'label': 'Total Scans',      'value': str(total),               'color': 'text-primary',   'accentColor': 'bg-primary'},
            {'id': 'early',    'label': 'Early Detections',  'value': str(grade_counts[0] + grade_counts[1]), 'color': 'text-tertiary',  'accentColor': 'bg-tertiary'},
            {'id': 'accuracy', 'label': 'AI Accuracy',       'value': f'{avg_conf:.1f}%',       'color': 'text-secondary', 'accentColor': 'bg-secondary'},
            {'id': 'backlog',  'label': 'Backlog',           'value': str(grade_counts[3] + grade_counts[4]), 'color': 'text-error',     'accentColor': 'bg-error'},
        ],
        'recent': [_scan_to_activity(s) for s in scans[:10]],
    }), 200

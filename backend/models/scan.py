import json
from datetime import datetime, timezone
from backend.extensions import db


class Scan(db.Model):
    __tablename__ = 'scans'

    id                = db.Column(db.Integer, primary_key=True)
    user_id           = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    patient_name      = db.Column(db.String(256), nullable=False, default='')
    patient_age       = db.Column(db.Integer, nullable=True)
    filename          = db.Column(db.String(256), nullable=False)
    uploaded_at       = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    grade             = db.Column(db.Integer, nullable=False)
    grade_label       = db.Column(db.String(64), nullable=False)
    confidence        = db.Column(db.Float, nullable=False)
    prob_ma           = db.Column(db.Float, nullable=False)
    prob_he           = db.Column(db.Float, nullable=False)
    prob_ex           = db.Column(db.Float, nullable=False)
    prob_se           = db.Column(db.Float, nullable=False)
    heatmap_path      = db.Column(db.String(512), nullable=True)
    original_image_path = db.Column(db.String(512), nullable=True)
    seg_overlay_url   = db.Column(db.String(512), nullable=True)
    seg_masks_json    = db.Column(db.Text, nullable=True)

    def to_dict(self):
        d = {
            'id': self.id,
            'user_id': self.user_id,
            'patient_name': self.patient_name,
            'patient_age': self.patient_age,
            'filename': self.filename,
            'uploaded_at': self.uploaded_at.isoformat() if self.uploaded_at else None,
            'grade': self.grade,
            'grade_label': self.grade_label,
            'confidence': self.confidence,
            'prob_ma': self.prob_ma,
            'prob_he': self.prob_he,
            'prob_ex': self.prob_ex,
            'prob_se': self.prob_se,
            'heatmap_path': self.heatmap_path,
            'lesions': {
                'MA': self.prob_ma,
                'HE': self.prob_he,
                'EX': self.prob_ex,
                'SE': self.prob_se,
            },
            'original_image_path': self.original_image_path,
            'seg_overlay_url': self.seg_overlay_url,
            'seg_masks': json.loads(self.seg_masks_json) if self.seg_masks_json else None,
        }
        return d

import os
import re
import sqlalchemy as sa
from flask import Flask, send_from_directory
from backend.config import (SECRET_KEY, JWT_SECRET_KEY, DATABASE_URL,
                            CHECKPOINT_PATH, SEG_CHECKPOINT_PATH, FRONTEND_URL)
from backend.extensions import db, jwt, cors
from backend.routes.auth import auth_bp
from backend.routes.predict import predict_bp
from backend.routes.scans import scans_bp

# Allow any localhost port (covers 5173, 5174, etc.) plus whatever FRONTEND_URL is set to
_ALLOWED_ORIGINS = [re.compile(r'http://localhost:\d+'), FRONTEND_URL]


def create_app():
    app = Flask(__name__, static_folder='static', static_url_path='/static')

    app.config['SECRET_KEY'] = SECRET_KEY
    app.config['JWT_SECRET_KEY'] = JWT_SECRET_KEY
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    app.config['JWT_ALGORITHM'] = 'HS256'
    app.config['JWT_HEADER_NAME'] = 'Authorization'
    app.config['JWT_HEADER_TYPE'] = 'Bearer'

    db.init_app(app)
    jwt.init_app(app)
    cors.init_app(
        app,
        resources={r'/api/*': {
            'origins': _ALLOWED_ORIGINS,
            'methods': ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
            'allow_headers': ['Content-Type', 'Authorization'],
        }},
    )

    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(predict_bp, url_prefix='/api')
    app.register_blueprint(scans_bp, url_prefix='/api')

    def _resolve_path(path_from_config):
        if os.path.isabs(path_from_config):
            return path_from_config
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.normpath(os.path.join(project_root, path_from_config))

    with app.app_context():
        from backend.models import user, scan  # noqa: F401 — registers models
        db.create_all()

        # SQLite performance pragmas
        if 'sqlite' in db.engine.url.drivername:
            from sqlalchemy import event
            @event.listens_for(db.engine, 'connect')
            def _set_pragmas(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute('PRAGMA journal_mode=WAL')
                cursor.execute('PRAGMA synchronous=NORMAL')
                cursor.execute('PRAGMA foreign_keys=ON')
                cursor.close()

        # ── Load DR grading model ────────────────────────────────────────
        grading_ckpt = _resolve_path(CHECKPOINT_PATH)
        try:
            from backend.ml.inference import init_model
            init_model(grading_ckpt)
        except Exception as exc:
            print(f'[WARNING] Grading model failed to load: {exc}')
            print('[WARNING] /api/predict will return 500 until the model is available.')

        # ── Load segmentation model ──────────────────────────────────────
        seg_ckpt = _resolve_path(SEG_CHECKPOINT_PATH)
        try:
            from backend.ml.segmentation import init_model as init_seg
            init_seg(seg_ckpt)
        except Exception as exc:
            print(f'[WARNING] Segmentation model failed to load: {exc}')
            print('[WARNING] Lesion segmentation will be skipped during inference.')

        # Ensure static output directories exist
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
        for sub in ['heatmaps', 'segmentations', 'uploads']:
            os.makedirs(os.path.join(static_dir, sub), exist_ok=True)

    return app


if __name__ == '__main__':
    application = create_app()
    application.run(host='0.0.0.0', port=5000, debug=False)

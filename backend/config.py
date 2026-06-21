import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

SECRET_KEY = os.environ.get('SECRET_KEY', 'fallback-secret-key')
JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'fallback-jwt-secret')
DATABASE_URL = os.environ.get('DATABASE_URL')
CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH', './checkpoints/best_model_lambda0.4.pth')
SEG_CHECKPOINT_PATH = os.environ.get('SEG_CHECKPOINT_PATH', './checkpoints/lesion_seg_v2.pth')
FRONTEND_URL = os.environ.get('FRONTEND_URL', 'http://localhost:5173')

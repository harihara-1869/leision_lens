"""
Migration script: add new columns to the scans table using SQLAlchemy.

Usage — from project root:
    cd leison-lens
    python backend/migrate_add_columns.py

Safe to run multiple times — skips existing columns.
"""
import os
import sys

# Ensure the project root is on sys.path so `backend.*` imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app import create_app
from backend.extensions import db

COLUMNS = [
    ('original_image_path', 'VARCHAR(512)'),
    ('seg_overlay_url', 'VARCHAR(512)'),
    ('seg_masks_json', 'TEXT'),
]


def get_existing_columns(inspector):
    return {col['name'] for col in inspector.get_columns('scans')}


def migrate():
    app = create_app()
    with app.app_context():
        from sqlalchemy import inspect, text

        inspector = inspect(db.engine)
        existing = get_existing_columns(inspector)

        added = 0
        for col_name, col_type in COLUMNS:
            if col_name not in existing:
                with db.engine.connect() as conn:
                    conn.execute(
                        text(f'ALTER TABLE scans ADD COLUMN {col_name} {col_type}')
                    )
                    conn.commit()
                added += 1
                print(f'  + Added column: {col_name} ({col_type})')
            else:
                print(f'  - Already exists: {col_name}')

        if added:
            print(f'\n[OK] Added {added} new column(s). Restart the backend to pick up the changes.')
        else:
            print('\n[OK] No new columns needed — schema is up to date.')


if __name__ == '__main__':
    migrate()

#!/usr/bin/env python
"""Initialize the database with all tables."""

import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from backend.app.database import create_db_and_tables

if __name__ == "__main__":
    try:
        create_db_and_tables()
        print("✅ Database initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        sys.exit(1)

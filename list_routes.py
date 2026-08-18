#!/usr/bin/env python
"""List all API routes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path('.').resolve()))

from backend.app.main import app

print("\n📋 Registered API Routes:\n")
print(f"{'Method':<10} {'Path':<50} {'Name':<30}")
print("-" * 90)

for route in app.routes:
    methods = ", ".join(sorted(route.methods)) if hasattr(route, "methods") else "N/A"
    path = route.path or "N/A"
    name = route.name if hasattr(route, "name") else "N/A"
    print(f"{methods:<10} {path:<50} {name:<30}")

print(f"\n✅ Total routes: {len(app.routes)}")

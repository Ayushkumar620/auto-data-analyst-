import sys
sys.path.insert(0, '/c/users/ayush/Desktop/auto-data-analyst')

from backend.app.main import app

print(f"App title: {app.title}")
print(f"App version: {app.version}")
print(f"Routes: {[r.path for r in app.routes]}")
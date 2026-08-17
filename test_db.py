import sys
sys.path.insert(0, '/c/users/ayush/Desktop/auto-data-analyst')

# Quick test to verify database is working
from backend.app.database import engine, SessionLocal
from backend.app.models import Base

# Create tables
Base.metadata.create_all(bind=engine)
print('Database tables created successfully')

# Test session
db = SessionLocal()
try:
    from backend.app.models import User, Project, Dataset
    print('Models: User, Project, Dataset - OK')
    print(f'User columns: {[c.name for c in User.__table__.columns]}')
    print(f'Project columns: {[c.name for c in Project.__table__.columns]}')
    print(f'Dataset columns: {[c.name for c in Dataset.__table__.columns]}')
finally:
    db.close()
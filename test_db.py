from backend.app.database import SessionLocal, Base, engine


def test_database_tables_can_be_created():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        assert db is not None
    finally:
        db.close()


def test_database_models_are_available():
    from backend.app.models import Dataset, Project, User

    assert User.__tablename__ == 'users'
    assert Project.__tablename__ == 'projects'
    assert Dataset.__tablename__ == 'datasets'

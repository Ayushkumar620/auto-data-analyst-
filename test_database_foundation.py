from backend.app.database import create_db_and_tables
from backend.app.models import Project, Dataset


def test_database_models_can_be_created():
    create_db_and_tables("sqlite:///:memory:")

    project = Project(name="Demo Project")
    dataset = Dataset(name="sales.csv", file_type="csv", rows=12, columns=5, project=project)

    assert project.name == "Demo Project"
    assert dataset.name == "sales.csv"
    assert dataset.file_type == "csv"
    assert dataset.project is project

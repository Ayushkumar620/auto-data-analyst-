"""Comprehensive test suite for Phase 7: Deep Learning & ML Model Registry."""
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from agent.schemas import AgentResult, AgentStatus, ClaimType
from agent.registry_agent import ModelRegistryAgent
from agent.planner import PlannerAgent
from backend.app.main import app
from backend.app.ml.registry import ModelArtifactMetadata, ModelRegistry


@pytest.fixture
def temp_registry_dir():
    """Create a clean isolated temporary directory for model registry testing."""
    temp_dir = tempfile.mkdtemp(prefix="test_registry_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_trained_rf():
    """Create and fit a sample RandomForest model."""
    np.random.seed(42)
    X = np.random.uniform(0, 10, (50, 3))
    y = (X[:, 0] + X[:, 1] > 10).astype(int)
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)
    return rf, ["feat_1", "feat_2", "feat_3"]


# ==============================================================================
# 1. Model Registration & Versioning Tests
# ==============================================================================

def test_model_registration_and_versioning(temp_registry_dir, sample_trained_rf):
    """Verify model registration, artifact persistence, and automatic version incrementing."""
    registry = ModelRegistry(registry_dir=temp_registry_dir)
    rf_model, feature_cols = sample_trained_rf

    # Register Version 1
    meta_v1 = registry.register_model(
        name="Churn_Predictor",
        model_object=rf_model,
        model_family="traditional_ml",
        algorithm="Random Forest",
        problem_type="binary_classification",
        target_column="is_churned",
        feature_columns=feature_cols,
        validation_metrics={"accuracy": 0.92, "f1_score": 0.90},
        primary_metric_name="accuracy",
        primary_metric_value=0.92,
        tags=["production", "v1"],
    )

    assert isinstance(meta_v1, ModelArtifactMetadata)
    assert meta_v1.version == 1
    assert meta_v1.name == "Churn_Predictor"
    assert meta_v1.primary_metric_value == 0.92

    # Register Version 2 of same model name
    meta_v2 = registry.register_model(
        name="Churn_Predictor",
        model_object=rf_model,
        model_family="traditional_ml",
        algorithm="Random Forest",
        problem_type="binary_classification",
        target_column="is_churned",
        feature_columns=feature_cols,
        validation_metrics={"accuracy": 0.96, "f1_score": 0.95},
        primary_metric_name="accuracy",
        primary_metric_value=0.96,
        tags=["production", "v2"],
    )

    assert meta_v2.version == 2
    assert meta_v2.model_id != meta_v1.model_id


def test_get_model_and_metadata(temp_registry_dir, sample_trained_rf):
    """Verify loading model artifact and verifying metadata."""
    registry = ModelRegistry(registry_dir=temp_registry_dir)
    rf_model, feature_cols = sample_trained_rf

    meta = registry.register_model(
        name="Test_RF",
        model_object=rf_model,
        model_family="traditional_ml",
        algorithm="Random Forest",
        problem_type="binary_classification",
        target_column="label",
        feature_columns=feature_cols,
        primary_metric_name="accuracy",
        primary_metric_value=0.90,
    )

    loaded_model, preprocessor, loaded_meta = registry.get_model(meta.model_id)
    assert hasattr(loaded_model, "predict")
    assert loaded_meta.name == "Test_RF"
    assert loaded_meta.feature_columns == feature_cols


def test_list_models_filtering(temp_registry_dir, sample_trained_rf):
    """Verify model listing and family/problem_type filtering."""
    registry = ModelRegistry(registry_dir=temp_registry_dir)
    rf_model, feature_cols = sample_trained_rf

    registry.register_model(
        name="Model_ML",
        model_object=rf_model,
        model_family="traditional_ml",
        algorithm="Random Forest",
        problem_type="binary_classification",
        target_column="label",
        feature_columns=feature_cols,
    )
    registry.register_model(
        name="Model_ANN",
        model_object=rf_model,
        model_family="ann",
        algorithm="MLP",
        problem_type="regression",
        target_column="target",
        feature_columns=feature_cols,
    )

    all_models = registry.list_models()
    assert len(all_models) == 2

    ann_models = registry.list_models(family="ann")
    assert len(ann_models) == 1
    assert ann_models[0]["name"] == "Model_ANN"


# ==============================================================================
# 2. Live Inference & Lifecycle Tests
# ==============================================================================

def test_model_predict_inference(temp_registry_dir, sample_trained_rf):
    """Verify running live batch and single-record predictions."""
    registry = ModelRegistry(registry_dir=temp_registry_dir)
    rf_model, feature_cols = sample_trained_rf

    meta = registry.register_model(
        name="RF_Infer",
        model_object=rf_model,
        model_family="traditional_ml",
        algorithm="Random Forest",
        problem_type="binary_classification",
        target_column="label",
        feature_columns=feature_cols,
    )

    # Test single dictionary record
    single_record = {"feat_1": 5.0, "feat_2": 6.0, "feat_3": 1.0}
    res_single = registry.predict(meta.model_id, single_record)
    assert res_single["sample_count"] == 1
    assert len(res_single["predictions"]) == 1
    assert "probabilities" in res_single

    # Test batch DataFrame
    df_batch = pd.DataFrame({
        "feat_1": [1.0, 8.0, 9.0],
        "feat_2": [2.0, 7.0, 9.0],
        "feat_3": [0.5, 1.5, 2.0],
    })
    res_batch = registry.predict(meta.model_id, df_batch)
    assert res_batch["sample_count"] == 3
    assert len(res_batch["predictions"]) == 3


def test_predict_missing_columns_validation(temp_registry_dir, sample_trained_rf):
    """Verify that predict raises a clear ValueError when required features are missing."""
    registry = ModelRegistry(registry_dir=temp_registry_dir)
    rf_model, feature_cols = sample_trained_rf

    meta = registry.register_model(
        name="RF_Validate",
        model_object=rf_model,
        model_family="traditional_ml",
        algorithm="Random Forest",
        problem_type="binary_classification",
        target_column="label",
        feature_columns=feature_cols,
    )

    bad_data = {"feat_1": 5.0}  # Missing feat_2 and feat_3
    with pytest.raises(ValueError, match="missing required feature columns"):
        registry.predict(meta.model_id, bad_data)


def test_model_lifecycle_status_and_delete(temp_registry_dir, sample_trained_rf):
    """Verify status update and model deletion."""
    registry = ModelRegistry(registry_dir=temp_registry_dir)
    rf_model, feature_cols = sample_trained_rf

    meta = registry.register_model(
        name="RF_Lifecycle",
        model_object=rf_model,
        model_family="traditional_ml",
        algorithm="Random Forest",
        problem_type="binary_classification",
        target_column="label",
        feature_columns=feature_cols,
    )

    # Update status to archived
    assert registry.set_status(meta.model_id, "archived") is True
    assert registry.get_metadata(meta.model_id).status == "archived"

    # Delete model
    assert registry.delete_model(meta.model_id) is True
    assert registry.get_metadata(meta.model_id) is None


# ==============================================================================
# 3. ModelRegistryAgent & Planner Integration Tests
# ==============================================================================

def test_model_registry_agent(sample_trained_rf):
    """Verify ModelRegistryAgent runs actions list, get, and predict."""
    rf_model, feature_cols = sample_trained_rf
    agent = ModelRegistryAgent()

    # Pre-register a model in global registry
    meta = agent.registry.register_model(
        name="Agent_Test_Model",
        model_object=rf_model,
        model_family="traditional_ml",
        algorithm="Random Forest",
        problem_type="binary_classification",
        target_column="label",
        feature_columns=feature_cols,
    )

    # Test list action
    res_list = agent.run({"action": "list"})
    assert res_list.is_success is True
    assert "models" in res_list.output
    assert res_list.output["count"] >= 1

    # Test get action
    res_get = agent.run({"action": "get", "model_id": meta.model_id})
    assert res_get.is_success is True
    assert res_get.output["metadata"]["model_id"] == meta.model_id

    # Test predict action
    test_df = pd.DataFrame({"feat_1": [5.0], "feat_2": [6.0], "feat_3": [1.0]})
    res_pred = agent.run({"action": "predict", "model_id": meta.model_id, "data": test_df})
    assert res_pred.is_success is True
    assert len(res_pred.output["predictions"]) == 1


def test_planner_agent_registry_action():
    """Verify PlannerAgent routing for 'registry' action."""
    planner = PlannerAgent()
    result = planner.run_agent({"action": "registry", "sub_action": "list"})

    assert isinstance(result, AgentResult)
    assert result.is_success is True
    assert result.validation is not None
    assert result.validation.passed is True


# ==============================================================================
# 4. FastAPI REST Endpoints Integration Tests
# ==============================================================================

def test_fastapi_model_endpoints(sample_trained_rf):
    """Test full FastAPI REST endpoints for Model Registry using TestClient."""
    client = TestClient(app)
    rf_model, feature_cols = sample_trained_rf

    # Pre-register a model using global registry
    registry = ModelRegistry()
    meta = registry.register_model(
        name="API_Test_Model",
        model_object=rf_model,
        model_family="traditional_ml",
        algorithm="Random Forest",
        problem_type="binary_classification",
        target_column="label",
        feature_columns=feature_cols,
    )

    # 1. GET /api/v1/models
    r_list = client.get("/api/v1/models")
    assert r_list.status_code == 200
    assert any(m["model_id"] == meta.model_id for m in r_list.json())

    # 2. GET /api/v1/models/{model_id}
    r_get = client.get(f"/api/v1/models/{meta.model_id}")
    assert r_get.status_code == 200
    assert r_get.json()["name"] == "API_Test_Model"

    # 3. POST /api/v1/models/{model_id}/predict
    payload = {"data": {"feat_1": 4.0, "feat_2": 5.0, "feat_3": 1.0}}
    r_pred = client.post(f"/api/v1/models/{meta.model_id}/predict", json=payload)
    assert r_pred.status_code == 200
    assert len(r_pred.json()["predictions"]) == 1

    # 4. PATCH /api/v1/models/{model_id}/status
    r_status = client.patch(f"/api/v1/models/{meta.model_id}/status", json={"status": "staging"})
    assert r_status.status_code == 200
    assert r_status.json()["new_status"] == "staging"

    # 5. DELETE /api/v1/models/{model_id}
    r_del = client.delete(f"/api/v1/models/{meta.model_id}")
    assert r_del.status_code == 200

    # Verify 404 after delete
    r_after = client.get(f"/api/v1/models/{meta.model_id}")
    assert r_after.status_code == 404


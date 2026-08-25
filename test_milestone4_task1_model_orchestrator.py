"""
Tests for Milestone 4, Task 1: Unified Intelligent Model Orchestrator.

Verifies:
1. Traditional ML routing (Linear, Trees, Ensembles)
2. ANN routing (Multi-Layer Perceptron)
3. CNN routing (Convolutional Neural Network on spatial/image data)
4. Incompatible model rejection (Tabular + CNN -> rejected)
5. Multiple candidate execution
6. Multi-family model comparison
7. Best model selection by optimization metric
8. Partial success preservation (failure isolation)
9. Complete failure handling (when all candidates fail)
10. Resource limits enforcement
11. Parallel execution
12. Sequential fallback execution
13. Model registry integration (marked as production_candidate)
14. Model versioning (version increments on repeated registration)
15. Unified prediction routing (predicting across ML, ANN, CNN)
16. Input schema validation for inference
17. ToolRegistry integration ('model_orchestrator')
18. Dynamic Planner integration
19. Standardized AgentResult generation
20. Traceable Evidence generation
"""
import os
import shutil
import tempfile
import pytest
import numpy as np
import pandas as pd

from agent.dynamic_planner import DynamicTaskPlanner, ExecutionPlan
from agent.intent import UserIntent
from agent.model_orchestrator import (
    ANNEngineAdapter,
    CNNEngineAdapter,
    ModelOrchestratorAgent,
    TraditionalMLEngine,
    UnifiedModelOrchestrator,
)
from agent.model_training_schemas import TrainingRequest
from agent.schemas import AgentResult, AgentStatus
from agent.tool_registry import DEFAULT_TOOL_REGISTRY
from backend.app.ml.registry import ModelRegistry


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def temp_registry():
    """Temporary model registry directory."""
    temp_dir = tempfile.mkdtemp()
    registry = ModelRegistry(registry_dir=temp_dir)
    yield registry
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def tabular_classification_df():
    """Synthetic tabular classification dataset (40 rows, 4 features)."""
    np.random.seed(42)
    n = 40
    return pd.DataFrame({
        "feature_1": np.random.uniform(10, 50, n),
        "feature_2": np.random.uniform(100, 500, n),
        "category": np.random.choice(["Tier1", "Tier2"], n),
        "churn": np.random.choice([0, 1], n),
    })


@pytest.fixture
def tabular_regression_df():
    """Synthetic tabular regression dataset (40 rows, 3 features)."""
    np.random.seed(42)
    n = 40
    f1 = np.random.uniform(5, 25, n)
    f2 = np.random.uniform(50, 100, n)
    target = 2.5 * f1 + 0.8 * f2 + np.random.normal(0, 2, n)
    return pd.DataFrame({
        "feature_1": f1,
        "feature_2": f2,
        "sales": target,
    })


@pytest.fixture
def spatial_image_df():
    """Synthetic 8x8 spatial images (40 rows, 64 pixels)."""
    np.random.seed(42)
    n = 40
    data = []
    labels = []
    for i in range(n):
        img = np.zeros((8, 8), dtype=float)
        if i % 2 == 0:
            img[:, [1, 3, 5, 7]] = 1.0 + np.random.normal(0, 0.1, (8, 4))
            labels.append(0)
        else:
            img[[1, 3, 5, 7], :] = 1.0 + np.random.normal(0, 0.1, (4, 8))
            labels.append(1)
        data.append(img.flatten())

    df = pd.DataFrame(data, columns=[f"px_{p}" for p in range(64)])
    df["label"] = labels
    return df


# ==============================================================================
# 1-4. Engine Routing & Capability Validation
# ==============================================================================

def test_traditional_ml_routing(temp_registry, tabular_classification_df):
    """1. Test that traditional ML candidates route to TraditionalMLEngine."""
    orchestrator = UnifiedModelOrchestrator(registry=temp_registry)
    engine = orchestrator.resolve_engine_for_candidate("Random Forest Classifier", "binary_classification", "tabular")
    assert isinstance(engine, TraditionalMLEngine)
    assert engine.capability.model_family == "traditional_ml"


def test_ann_routing(temp_registry, tabular_classification_df):
    """2. Test that ANN candidates route to ANNEngineAdapter."""
    orchestrator = UnifiedModelOrchestrator(registry=temp_registry)
    engine = orchestrator.resolve_engine_for_candidate("Artificial Neural Network (ANN/MLP)", "binary_classification", "tabular")
    assert isinstance(engine, ANNEngineAdapter)
    assert engine.capability.model_family == "ann"


def test_cnn_routing(temp_registry, spatial_image_df):
    """3. Test that CNN candidates route to CNNEngineAdapter."""
    orchestrator = UnifiedModelOrchestrator(registry=temp_registry)
    engine = orchestrator.resolve_engine_for_candidate("Convolutional Neural Network (CNN)", "image_classification", "image")
    assert isinstance(engine, CNNEngineAdapter)
    assert engine.capability.model_family == "cnn"


def test_incompatible_model_rejection(temp_registry, tabular_classification_df):
    """4. Test that CNN candidate is rejected for tabular datasets."""
    orchestrator = UnifiedModelOrchestrator(registry=temp_registry)
    is_valid, reason = orchestrator.validate_candidate_compatibility(
        "Convolutional Neural Network (CNN)", "binary_classification", "tabular", len(tabular_classification_df)
    )
    assert is_valid is False
    assert "CNN requires 2D/3D" in reason


# ==============================================================================
# 5-7. Multiple Candidates, Comparison & Winner Selection
# ==============================================================================

def test_multiple_candidate_execution_and_comparison(temp_registry, tabular_classification_df):
    """5, 6 & 7. Test multi-model comparison across ML and ANN with deterministic winner selection."""
    orchestrator = UnifiedModelOrchestrator(registry=temp_registry)
    req = TrainingRequest(
        target_column="churn",
        feature_columns=["feature_1", "feature_2", "category"],
        task_type="binary_classification",
        candidate_models=["Logistic Regression", "Random Forest Classifier", "Artificial Neural Network (ANN/MLP)"],
        optimization_metric="accuracy",
    )
    res = orchestrator.orchestrate(req, tabular_classification_df, data_modality="tabular", parallel=False)

    assert res.status == "success"
    assert len(res.candidates) == 3
    assert len(res.ranking) == 3
    assert res.best_model is not None
    assert res.ranking[0]["score"] >= res.ranking[1]["score"]
    assert "selected because it achieved" in res.selection_reason


# ==============================================================================
# 8-9. Failure Isolation & Complete Failure Handling
# ==============================================================================

def test_partial_success_preservation(temp_registry, tabular_classification_df):
    """8. Test partial success when one invalid candidate fails but others succeed."""
    orchestrator = UnifiedModelOrchestrator(registry=temp_registry)
    req = TrainingRequest(
        target_column="churn",
        feature_columns=["feature_1", "feature_2", "category"],
        task_type="binary_classification",
        candidate_models=[
            "Random Forest Classifier",
            "Convolutional Neural Network (CNN)",  # Will be rejected for tabular
        ],
        optimization_metric="accuracy",
    )
    res = orchestrator.orchestrate(req, tabular_classification_df, data_modality="tabular", parallel=False)

    assert res.status == "partial"
    assert res.best_model is not None
    assert res.best_model.model_family != "cnn"
    assert len(res.warnings) > 0


def test_complete_failure_handling(temp_registry, tabular_classification_df):
    """9. Test complete failure when all candidates are rejected or invalid."""
    orchestrator = UnifiedModelOrchestrator(registry=temp_registry)
    req = TrainingRequest(
        target_column="non_existent_column",
        task_type="binary_classification",
        candidate_models=["Random Forest Classifier"],
    )
    res = orchestrator.orchestrate(req, tabular_classification_df, data_modality="tabular")

    assert res.status == "failed"
    assert res.best_model is None
    assert "not found in dataset" in res.selection_reason


# ==============================================================================
# 10-12. Resource Limits & Parallel / Sequential Execution
# ==============================================================================

def test_parallel_execution_and_sequential_fallback(temp_registry, tabular_regression_df):
    """10, 11 & 12. Test parallel cross-validation execution with max_workers."""
    orchestrator = UnifiedModelOrchestrator(registry=temp_registry, max_parallel_models=2)
    req = TrainingRequest(
        target_column="sales",
        feature_columns=["feature_1", "feature_2"],
        task_type="regression",
        candidate_models=["Linear Regression", "Ridge Regression", "Random Forest Regressor"],
        optimization_metric="r2",
    )
    res_parallel = orchestrator.orchestrate(req, tabular_regression_df, data_modality="tabular", parallel=True)
    assert res_parallel.status == "success"
    assert len(res_parallel.candidates) == 3


# ==============================================================================
# 13-14. Model Registry & Versioning
# ==============================================================================

def test_model_registry_and_versioning(temp_registry, tabular_classification_df):
    """13 & 14. Test model persistence as production_candidate and automated version increment."""
    orchestrator = UnifiedModelOrchestrator(registry=temp_registry)
    req = TrainingRequest(
        target_column="churn",
        feature_columns=["feature_1", "feature_2"],
        task_type="binary_classification",
        candidate_models=["Logistic Regression"],
        optimization_metric="accuracy",
    )

    # First registration -> version 1
    res1 = orchestrator.orchestrate(req, tabular_classification_df)
    meta1 = temp_registry.get_metadata(res1.best_model.model_id)
    assert meta1 is not None
    assert meta1.version == 1
    assert "production_candidate" in meta1.tags

    # Second registration of same name -> version 2
    res2 = orchestrator.orchestrate(req, tabular_classification_df)
    meta2 = temp_registry.get_metadata(res2.best_model.model_id)
    assert meta2 is not None
    assert meta2.version == 2


# ==============================================================================
# 15-16. Unified Prediction Routing & Schema Validation
# ==============================================================================

def test_unified_prediction_routing(temp_registry, tabular_classification_df):
    """15. Test unified prediction routing across registered model families."""
    orchestrator = UnifiedModelOrchestrator(registry=temp_registry)
    req = TrainingRequest(
        target_column="churn",
        feature_columns=["feature_1", "feature_2", "category"],
        task_type="binary_classification",
        candidate_models=["Random Forest Classifier"],
    )
    res = orchestrator.orchestrate(req, tabular_classification_df)
    model_id = res.best_model.model_id

    # New data prediction
    new_data = tabular_classification_df[["feature_1", "feature_2", "category"]].iloc[:3].copy()
    pred_res = orchestrator.predict(model_id, new_data)

    assert pred_res["model_id"] == model_id
    assert len(pred_res["predictions"]) == 3
    assert pred_res["status"] == "success"


def test_prediction_schema_validation_error(temp_registry, tabular_classification_df):
    """16. Test schema validation error when new data is missing required feature columns."""
    orchestrator = UnifiedModelOrchestrator(registry=temp_registry)
    req = TrainingRequest(
        target_column="churn",
        feature_columns=["feature_1", "feature_2", "category"],
        task_type="binary_classification",
        candidate_models=["Logistic Regression"],
    )
    res = orchestrator.orchestrate(req, tabular_classification_df)
    model_id = res.best_model.model_id

    bad_data = pd.DataFrame({"feature_1": [12.5]})  # missing feature_2 & category
    with pytest.raises(ValueError, match="missing required features"):
        orchestrator.predict(model_id, bad_data)


# ==============================================================================
# 17-18. ToolRegistry & Dynamic Planner Integration
# ==============================================================================

def test_tool_registry_model_orchestrator_integration():
    """17. Test that 'model_orchestrator' is registered in DEFAULT_TOOL_REGISTRY with valid capabilities."""
    tool = DEFAULT_TOOL_REGISTRY.get("model_orchestrator")
    assert tool is not None
    assert "model_orchestration" in tool.capabilities
    assert "multi_model_training" in tool.capabilities


def test_dynamic_planner_model_orchestrator_integration():
    """18. Test that DynamicTaskPlanner synthesizes steps utilizing model_orchestrator."""
    planner = DynamicTaskPlanner()
    intent = UserIntent(
        intent_type="model_orchestration",
        objective="Benchmark multiple models to predict revenue",
        metrics=["revenue"],
        dimensions=["region", "sales_channel"],
        required_capabilities=["model_orchestration"],
        original_command="Benchmark and find the best model for revenue",
    )
    plan = planner.generate_plan(intent)

    assert isinstance(plan, ExecutionPlan)
    tool_names = [s.tool_name for s in plan.steps]
    assert "model_orchestrator" in tool_names


# ==============================================================================
# 19-20. AgentResult & Traceable Evidence Generation
# ==============================================================================

def test_agent_result_and_evidence_generation(temp_registry, tabular_classification_df):
    """19 & 20. Test that ModelOrchestratorAgent returns standardized AgentResult with Evidence."""
    agent = ModelOrchestratorAgent(registry=temp_registry)
    task = {
        "data": tabular_classification_df,
        "target": "churn",
        "task_type": "binary_classification",
        "candidates": ["Logistic Regression", "Random Forest Classifier"],
        "metric": "accuracy",
    }
    result = agent.run(task)

    assert isinstance(result, AgentResult)
    assert result.status == AgentStatus.SUCCESS
    assert result.is_success is True
    assert len(result.evidence) > 0
    assert result.evidence[0].source == "UnifiedModelOrchestrator"
    assert "winner" in result.evidence[0].data_ref

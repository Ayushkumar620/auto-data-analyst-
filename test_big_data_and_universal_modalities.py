"""Comprehensive Test Suite for Enterprise Big Data Scalability & Universal Modalities.

Tests:
1. MemoryOptimizer (downcasting & memory footprint reduction)
2. StreamingAggregator (chunked streaming running statistics)
3. StratifiedRepresentativeSampler (Cochran's formula & class balance)
4. UniversalDatasetLoader (Parquet, Feather, NDJSON, SQLite, TSV, NumPy)
5. TextModalityEngine (NLP Sentiment, TF-IDF n-grams & keywords)
6. RelationalModalityEngine (Multi-table FK discovery & auto-joins)
7. End-to-End Big Dataset Command Execution
"""
import io
import json
import os
import sqlite3
import tempfile
import numpy as np
import pandas as pd
import pytest

from backend.app.core.big_data_engine import (
    MemoryOptimizer,
    MemoryProfile,
    StreamingAggregator,
    StratifiedRepresentativeSampler,
)
from backend.app.core.universal_loader import (
    UniversalDatasetLoader,
    UniversalLoadError,
)
from backend.app.core.modality_engines import (
    TextModalityEngine,
    TextAnalysisReport,
    RelationalModalityEngine,
    HierarchicalJSONEngine,
)
from agent.command_orchestrator import AutonomousCommandOrchestrator


# ==============================================================================
# 1. Big Data Memory Optimization Tests
# ==============================================================================

def test_memory_optimizer_downcasting():
    """Verify aggressive memory downcasting on integers, floats, and low-cardinality strings."""
    n = 10000
    df = pd.DataFrame({
        "tiny_int": np.random.randint(-100, 100, n, dtype=np.int64),
        "med_int": np.random.randint(-20000, 20000, n, dtype=np.int64),
        "big_float": np.random.uniform(10.0, 5000.0, n).astype(np.float64),
        "low_card_cat": np.random.choice(["Tier1", "Tier2", "Tier3"], n),
        "high_card_str": [f"ID_{i}" for i in range(n)],
    })

    orig_mem = df.memory_usage(deep=True).sum()
    opt_df, profile = MemoryOptimizer.optimize(df)

    assert isinstance(profile, MemoryProfile)
    assert profile.reduction_percentage > 30.0  # At least 30% RAM savings
    assert opt_df["tiny_int"].dtype == np.int8
    assert opt_df["med_int"].dtype == np.int16
    assert opt_df["big_float"].dtype == np.float32
    assert str(opt_df["low_card_cat"].dtype) == "category"
    assert opt_df.shape == df.shape


# ==============================================================================
# 2. Cochran Stratified Sampling Tests
# ==============================================================================

def test_stratified_representative_sampler():
    """Verify Cochran formula sample size calculation and class balance retention."""
    n = 100000
    df = pd.DataFrame({
        "feature_1": np.random.randn(n),
        "target_class": np.random.choice(["ClassA", "ClassB", "ClassC"], n, p=[0.70, 0.20, 0.10]),
    })

    sampled_df, info = StratifiedRepresentativeSampler.sample_dataframe(
        df, target_column="target_class", max_rows=15000
    )

    assert info["is_sampled"] is True
    assert len(sampled_df) <= 15000
    assert len(sampled_df) >= 10000

    # Verify class balance preserved within 2% margin
    orig_props = df["target_class"].value_counts(normalize=True)
    sample_props = sampled_df["target_class"].value_counts(normalize=True)

    for c in ["ClassA", "ClassB", "ClassC"]:
        assert abs(orig_props[c] - sample_props[c]) < 0.02


# ==============================================================================
# 3. Streaming Aggregator Tests
# ==============================================================================

def test_streaming_aggregator_large_csv():
    """Verify exact running aggregates over chunked CSV streams."""
    n = 5000
    df = pd.DataFrame({
        "sales": np.random.uniform(10, 100, n),
        "units": np.random.randint(1, 10, n),
    })

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name

    try:
        summary = StreamingAggregator.stream_csv_summary(temp_path, chunk_size=1000)
        assert summary["total_rows"] == n
        assert abs(summary["numeric_means"]["sales"] - df["sales"].mean()) < 0.01
        assert abs(summary["numeric_mins"]["sales"] - df["sales"].min()) < 0.01
        assert abs(summary["numeric_maxs"]["sales"] - df["sales"].max()) < 0.01
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ==============================================================================
# 4. Universal Dataset Loader Tests (Parquet, Feather, JSON, SQLite, NumPy)
# ==============================================================================

def test_universal_loader_parquet():
    """Verify Parquet columnar loading."""
    df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": [4.0, 5.0, 6.0]})
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        df.to_parquet(f.name)
        temp_path = f.name

    try:
        loaded, profile = UniversalDatasetLoader.load(temp_path)
        assert isinstance(loaded, pd.DataFrame)
        assert loaded.shape == (3, 2)
        assert list(loaded.columns) == ["col_a", "col_b"]
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_universal_loader_feather():
    """Verify Arrow Feather loading."""
    df = pd.DataFrame({"x": [10, 20], "y": ["A", "B"]})
    with tempfile.NamedTemporaryFile(suffix=".feather", delete=False) as f:
        df.to_feather(f.name)
        temp_path = f.name

    try:
        loaded, _ = UniversalDatasetLoader.load(temp_path)
        assert isinstance(loaded, pd.DataFrame)
        assert loaded.shape == (2, 2)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_universal_loader_hierarchical_json():
    """Verify nested JSON normalization."""
    nested_json = [
        {"id": 1, "user": {"name": "Alice", "country": "US"}, "orders": 3},
        {"id": 2, "user": {"name": "Bob", "country": "India"}, "orders": 5},
    ]
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump(nested_json, f)
        temp_path = f.name

    try:
        loaded, _ = UniversalDatasetLoader.load(temp_path)
        assert isinstance(loaded, pd.DataFrame)
        assert "user.name" in loaded.columns or "name" in loaded.columns
        assert len(loaded) == 2
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_universal_loader_sqlite():
    """Verify multi-table SQLite ingestion."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_path = f.name

    conn = sqlite3.connect(temp_path)
    pd.DataFrame({"cust_id": [1, 2], "name": ["Alice", "Bob"]}).to_sql("customers", conn, index=False)
    pd.DataFrame({"order_id": [101, 102], "cust_id": [1, 2], "amount": [50.0, 75.0]}).to_sql("orders", conn, index=False)
    conn.close()

    try:
        tables, _ = UniversalDatasetLoader.load(temp_path)
        assert isinstance(tables, dict)
        assert "customers" in tables
        assert "orders" in tables

        # Verify auto-join
        joined = RelationalModalityEngine.auto_join_tables(tables)
        assert len(joined) == 2
        assert "amount" in joined.columns
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_universal_loader_numpy_matrix():
    """Verify NumPy .npy matrix ingestion."""
    mat = np.random.randn(20, 5)
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f.name, mat)
        temp_path = f.name

    try:
        loaded, _ = UniversalDatasetLoader.load(temp_path)
        assert isinstance(loaded, pd.DataFrame)
        assert loaded.shape == (20, 5)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ==============================================================================
# 5. NLP Text Modality Engine Tests
# ==============================================================================

def test_text_modality_engine_sentiment_and_keywords():
    """Verify NLP sentiment profiling, keyword extraction, and vocabulary diversity."""
    reviews = pd.Series([
        "This product is amazing, fast, helpful, and reliable with excellent customer service!",
        "Terrible experience, very bad and slow support with broken items.",
        "Great quality and wonderful features. Loved the boost in our team profit.",
        "Neutral performance, average quality overall without complaints.",
    ])

    assert TextModalityEngine.is_text_column(reviews) is True

    report = TextModalityEngine.analyze_text_column(reviews, column_name="customer_feedback")
    assert isinstance(report, TextAnalysisReport)
    assert report.total_documents == 4
    assert report.sentiment_distribution["positive"] >= 0.50
    assert report.sentiment_distribution["negative"] >= 0.25
    assert len(report.top_keywords) > 0


# ==============================================================================
# 6. End-to-End Command Orchestration on Big Data & Text Modalities
# ==============================================================================

def test_command_orchestrator_on_big_dataset():
    """Verify AutonomousCommandOrchestrator runs smoothly on 60,000-row dataset."""
    n = 60000
    df = pd.DataFrame({
        "revenue": np.random.uniform(100, 1000, n),
        "churn": np.random.choice([0, 1], n, p=[0.8, 0.2]),
        "category": np.random.choice(["A", "B", "C"], n),
    })

    orchestrator = AutonomousCommandOrchestrator()
    res = orchestrator.execute_command("Build a machine learning model to predict churn", df)

    assert res.user_intent == "prediction"
    assert any("sampling" in op for op in res.required_operations)
    assert res.validation_summary["status"] in ("PASSED", "PASSED_WITH_WARNINGS")


def test_command_orchestrator_on_text_dataset():
    """Verify AutonomousCommandOrchestrator extracts NLP sentiment when text is queried."""
    df = pd.DataFrame({
        "customer_id": [1, 2, 3, 4],
        "feedback": [
            "Excellent and superb quality, absolutely loved the fast shipping!",
            "Terrible customer service and broken product, very unhappy.",
            "Great tool, helpful and very reliable boost for our analytics.",
            "Average experience with some minor issues.",
        ],
    })

    orchestrator = AutonomousCommandOrchestrator()
    res = orchestrator.execute_command("Analyze customer feedback sentiment and keywords", df)

    assert "sentiment" in res.final_explanation.lower()
    assert "positive" in res.final_explanation.lower()
    assert len(res.evidence) > 0

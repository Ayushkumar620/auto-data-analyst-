"""Test Quick Charts Multi-Type Generation and Automated Summaries.

Tests:
1. DataVisualizer chart generation across all 8 chart types (Bar, Line, Scatter, Box, Pie, Histogram, Heatmap, Area)
2. Automated Evidence-Backed Chart Summary generation
3. CommandParser routing of chart commands and type parameters
"""
import numpy as np
import pandas as pd
import pytest

from agent.visualizer import DataVisualizer
from agent.command_parser import CommandParser


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "Country": np.random.choice(["USA", "India", "Germany", "Japan", "Brazil"], n),
        "Revenue": np.random.uniform(500, 5000, n),
        "Spend": np.random.uniform(100, 2000, n),
        "Score": np.random.normal(75, 10, n),
    })


def test_visualizer_all_chart_types_with_summaries(sample_df):
    """Verify DataVisualizer generates base64 images and summaries for all supported chart types."""
    vis = DataVisualizer(sample_df)

    for c_type in ["bar", "line", "scatter", "box", "pie", "histogram", "heatmap", "area"]:
        res = vis.chart(chart_type=c_type)
        assert len(res) >= 1
        item = res[0]
        assert item["chart_type"] in (c_type, "histogram")
        assert len(item["image"]) > 100  # Valid base64 image data
        assert isinstance(item["summary"], str)
        assert len(item["summary"]) > 10  # Meaningful narrative summary
        assert "available_types" in item
        assert "bar" in item["available_types"]


def test_command_parser_chart_types_routing(sample_df):
    """Verify CommandParser correctly extracts type= parameter and returns available types."""
    parser = CommandParser(sample_df)

    # 1. Default chart command
    res1 = parser.parse("chart")
    assert res1["type"] == "chart"
    assert len(res1["charts"]) >= 1
    assert "available_types" in res1
    assert res1["charts"][0]["summary"] != ""

    # 2. Specific chart type parameter: chart type=bar
    res2 = parser.parse("chart type=bar")
    assert res2["type"] == "chart"
    assert res2["charts"][0]["chart_type"] == "bar"
    assert "Bar Chart" in res2["charts"][0]["summary"]

    # 3. Direct keyword: pie
    res3 = parser.parse("pie")
    assert res3["type"] == "chart"
    assert res3["charts"][0]["chart_type"] == "pie"
    assert "Pie Chart" in res3["charts"][0]["summary"]

    # 4. Direct keyword: scatter
    res4 = parser.parse("scatter")
    assert res4["type"] == "chart"
    assert res4["charts"][0]["chart_type"] == "scatter"
    assert "Scatter Plot" in res4["charts"][0]["summary"]


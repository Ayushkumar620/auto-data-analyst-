import pandas as pd
import pytest

from backend.app.visualization import VisualizationEngine


@pytest.fixture
def dataframe():
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        "region": ["North", "South", "North"],
        "sales": [100, 200, 150],
        "profit": [20, 40, 30],
        "empty": [None, None, None],
    })


def test_recommendations_follow_column_types(dataframe):
    recommendations = VisualizationEngine().recommend(dataframe)
    by_type = {item["chart_type"]: item for item in recommendations}
    assert by_type["line"]["x"] == "date"
    assert by_type["bar"]["x"] == "region"
    assert by_type["scatter"]["x"] == "sales"
    assert by_type["histogram"]["x"] == "sales"
    assert by_type["heatmap"]["columns"] == ["sales", "profit"]


@pytest.mark.parametrize("chart_type,x,y", [
    ("bar", "region", "sales"), ("line", "date", "sales"),
    ("scatter", "sales", "profit"), ("histogram", "sales", None),
    ("box", "sales", None), ("heatmap", None, None),
])
def test_supported_charts_return_json_ready_response(dataframe, chart_type, x, y):
    chart = VisualizationEngine().generate(dataframe, {"chart_type": chart_type, "x": x, "y": y})
    assert chart["type"] == chart_type
    assert isinstance(chart["figure"], dict)


def test_missing_and_empty_columns_do_not_crash(dataframe):
    chart = VisualizationEngine().generate(dataframe, {"chart_type": "histogram", "x": "sales"})
    assert chart["figure"]["data"]
    assert VisualizationEngine().recommend(dataframe)


def test_invalid_column_has_useful_error(dataframe):
    with pytest.raises(ValueError, match="does not exist"):
        VisualizationEngine().generate(dataframe, {"chart_type": "bar", "x": "unknown", "y": "sales"})

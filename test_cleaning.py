import pandas as pd

from backend.app.cleaning.cleaner import DataCleaner


def test_cleaning_pipeline_reports_actions():
    dataframe = pd.DataFrame(
        {
            "Customer Name": ["Alice", "Alice", "Bob", None],
            "Customer Age": [25, 25, None, 40],
            "Sales": [100, 100, 200, 1000],
            "Order Date": ["2025/01/01", "2025/01/01", "2025/02/01", "bad-date"],
        }
    )

    cleaner = DataCleaner(dataframe)
    result = cleaner.clean()

    assert result["status"] == "success"
    assert result["rows_removed"] >= 0
    assert result["quality_after"] >= result["quality_before"]
    assert result["cleaning_report"]

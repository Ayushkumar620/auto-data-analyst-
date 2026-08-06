from __future__ import annotations

from typing import List, Tuple

import pandas as pd


class MissingValueCleaner:
    def clean(self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        cleaned = dataframe.copy()
        filled_count = 0
        for column in cleaned.columns:
            if cleaned[column].isna().any():
                dtype = cleaned[column].dtype
                if pd.api.types.is_numeric_dtype(dtype):
                    fill_value = cleaned[column].median()
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    fill_value = None
                else:
                    fill_value = cleaned[column].mode(dropna=True).iloc[0] if not cleaned[column].mode(dropna=True).empty else ""
                if fill_value is not None:
                    cleaned[column] = cleaned[column].fillna(fill_value)
                    filled_count += int(cleaned[column].isna().sum())
                else:
                    filled_count += 0
        messages = [f"Filled missing values in {filled_count} cells"] if filled_count else ["No missing values required filling"]
        return cleaned, messages

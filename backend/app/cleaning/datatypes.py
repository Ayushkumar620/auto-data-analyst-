from __future__ import annotations

from typing import List, Tuple

import pandas as pd


class DataTypeCleaner:
    def clean(self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        cleaned = dataframe.copy()
        conversions = 0
        for column in cleaned.columns:
            if cleaned[column].dtype == object:
                try:
                    converted = pd.to_numeric(cleaned[column], errors="raise")
                    cleaned[column] = converted
                    conversions += 1
                except Exception:
                    try:
                        converted = pd.to_datetime(cleaned[column], errors="coerce")
                        if converted.notna().sum() > 0:
                            cleaned[column] = converted
                            conversions += 1
                    except Exception:
                        continue
        if conversions:
            return cleaned, [f"Converted {conversions} columns to more appropriate dtypes"]
        return cleaned, ["No datatype conversions were needed"]

from __future__ import annotations

from typing import List, Tuple

import pandas as pd


class DuplicateCleaner:
    def clean(self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        cleaned = dataframe.drop_duplicates().copy()
        removed = int(len(dataframe) - len(cleaned))
        if removed:
            return cleaned, [f"Removed {removed} duplicate rows"]
        return cleaned, ["No duplicate rows found"]

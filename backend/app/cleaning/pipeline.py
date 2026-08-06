from __future__ import annotations

from typing import Callable, List, Tuple

import pandas as pd


class CleaningPipeline:
    def __init__(self, step1: Callable[[pd.DataFrame], Tuple[pd.DataFrame, List[str]]], step2: Callable[[pd.DataFrame], Tuple[pd.DataFrame, List[str]]], step3: Callable[[pd.DataFrame], Tuple[pd.DataFrame, List[str]]], step4: Callable[[pd.DataFrame], Tuple[pd.DataFrame, List[str]]], step5: Callable[[pd.DataFrame], Tuple[pd.DataFrame, List[str]]]):
        self.steps = [step1, step2, step3, step4, step5]

    def run(self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        result = dataframe.copy()
        messages: List[str] = []
        missing_values_fixed = 0
        datatype_conversions = 0
        outliers_detected = 0

        for step in self.steps:
            result, step_messages = step(result)
            messages.extend(step_messages)
            if any("Filled" in message for message in step_messages):
                missing_values_fixed += 1
            if any("Converted" in message for message in step_messages):
                datatype_conversions += 1
            if any("Flagged" in message for message in step_messages):
                outliers_detected += 1

        return result, {
            "messages": messages,
            "missing_values_fixed": missing_values_fixed,
            "datatype_conversions": datatype_conversions,
            "outliers_detected": outliers_detected,
        }

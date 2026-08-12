"""Safe prompt construction for an optional external interpretation layer."""

from __future__ import annotations

import json
from typing import Any, Dict


def build_interpretation_prompt(dataset_name: str, facts: Dict[str, Any]) -> str:
    return (
        f"Dataset: {dataset_name}\n\nEvidence (JSON):\n{json.dumps(facts, default=str)}\n\n"
        "Task: Write concise business explanations using only this evidence. "
        "Do not calculate values, invent facts, claim causation from correlation, or make claims where evidence is absent. "
        "Clearly mark recommendations as conditional."
    )

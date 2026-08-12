"""Serialization boundaries between Plotly Python figures and API clients."""

from __future__ import annotations

import json
from typing import Any, Dict

from plotly.utils import PlotlyJSONEncoder


def figure_to_json(figure: Any) -> Dict[str, Any]:
    """Return plain JSON-compatible data, never a Plotly Python object."""
    return json.loads(json.dumps(figure.to_plotly_json(), cls=PlotlyJSONEncoder))

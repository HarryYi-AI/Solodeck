from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def safe_json_dumps(value: Any) -> str:
    def default(obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict("records")
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        return str(obj)

    return json.dumps(value, ensure_ascii=False, default=default, indent=2)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

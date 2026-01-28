from __future__ import annotations

import json
import re
from typing import Any, Dict, List


def strip_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```json"):
        return text[7:-3].strip()
    if text.startswith("```"):
        return text[3:-3].strip()
    return text


def extract_json(text: str) -> Dict[str, Any]:
    cleaned = strip_fences(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("LLM response did not contain JSON.")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("LLM response JSON was not an object.")
    return data


def coerce_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        item = value.strip()
        return [item] if item else []
    return []


def coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

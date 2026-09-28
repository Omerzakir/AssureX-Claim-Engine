"""AssureX SRS XXII-XXIV model comparison and consistency contract."""
from __future__ import annotations

VERSION = "24.0.0"
DEFAULT_THRESHOLDS = {
    "strong_match_max_difference": 0.10,
    "acceptable_match_max_difference": 0.20,
    "weak_match_max_difference": 0.35,
    "minimum_confidence": 0.55,
}


def _label(output: dict) -> str:
    value = output.get("label", output.get("predicted_class"))
    if value is None:
        raise ValueError("Model output is missing label")
    return str(value)


def _confidence(output: dict) -> float:
    value = output.get("confidence", output.get("top_confidence"))
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Model output is missing numeric confidence") from exc
    if not 0.0 <= value <= 1.0:
        raise ValueError("Model confidence must be between 0 and 1")
    return value


def _thresholds(config: dict | None) -> dict:
    """Load configurable thresholds while keeping safe defaults."""
    raw = config or {}
    source = raw.get("model_consistency") if isinstance(raw, dict) else None
    source = source if isinstance(source, dict) else raw if isinstance(raw, dict) else {}
    result = dict(DEFAULT_THRESHOLDS)
    for key in result:
        if key in source:
            value = float(source[key])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Threshold {key} must be between 0 and 1")
            result[key] = value
    if not (result["strong_match_max_difference"] <= result["acceptable_match_max_difference"] <= result["weak_match_max_difference"]):
        raise ValueError("Consistency difference thresholds must be ordered strong <= acceptable <= weak")
    return result


def compare_predictions(python_model: dict, teachable_machine: dict, config: dict | None = None) -> dict:
    """Implement SRS XXII-XXIV using already-produced model outputs only.

    XXII: prediction match/mismatch.
    XXIII: absolute top-confidence difference.
    XXIV: one of the five SRS statuses, using configurable thresholds.
    """
    thresholds = _thresholds(config)
    python_label = _label(python_model)
    tm_label = _label(teachable_machine)
    python_conf = _confidence(python_model)
    tm_conf = _confidence(teachable_machine)
    confidence_difference = round(abs(python_conf - tm_conf), 6)
    same_prediction = python_label == tm_label
    min_confidence = min(python_conf, tm_conf)

    if min_confidence < thresholds["minimum_confidence"]:
        status = "Uncertain Result"
    elif same_prediction and confidence_difference <= thresholds["strong_match_max_difference"]:
        status = "Strong Match"
    elif same_prediction and confidence_difference <= thresholds["acceptable_match_max_difference"]:
        status = "Acceptable Match"
    elif same_prediction and confidence_difference <= thresholds["weak_match_max_difference"]:
        status = "Weak Match"
    else:
        status = "Model Disagreement"

    return {
        "version": VERSION,
        "python_prediction": python_label,
        "teachable_machine_prediction": tm_label,
        "predictions_match": same_prediction,
        "python_confidence": python_conf,
        "teachable_machine_confidence": tm_conf,
        "confidence_difference": confidence_difference,
        "comparison": status,
        "consistency_status": status,
        "minimum_confidence": min_confidence,
        "thresholds": thresholds,
    }

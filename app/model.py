import json
import math
from pathlib import Path

from .features import SPECS

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
LEVELS = [(30, "Low", "Keep up regular check-ups and a healthy routine."),
          (60, "Moderate", "Worth discussing with a doctor at your next visit."),
          (101, "High", "Book an appointment with a doctor soon to review these results.")]


def _load():
    models = {}
    for d in SPECS:
        p = MODELS_DIR / f"{d}.json"
        if not p.exists():
            raise RuntimeError(f"Missing {p}. Run: python ml/train.py")
        models[d] = json.loads(p.read_text())
    return models


MODELS = _load()


def load_metrics():
    p = MODELS_DIR / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


def validate(disease, values):
    """Return (clean_values, errors)."""
    clean, errors = {}, []
    for f in SPECS[disease]["features"]:
        v = values.get(f["name"])
        if v is None:
            errors.append(f"{f['label']} is required")
        elif not (f["min"] <= v <= f["max"]):
            errors.append(f"{f['label']} must be between {f['min']} and {f['max']}")
        else:
            clean[f["name"]] = float(v)
    return clean, errors


def predict(disease, values):
    m = MODELS[disease]
    labels = {f["name"]: f["label"] for f in SPECS[disease]["features"]}
    logit, factors = m["intercept"], []
    for name, mean, std, coef in zip(m["features"], m["mean"], m["std"], m["coef"]):
        contribution = coef * (values[name] - mean) / std  # effect vs. an average patient
        logit += contribution
        factors.append({"name": name, "label": labels[name], "value": values[name],
                        "contribution": round(contribution, 3)})
    risk = 100 / (1 + math.exp(-logit))
    level, advice = next((n, a) for cap, n, a in LEVELS if risk < cap)
    factors.sort(key=lambda f: abs(f["contribution"]), reverse=True)
    return {"disease": disease, "risk_percent": round(risk, 1), "level": level, "advice": advice,
            "factors": factors[:6],
            "disclaimer": "Educational decision-support only. Not a medical diagnosis."}

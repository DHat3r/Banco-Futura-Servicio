"""
Carga del modelo de churn (XGBoost) e inferencia con explicabilidad SHAP.

Este módulo concentra TODO lo que toca al modelo de churn, a propósito: el
día que migres a Databricks, esta es la única pieza que cambia. En vez de
``joblib.load(...)`` acá adentro pasarías a algo como
``mlflow.pyfunc.load_model("models:/churn_banco_futura/Production")`` o a
una llamada HTTP al endpoint de Databricks Model Serving — el resto de la
API (schemas, main.py, el frontend) no se entera del cambio.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import shap

from app.features import ALL_FEATURES, FEATURE_LABELS_ES, build_inference_row

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "churn_model"
PIPELINE_PATH = ARTIFACTS_DIR / "pipeline.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

_pipeline = None
_explainer = None
_metrics: dict | None = None


class ModeloNoEntrenadoError(RuntimeError):
    pass


def cargar_modelo() -> None:
    """Carga el pipeline entrenado y construye el explainer de SHAP una sola vez."""
    global _pipeline, _explainer, _metrics

    import joblib  # import local: evita cargar xgboost si el modelo nunca se usa

    if not PIPELINE_PATH.exists():
        raise ModeloNoEntrenadoError(
            f"No se encontró {PIPELINE_PATH}. Corre primero: "
            f"python scripts/train_xgboost.py"
        )

    _pipeline = joblib.load(PIPELINE_PATH)
    xgb_model = _pipeline.named_steps["model"]
    _explainer = shap.TreeExplainer(xgb_model)

    if METRICS_PATH.exists():
        _metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def modelo_listo() -> bool:
    return _pipeline is not None


def metricas_modelo() -> dict | None:
    return _metrics


def _nivel_riesgo(prob: float) -> str:
    if prob <= 0.40:
        return "BAJO"
    if prob <= 0.70:
        return "MEDIO"
    return "ALTO"


def predecir_churn(cliente: dict, top_n: int = 4) -> dict:
    if _pipeline is None or _explainer is None:
        raise ModeloNoEntrenadoError("El modelo de churn no está cargado todavía.")

    row = build_inference_row(cliente)

    prob_churn = float(_pipeline.predict_proba(row)[0, 1])
    nivel = _nivel_riesgo(prob_churn)

    preprocessor = _pipeline.named_steps["preprocessor"]
    row_processed = preprocessor.transform(row)
    cat_names = preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(
        ["Gender", "subscription_type", "contract_length"]
    )
    feature_names = [
        "Age", "Tenure", "usage_frequency", "support_calls", "payment_delay",
        "total_spend", "last_interaction", "support_calls_per_tenure",
        "spend_per_tenure", "risk_score_flags",
    ] + list(cat_names)

    shap_values = _explainer.shap_values(row_processed)[0]

    factores = sorted(
        zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True
    )[:top_n]

    factores_out = []
    for feat_name, impacto in factores:
        base_feat = next((f for f in ALL_FEATURES if feat_name == f or feat_name.startswith(f + "_")), feat_name)
        valor = row.iloc[0][base_feat] if base_feat in row.columns else feat_name.split("_")[-1]
        factores_out.append({
            "feature": base_feat,
            "feature_label": FEATURE_LABELS_ES.get(base_feat, base_feat),
            "impacto": round(float(impacto), 4),
            "valor_cliente": str(valor),
        })

    return {
        "prob_churn": round(prob_churn, 4),
        "nivel_riesgo": nivel,
        "factores_principales": factores_out,
    }

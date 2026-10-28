"""
Feature engineering — Banco Futura · Churn Scoring Service

Este módulo es la versión local (pandas) de lo que en Databricks corre como
las capas Silver y Gold de la pipeline Medallion (ver
pipeline/pipeline_02_silver.py y pipeline/pipeline_03_gold.py del proyecto
Databricks). Las reglas de negocio son EXACTAMENTE las mismas — esto es a
propósito, para que cuando migres a Databricks no tengas que re-derivar nada:
el modelo entrenado acá y el entrenado allá ven las mismas columnas.

Se usa en dos lugares:
1. scripts/train_xgboost.py — para construir el dataset de entrenamiento.
2. app/churn_model.py — para transformar el request de la API (datos crudos
   de un cliente) en las mismas features que el modelo espera.
"""

from __future__ import annotations

import pandas as pd

TARGET = "Churn"

# Columnas tal como llegan del cliente / del CSV crudo (snake_case ya aplicado,
# equivalente a lo que hace bronze_churn + el rename de silver_churn).
RAW_NUM_COLUMNS = [
    "Age",
    "Tenure",
    "usage_frequency",
    "support_calls",
    "payment_delay",
    "total_spend",
    "last_interaction",
]
RAW_CAT_COLUMNS = ["Gender", "subscription_type", "contract_length"]

# Features derivadas que agrega la capa Silver.
DERIVED_NUM_COLUMNS = [
    "support_calls_per_tenure",
    "spend_per_tenure",
    "risk_score_flags",
]

# Mismo orden y mismos nombres que NUM_FEATURES / CAT_FEATURES en
# pipeline_03_gold.py y en model_A_xgboost.py — no reordenar.
NUM_FEATURES = RAW_NUM_COLUMNS + DERIVED_NUM_COLUMNS
CAT_FEATURES = RAW_CAT_COLUMNS
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

# Valores válidos — los mismos que existen en el dataset de Kaggle. Útiles
# para los <select> del frontend y para validar el request en la API.
GENDER_OPTIONS = ["Female", "Male"]
SUBSCRIPTION_OPTIONS = ["Basic", "Standard", "Premium"]
CONTRACT_OPTIONS = ["Monthly", "Quarterly", "Annual"]

# Etiquetas en español para mostrar en el frontend (SHAP, tarjetas, etc.)
FEATURE_LABELS_ES = {
    "Age": "Edad",
    "Tenure": "Antigüedad (meses)",
    "usage_frequency": "Frecuencia de uso",
    "support_calls": "Llamadas a soporte",
    "payment_delay": "Atraso en pagos (días)",
    "total_spend": "Gasto total",
    "last_interaction": "Días desde última interacción",
    "support_calls_per_tenure": "Llamadas a soporte por antigüedad",
    "spend_per_tenure": "Gasto por antigüedad",
    "risk_score_flags": "Banderas de riesgo acumuladas",
    "Gender": "Género",
    "subscription_type": "Tipo de suscripción",
    "contract_length": "Duración del contrato",
}


def rename_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica el mismo rename que bronze_churn / silver_churn en Databricks."""
    return df.rename(
        columns={
            "Usage Frequency": "usage_frequency",
            "Support Calls": "support_calls",
            "Payment Delay": "payment_delay",
            "Subscription Type": "subscription_type",
            "Contract Length": "contract_length",
            "Total Spend": "total_spend",
            "Last Interaction": "last_interaction",
        }
    )


def add_silver_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replica exactamente las columnas derivadas de pipeline_02_silver.py:
    support_calls_per_tenure, spend_per_tenure, is_new_client, high_friction,
    high_payment_delay, is_inactive y risk_score_flags (suma de las 4 banderas).
    """
    df = df.copy()

    tenure_positive = df["Tenure"] > 0
    df["support_calls_per_tenure"] = df["support_calls"].astype(float)
    df.loc[tenure_positive, "support_calls_per_tenure"] = (
        df.loc[tenure_positive, "support_calls"] / df.loc[tenure_positive, "Tenure"]
    )

    df["spend_per_tenure"] = df["total_spend"].astype(float)
    df.loc[tenure_positive, "spend_per_tenure"] = (
        df.loc[tenure_positive, "total_spend"] / df.loc[tenure_positive, "Tenure"]
    )

    is_new_client = (df["Tenure"] <= 6).astype(int)
    high_friction = (df["support_calls"] >= 4).astype(int)
    high_payment_delay = (df["payment_delay"] >= 15).astype(int)
    is_inactive = (df["last_interaction"] > 20).astype(int)

    df["risk_score_flags"] = is_new_client + high_friction + high_payment_delay + is_inactive

    return df


def build_training_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completa Bronze -> Silver -> Gold para entrenamiento, a partir
    del CSV crudo de Kaggle (mismo dataset que usa el proyecto Databricks).
    """
    df = rename_raw_columns(raw_df)

    df = df.dropna(subset=["CustomerID"]) if "CustomerID" in df.columns else df
    df = df.drop(columns=["CustomerID"], errors="ignore")

    for col in ["Age", "Tenure", "usage_frequency", "support_calls", "payment_delay", "last_interaction", TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df["total_spend"] = pd.to_numeric(df["total_spend"], errors="coerce")

    df = df.dropna(subset=RAW_NUM_COLUMNS + [TARGET])
    df = df[(df["Age"] >= 0) & (df["Tenure"] >= 0)]
    df = df.drop_duplicates()

    for col in ["Age", "Tenure", "usage_frequency", "support_calls", "payment_delay", "last_interaction", TARGET]:
        df[col] = df[col].astype(int)

    df = add_silver_features(df)

    return df[ALL_FEATURES + [TARGET]]


def build_inference_row(client: dict) -> pd.DataFrame:
    """
    Convierte el payload crudo de un cliente (lo que manda el frontend) en
    una fila con exactamente las columnas que espera el pipeline de sklearn
    (mismo orden / mismos nombres que ALL_FEATURES).
    """
    row = pd.DataFrame([client])
    row = add_silver_features(row)
    return row[ALL_FEATURES]

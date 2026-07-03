"""
Entrenamiento local del Modelo A (XGBoost) — Banco Futura

Equivalente local a models/model_A_xgboost.py del proyecto Databricks: mismo
preprocesamiento, mismo manejo de desbalance (scale_pos_weight), misma
búsqueda de hiperparámetros con RandomizedSearchCV y misma explicabilidad
con SHAP. La diferencia es solo el origen de los datos: aquí se lee
directamente el CSV de Kaggle en vez de la tabla gold_churn_features de
Spark, y el modelo se guarda en disco con joblib en vez de en MLflow.

Uso:
    python scripts/train_xgboost.py                  # corrida rápida (muestreada)
    python scripts/train_xgboost.py --full            # con todos los datos
    python scripts/train_xgboost.py --sample 100000   # tamaño de muestra custom
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.features import ALL_FEATURES, CAT_FEATURES, NUM_FEATURES, TARGET, build_training_frame  # noqa: E402

RANDOM_STATE = 42
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "churn_model"


def load_raw_data() -> pd.DataFrame:
    train_path = DATA_DIR / "customer_churn_dataset-training-master.csv"
    test_path = DATA_DIR / "customer_churn_dataset-testing-master.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return pd.concat([train_df, test_df], ignore_index=True)


def main(sample_size: int | None, n_iter: int) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Leyendo CSV crudos y aplicando Bronze -> Silver -> Gold (pandas)...")
    raw_df = load_raw_data()
    gold_df = build_training_frame(raw_df)
    print(f"   Filas tras limpieza: {len(gold_df):,}")

    if sample_size and sample_size < len(gold_df):
        gold_df = gold_df.sample(n=sample_size, random_state=RANDOM_STATE)
        print(f"   Muestreado a {len(gold_df):,} filas para esta corrida local")

    X = gold_df[ALL_FEATURES]
    y = gold_df[TARGET]
    print(f"   Distribución del target:\n{y.value_counts().to_string()}")

    print("\n2. Split 80/20 estratificado...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    scale_pos_weight = n_negative / n_positive
    print(f"   scale_pos_weight: {scale_pos_weight:.4f}")

    print("\n3. Definiendo pipeline de preprocesamiento + XGBoost...")
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUM_FEATURES),
            ("cat", categorical_transformer, CAT_FEATURES),
        ],
        remainder="drop",
    )

    xgb_base = XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_jobs=-1,
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", xgb_base)])

    param_dist = {
        "model__n_estimators": [200, 300, 400],
        "model__max_depth": [3, 4, 5],
        "model__learning_rate": [0.03, 0.05, 0.08],
        "model__subsample": [0.7, 0.8, 0.9],
        "model__colsample_bytree": [0.7, 0.8, 0.9],
        "model__reg_alpha": [0.1, 0.5, 1.0],
        "model__reg_lambda": [1.0, 2.0, 5.0],
        "model__min_child_weight": [3, 5, 7],
        "model__gamma": [0.1, 0.2, 0.5],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE,
    )

    print(f"\n4. Buscando hiperparámetros ({n_iter} combinaciones x 3 folds)...")
    start = datetime.now()
    search.fit(X_train, y_train)
    elapsed = (datetime.now() - start).seconds
    best_model = search.best_estimator_
    print(f"   Listo en {elapsed}s | Mejor AUC-ROC (CV): {search.best_score_:.4f}")

    print("\n5. Evaluando sobre el conjunto de prueba...")
    y_pred = best_model.predict(X_test)
    y_pred_prob = best_model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_prob),
        "scale_pos_weight": float(scale_pos_weight),
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "trained_at": datetime.now().isoformat(),
        "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
    }
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(classification_report(y_test, y_pred))

    print("\n6. Guardando artefactos en", ARTIFACTS_DIR)
    joblib.dump(best_model, ARTIFACTS_DIR / "pipeline.joblib")
    with open(ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(ARTIFACTS_DIR / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump({"num_features": NUM_FEATURES, "cat_features": CAT_FEATURES}, f, indent=2, ensure_ascii=False)

    print("✓ Modelo guardado. Ya puedes levantar la API (uvicorn app.main:app).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=60_000, help="Filas a muestrear (omite con --full)")
    parser.add_argument("--full", action="store_true", help="Usar el dataset completo, sin muestreo")
    parser.add_argument("--n_iter", type=int, default=15, help="Combinaciones para RandomizedSearchCV")
    args = parser.parse_args()
    main(sample_size=None if args.full else args.sample, n_iter=args.n_iter)

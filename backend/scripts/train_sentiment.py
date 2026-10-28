"""
Entrenamiento de la cabeza de clasificación de sentimiento — Banco Futura

Usa BETO (dccuchile/bert-base-spanish-wwm-cased) para generar embeddings del
token [CLS] de cada reseña, y entrena encima una red densa pequeña (16 -> 8 -> 1,
sigmoide) para clasificar sentimiento positivo/negativo.

Los datos son reseñas reales de apps de bancos en Google Play, etiquetadas por
estrellas (supervisión débil): generarlas antes con

    python scripts/review_scraper.py

Requiere internet la primera vez (descarga BETO desde Hugging Face, ~440MB,
después queda cacheado). Instala antes: pip install -r requirements-sentiment.txt

Uso:
    python scripts/train_sentiment.py                     # 2000 reseñas por clase
    python scripts/train_sentiment.py --max-por-clase 500  # prueba rápida
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertTokenizer, logging

sys.path.append(str(Path(__file__).resolve().parent.parent))

logging.set_verbosity_error()

MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "resenas_bancos.csv"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "sentiment_model"
RANDOM_STATE = 42
TAMANO_LOTE = 32


def cargar_dataset(max_por_clase: int) -> pd.DataFrame:
    """Lee las reseñas y toma la misma cantidad de cada clase, para que el modelo no se sesgue."""
    if not DATASET_PATH.exists():
        raise SystemExit(f"No se encontró {DATASET_PATH}. Corre primero: python scripts/review_scraper.py")
    df = pd.read_csv(DATASET_PATH)
    n = min(max_por_clase, df["etiqueta"].value_counts().min())
    return df.groupby("etiqueta").sample(n=n, random_state=RANDOM_STATE)


def generar_embeddings(textos: list[str], tokenizador, bert_modelo) -> np.ndarray:
    """Embedding del token [CLS] de cada texto, procesando por lotes para ir más rápido."""
    embeddings = []
    for inicio in range(0, len(textos), TAMANO_LOTE):
        lote = textos[inicio:inicio + TAMANO_LOTE]
        encoded = tokenizador(lote, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            salida = bert_modelo(**encoded)
        embeddings.append(salida.last_hidden_state[:, 0, :].numpy())
        print(f"   {min(inicio + TAMANO_LOTE, len(textos))}/{len(textos)}", end="\r")
    print()
    return np.vstack(embeddings)


def main(max_por_clase: int) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Cargando reseñas...")
    df = cargar_dataset(max_por_clase)
    print(f"   {len(df)} reseñas ({len(df) // 2} por clase)")

    print(f"2. Cargando BETO ({MODEL_NAME})... esto descarga ~440MB la primera vez")
    tokenizador = BertTokenizer.from_pretrained(MODEL_NAME)
    bert_modelo = BertModel.from_pretrained(MODEL_NAME, ignore_mismatched_sizes=True)
    bert_modelo.eval()

    print("3. Generando embeddings...")
    X = generar_embeddings(df["texto"].tolist(), tokenizador, bert_modelo)
    y = df["etiqueta"].to_numpy(dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print("4. Construyendo y entrenando la red (16 -> 8 -> 1, sigmoide)...")
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    modelo = tf.keras.Sequential([
        tf.keras.Input(shape=[768]),
        tf.keras.layers.Dense(units=16, activation="relu"),
        tf.keras.layers.Dense(units=8, activation="relu"),
        tf.keras.layers.Dense(units=1, activation="sigmoid"),
    ])
    modelo.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
    # Corta cuando la pérdida de validación deja de mejorar, para no sobreajustar.
    parada_temprana = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    modelo.fit(X_train, y_train, validation_split=0.1, epochs=100, callbacks=[parada_temprana], verbose=0)

    print("5. Evaluando en el conjunto de test...")
    y_pred = (modelo.predict(X_test, verbose=0).ravel() >= 0.5).astype(int)
    metricas = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "dataset": DATASET_PATH.name,
        "trained_at": datetime.now().isoformat(),
    }
    print(json.dumps(metricas, indent=2, ensure_ascii=False))

    out_path = ARTIFACTS_DIR / "red_sentimiento.keras"
    modelo.save(out_path)
    with open(ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    print(f"✓ Cabeza de sentimiento guardada en {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-por-clase", type=int, default=2000, help="Reseñas por clase a usar")
    args = parser.parse_args()
    main(max_por_clase=args.max_por_clase)

"""
Entrenamiento de la cabeza de clasificación de sentimiento — Banco Futura

Equivalente a tu notebook BERT_Entrenamiento: usa BETO
(dccuchile/bert-base-spanish-wwm-cased) para generar embeddings del token
[CLS] de cada frase, y entrena encima una red densa pequeña (16 -> 8 -> 1,
sigmoide) para clasificar sentimiento positivo/negativo. Las 40 frases de
ejemplo son las mismas de tu notebook original.

Requiere internet la primera vez (descarga BETO desde Hugging Face, ~440MB,
después queda cacheado). Instala antes: pip install transformers torch
tensorflow

Uso:
    python scripts/train_sentiment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
from transformers import BertModel, BertTokenizer, logging

sys.path.append(str(Path(__file__).resolve().parent.parent))

logging.set_verbosity_error()

MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "sentiment_model"

# Mismas 40 frases de tu notebook BERT_Entrenamiento (20 positivas + 20 negativas).
ORACIONES = [
    "Buena atención del ejecutivo",
    "Me ayudaron a solucionar mi problema rápidamente",
    "La aplicación del banco funciona muy bien",
    "Estoy muy satisfecho con los servicios de Banco Futura",
    "El proceso para abrir mi cuenta fue muy fácil",
    "La atención fue amable y eficiente",
    "Resolvieron mi consulta en pocos minutos",
    "El sitio web es rápido y fácil de usar",
    "Recibí una excelente orientación para mi crédito",
    "Las transferencias funcionan sin problemas",
    "Me respondieron de inmediato por teléfono",
    "El personal fue muy cordial conmigo",
    "La apertura de cuenta fue sencilla y rápida",
    "Los trámites fueron claros y transparentes",
    "Estoy conforme con la atención recibida",
    "La sucursal estaba bien organizada",
    "Obtuvieron una solución efectiva para mi caso",
    "El sistema de pagos funciona perfectamente",
    "Siempre responden mis dudas oportunamente",
    "La experiencia general fue excelente",
    "Pésima atención al cliente, no resolvieron nada",
    "La aplicación se cae constantemente",
    "Llevo semanas esperando respuesta y nadie me contacta",
    "Me cobraron comisiones que no corresponden",
    "El servicio es muy lento y deficiente",
    "Nunca solucionaron mi problema",
    "La plataforma presenta errores todos los días",
    "Tuve que llamar varias veces para obtener ayuda",
    "La información entregada fue confusa",
    "Perdí mucho tiempo esperando atención",
    "No respetaron los plazos prometidos",
    "La sucursal estaba desorganizada",
    "El ejecutivo no supo responder mis preguntas",
    "Mi transferencia fue rechazada sin explicación",
    "La atención telefónica fue muy mala",
    "La página web funciona extremadamente lento",
    "No recibí respuesta a mis reclamos",
    "El proceso fue complicado y frustrante",
    "Me entregaron información incorrecta",
    "La experiencia general fue decepcionante",
]
ETIQUETAS = np.array([1] * 20 + [0] * 20, dtype=float)


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"1. Cargando BETO ({MODEL_NAME})... esto descarga ~440MB la primera vez")
    tokenizador = BertTokenizer.from_pretrained(MODEL_NAME)
    bert_modelo = BertModel.from_pretrained(MODEL_NAME, ignore_mismatched_sizes=True)
    bert_modelo.eval()

    def get_embedding(oracion: str):
        encoded = tokenizador(oracion, return_tensors="pt")
        with torch.no_grad():
            salida = bert_modelo(**encoded)
        return salida.last_hidden_state[0, 0, :].numpy()

    print("2. Generando embeddings de las 40 frases de ejemplo...")
    X = np.array([get_embedding(o) for o in ORACIONES])
    print(f"   Forma de X: {X.shape}")

    print("3. Construyendo y entrenando la red (16 -> 8 -> 1, sigmoide)...")
    modelo = tf.keras.Sequential([
        tf.keras.Input(shape=[768]),
        tf.keras.layers.Dense(units=16, activation="relu"),
        tf.keras.layers.Dense(units=8, activation="relu"),
        tf.keras.layers.Dense(units=1, activation="sigmoid"),
    ])
    modelo.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
    historial = modelo.fit(X, ETIQUETAS, epochs=100, verbose=0)
    print(f"   Pérdida final: {historial.history['loss'][-1]:.4f}")

    out_path = ARTIFACTS_DIR / "red_sentimiento.keras"
    modelo.save(out_path)
    print(f"✓ Cabeza de sentimiento guardada en {out_path}")
    print("  Ya puedes levantar la API — el endpoint /predict/sentimiento descargará")
    print("  BETO la primera vez que reciba una solicitud.")


if __name__ == "__main__":
    main()

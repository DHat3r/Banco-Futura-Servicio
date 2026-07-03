"""
Carga del modelo de sentimiento (BETO + cabeza de clasificación) e inferencia.

Es la versión "productizada" de tu notebook BERT_Entrenamiento: mismo BETO
(``dccuchile/bert-base-spanish-wwm-cased``), mismo embedding (token [CLS]),
misma arquitectura de red encima (16 -> 8 -> 1, sigmoide). La diferencia es
que aquí la red ya entrenada se carga una sola vez al iniciar la API, en vez
de entrenarse cada vez.

Importante: BETO se descarga la primera vez desde Hugging Face, así que esto
necesita conexión a internet la primera vez que se ejecuta (después queda
cacheado localmente). Antes de levantar la API por primera vez, corre:

    python scripts/train_sentiment.py

Eso entrena la cabeza de clasificación con las mismas 40 frases de ejemplo
de tu notebook y la deja guardada en artifacts/sentiment_model/.
"""

from __future__ import annotations

from pathlib import Path

MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "sentiment_model"
HEAD_PATH = ARTIFACTS_DIR / "red_sentimiento.keras"

_tokenizer = None
_bert_modelo = None
_red = None


class ModeloNoEntrenadoError(RuntimeError):
    pass


def cargar_modelo() -> None:
    """Carga BETO (tokenizer + encoder) y la cabeza de clasificación entrenada."""
    global _tokenizer, _bert_modelo, _red

    if not HEAD_PATH.exists():
        raise ModeloNoEntrenadoError(
            f"No se encontró {HEAD_PATH}. Corre primero: python scripts/train_sentiment.py"
        )

    import tensorflow as tf
    import torch
    from transformers import BertModel, BertTokenizer, logging

    logging.set_verbosity_error()

    _tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    _bert_modelo = BertModel.from_pretrained(MODEL_NAME)
    _bert_modelo.eval()
    _red = tf.keras.models.load_model(HEAD_PATH)


def modelo_listo() -> bool:
    return _red is not None


def _obtener_embedding(texto: str):
    import torch

    encoded = _tokenizer(texto, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        salida = _bert_modelo(**encoded)
    return salida.last_hidden_state[0, 0, :].numpy()  # token [CLS]


def predecir_sentimiento(texto: str) -> dict:
    if _red is None:
        raise ModeloNoEntrenadoError("El modelo de sentimiento no está cargado todavía.")

    embedding = _obtener_embedding(texto).reshape(1, -1)
    score = float(_red.predict(embedding, verbose=0)[0][0])
    sentimiento = "positivo" if score >= 0.5 else "negativo"

    return {
        "texto": texto,
        "sentimiento": sentimiento,
        "score": round(score, 4),
    }

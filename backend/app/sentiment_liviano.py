"""
Modelo de sentimiento "BETO liviano": BETO cuantizado a int8 en ONNX + la misma cabeza en NumPy.

Es un segundo modelo, no un reemplazo de app/sentiment_model.py. Da casi las mismas
respuestas que el completo, pero al predecir no necesita PyTorch ni TensorFlow: solo
onnxruntime, tokenizers y numpy. Por eso cabe en el plan gratis de Render (512 MB).

Los artefactos se crean con scripts/export_sentiment_liviano.py. tokenizer.json y
cabeza.npz están en el repo; encoder_int8.onnx (~110 MB) no, porque GitHub rechaza
archivos de más de 100 MB: el Dockerfile lo descarga de un release del repo.
"""

from __future__ import annotations

import os
from pathlib import Path

from app.sentiment_model import ModeloNoEntrenadoError, quitar_tildes

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "sentiment_liviano"
ENCODER_PATH = Path(os.getenv("SENTIMENT_LIVIANO_ENCODER", ARTIFACTS_DIR / "encoder_int8.onnx"))
TOKENIZER_PATH = ARTIFACTS_DIR / "tokenizer.json"
CABEZA_PATH = ARTIFACTS_DIR / "cabeza.npz"

_tokenizador = None
_sesion = None
_cabeza: dict | None = None


def cargar_modelo() -> None:
    """Carga el tokenizador, la sesión de ONNX y los pesos de la cabeza."""
    global _tokenizador, _sesion, _cabeza

    if os.getenv("ENABLE_SENTIMENT_LIVIANO", "false").lower() != "true":
        raise ModeloNoEntrenadoError("BETO liviano desactivado (ENABLE_SENTIMENT_LIVIANO != true)")

    faltan = [ruta.name for ruta in (ENCODER_PATH, TOKENIZER_PATH, CABEZA_PATH) if not ruta.exists()]
    if faltan:
        raise ModeloNoEntrenadoError(
            f"Faltan artefactos de BETO liviano: {', '.join(faltan)}. "
            "Corre scripts/export_sentiment_liviano.py o construye la imagen con INSTALL_SENTIMENT_LIVIANO=true."
        )

    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    opciones = ort.SessionOptions()
    # Un hilo: en un servidor chico, varios hilos compiten por la misma CPU y no ganan velocidad.
    opciones.intra_op_num_threads = 1
    _sesion = ort.InferenceSession(str(ENCODER_PATH), opciones, providers=["CPUExecutionProvider"])
    _tokenizador = Tokenizer.from_file(str(TOKENIZER_PATH))
    with np.load(CABEZA_PATH) as pesos:
        _cabeza = {nombre: pesos[nombre] for nombre in pesos.files}


def modelo_listo() -> bool:
    return _sesion is not None


def aplicar_cabeza(cabeza: dict, cls):
    """La red de Keras (768 -> 16 -> 8 -> 1) escrita en NumPy: relu, relu y sigmoide."""
    import numpy as np

    h = np.maximum(cls @ cabeza["W1"] + cabeza["b1"], 0)
    h = np.maximum(h @ cabeza["W2"] + cabeza["b2"], 0)
    z = h @ cabeza["W3"] + cabeza["b3"]
    return (1 / (1 + np.exp(-z))).ravel()


def predecir_sentimiento(texto: str) -> dict:
    if _sesion is None:
        raise ModeloNoEntrenadoError("BETO liviano no está cargado todavía.")

    import numpy as np

    # Igual que el modelo completo: se predice sobre el texto sin tildes y se devuelve el original.
    codificado = _tokenizador.encode(quitar_tildes(texto))
    entradas = {
        "input_ids": np.array([codificado.ids], dtype=np.int64),
        "attention_mask": np.array([codificado.attention_mask], dtype=np.int64),
        "token_type_ids": np.array([codificado.type_ids], dtype=np.int64),
    }
    cls = _sesion.run(["cls"], entradas)[0]
    score = float(aplicar_cabeza(_cabeza, cls)[0])

    return {
        "texto": texto,
        "sentimiento": "positivo" if score >= 0.5 else "negativo",
        "score": round(score, 4),
    }

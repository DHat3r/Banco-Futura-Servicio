"""Crea "BETO liviano": la misma red de sentimiento, sin PyTorch ni TensorFlow al predecir.

BETO se exporta a ONNX y se cuantiza a int8 (cada peso pasa de 4 bytes a 1). La cabeza
entrenada (768 -> 16 -> 8 -> 1) se guarda como matrices de NumPy. En la API basta con
onnxruntime + tokenizers + numpy, que caben en el plan gratis de Render (512 MB).

No reemplaza al modelo completo (app/sentiment_model.py): es un segundo modelo que se
compara con él sobre las mismas 800 reseñas de prueba del entrenamiento.

Uso (desde backend/, con torch, transformers, onnx, onnxruntime y h5py instalados):
    python scripts/export_sentiment_liviano.py

Genera en artifacts/sentiment_liviano/:
    encoder_int8.onnx   BETO cuantizado (~110 MB, NO va al repo: se publica en un release)
    tokenizer.json      tokenizador de BETO para la librería tokenizers
    cabeza.npz          pesos de la red de clasificación
    metrics.json        métricas de los dos modelos, tamaño y latencia
"""

import io
import json
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer
from transformers import BertModel, BertTokenizer, BertTokenizerFast, logging

sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.sentiment_model import MODEL_NAME, quitar_tildes  # noqa: E402 - mismo modelo que la API

logging.set_verbosity_error()

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BACKEND_DIR / "data" / "resenas_bancos.csv"
CABEZA_KERAS = BACKEND_DIR / "artifacts" / "sentiment_model" / "red_sentimiento.keras"
SALIDA = BACKEND_DIR / "artifacts" / "sentiment_liviano"

# Deben ser los mismos valores de scripts/train_sentiment.py para obtener las mismas 800 reseñas.
# No se importa ese script porque carga TensorFlow al importarse.
RANDOM_STATE = 42
MAX_POR_CLASE = 2000
MAX_TOKENS = 128


class EncoderCLS(torch.nn.Module):
    """BETO que devuelve solo el vector del token [CLS], lo único que usa la cabeza."""

    def __init__(self, bert: BertModel):
        super().__init__()
        self.bert = bert

    def forward(self, input_ids, attention_mask, token_type_ids):
        salida = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return salida.last_hidden_state[:, 0, :]


def split_de_prueba() -> pd.DataFrame:
    """Las mismas 800 reseñas de prueba que usó train_sentiment.py."""
    df = pd.read_csv(DATASET_PATH)
    n = min(MAX_POR_CLASE, df["etiqueta"].value_counts().min())
    df = df.groupby("etiqueta").sample(n=n, random_state=RANDOM_STATE)
    _, df_test = train_test_split(df, test_size=0.2, stratify=df["etiqueta"], random_state=RANDOM_STATE)
    return df_test


def leer_cabeza() -> dict[str, np.ndarray]:
    """Lee los pesos de las 3 capas densas desde el .keras (un zip con un HDF5), sin TensorFlow."""
    with zipfile.ZipFile(CABEZA_KERAS) as archivo:
        pesos = h5py.File(io.BytesIO(archivo.read("model.weights.h5")), "r")
        capas = ["dense", "dense_1", "dense_2"]
        return {
            nombre: np.array(pesos[f"layers/{capa}/vars/{i}"], dtype=np.float32)
            for n, capa in enumerate(capas, start=1)
            for i, nombre in ((0, f"W{n}"), (1, f"b{n}"))
        }


def aplicar_cabeza(cabeza: dict[str, np.ndarray], cls: np.ndarray) -> np.ndarray:
    """La misma red de Keras escrita en NumPy: relu, relu y sigmoide."""
    h = np.maximum(cls @ cabeza["W1"] + cabeza["b1"], 0)
    h = np.maximum(h @ cabeza["W2"] + cabeza["b2"], 0)
    z = h @ cabeza["W3"] + cabeza["b3"]
    return (1 / (1 + np.exp(-z))).ravel()


def exportar_onnx(bert: BertModel, tokenizador: BertTokenizer, ruta: Path) -> None:
    ejemplo = tokenizador(["hola mundo"], return_tensors="pt")
    entradas = (ejemplo["input_ids"], ejemplo["attention_mask"], ejemplo["token_type_ids"])
    # Ejes que pueden cambiar de tamaño: cuántos textos (0) y cuántos tokens (1).
    ejes = {0: "batch", 1: "tokens"}
    # .eval() es obligatorio: un módulo nuevo nace en modo entrenamiento, y al terminar la
    # exportación PyTorch lo deja así, arrastrando a BETO (dropout activo en el resto del script).
    torch.onnx.export(
        EncoderCLS(bert).eval(),
        entradas,
        str(ruta),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["cls"],
        dynamic_axes={
            "input_ids": ejes,
            "attention_mask": ejes,
            "token_type_ids": ejes,
            "cls": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )


def predecir_original(textos, tokenizador, bert, cabeza) -> np.ndarray:
    """Modelo completo: BETO en PyTorch (float32) + la cabeza."""
    scores = []
    for texto in textos:
        encoded = tokenizador(texto, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
        with torch.no_grad():
            cls = bert(**encoded).last_hidden_state[:, 0, :].numpy()
        scores.append(aplicar_cabeza(cabeza, cls)[0])
    return np.array(scores)


def predecir_liviano(textos, tokenizador: Tokenizer, sesion: ort.InferenceSession, cabeza) -> np.ndarray:
    """Modelo liviano: tokenizers + ONNX int8 + la cabeza. Es lo mismo que hará la API."""
    scores = []
    for texto in textos:
        codificado = tokenizador.encode(texto)
        entradas = {
            "input_ids": np.array([codificado.ids], dtype=np.int64),
            "attention_mask": np.array([codificado.attention_mask], dtype=np.int64),
            "token_type_ids": np.array([codificado.type_ids], dtype=np.int64),
        }
        cls = sesion.run(["cls"], entradas)[0]
        scores.append(aplicar_cabeza(cabeza, cls)[0])
    return np.array(scores)


def metricas(y: np.ndarray, scores: np.ndarray) -> dict:
    y_pred = (scores >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1_score": f1_score(y, y_pred),
    }


def latencia_ms(predecir, textos) -> float:
    """Tiempo medio por reseña, de a una (como llega a la API)."""
    predecir(textos[:5])  # calentamiento
    inicio = time.perf_counter()
    predecir(textos)
    return round((time.perf_counter() - inicio) / len(textos) * 1000, 1)


def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    ruta_fp32 = SALIDA / "encoder_fp32.onnx"
    ruta_int8 = SALIDA / "encoder_int8.onnx"

    print("1. Cargando BETO y la cabeza entrenada...")
    tokenizador = BertTokenizer.from_pretrained(MODEL_NAME)
    bert = BertModel.from_pretrained(MODEL_NAME, ignore_mismatched_sizes=True).eval()
    cabeza = leer_cabeza()
    np.savez(SALIDA / "cabeza.npz", **cabeza)

    print("2. Guardando el tokenizador rápido (librería tokenizers)...")
    rapido = BertTokenizerFast.from_pretrained(MODEL_NAME).backend_tokenizer
    rapido.enable_truncation(max_length=MAX_TOKENS)
    rapido.save(str(SALIDA / "tokenizer.json"))
    tokenizador_liviano = Tokenizer.from_file(str(SALIDA / "tokenizer.json"))

    print("3. Exportando a ONNX y cuantizando a int8...")
    exportar_onnx(bert, tokenizador, ruta_fp32)
    quantize_dynamic(str(ruta_fp32), str(ruta_int8), weight_type=QuantType.QInt8)
    tamano_fp32 = ruta_fp32.stat().st_size
    ruta_fp32.unlink()  # solo era un paso intermedio
    sesion = ort.InferenceSession(str(ruta_int8), providers=["CPUExecutionProvider"])

    print("4. Evaluando los dos modelos con las mismas 800 reseñas...")
    df_test = split_de_prueba()
    y = df_test["etiqueta"].to_numpy(dtype=int)
    conjuntos = {
        "test": df_test["texto"].tolist(),
        # La API siempre quita las tildes antes de predecir: este es el caso real.
        "test_sin_tildes": df_test["texto"].map(quitar_tildes).tolist(),
    }

    resultados = {"original": {}, "liviano": {}}
    for nombre, textos in conjuntos.items():
        s_original = predecir_original(textos, tokenizador, bert, cabeza)
        s_liviano = predecir_liviano(textos, tokenizador_liviano, sesion, cabeza)
        resultados["original"][nombre] = metricas(y, s_original)
        resultados["liviano"][nombre] = metricas(y, s_liviano)
        coincide = float(np.mean((s_original >= 0.5) == (s_liviano >= 0.5)))
        resultados.setdefault("coincidencia_entre_modelos", {})[nombre] = coincide
        print(f"   {nombre}: F1 original {resultados['original'][nombre]['f1_score']:.4f}"
              f" | F1 liviano {resultados['liviano'][nombre]['f1_score']:.4f}"
              f" | misma respuesta en {coincide:.1%}")

    muestra = conjuntos["test_sin_tildes"][:100]
    resultados["latencia_ms_por_resena"] = {
        "original": latencia_ms(lambda t: predecir_original(t, tokenizador, bert, cabeza), muestra),
        "liviano": latencia_ms(lambda t: predecir_liviano(t, tokenizador_liviano, sesion, cabeza), muestra),
    }
    resultados["tamano_mb"] = {
        "encoder_fp32": round(tamano_fp32 / 1e6, 1),
        "encoder_int8": round(ruta_int8.stat().st_size / 1e6, 1),
    }
    resultados.update({
        "modelo_base": MODEL_NAME,
        "cuantizacion": "dinámica int8 (onnxruntime.quantization.quantize_dynamic)",
        "n_test": len(df_test),
        "exportado_en": datetime.now().isoformat(timespec="seconds"),
    })

    (SALIDA / "metrics.json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"5. Listo. Latencia: {resultados['latencia_ms_por_resena']} ms | Tamaño: {resultados['tamano_mb']} MB")


if __name__ == "__main__":
    main()

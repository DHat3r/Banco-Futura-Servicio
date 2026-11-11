"""Tests del modelo de sentimiento liviano (BETO int8 + cabeza en NumPy), sin cargar el encoder real."""

import csv

import numpy as np
import pytest

import app.db as db
from app import sentiment_liviano, sentiment_model
from app.sentiment_model import ModeloNoEntrenadoError


def _primer_customer_id() -> int:
    """Lee el primer CustomerID de la muestra versionada, directamente desde el CSV."""
    with open(db.SAMPLE_CSV, newline="", encoding="utf-8") as archivo:
        primera_fila = next(csv.DictReader(archivo))
    return int(primera_fila["CustomerID"])


def test_aplicar_cabeza_calcula_relu_relu_sigmoide_a_mano():
    # Red 4 -> 2 -> 2 -> 1. Con cls = [1, 0, 0.5, 0]:
    #   capa 1: [1 + 0.25, -1] + [0, 0.5] = [1.25, -0.5]  -> relu -> [1.25, 0]   (la relu corta el -0.5)
    #   capa 2: [1.25*2, 1.25*1] + [0.5, -1] = [3.0, 0.25] -> relu -> [3.0, 0.25]
    #   salida: 3.0*0.5 + 0.25*1 - 1 = 0.75                 -> sigmoide(0.75)
    cabeza = {
        "W1": np.array([[1.0, -1.0], [0.0, 0.0], [0.5, 0.0], [0.0, 0.0]]),
        "b1": np.array([0.0, 0.5]),
        "W2": np.array([[2.0, 1.0], [-3.0, 4.0]]),
        "b2": np.array([0.5, -1.0]),
        "W3": np.array([[0.5], [1.0]]),
        "b3": np.array([-1.0]),
    }
    cls = np.array([[1.0, 0.0, 0.5, 0.0]])

    resultado = sentiment_liviano.aplicar_cabeza(cabeza, cls)

    esperado = np.array([1 / (1 + np.exp(-0.75))])
    np.testing.assert_allclose(resultado, esperado)


def test_cargar_modelo_falla_si_el_liviano_esta_desactivado(monkeypatch):
    monkeypatch.delenv("ENABLE_SENTIMENT_LIVIANO", raising=False)

    with pytest.raises(ModeloNoEntrenadoError):
        sentiment_liviano.cargar_modelo()


def test_cargar_modelo_falla_si_falta_el_encoder(monkeypatch, tmp_path):
    monkeypatch.setenv("ENABLE_SENTIMENT_LIVIANO", "true")
    monkeypatch.setattr(sentiment_liviano, "ENCODER_PATH", tmp_path / "encoder_int8.onnx")

    with pytest.raises(ModeloNoEntrenadoError) as error:
        sentiment_liviano.cargar_modelo()

    assert "encoder_int8.onnx" in str(error.value)


def test_predecir_sentimiento_sin_modelo_cargado_falla(monkeypatch):
    monkeypatch.setattr(sentiment_liviano, "_sesion", None)

    with pytest.raises(ModeloNoEntrenadoError):
        sentiment_liviano.predecir_sentimiento("hola mundo")


def test_endpoint_liviano_responde_503_si_el_modelo_no_esta_listo(client, monkeypatch):
    monkeypatch.setattr(sentiment_liviano, "modelo_listo", lambda: False)

    respuesta = client.post("/predict/sentimiento/liviano", json={"texto": "hola mundo"})

    assert respuesta.status_code == 503


def test_endpoint_liviano_devuelve_la_prediccion_del_modelo(client, monkeypatch):
    prediccion = {"texto": "Excelente servicio", "sentimiento": "positivo", "score": 0.9123}
    monkeypatch.setattr(sentiment_liviano, "modelo_listo", lambda: True)
    monkeypatch.setattr(sentiment_liviano, "predecir_sentimiento", lambda texto: prediccion)

    respuesta = client.post("/predict/sentimiento/liviano", json={"texto": "Excelente servicio"})

    assert respuesta.status_code == 200
    assert respuesta.json() == prediccion


def test_health_incluye_sentiment_liviano_listo(client):
    respuesta = client.get("/health")

    assert respuesta.status_code == 200
    assert "sentiment_liviano_listo" in respuesta.json()


def test_mensajes_usan_el_liviano_si_el_completo_no_esta_listo(client, monkeypatch):
    prediccion = {"texto": "mensaje", "sentimiento": "negativo", "score": 0.1234}
    monkeypatch.setattr(sentiment_model, "modelo_listo", lambda: False)
    monkeypatch.setattr(sentiment_liviano, "modelo_listo", lambda: True)
    monkeypatch.setattr(sentiment_liviano, "predecir_sentimiento", lambda texto: prediccion)
    id_cliente = _primer_customer_id()
    client.post(f"/cliente/{id_cliente}/mensaje", json={"texto": "Quiero presentar un reclamo"})

    respuesta = client.get(f"/cliente/{id_cliente}/mensajes")

    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert len(datos["analisis"]) == len(datos["mensajes"]) >= 1
    assert all(analisis == prediccion for analisis in datos["analisis"])

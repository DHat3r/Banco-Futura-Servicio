"""Tests de integración de la API con TestClient (sin levantar uvicorn)."""

import csv

import app.db as db


def _primer_customer_id() -> int:
    """Lee el primer CustomerID de la muestra versionada, directamente desde el CSV."""
    with open(db.SAMPLE_CSV, newline="", encoding="utf-8") as archivo:
        primera_fila = next(csv.DictReader(archivo))
    return int(primera_fila["CustomerID"])


def test_health_reporta_churn_listo_y_sentimiento_apagado(client):
    respuesta = client.get("/health")

    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["churn_model_listo"] is True
    assert datos["sentiment_model_listo"] is False


def test_metadata_cliente_devuelve_las_opciones_de_los_selects(client):
    respuesta = client.get("/metadata/cliente")

    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert isinstance(datos["gender_options"], list)
    assert isinstance(datos["subscription_options"], list)
    assert isinstance(datos["contract_options"], list)


def test_predict_churn_con_ejemplo_devuelve_prediccion_valida(client, cliente_ejemplo):
    respuesta = client.post("/predict/churn", json=cliente_ejemplo)

    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert 0 <= datos["prob_churn"] <= 1
    assert datos["nivel_riesgo"] in {"BAJO", "MEDIO", "ALTO"}
    assert len(datos["factores_principales"]) == 4


def test_predict_churn_rechaza_edad_negativa(client, cliente_ejemplo):
    cliente_ejemplo["Age"] = -1

    respuesta = client.post("/predict/churn", json=cliente_ejemplo)

    assert respuesta.status_code == 422


def test_predict_churn_rechaza_genero_fuera_de_opciones(client, cliente_ejemplo):
    cliente_ejemplo["Gender"] = "Otro"

    respuesta = client.post("/predict/churn", json=cliente_ejemplo)

    assert respuesta.status_code == 422


def test_predict_sentimiento_responde_503_si_el_modelo_esta_apagado(client):
    respuesta = client.post("/predict/sentimiento", json={"texto": "hola mundo"})

    assert respuesta.status_code == 503


def test_cliente_churn_existente_devuelve_prediccion(client):
    respuesta = client.get(f"/cliente/{_primer_customer_id()}/churn")

    assert respuesta.status_code == 200


def test_cliente_churn_inexistente_devuelve_404(client):
    respuesta = client.get("/cliente/999999999/churn")

    assert respuesta.status_code == 404


def test_mensaje_se_guarda_pero_listar_con_sentimiento_responde_503(client):
    id_cliente = _primer_customer_id()

    respuesta_post = client.post(
        f"/cliente/{id_cliente}/mensaje", json={"texto": "Quiero presentar un reclamo"}
    )
    respuesta_get = client.get(f"/cliente/{id_cliente}/mensajes")

    assert respuesta_post.status_code == 200
    assert respuesta_post.json()["ok"] is True
    assert respuesta_get.status_code == 503

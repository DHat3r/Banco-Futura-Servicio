"""Tests de la cartera paginada (GET /clientes) contra la muestra versionada."""

import csv
import math

import app.db as db

CLAVES_CLIENTE = {"id", "edad", "suscripcion", "contrato", "atraso_pago", "gasto_total"}


def _ids_de_la_muestra() -> list[int]:
    """Todos los CustomerID de la muestra versionada, ordenados, leídos directamente desde el CSV."""
    with open(db.SAMPLE_CSV, newline="", encoding="utf-8") as archivo:
        ids = {int(fila["CustomerID"]) for fila in csv.DictReader(archivo) if fila["CustomerID"]}
    return sorted(ids)


def _ids_de(respuesta_json: dict) -> list[int]:
    return [cliente["id"] for cliente in respuesta_json["clientes"]]


def test_primera_pagina_trae_10_clientes_ordenados_y_totales_correctos(client):
    esperados = _ids_de_la_muestra()

    respuesta = client.get("/clientes?pagina=1")

    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert _ids_de(datos) == esperados[:10]
    assert _ids_de(datos) == sorted(_ids_de(datos))
    assert datos["total"] == len(esperados)
    assert datos["total_paginas"] == math.ceil(len(esperados) / 10)


def test_cada_cliente_trae_solo_las_claves_de_la_cartera_sin_churn(client):
    respuesta = client.get("/clientes?pagina=1")

    for cliente in respuesta.json()["clientes"]:
        assert set(cliente.keys()) == CLAVES_CLIENTE
        assert "Churn" not in cliente
        assert "churn" not in cliente


def test_segunda_pagina_no_repite_y_continua_el_orden(client):
    esperados = _ids_de_la_muestra()

    primera = client.get("/clientes?pagina=1").json()
    segunda = client.get("/clientes?pagina=2").json()

    assert set(_ids_de(primera)).isdisjoint(_ids_de(segunda))
    assert _ids_de(segunda) == esperados[10:20]


def test_pagina_mas_alla_del_total_devuelve_lista_vacia_con_200(client):
    total_paginas = client.get("/clientes?pagina=1").json()["total_paginas"]

    respuesta = client.get(f"/clientes?pagina={total_paginas + 1}")

    assert respuesta.status_code == 200
    assert respuesta.json()["clientes"] == []


def test_filtro_por_prefijo_de_id_devuelve_solo_ids_que_empiezan_con_el(client):
    esperados = _ids_de_la_muestra()
    prefijo = str(esperados[0])[:2]
    ids_con_prefijo = [i for i in esperados if str(i).startswith(prefijo)]

    respuesta = client.get(f"/clientes?pagina=1&id={prefijo}")

    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert all(str(id_cliente).startswith(prefijo) for id_cliente in _ids_de(datos))
    assert datos["total"] == len(ids_con_prefijo)


def test_filtro_sin_coincidencias_devuelve_total_cero_y_lista_vacia(client):
    # Ningún ID entero se escribe con un 0 al principio, así que el prefijo "0" no coincide con nada.
    respuesta = client.get("/clientes?pagina=1&id=0")

    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["total"] == 0
    assert datos["total_paginas"] == 1
    assert datos["clientes"] == []


def test_pagina_cero_responde_422(client):
    respuesta = client.get("/clientes?pagina=0")

    assert respuesta.status_code == 422


def test_filtro_con_letras_responde_422(client):
    respuesta = client.get("/clientes?id=abc")

    assert respuesta.status_code == 422

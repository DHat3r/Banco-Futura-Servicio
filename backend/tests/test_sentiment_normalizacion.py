"""Tests de la normalización de texto del modelo de sentimiento (no cargan BETO)."""

import pytest

from app.sentiment_model import MODEL_NAME, quitar_tildes


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("pésimo servicio", "pesimo servicio"),
        ("PÉSIMO SERVICIO", "PESIMO SERVICIO"),
        ("atención rápida", "atencion rapida"),
        ("pingüino", "pinguino"),
        ("sin tildes", "sin tildes"),
    ],
)
def test_quitar_tildes(texto, esperado):
    assert quitar_tildes(texto) == esperado


def test_quitar_tildes_conserva_la_enie():
    # "año" no puede pasar a "ano": cambia el significado.
    assert quitar_tildes("Llevo un año esperando, Ñuñoa") == "Llevo un año esperando, Ñuñoa"


def test_el_modelo_base_ignora_mayusculas():
    # La versión cased trataba "PESIMO" como pedazos sin sentido; la uncased no.
    assert MODEL_NAME.endswith("-uncased")

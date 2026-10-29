"""Tests del etiquetado y la limpieza del dataset de reseñas (sin llamar a Google Play)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from review_scraper import construir_dataset, etiquetar  # noqa: E402


@pytest.mark.parametrize(
    ("estrellas", "esperado"),
    [(1, 0), (2, 0), (3, None), (4, 1), (5, 1)],
)
def test_etiquetar_por_estrellas(estrellas, esperado):
    assert etiquetar(estrellas) == esperado


def test_construir_dataset_descarta_tres_estrellas_duplicados_y_textos_cortos():
    filas = [
        {"texto": "Pésima app, se cae siempre", "estrellas": 1, "app": "a"},
        {"texto": "Pésima app, se cae siempre", "estrellas": 1, "app": "b"},  # duplicado
        {"texto": "Excelente atención del banco", "estrellas": 5, "app": "a"},
        {"texto": "Está bien, nada especial", "estrellas": 3, "app": "a"},    # neutra
        {"texto": "malo", "estrellas": 1, "app": "a"},                        # muy corta
    ]

    df = construir_dataset(filas)

    assert list(df["etiqueta"]) == [0, 1]
    assert df["etiqueta"].dtype == int

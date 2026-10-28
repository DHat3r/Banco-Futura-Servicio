"""Fixtures compartidas por los tests de la API de churn."""

import copy
import os

# Sentimiento desactivado en tests: no se descarga BETO ni se carga TensorFlow.
os.environ["ENABLE_SENTIMENT"] = "false"

import pytest
from fastapi.testclient import TestClient

import app.db as db
from app.main import app as aplicacion
from app.schemas import ClienteInput


@pytest.fixture(scope="module")
def client():
    """Cliente HTTP con el lifespan ejecutado, forzando la muestra versionada como base de clientes.

    Apuntamos FULL_CSV a un archivo inexistente para que el test sea determinista,
    también en máquinas donde existe el CSV completo de Kaggle.
    """
    csv_original = db.FULL_CSV
    db.FULL_CSV = db.DATA_DIR / "no_existe.csv"
    try:
        with TestClient(aplicacion) as c:
            yield c
    finally:
        db.FULL_CSV = csv_original


@pytest.fixture
def cliente_ejemplo() -> dict:
    """Copia del ejemplo del schema: cada test puede modificarla sin afectar a los demás."""
    return copy.deepcopy(ClienteInput.model_config["json_schema_extra"]["example"])

"""
Acceso a datos de clientes y mensajes (almacenamiento en memoria).

Los clientes se cargan desde el CSV completo si existe; si no, desde la muestra
versionada en git (`sample_clientes.csv`). Los mensajes solo viven en memoria.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.features import rename_raw_columns

_clientes: dict = {}
_mensajes: dict[int, list[str]] = {}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FULL_CSV = DATA_DIR / "customer_churn_dataset-training-master.csv"
SAMPLE_CSV = DATA_DIR / "sample_clientes.csv"


def cargar_clientes():
    global _clientes
    csv = FULL_CSV if FULL_CSV.exists() else SAMPLE_CSV
    df = pd.read_csv(csv)
    df = rename_raw_columns(df)
    df = df.dropna(subset=["CustomerID"])
    # Sin la fila vacía, las columnas vuelven a ser enteras (62 y no 62.0).
    df = df.convert_dtypes()
    _clientes = df.set_index("CustomerID").to_dict(orient="index")

def buscar_cliente(id_cliente: int) -> dict | None:
    return _clientes.get(id_cliente)

def guardar_mensaje(id_cliente: int, texto: str):
    _mensajes.setdefault(id_cliente, []).append(texto)

def obtener_mensajes(id_cliente: int) -> list[str]:
    return _mensajes.get(id_cliente, [])

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
_ids_ordenados: list[int] = []  # se ordenan una vez al cargar, no en cada página
_mensajes: dict[int, list[str]] = {}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FULL_CSV = DATA_DIR / "customer_churn_dataset-training-master.csv"
SAMPLE_CSV = DATA_DIR / "sample_clientes.csv"


def cargar_clientes():
    global _clientes, _ids_ordenados
    csv = FULL_CSV if FULL_CSV.exists() else SAMPLE_CSV
    df = pd.read_csv(csv)
    df = rename_raw_columns(df)
    df = df.dropna(subset=["CustomerID"])
    # Sin la fila vacía, las columnas vuelven a ser enteras (62 y no 62.0).
    df = df.convert_dtypes()
    _clientes = df.set_index("CustomerID").to_dict(orient="index")
    _ids_ordenados = sorted(int(id_cliente) for id_cliente in _clientes)

def buscar_cliente(id_cliente: int) -> dict | None:
    return _clientes.get(id_cliente)

def listar_clientes(pagina: int, por_pagina: int, filtro_id: str = "") -> tuple[list[dict], int]:
    """Una página de la cartera, ordenada por ID, y el total de clientes que cumplen el filtro.

    El filtro busca los IDs que empiezan con ese texto. No incluye la columna Churn:
    es la respuesta real y no debe aparecer junto a la predicción.
    """
    ids = [i for i in _ids_ordenados if str(i).startswith(filtro_id)] if filtro_id else _ids_ordenados
    inicio = (pagina - 1) * por_pagina
    clientes = [
        {
            "id": id_cliente,
            "edad": int(_clientes[id_cliente]["Age"]),
            "suscripcion": str(_clientes[id_cliente]["subscription_type"]),
            "contrato": str(_clientes[id_cliente]["contract_length"]),
            "atraso_pago": int(_clientes[id_cliente]["payment_delay"]),
            "gasto_total": float(_clientes[id_cliente]["total_spend"]),
        }
        for id_cliente in ids[inicio:inicio + por_pagina]
    ]
    return clientes, len(ids)

def guardar_mensaje(id_cliente: int, texto: str):
    _mensajes.setdefault(id_cliente, []).append(texto)

def obtener_mensajes(id_cliente: int) -> list[str]:
    return _mensajes.get(id_cliente, [])

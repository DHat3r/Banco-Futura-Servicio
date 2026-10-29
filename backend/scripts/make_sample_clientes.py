"""
Genera una muestra liviana de clientes (300 filas) para el repositorio.

El CSV completo pesa ~23 MB y está ignorado por git, así que la API no lo tiene
al clonar el proyecto. Esta muestra sí se versiona, para que la búsqueda por
ID (`app/db.py`) funcione sin el CSV completo.

Uso (desde la carpeta `backend/`):

    python scripts/make_sample_clientes.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FULL_CSV = DATA_DIR / "customer_churn_dataset-training-master.csv"
SAMPLE_CSV = DATA_DIR / "sample_clientes.csv"

N_MUESTRA = 300
RANDOM_STATE = 42


def main() -> None:
    df = pd.read_csv(FULL_CSV)
    # Sin CustomerID no se puede buscar el cliente, así que esas filas se descartan.
    df = df.dropna(subset=["CustomerID"])
    # Sin la fila vacía, las columnas vuelven a ser enteras (62 y no 62.0).
    df = df.convert_dtypes()

    muestra = df.sample(n=N_MUESTRA, random_state=RANDOM_STATE)
    muestra = muestra.sort_values("CustomerID")
    muestra.to_csv(SAMPLE_CSV, index=False)
    print(f"Muestra guardada en {SAMPLE_CSV} ({len(muestra)} clientes)")


if __name__ == "__main__":
    main()

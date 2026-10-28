import pandas as pd
from pathlib import Path

_clientes: dict = {}
_mensajes: dict[int, list[str]] = {}

def cargar_clientes():
    global _clientes
    csv = Path(__file__).parent.parent / "data" / "customer_churn_dataset-training-master.csv"
    df = pd.read_csv(csv)
    # Renombrar columnas igual que el pipeline
    df = df.rename(columns={
        "Usage Frequency": "usage_frequency",
        "Support Calls": "support_calls",
        "Payment Delay": "payment_delay",
        "Subscription Type": "subscription_type",
        "Contract Length": "contract_length",
        "Total Spend": "total_spend",
        "Last Interaction": "last_interaction",
    })
    _clientes = df.set_index("CustomerID").to_dict(orient="index")

def buscar_cliente(id_cliente: int) -> dict | None:
    return _clientes.get(id_cliente)

def guardar_mensaje(id_cliente: int, texto: str):
    _mensajes.setdefault(id_cliente, []).append(texto)

def obtener_mensajes(id_cliente: int) -> list[str]:
    return _mensajes.get(id_cliente, [])
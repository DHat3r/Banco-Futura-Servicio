"""Esquemas Pydantic — request/response de la API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.features import CONTRACT_OPTIONS, GENDER_OPTIONS, SUBSCRIPTION_OPTIONS


class ClienteInput(BaseModel):
    """Datos crudos de un cliente, tal como los ingresaría un ejecutivo en el frontend."""

    Age: int = Field(..., ge=0, le=100, description="Edad del cliente")
    Tenure: int = Field(..., ge=0, le=120, description="Antigüedad en meses")
    usage_frequency: int = Field(..., ge=0, le=60, description="Frecuencia de uso mensual")
    support_calls: int = Field(..., ge=0, le=50, description="Llamadas a soporte")
    payment_delay: int = Field(..., ge=0, le=90, description="Días de atraso en pagos")
    total_spend: float = Field(..., ge=0, description="Gasto total acumulado")
    last_interaction: int = Field(..., ge=0, le=365, description="Días desde la última interacción")
    Gender: Literal[tuple(GENDER_OPTIONS)]  # type: ignore[valid-type]
    subscription_type: Literal[tuple(SUBSCRIPTION_OPTIONS)]  # type: ignore[valid-type]
    contract_length: Literal[tuple(CONTRACT_OPTIONS)]  # type: ignore[valid-type]

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 42,
                "Tenure": 5,
                "usage_frequency": 8,
                "support_calls": 6,
                "payment_delay": 22,
                "total_spend": 480.0,
                "last_interaction": 35,
                "Gender": "Female",
                "subscription_type": "Basic",
                "contract_length": "Monthly",
            }
        }


class FactorRiesgo(BaseModel):
    feature: str
    feature_label: str
    impacto: float = Field(..., description="Contribución SHAP (positivo = empuja hacia churn)")
    valor_cliente: str


class ChurnPrediction(BaseModel):
    prob_churn: float
    nivel_riesgo: Literal["BAJO", "MEDIO", "ALTO"]
    factores_principales: list[FactorRiesgo]


class SentimientoInput(BaseModel):
    texto: str = Field(..., min_length=3, max_length=2000, description="Mensaje o reclamo del cliente")

    class Config:
        json_schema_extra = {"example": {"texto": "Llevo dos semanas esperando que me respondan el reclamo"}}


class SentimientoPrediction(BaseModel):
    texto: str
    sentimiento: Literal["positivo", "negativo"]
    score: float = Field(..., description="Probabilidad de sentimiento positivo (0 a 1)")

class MensajeInput(BaseModel):
    texto: str = Field(..., min_length=3, max_length=2000)

class MensajesSentimientoResponse(BaseModel):
    id_cliente: int
    mensajes: list[str]
    analisis: list[SentimientoPrediction]
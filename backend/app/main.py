"""
API — Banco Futura · Churn & Sentimiento

Dos modelos independientes detrás de un mismo API gateway:
  - POST /predict/churn         -> XGBoost, predicción de abandono de clientes
  - POST /predict/sentimiento   -> BETO, sentimiento de un reclamo/mensaje

Local dev:
    uvicorn app.main:app --reload --port 8000

Documentación interactiva una vez levantada: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app import churn_model, sentiment_model
from app import db as db_module
from app.db import buscar_cliente, cargar_clientes, guardar_mensaje, obtener_mensajes
from app.features import CONTRACT_OPTIONS, GENDER_OPTIONS, SUBSCRIPTION_OPTIONS
from app.schemas import (
    ChurnPrediction,
    ClienteInput,
    MensajeInput,
    MensajesSentimientoResponse,
    SentimientoInput,
    SentimientoPrediction,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("banco_futura_api")


def cargar_modelos() -> None:
    try:
        churn_model.cargar_modelo()
        logger.info("Modelo de churn (XGBoost) cargado correctamente.")
    except churn_model.ModeloNoEntrenadoError as e:
        logger.warning("Churn no disponible todavía: %s", e)

    try:
        sentiment_model.cargar_modelo()
        logger.info("Modelo de sentimiento (BETO) cargado correctamente.")
    except (sentiment_model.ModeloNoEntrenadoError, ImportError) as e:
        logger.warning("Sentimiento no disponible todavía: %s", e)

    try:
        cargar_clientes()
        logger.info("Base de clientes cargada: %d registros. Primeros IDs: %s",
                    len(db_module._clientes), list(db_module._clientes.keys())[:5])
    except Exception as e:
        logger.error("Error cargando clientes: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cargar_modelos()
    yield


app = FastAPI(
    title="Banco Futura · Churn & Sentimiento API",
    description="Scoring de riesgo de abandono (XGBoost) y análisis de sentimiento de reclamos (BETO).",
    version="1.0.0",
    lifespan=lifespan,
)

# Orígenes permitidos para CORS, separados por coma en la variable de entorno CORS_ORIGINS.
# Por defecto solo Vite en local (5173). En producción, define CORS_ORIGINS con el dominio real del frontend.
CORS_ORIGINS = [
    origen.strip()
    for origen in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
    if origen.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "churn_model_listo": churn_model.modelo_listo(),
        "sentiment_model_listo": sentiment_model.modelo_listo(),
    }


@app.get("/metadata/cliente")
def metadata_cliente() -> dict:
    """Opciones válidas para los <select> del frontend, sacadas de una sola fuente de verdad."""
    return {
        "gender_options": GENDER_OPTIONS,
        "subscription_options": SUBSCRIPTION_OPTIONS,
        "contract_options": CONTRACT_OPTIONS,
    }


@app.post("/predict/churn", response_model=ChurnPrediction)
def predict_churn(cliente: ClienteInput) -> dict:
    if not churn_model.modelo_listo():
        raise HTTPException(
            status_code=503,
            detail="El modelo de churn no está entrenado todavía. Corre scripts/train_xgboost.py",
        )
    try:
        return churn_model.predecir_churn(cliente.model_dump())
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error prediciendo churn")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/sentimiento", response_model=SentimientoPrediction)
def predict_sentimiento(payload: SentimientoInput) -> dict:
    if not sentiment_model.modelo_listo():
        raise HTTPException(
            status_code=503,
            detail="El modelo de sentimiento no está entrenado todavía. Corre scripts/train_sentiment.py",
        )
    try:
        return sentiment_model.predecir_sentimiento(payload.texto)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error prediciendo sentimiento")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.get("/cliente/{id_cliente}/churn", response_model=ChurnPrediction)
def churn_por_id(id_cliente: int):
    cliente = buscar_cliente(id_cliente)
    if not cliente:
        raise HTTPException(status_code=404, detail=f"Cliente {id_cliente} no encontrado")
    if not churn_model.modelo_listo():
        raise HTTPException(status_code=503, detail="Modelo de churn no disponible")
    return churn_model.predecir_churn(cliente)

@app.post("/cliente/{id_cliente}/mensaje")
def agregar_mensaje(id_cliente: int, payload: MensajeInput):
    if not buscar_cliente(id_cliente):
        raise HTTPException(status_code=404, detail=f"Cliente {id_cliente} no encontrado")
    guardar_mensaje(id_cliente, payload.texto)
    return {"ok": True, "total_mensajes": len(obtener_mensajes(id_cliente))}

@app.get("/cliente/{id_cliente}/mensajes", response_model=MensajesSentimientoResponse)
def mensajes_con_sentimiento(id_cliente: int):
    if not buscar_cliente(id_cliente):
        raise HTTPException(status_code=404, detail=f"Cliente {id_cliente} no encontrado")
    if not sentiment_model.modelo_listo():
        raise HTTPException(status_code=503, detail="Modelo de sentimiento no disponible")
    mensajes = obtener_mensajes(id_cliente)
    analisis = [sentiment_model.predecir_sentimiento(m) for m in mensajes]
    return {"id_cliente": id_cliente, "mensajes": mensajes, "analisis": analisis}
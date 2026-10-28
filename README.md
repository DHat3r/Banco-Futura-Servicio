# Banco Futura · Predicción de churn bancario

> Plataforma de extremo a extremo que predice qué clientes de un banco están por irse y prioriza a quién contactar, combinando un modelo de churn (XGBoost) con el sentimiento de sus reclamos (BETO).

## Stack

| Capa | Tecnología |
|------|------------|
| Datos y pipeline | Databricks (Free Edition), Lakeflow Pipeline, Delta Lake, arquitectura medallion |
| Modelos | XGBoost + SHAP, Spark MLlib, BETO (BERT en español) |
| Backend | FastAPI, Pydantic |
| Frontend | React + Vite |

## Arquitectura

```
Kaggle CSV ──► databricks/   Bronze → Silver → Gold  ──► modelo A (XGBoost) / modelo B (Spark MLlib)
                                                              │
notebooks/  (exploración: BERT, primeras pruebas)             ▼
                                              backend/  API FastAPI ──► frontend/  React
                                                (churn + sentimiento)
```

| Carpeta | Qué contiene |
|---------|--------------|
| `notebooks/` | Exploración inicial: XGBoost y BERT aplicados al churn |
| `databricks/` | Pipeline medallion y los dos modelos, orquestados por un Job ([detalles](databricks/README.md)) |
| `backend/` | API REST que sirve el modelo de churn y el de sentimiento |
| `frontend/` | Interfaz que junta ambas señales para priorizar clientes |
| `docs/` | Documentación de la entrega API + frontend ([detalles](docs/entrega-api-frontend.md)) |

## Cómo ejecutarlo

Backend (el modelo de churn ya viene entrenado en `backend/artifacts/`):

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # en Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Swagger en `http://localhost:8000/docs`.

Frontend:

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

La interfaz queda en `http://localhost:5173`.

Para reentrenar, descarga los datos según [`backend/data/README.md`](backend/data/README.md) y corre `python scripts/train_xgboost.py`. El modelo de sentimiento se entrena con `python scripts/train_sentiment.py` (descarga BETO, ~440 MB).

## Decisiones técnicas

- **Dos modelos separados y no uno solo:** los datos de churn son tabulares y XGBoost los aprovecha mejor que BERT. BETO aporta una señal distinta, el sentimiento de un reclamo, y el frontend combina ambas.
- **Features con las mismas reglas en todos lados:** `backend/app/features.py` replica la lógica de `databricks/pipeline/pipeline_02_silver.py`, para que la API calcule lo mismo que el pipeline.
- **Volumen de Unity Catalog en vez de descargar de Kaggle en vivo:** Databricks Free Edition restringe la salida a internet.

## Resultados

Modelo de churn (XGBoost), evaluado sobre 101.042 registros de test (`backend/artifacts/churn_model/metrics.json`):

| Métrica | Valor |
|---------|-------|
| AUC-ROC | 0,953 |
| F1 | 0,939 |
| Precisión | 0,898 |
| Recall | 0,984 |

## Qué mejoraría con más tiempo

- `docker compose up` para levantar backend y frontend con un solo comando.
- Tests de la API y del cálculo de features.
- CI con GitHub Actions.
- Cargar el modelo desde MLflow / Unity Catalog en lugar del `.joblib` local.
- Restringir CORS (hoy está abierto a cualquier origen).

## Historia del repositorio

Este repo une tres repos anteriores, conservando sus commits: `bank-churn-prediction` (notebooks, mayo 2026), `databricks` (pipeline, junio 2026) y `Banco-futura` (API + frontend, julio 2026). Al fusionarlos se eliminaron del historial los datasets, `node_modules` y otros archivos pesados.

## Autor

Leandro · [GitHub](https://github.com/DHat3r)

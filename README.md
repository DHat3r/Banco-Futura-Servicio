# Banco Futura · Predicción de churn bancario

[![CI](https://github.com/DHat3r/Banco-Futura-Servicio/actions/workflows/ci.yml/badge.svg)](https://github.com/DHat3r/Banco-Futura-Servicio/actions/workflows/ci.yml)

> Plataforma de extremo a extremo que predice qué clientes de un banco están por irse y prioriza a quién contactar, combinando un modelo de churn (XGBoost) con el sentimiento de sus reclamos (BETO).

## Stack

| Capa | Tecnología |
|------|------------|
| Datos y pipeline | Databricks (Free Edition), Lakeflow Pipeline, Delta Lake, arquitectura medallion |
| Modelos | XGBoost + SHAP, Spark MLlib, BETO (BERT en español) |
| Backend | FastAPI, Pydantic |
| Frontend | React + Vite |
| Infraestructura | Docker Compose, nginx, GitHub Actions |

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

### Con Docker (recomendado)

```bash
git clone https://github.com/DHat3r/Banco-Futura-Servicio.git
cd Banco-Futura-Servicio
cp .env.example .env
docker compose up --build
```

En `.env` se configura `INSTALL_SENTIMENT` (activar el modelo de sentimiento) y `API_PORT` (puerto de la API). También se pueden pasar como variables de entorno, como en los ejemplos de abajo.

La interfaz queda en `http://localhost:5173` (nginx reenvía `/api` a la API) y Swagger en `http://localhost:8000/docs`.

Si el puerto 8000 está ocupado, cambia el puerto de la API (Swagger queda en `http://localhost:18000/docs`):

```bash
API_PORT=18000 docker compose up --build
```

En PowerShell: `$env:API_PORT=18000; docker compose up --build`.

Para incluir el modelo de sentimiento (imagen pesada; descarga BETO, ~440 MB, en el primer arranque):

```bash
INSTALL_SENTIMENT=true docker compose up --build
```

### Sin Docker

Backend (el modelo de churn ya viene entrenado en `backend/artifacts/`):

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # en Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Swagger en `http://localhost:8000/docs`. El modelo de sentimiento necesita `pip install -r requirements-sentiment.txt` y `ENABLE_SENTIMENT=true`.

Frontend:

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

La interfaz queda en `http://localhost:5173`.

Para reentrenar, descarga los datos según [`backend/data/README.md`](backend/data/README.md) y corre `python scripts/train_xgboost.py`. El modelo de sentimiento se entrena con reseñas de apps de bancos: primero `python scripts/review_scraper.py` (descarga ~13 mil reseñas de Google Play) y luego `python scripts/train_sentiment.py` (descarga BETO, ~440 MB). Con Docker, para que el modelo quede en tu disco:

```bash
docker compose run --rm -v "${PWD}/backend/scripts:/app/scripts" -v "${PWD}/backend/artifacts:/app/artifacts" api python scripts/train_sentiment.py
```

## Tests

```bash
cd backend
pip install -r requirements-dev.txt
pytest -q
```

CI (GitHub Actions) ejecuta los tests del backend y el build del frontend en cada push y pull request.

## Decisiones técnicas

- **Dos modelos separados y no uno solo:** los datos de churn son tabulares y XGBoost los aprovecha mejor que BERT. BETO aporta una señal distinta, el sentimiento de un reclamo, y el frontend combina ambas.
- **Features con las mismas reglas en todos lados:** `backend/app/features.py` replica la lógica de `databricks/pipeline/pipeline_02_silver.py`, para que la API calcule lo mismo que el pipeline.
- **Volumen de Unity Catalog en vez de descargar de Kaggle en vivo:** Databricks Free Edition restringe la salida a internet.
- **Supervisión débil para el sentimiento:** en vez de etiquetar a mano, las reseñas de Google Play se etiquetan por sus estrellas (1-2 negativo, 4-5 positivo, 3 se descarta). Es ruidoso, pero con miles de ejemplos el ruido se compensa. Se usan 10 apps de bancos de Chile, España, México y Argentina, y solo se guarda el texto (sin datos del autor).

## Resultados

Modelo de churn (XGBoost), evaluado sobre 101.042 registros de test (`backend/artifacts/churn_model/metrics.json`):

| Métrica | Valor |
|---------|-------|
| AUC-ROC | 0,953 |
| F1 | 0,939 |
| Precisión | 0,898 |
| Recall | 0,984 |

Modelo de sentimiento (BETO uncased + red densa), entrenado con 3.200 reseñas balanceadas más sus copias sin tildes, y evaluado sobre 800 que no vio, sin tildes como las procesa la API (`backend/artifacts/sentiment_model/metrics.json`):

| Métrica | Valor |
|---------|-------|
| Accuracy | 0,895 |
| F1 | 0,886 |
| Precisión | 0,970 |
| Recall | 0,815 |

### Limitación encontrada y cómo se resolvió

Probando la interfaz, el reclamo *"me gustaria saber si me puedo desvincular a este banco DADO SU PESIMO SERVICIO"* salía **positivo**. Comparando variantes de la frase:

- El modelo original usaba BETO **cased**, que parte las palabras en MAYÚSCULAS en pedazos sin sentido (`PESIMO` → `PES ##IM ##O`). Los clientes enojados escriben justamente así.
- Al cambiar a BETO **uncased** se arreglaron las mayúsculas, pero las palabras **con** tilde empezaron a fallar (`pésimo servicio` → positivo), probablemente porque BETO uncased se preentrenó sin tildes.
- Solución: el entrenamiento agrega copias sin tildes de cada reseña (*data augmentation*, aplicada solo a train para evitar *data leakage*), y la API quita las tildes antes de predecir. Ahora `PESIMO SERVICIO`, `pésimo servicio` y `Pésimo servicio` dan el mismo resultado.

La función que quita las tildes (`quitar_tildes`) conserva la ñ, y es la misma en el entrenamiento y en la API para que el modelo vea en producción lo mismo que aprendió.

## Qué mejoraría con más tiempo

- Cargar el modelo desde MLflow / Unity Catalog en lugar del `.joblib` local.
- Persistir los mensajes de clientes en una base de datos (hoy viven en memoria).
- Migrar `class Config` de los schemas a `model_config` (Pydantic v2).

## Historia del repositorio

Este repo une tres repos anteriores, conservando sus commits: `bank-churn-prediction` (notebooks, mayo 2026), `databricks` (pipeline, junio 2026) y `Banco-futura` (API + frontend, julio 2026). Al fusionarlos se eliminaron del historial los datasets, `node_modules` y otros archivos pesados.

## Autor

Leandro · [GitHub](https://github.com/DHat3r)

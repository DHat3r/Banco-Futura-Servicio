# Banco Futura · Churn & Sentimiento — API + Frontend

Última entrega: cumple el requisito de **uso de API (REST)** con un gateway en
FastAPI que sirve dos modelos independientes, más un **frontend en React**
que los consume.

```
backend/
  app/
    main.py            ← FastAPI: /predict/churn, /predict/sentimiento, /health
    features.py         ← feature engineering (equivalente pandas de Silver+Gold)
    churn_model.py       ← carga XGBoost + SHAP, inferencia
    sentiment_model.py    ← carga BETO + cabeza entrenada, inferencia
    schemas.py            ← contratos Pydantic de la API
  scripts/
    train_xgboost.py    ← entrena el modelo de churn (YA CORRIDO, ver abajo)
    train_sentiment.py   ← entrena la cabeza de sentimiento sobre embeddings BETO
  artifacts/
    churn_model/         ← pipeline.joblib + metrics.json (modelo ya entrenado)
    sentiment_model/      ← se llena al correr train_sentiment.py
  data/                 ← los 2 CSV de Kaggle (train+test)
  requirements.txt

frontend/
  src/
    App.jsx, components/, api.js, styles.css
  package.json
```

## Por qué dos modelos por separado (y no uno solo)

El modelo de churn (XGBoost, datos tabulares) y el de sentimiento (BETO,
texto) **no se mezclan en un solo pipeline** — y es intencional. BERT/BETO
no tiene sentido como reemplazo del modelo de churn (forzar texto donde ya
hay datos tabulares perfectamente buenos no agrega nada; de hecho es la
razón por la que tu notebook `bert_churn_bancario` rinde peor que XGBoost
para ese problema). Donde BETO sí aporta es como **señal complementaria**:
sentimiento de un reclamo reciente. El frontend junta ambas señales
visualmente (cliente de riesgo ALTO + reclamo con sentimiento negativo =
prioridad de contacto), sin que los modelos se toquen entre sí. Por eso son
dos endpoints / dos "microservicios" lógicos detrás del mismo gateway.

## Cómo correrlo

### 1. Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # o el venv que ya uses
pip install -r requirements.txt
```

**El modelo de churn ya viene entrenado** (`artifacts/churn_model/pipeline.joblib`,
AUC-ROC ≈ 0.95 sobre una muestra de 60k filas). Si quieres reentrenarlo con
más datos o ajustar hiperparámetros:

```bash
python scripts/train_xgboost.py --full          # con los ~505k registros completos
python scripts/train_xgboost.py --sample 150000  # con una muestra más grande
```

**El modelo de sentimiento (BETO) necesita que lo entrenes tú**, porque
descarga los pesos de BETO desde Hugging Face (~440MB) y este entorno
donde te estoy ayudando no tiene salida a internet hacia ahí:

```bash
python scripts/train_sentiment.py
```

Después levanta la API:

```bash
uvicorn app.main:app --reload --port 8000
```

Docs interactivas en http://localhost:8000/docs

### 2. Frontend

```bash
cd frontend
npm install
cp .env.example .env     # ajusta VITE_API_URL si tu backend no está en localhost:8000
npm run dev
```

Abre http://localhost:5173

## Migrar a Databricks (próxima etapa)

Todo el código del modelo de churn está aislado en `app/churn_model.py`.
Cuando tengas el modelo registrado en MLflow / Unity Catalog en Databricks
(usando tu pipeline `databricks-master/`), lo único que cambia es la función
`cargar_modelo()` de ese archivo: en vez de `joblib.load(...)`, harías algo
como:

```python
import mlflow
modelo = mlflow.pyfunc.load_model("models:/churn_banco_futura/Production")
```

o, si prefieres consumir el modelo vía Databricks Model Serving en vez de
cargarlo en el proceso de la API, reemplazarías `predecir_churn()` por una
llamada HTTP al endpoint de serving. El resto de `main.py`, `schemas.py` y
todo el frontend queda intacto — por eso vale la pena mantener esa
separación desde ahora.

## Notas

- CORS está abierto (`allow_origins=["*"]`) para que el desarrollo local sea
  simple. Antes de "producción" (o de la demo, si el profesor pregunta),
  restringe esto al dominio real del frontend.
- Las features derivadas (`support_calls_per_tenure`, `spend_per_tenure`,
  `risk_score_flags`) se calculan en `app/features.py` con las MISMAS reglas
  que `pipeline_02_silver.py` de tu proyecto Databricks — no las dupliques
  ni las redefinas en el frontend.

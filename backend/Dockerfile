# Imagen de la API (FastAPI + modelo XGBoost). Base slim para mantener la imagen ligera.
FROM python:3.12-slim

WORKDIR /app

# Sin .pyc y con salida de logs sin buffer, para ver los logs al instante en Docker.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Por defecto no se instala el modelo de sentimiento (TensorFlow/torch pesan varios GB).
# Se activa con: INSTALL_SENTIMENT=true docker compose up --build
ARG INSTALL_SENTIMENT=false

# Primero solo los requirements: así Docker reutiliza la caché de esta capa
# mientras no cambien las dependencias, aunque cambie el código.
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Índice CPU de PyTorch para no bajar las librerías CUDA (varios GB innecesarios en un contenedor sin GPU).
RUN if [ "$INSTALL_SENTIMENT" = "true" ]; then \
        pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-sentiment.txt; \
    fi

# Código de la aplicación y artefactos del modelo.
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

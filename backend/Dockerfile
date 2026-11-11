# Imagen de la API (FastAPI + modelo XGBoost). Base slim para mantener la imagen ligera.
FROM python:3.12-slim

WORKDIR /app

# Sin .pyc y con salida de logs sin buffer, para ver los logs al instante en Docker.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Por defecto no se instala el modelo de sentimiento (TensorFlow/torch pesan varios GB).
# Se activa con: INSTALL_SENTIMENT=true docker compose up --build
ARG INSTALL_SENTIMENT=false

# Por defecto no se instala el modelo liviano (BETO en ONNX, cuantizado).
# Se activa con: INSTALL_SENTIMENT_LIVIANO=true docker compose up --build
ARG INSTALL_SENTIMENT_LIVIANO=false

# El encoder pesa ~110 MB y GitHub rechaza archivos de más de 100 MB en el repo, por eso se descarga de un release.
# El sha256 detecta descargas corruptas o archivos alterados: si no coincide, el build falla.
ARG ENCODER_LIVIANO_URL=https://github.com/DHat3r/Banco-Futura-Servicio/releases/download/sentimiento-liviano-v1/encoder_int8.onnx
ARG ENCODER_LIVIANO_SHA256=73c945c706a11b9df756a135b65ccdb963348249c1d4b9352241f82a671301f4

# El modelo liviano queda activo en el contenedor solo si se instaló.
ENV ENABLE_SENTIMENT_LIVIANO=${INSTALL_SENTIMENT_LIVIANO}

# Primero solo los requirements: así Docker reutiliza la caché de esta capa
# mientras no cambien las dependencias, aunque cambie el código.
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Índice CPU de PyTorch para no bajar las librerías CUDA (varios GB innecesarios en un contenedor sin GPU).
RUN if [ "$INSTALL_SENTIMENT" = "true" ]; then \
        pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-sentiment.txt; \
    fi

# Dependencias del modelo liviano (ONNX Runtime y tokenizador), sin las de PyTorch.
RUN if [ "$INSTALL_SENTIMENT_LIVIANO" = "true" ]; then \
        pip install --no-cache-dir -r requirements-liviano.txt; \
    fi

# Código de la aplicación y artefactos del modelo.
COPY . .

# Descarga el encoder del release y verifica su sha256 (Python, porque la imagen slim no trae curl ni sha256sum).
RUN if [ "$INSTALL_SENTIMENT_LIVIANO" = "true" ]; then \
        python -c "import hashlib, sys, urllib.request; url, esperado, destino = sys.argv[1], sys.argv[2], 'artifacts/sentiment_liviano/encoder_int8.onnx'; urllib.request.urlretrieve(url, destino); real = hashlib.sha256(open(destino, 'rb').read()).hexdigest(); sys.exit(0) if real == esperado else sys.exit('ERROR: el sha256 del encoder no coincide (esperado ' + esperado + ', obtenido ' + real + ')')" "$ENCODER_LIVIANO_URL" "$ENCODER_LIVIANO_SHA256"; \
    fi

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

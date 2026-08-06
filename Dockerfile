# BioVision AI — Cell Analysis Platform
# One image, two possible entry points:
#   - Streamlit UI (app/app.py)       -> default
#   - FastAPI REST layer (api/main.py) -> override CMD, see docker-compose.yml

FROM python:3.10-slim

# System libraries required by opencv-python and torch at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so this layer is cached
# unless requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# data/ holds the SQLite DB and generated PDF reports (see utils/database.py)
# models/ holds best.pt if you have a fine-tuned model
RUN mkdir -p data models

EXPOSE 8501 8000

# Health check targets the Streamlit port by default; the API service
# in docker-compose.yml overrides this with its own /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Default: Streamlit UI. Override CMD to run the API instead:
#   docker run <image> uvicorn api.main:app --host 0.0.0.0 --port 8000
CMD ["streamlit", "run", "app/app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0"]

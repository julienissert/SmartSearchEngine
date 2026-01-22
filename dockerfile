# 1. Base légère Debian
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 2. Installation des dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    pkg-config \
    libhdf5-dev \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    sqlite3 \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    export CFLAGS="-Wno-error=incompatible-pointer-types" && \
    pip install --no-cache-dir -r requirements.txt

# 4. Copie du code (SoC)
COPY src/ ./src/
COPY .env .

# 5. Création des dossiers nécessaires [cite: 1, 18, 19]
RUN mkdir -p raw-datasets computed-data/indexes computed-data/metadata logs

# Configuration du PYTHONPATH [cite: 27]
ENV PYTHONPATH=/app

# Désactivation du pre-download pendant le build pour éviter le Segfault sur Mac
# Les modèles seront téléchargés au premier RUN du conteneur
ENTRYPOINT ["python", "src/main.py"]
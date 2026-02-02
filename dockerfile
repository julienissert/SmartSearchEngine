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
    procps \ 
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Installation Stratégique des dépendances (Split)

# A. Mise à jour de pip
RUN pip install --no-cache-dir --upgrade pip

# B. Installation des "Lourds" (Torch & Nvidia) en premier
# On force l'installation ici pour gérer l'espace disque étape par étape.
# Note : On utilise l'index CUDA 12.1 pour être sûr d'avoir la version GPU compatible
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# C. Installation de Paddle (OCR) séparément
RUN pip install --no-cache-dir paddlepaddle-gpu==2.6.0 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html || \
    pip install --no-cache-dir paddlepaddle==2.6.0

# D. Installation du reste des requirements
COPY requirements.txt .
# On ignore les erreurs de dépendances pointer-types et on demande à pip d'ignorer
# torch/paddle s'ils sont déjà installés (grâce aux étapes précédentes)
RUN export CFLAGS="-Wno-error=incompatible-pointer-types" && \
    pip install --no-cache-dir -r requirements.txt

# --- AJOUT SÉCURITÉ THREADS ---
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

# 4. Copie du code (SoC)
COPY src/ ./src/
COPY .env .

# 5. Création explicite de l'arborescence
RUN mkdir -p \
    raw-datasets \
    computed-data/indexes \
    computed-data/metadata \
    logs \
    /root/.cache/huggingface \
    /root/.paddleocr

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "src/main.py"]
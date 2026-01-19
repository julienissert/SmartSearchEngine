# 1. Base avec support GPU NVIDIA
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 2. Installation groupée (Python 3.12 + Dépendances système)
RUN apt-get update && apt-get install -y \
    software-properties-common curl gcc libgl1-mesa-glx libglib2.0-0 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Installation de PIP pour Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

WORKDIR /app

# 3. Dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# --- PRÉ-CHARGEMENT DES MODÈLES (Zéro attente au lancement) ---
ARG OCR_LANG=latin
ENV OCR_LANG=$OCR_LANG

# Pré-téléchargement PaddleOCR et CLIP
RUN python3 -c "from paddleocr import PaddleOCR; PaddleOCR(lang='${OCR_LANG}', use_angle_cls=True, show_log=False)" \
    && python3 -c "from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor; \
       m='openai/clip-vit-base-patch32'; \
       CLIPModel.from_pretrained(m); CLIPTokenizer.from_pretrained(m); CLIPImageProcessor.from_pretrained(m)"

# 4. Copie du code et fichiers
COPY src/ ./src/
COPY .env .

# 5. Dossiers de données
RUN mkdir -p raw-datasets computed-data logs

ENTRYPOINT ["python3", "src/main.py"]
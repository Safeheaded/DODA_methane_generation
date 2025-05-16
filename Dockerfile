FROM python:3.8-slim

# Ustaw zmienne środowiskowe
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1

# Instalacja zależności systemowych
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Utwórz katalog aplikacji
WORKDIR /app

# Skopiuj plik z zależnościami
COPY . .

RUN python -m pip install numpy

RUN python -m pip install torch==1.13.0 torchvision==0.14.0 --index-url https://download.pytorch.org/whl/cu128

# Zainstaluj zależności
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN python -m pip install pytorch-lightning==1.5.0 taming-transformers

# Skopiuj resztę plików aplikacji (jeśli chcesz)
# COPY . .

# Domyślny command (do modyfikacji wg potrzeb)
CMD ["bash"]

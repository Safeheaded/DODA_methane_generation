# Bazowy obraz z Ubuntu
FROM ubuntu:20.04

# Ustawienia środowiskowe (żeby uniknąć pytań przy instalacji)
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        python3.8 \
        python3.8-venv \
        python3.8-distutils \
        curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Zainstaluj pip kompatybilny z Pythonem 3.8
RUN curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py | python3.8

# Ustaw python3 i pip jako domyślne polecenia
RUN ln -s /usr/bin/python3.8 /usr/bin/python && \
    ln -s /usr/local/bin/pip /usr/bin/pip

# Utwórz katalog aplikacji
WORKDIR /app

# Skopiuj plik z zależnościami
COPY . .

RUN python -m pip install numpy

RUN python -m pip install torch==1.13.0 torchvision==0.14.0 --index-url https://download.pytorch.org/whl/cu128

# Zainstaluj zależności
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN python -m pip install pytorch-lightning taming-transformers

RUN python -m pip install pytorch-lightning taming-transformers

# Skopiuj resztę plików aplikacji (jeśli chcesz)
# COPY . .

# Domyślny command (do modyfikacji wg potrzeb)
CMD ["bash"]

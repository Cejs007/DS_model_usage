# Použijeme officiální Python image s podporou GPU (volitelné)
FROM tensorflow/tensorflow:2.12.0-gpu

# Nastavení pracovního adresáře
WORKDIR /app

# Instalace systémových závislostí
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Kopírování souborů projektu
COPY requirements.txt .

# Instalace Python závislostí
RUN pip install --no-cache-dir -r requirements.txt

# Kopírování zbytku aplikace
COPY . .

# Expose port pro Streamlit
EXPOSE 8501

# Spuštění Streamlit aplikace
ENTRYPOINT ["streamlit", "run", "main.py"]
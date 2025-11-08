# Utiliser l'image Python officielle
FROM python:3.11-slim

# Installer git et build-essential pour compiler certains packages
RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

# Installer pip à jour
RUN pip install --upgrade pip

# Installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Définir le dossier de travail
WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copier tout le projet ensuite
COPY . .

# Définir variable d'environnement pour logs
ENV PYTHONUNBUFFERED=1

# Commande par défaut pour lancer le bot
CMD ["python", "signal_bot.py"]

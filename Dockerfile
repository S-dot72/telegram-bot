# Image de base (Python 3.11 stable)
FROM python:3.11-slim

# Installer utilitaires nécessaires (git, build tools si besoin)
# On installe aussi curl and ca-certificates pour éviter erreurs réseau
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git build-essential gcc g++ curl ca-certificates libatlas3-base && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Copier requirements et installer dépendances
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copier le projet
COPY . /app
ENV PYTHONUNBUFFERED=1

# Commande par défaut
CMD ["python", "signal_bot.py"]

# Image de base (Python 3.11 stable)
FROM python:3.11-slim

# Installer utilitaires nécessaires
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git build-essential gcc g++ curl ca-certificates libatlas3-base && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Cloner pandas-ta depuis GitHub
RUN git clone https://github.com/twopirllc/pandas-ta.git /tmp/pandas-ta

# Copier requirements
WORKDIR /app
COPY requirements.txt /app/requirements.txt

# Installer d'abord les dépendances de base
RUN pip install --no-cache-dir -r /app/requirements.txt

# Installer pandas-ta depuis le clone local
RUN pip install --no-cache-dir /tmp/pandas-ta

# Nettoyer le clone
RUN rm -rf /tmp/pandas-ta

# Copier le projet
COPY . /app

ENV PYTHONUNBUFFERED=1

# Commande par défaut
CMD ["python", "signal_bot.py"]

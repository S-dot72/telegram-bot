FROM python:3.11-slim

# Installer Git et autres dépendances système
RUN apt-get update && apt-get install -y git build-essential libssl-dev libffi-dev python3-dev && rm -rf /var/lib/apt/lists/*

# Copier requirements.txt
COPY requirements.txt .

# Installer les packages Python
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copier le projet
COPY . .

# Définir le dossier de travail
WORKDIR /app

# Commande pour lancer le bot
CMD ["python", "signal_bot.py"]

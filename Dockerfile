# Image Python 3.11 slim
FROM python:3.11-slim

# Installer dépendances système
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Mettre à jour pip
RUN pip install --upgrade pip

# Copier requirements.txt (sans pandas-ta)
COPY requirements.txt .
RUN sed -i '/pandas-ta/d' requirements.txt

# Installer les autres dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le projet
COPY . /app
WORKDIR /app

# Installer pandas-ta depuis ZIP GitHub
RUN wget https://github.com/twopirllc/pandas-ta/archive/refs/heads/main.zip -O /tmp/pandas-ta.zip \
    && unzip /tmp/pandas-ta.zip -d /tmp \
    && pip install /tmp/pandas-ta-main \
    && rm -rf /tmp/pandas-ta.zip /tmp/pandas-ta-main

# Lancer le bot
CMD ["python", "signal_bot.py"]

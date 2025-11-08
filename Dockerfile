FROM python:3.11-slim

# Installer dépendances système
RUN apt-get update && apt-get install -y wget unzip && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copier requirements et installer autres packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Télécharger pandas-ta depuis GitHub et installer
RUN wget https://github.com/twopirllc/pandas-ta/archive/refs/heads/main.zip -O /tmp/pandas-ta.zip && \
    unzip /tmp/pandas-ta.zip -d /tmp && \
    mv /tmp/pandas-ta-main /tmp/pandas-ta && \
    rm /tmp/pandas-ta.zip && \
    pip install /tmp/pandas-ta

# Copier projet
COPY . /app
WORKDIR /app

CMD ["python", "signal_bot.py"]

FROM python:3.9-slim

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
WORKDIR /app
CMD ["python", "signal_bot.py"]

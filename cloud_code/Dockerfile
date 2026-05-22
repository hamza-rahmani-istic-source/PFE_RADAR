FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-plaques.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements-plaques.txt

COPY . .

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn cloud_api:app --host 0.0.0.0 --port ${PORT}"]

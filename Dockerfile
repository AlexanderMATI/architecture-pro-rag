FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt


COPY . /app

RUN python src/build_index.py --kb-dir knowledge_base --index-dir index || echo "build_index failed at build time, will build at runtime"

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

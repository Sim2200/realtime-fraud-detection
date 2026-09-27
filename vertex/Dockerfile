FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-vertex.txt .
RUN pip install --no-cache-dir -r requirements-vertex.txt

COPY fraud/ ./fraud/
COPY vertex/ ./vertex/

ENV PYTHONPATH=/app PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["uvicorn", "vertex.serve:app", "--host", "0.0.0.0", "--port", "8080"]

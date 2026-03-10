FROM python:3.11-slim

WORKDIR /app

RUN pip install fastapi uvicorn[standard]

COPY server.py .
COPY samples.json .
COPY static/ ./static/

RUN mkdir -p results

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
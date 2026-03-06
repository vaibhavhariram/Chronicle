# Chronicle Runtime - optional container for demo
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY chronicle_runtime/ chronicle_runtime/
COPY scripts/ scripts/
RUN pip install --no-cache-dir -e .

ENV DEVICE=cpu
ENV MODEL_NAME=gpt2

EXPOSE 8000

CMD ["uvicorn", "chronicle_runtime.server.main:app", "--host", "0.0.0.0", "--port", "8000"]

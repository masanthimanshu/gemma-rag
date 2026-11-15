FROM python:3.15-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p models

RUN hf download litert-community/gemma-4-E2B-it-litert-lm gemma-4-E2B-it.litertlm --local-dir models
RUN hf download litert-community/embeddinggemma-2-740m-litert-lm embeddinggemma-2-740m.litertlm --local-dir models

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3500"]

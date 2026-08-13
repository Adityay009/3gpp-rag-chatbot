FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python scripts/download_spec.py && \
    python -m app.ingest --pdf data/ts_123501.pdf --doc-id "3GPP TS 23.501" --version "18.5.0"

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

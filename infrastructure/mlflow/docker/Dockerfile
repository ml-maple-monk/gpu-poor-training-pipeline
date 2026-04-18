FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFLOW_HOME=/mlflow \
    PYTHONPATH=/mlflow

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates libpq5 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        'mlflow[extras]==2.17.2' \
        psycopg2-binary==2.9.9 \
        gunicorn==23.0.0 \
        boto3==1.35.49 \
        prometheus-client==0.21.0

WORKDIR /mlflow

EXPOSE 5000

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:5000/health || exit 1

CMD ["sh", "-c", "if [ \"${MLFLOW_SERVER_ARTIFACT_MODE:-proxy}\" = \"direct\" ]; then \
        exec mlflow server \
            --host 0.0.0.0 \
            --port 5000 \
            --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
            --default-artifact-root ${MLFLOW_ARTIFACTS_DESTINATION} \
            --no-serve-artifacts \
            --gunicorn-opts \"--workers ${MLFLOW_WORKERS:-4} --timeout ${MLFLOW_TIMEOUT:-120} --keep-alive 15 --log-level info --access-logfile -\"; \
    else \
        exec mlflow server \
            --host 0.0.0.0 \
            --port 5000 \
            --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
            --artifacts-destination ${MLFLOW_ARTIFACTS_DESTINATION} \
            --serve-artifacts \
            --gunicorn-opts \"--workers ${MLFLOW_WORKERS:-4} --timeout ${MLFLOW_TIMEOUT:-120} --keep-alive 15 --log-level info --access-logfile -\"; \
    fi"]

# MLflow tracking server image: the stock package plus the Postgres and S3 drivers.
FROM python:3.12-slim
ARG MLFLOW_VERSION
RUN pip install --no-cache-dir "mlflow==${MLFLOW_VERSION}" psycopg2-binary boto3
COPY ensure_bucket.py /ensure_bucket.py
EXPOSE 5000

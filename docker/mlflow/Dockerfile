FROM python:3.12-slim

RUN pip install --no-cache-dir \
    "mlflow==3.14.0" \
    "psycopg2-binary>=2.9"

EXPOSE 5000

ENTRYPOINT ["mlflow", "server"]

# Build from the repository root:  docker build -f deploy/Dockerfile -t elevator-pdm:1.0 .
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PDM_PORT=8080 PDM_MODEL_DIR=/app/deploy/model
WORKDIR /app
COPY deploy/requirements-service.txt deploy/requirements-service.txt
RUN pip install --no-cache-dir -r deploy/requirements-service.txt
COPY pdm/ pdm/
COPY deploy/service/ deploy/service/
COPY deploy/model/ deploy/model/
RUN useradd --create-home --uid 10001 pdm && chown -R pdm /app
USER pdm
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/health').status==200 else 1)"
CMD ["python", "deploy/service/app.py"]

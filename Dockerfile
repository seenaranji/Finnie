# Finnie: ONE image for the API, the UI and the one-off jobs (migrate, index). Which one runs is decided by the command in docker-compose.yml.
# Layers: the slow, rarely-changing part (installing packages) comes first so editing code does not reinstall them.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 HOME=/home/finnie
WORKDIR /app

# Layer 1: dependencies. Test and load-test tools are left out of the runtime image (pytest, coverage, locust).
COPY requirements.txt .
RUN grep -Ev '^(pytest|pytest-cov|locust)==' requirements.txt > requirements.runtime.txt \
    && pip install -r requirements.runtime.txt

# Layer 2: the code and data the app reads at run time (see .dockerignore for what is left out; secrets are NEVER copied in).
COPY config.yaml alembic.ini ./
COPY .streamlit .streamlit
COPY migrations migrations
COPY knowledge_base knowledge_base
COPY src src
COPY scripts scripts

# Run as an ordinary user, not root. /app/logs is where the app writes (mounted as a volume by compose).
RUN useradd --create-home --uid 1000 finnie && mkdir -p /app/logs && chown -R finnie /app /home/finnie
USER finnie

EXPOSE 8000 8501
# Default command = the API. Inside a container the app must listen on 0.0.0.0 (compose publishes it to localhost only).
CMD ["uvicorn", "src.web_app.api:app", "--host", "0.0.0.0", "--port", "8000"]

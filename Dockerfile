# syntax=docker/dockerfile:1

# -----------------------------------------------------------------------------
# Stage 1: Build the React frontend
# -----------------------------------------------------------------------------
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# -----------------------------------------------------------------------------
# Stage 2: Build the Python wheel
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS wheel-builder

WORKDIR /app
RUN pip install --no-cache-dir hatch

COPY pyproject.toml README.md LICENSE ./
COPY trainscope/ ./trainscope/
COPY --from=frontend-builder /app/trainscope/ui/static ./trainscope/ui/static
RUN python -m hatch build -t wheel

# -----------------------------------------------------------------------------
# Stage 3: Runtime image
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRAINSCOPE_RUNS_DIR=/data/trainscope_runs

WORKDIR /app

# Install the built wheel plus telemetry support.
COPY --from=wheel-builder /app/dist/*.whl ./
RUN pip install --no-cache-dir ./*.whl[telemetry] && rm -f ./*.whl

# Create the runs directory and make it a volume.
RUN mkdir -p "${TRAINSCOPE_RUNS_DIR}"
VOLUME ["/data/trainscope_runs"]

EXPOSE 7007

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7007/api/health')" || exit 1

ENTRYPOINT ["trainscope"]
CMD ["ui", "--run", "/data/trainscope_runs", "--host", "0.0.0.0", "--port", "7007"]

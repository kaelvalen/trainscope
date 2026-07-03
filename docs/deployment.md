# Deployment guide

This document covers running TrainScope in production with Docker,
docker-compose, and Kubernetes.

## Docker

Build the image locally. The Dockerfile is multi-stage: it builds the React
frontend, packages the Python wheel, and produces a small runtime image::

    docker build -t trainscope:latest .

Run the UI for an existing run directory on the host::

    docker run --rm -it \
        -p 7007:7007 \
        -v $(pwd)/trainscope_runs:/data/trainscope_runs:ro \
        trainscope:latest

Open http://localhost:7007 in your browser.

### Authentication

Set one or both environment variables when running the container::

    docker run --rm -it \
        -p 7007:7007 \
        -e TRAINSCOPE_API_KEY="your-secret-key" \
        -e TRAINSCOPE_BASIC_AUTH="admin:strong-password" \
        -v $(pwd)/trainscope_runs:/data/trainscope_runs:ro \
        trainscope:latest

Clients must then send either:

- Header ``X-API-Key: your-secret-key``
- Header ``Authorization: Bearer your-secret-key``
- Header ``Authorization: Basic $(echo -n admin:strong-password | base64)``

## docker-compose

A compose file is included for convenience. It exposes the UI on port ``7007``
and persists runs in a named Docker volume::

    docker compose up --build

To use authentication, set environment variables before starting::

    export TRAINSCOPE_API_KEY="your-secret-key"
    docker compose up --build

Or uncomment the relevant lines in ``docker-compose.yml``.

## Kubernetes (Helm)

A basic Helm chart is provided under ``k8s/helm/trainscope``.

Install it with default values::

    helm install trainscope ./k8s/helm/trainscope

Expose via a LoadBalancer service::

    helm install trainscope ./k8s/helm/trainscope \
        --set service.type=LoadBalancer

Or with an ingress::

    helm install trainscope ./k8s/helm/trainscope \
        --set ingress.enabled=true \
        --set ingress.host=trainscope.example.com

Override the runs volume size or storage class::

    helm install trainscope ./k8s/helm/trainscope \
        --set persistence.size=50Gi \
        --set persistence.storageClass=fast-ssd

## Metrics

The UI server exposes Prometheus metrics at ``/metrics``:

- ``trainscope_requests_total``
- ``trainscope_ws_connections``
- ``trainscope_runs_loaded``

Scrape this endpoint with your Prometheus server or a ServiceMonitor.

# syntax=docker/dockerfile:1
# Multi-stage Dockerfile for ShopHub Agent services.
# Stage 1: builder - installs Python dependencies into a venv.
# Stage 2: runtime - python:3.12-slim, non-root user, venv copied from builder.

# ---------------------------------------------------------------------------
# Builder stage
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install system dependencies needed for psycopg compilation.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and populate the venv.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Runtime stage
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Create non-root user.
RUN groupadd -r shophub && useradd -r -g shophub -d /home/shophub -s /sbin/nologin shophub

WORKDIR /home/shophub/app

# Copy venv from builder.
COPY --from=builder /opt/venv /opt/venv

# Copy application code.
COPY store_api.py orders_agent.py orchestrator.py llm.py guardrails.py config.py client.py evals.py ./
COPY tests ./tests

# Create directories and set ownership.
RUN mkdir -p /home/shophub/app && chown -R shophub:shophub /home/shophub

USER shophub

# ---------------------------------------------------------------------------
# Service entry points (command overridden by Compose)
# ---------------------------------------------------------------------------
EXPOSE 8000 8001 8002

CMD ["python", "-c", "print('Specify a service: store_api, orders_agent, or orchestrator')"]

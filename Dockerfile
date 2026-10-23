# ==============================================================================
# BANGLADESH CLIMATE INTELLIGENCE PLATFORM (BCIP) - PRODUCTION DOCKERFILE
# Multi-stage optimized, non-root, security-hardened container image
# ==============================================================================

FROM python:3.12-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system runtime dependencies & security updates
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install UV fast package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create non-root unprivileged service user
RUN groupadd -g 10001 bcip && \
    useradd -u 10000 -g bcip -m -s /bin/bash bcipuser

# Install python dependencies
COPY requirements.txt ./
RUN uv pip install --system -r requirements.txt

# Copy application source code
COPY --chown=bcipuser:bcip . .

# Set up runtime data directories
RUN mkdir -p /app/data/processed /app/data/spatial /app/models /app/assets /app/logs && \
    chown -R bcipuser:bcip /app

USER bcipuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

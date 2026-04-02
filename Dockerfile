# Multi-stage Docker build for the XAUUSD trading system.
# Stage 1: Build (with all dev dependencies)
# Stage 2: Runtime (minimal, production-only)

# --- Build stage ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Runtime stage ---
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY pyproject.toml .

# Create data directories
RUN mkdir -p data models exports logs

# Non-root user for security
RUN useradd -m -s /bin/bash trader
USER trader

# Health check
HEALTHCHECK --interval=60s --timeout=10s \
    CMD python -c "import src; print('OK')" || exit 1

# Default: paper trading with synthetic data
ENTRYPOINT ["python", "scripts/paper_trade.py"]
CMD ["--config", "configs/deployment/production.yaml", "--synthetic"]

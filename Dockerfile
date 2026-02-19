FROM python:3.10-slim AS builder

WORKDIR /app

# System deps for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

# Install torch CPU-only from PyTorch index (TERPISAH dari requirements.txt)
RUN pip install --user --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install sisanya dari requirements.txt
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ── Final image (slim, tanpa build tools) ─────────────
FROM python:3.10-slim

WORKDIR /app

# Runtime deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages dari builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# App code
COPY src/ src/
COPY inference/ inference/

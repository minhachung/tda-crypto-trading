FROM python:3.11-slim AS base

# System dependencies for Ripser, matplotlib, scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    libfreetype6-dev \
    libpng-dev \
    git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache dependencies separately from code
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy source last so code changes don't bust the dep cache
COPY src/ ./src/
COPY examples/ ./examples/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY main.py ./
COPY README.md ./

# Default: run the headline experiment
ENTRYPOINT ["python", "examples/run_validation_v6.py"]
CMD ["90"]

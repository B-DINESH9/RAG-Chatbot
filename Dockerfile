# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies for PDF parsing and chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY src/ ./src/
COPY data/ ./data/
COPY scripts/healthcheck.py ./scripts/healthcheck.py
COPY .streamlit/ ./.streamlit/ 2>/dev/null || true

# Create data directories
RUN mkdir -p data/tmp data/vector_stores

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python scripts/healthcheck.py || exit 1

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

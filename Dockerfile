# ==============================================================================
# Production Dockerfile for Emergency Vehicle Route & Congestion Predictor
# Deployable to: Hugging Face Spaces (Free), Google Cloud Run, AWS ECS, or Local
# ==============================================================================

FROM python:3.11-slim

# Prevent interactive prompts and set Python flags
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0
# Install minimal OS dependencies for OpenCV headless and networking
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up non-root user for security (required by Hugging Face Spaces & Enterprise K8s)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Upgrade pip and install CPU-optimized PyTorch first for fast, lightweight builds
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY --chown=appuser:appuser . /app

# Ensure proper permissions for runtime caches and saved weights
RUN chown -R appuser:appuser /app
USER appuser

# Expose port (7860 for Hugging Face Spaces / configurable via $PORT)
EXPOSE 10000
# Container healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

# Launch FastAPI cloud service
CMD ["sh", "-c", "python -m uvicorn backend.main:app --host ${HOST} --port ${PORT}"]

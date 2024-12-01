FROM chromadb/chroma:latest

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create data directory
RUN mkdir -p /chroma/data

# Copy configuration
COPY config/chromadb.yaml /chroma/config/

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/heartbeat || exit 1
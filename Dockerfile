# Builder stage
FROM python:3.9-slim-buster as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN python -m venv venv && \
    . venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.9-alpine

# Set environment variables
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install curl for health check
RUN apk add --no-cache curl

# Copy the virtual environment from the builder stage
COPY --from=builder /app/venv /app/venv
COPY . .

# Expose the port the application runs on
EXPOSE 5000

# Health check to ensure the application is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:5000/?health=check || exit 1

# Use a non-root user for security
RUN addgroup -S appgroup && adduser -S appuser -G appgroup
USER appuser

# Activate the virtual environment and run the application
CMD ["/app/venv/bin/python", "-m", "streamlit", "run", "ChatGPT.py"]

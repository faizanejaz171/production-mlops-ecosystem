# Stage 1: Build stage for installing dependencies
FROM python:3.10-slim AS builder
WORKDIR /app
COPY pyproject.toml uv.lock ./
# Install required production dependencies
RUN pip install --no-cache-dir pydantic pytest requests python-dotenv

# Stage 2: Final lightweight runtime stage
FROM python:3.10-slim
WORKDIR /app
# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
# Copy application scripts and sample data
COPY validate_schema.py .
COPY validate_batch.py .
COPY sample_data/ ./sample_data/

CMD ["python", "validate_batch.py"]

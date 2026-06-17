# Stage 1: Build/Install
FROM python:3.10-slim AS builder
WORKDIR /app
COPY pyproject.toml uv.lock ./
# uv install dependencies
RUN pip install --no-cache-dir pydantic pytest

# Stage 2: Final Runtime
FROM python:3.10-slim
WORKDIR /app
# Sirf zaroori files copy karo
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY validate_schema.py .
CMD ["python", "validate_schema.py"]

FROM python:3.10-slim

WORKDIR /app

# Sab se pehle dependencies install karne ke liye files copy karein
COPY pyproject.toml uv.lock ./

# Pip ke zariye dependencies install kar rahe hain simple rakhne ke liye aaj
RUN pip install --no-cache-dir pydantic pytest

# Baqi ka code copy karein
COPY validate_schema.py .

# Container chalte hi script run ho jaye
CMD ["python", "validate_schema.py"]

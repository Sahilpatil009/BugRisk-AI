FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
RUN pip install --no-cache-dir .
COPY app ./app

# The trained artifact is intentionally not committed to Git. Production images
# download one immutable release asset and verify it before the API can start.
ARG MODEL_URL="https://github.com/Sahilpatil009/BugRisk-AI/releases/download/apachejit-xgboost-v2/bugrisk_model.joblib"
ARG MODEL_SHA256="994f9dc30ebea9996f11852b21bb1d47932257bc7aeb876f18da1ff7bdcdb255"
RUN mkdir -p /models && python -c "import hashlib, pathlib, urllib.request; target=pathlib.Path('/models/bugrisk_model.joblib'); urllib.request.urlretrieve('${MODEL_URL}', target); actual=hashlib.sha256(target.read_bytes()).hexdigest(); assert actual == '${MODEL_SHA256}', f'model checksum mismatch: {actual}'"
ENV MODEL_PATH=/models/bugrisk_model.joblib

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

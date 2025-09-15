FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-serve.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements-serve.txt

COPY miniLLM ./miniLLM

EXPOSE 8000
CMD ["uvicorn", "miniLLM.service.app:app", "--host", "0.0.0.0", "--port", "8000"]




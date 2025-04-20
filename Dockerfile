FROM python:3.12-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

WORKDIR /app/builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --target=/install --no-cache-dir -r requirements.txt

FROM nvcr.io/nvidia/pytorch:25.03-py3 

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PYTHONPATH=/app/lib 

WORKDIR /app

COPY --from=builder /install /app/lib

COPY Main ./Main
COPY laty_gui.py .

RUN rm -rf /root/.cache /tmp/* /var/lib/apt/lists/*

EXPOSE 7866

CMD ["python", "laty_gui.py"]
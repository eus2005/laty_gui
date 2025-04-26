FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
	PYTHONUNBUFFERED=1 \
	PIP_DISABLE_PIP_VERSION_CHECK=on

RUN apt-get clean && apt-get update && apt-get install -y --no-install-recommends git ca-certificates supervisor wget unzip && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/laty_gui

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY Main ./Main
COPY laty_gui.py .
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

RUN mkdir -p Models Lora Datasets Output Config && rm -rf /tmp/* /root/.cache/pip && rm -rf /root/.cache/huggingface/hub/*

EXPOSE 7866 8888

CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
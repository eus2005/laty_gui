# Stage 1: Builder - Install dependencies
# Match Python version to the runtime image (nvcr.io/nvidia/pytorch:25.03-py3 uses Python 3.10)
FROM python:3.10-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

WORKDIR /app/builder

# Install build tools if needed (git for potential git dependencies in requirements)
# ca-certificates is good practice
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies into a specific directory
# Use --break-system-packages if needed on newer Debian/Python bases
# Note: Installing xformers might take time
RUN pip install --target=/install --no-cache-dir -r requirements.txt

# Stage 2: Runtime - Use Nvidia PyTorch base image
# This image includes PyTorch, CUDA, cuDNN, and Python 3.10
FROM nvcr.io/nvidia/pytorch:25.03-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    # Set PYTHONPATH to include the installed libraries
    PYTHONPATH=/app/lib

WORKDIR /app

# Copy installed libraries from the builder stage
COPY --from=builder /install /app/lib

# Copy application code
COPY Main ./Main
COPY laty_gui.py .
# Add any other necessary scripts or config files here if not ignored

# Optional: Install system dependencies if needed by any library (e.g., libstdc++ for some)
# RUN apt-get update && apt-get install -y --no-install-recommends <needed_package> && rm -rf /var/lib/apt/lists/*

# Clean up potentially large cache directories
RUN rm -rf /root/.cache /tmp/* /var/lib/apt/lists/*

# Expose the Gradio port
EXPOSE 7866

# Default command to run the Gradio GUI
CMD ["python", "laty_gui.py"]
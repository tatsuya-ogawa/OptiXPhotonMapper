FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    imagemagick \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64 /opt/OptiX

ENV OPTIX_ROOT=/opt/OptiX
ENV CUDA_HOME=/usr/local/cuda

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# CMD ["/bin/bash"]
ADD entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

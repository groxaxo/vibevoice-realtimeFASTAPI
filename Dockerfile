FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV ENABLE_FLASHSR=true
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV MAX_JOBS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3.11-distutils \
        git curl ca-certificates build-essential ninja-build libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

COPY pyproject.toml ./
COPY .python-version ./
COPY README.md ./
COPY main.py ./
COPY runner ./runner
COPY third_party/VibeVoice ./third_party/VibeVoice
COPY overrides ./overrides
COPY scripts ./scripts

RUN mkdir -p models
RUN uv venv --python /usr/bin/python3.11
RUN . .venv/bin/activate && uv sync --extra vibevoice
RUN . .venv/bin/activate && uv pip install -e third_party/VibeVoice
RUN . .venv/bin/activate && uv pip install --no-build-isolation flash-attn

RUN if [ -f overrides/app.py ]; then cp overrides/app.py third_party/VibeVoice/demo/web/app.py; fi
RUN if [ -f overrides/index.html ]; then cp overrides/index.html third_party/VibeVoice/demo/web/index.html; fi

EXPOSE 8000
ENV HF_HOME=/app/models
CMD [".venv/bin/python", "scripts/run_realtime_demo.py", "--port", "8000", "--host", "0.0.0.0"]

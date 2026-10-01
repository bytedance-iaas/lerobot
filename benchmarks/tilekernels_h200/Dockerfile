FROM nvidia/cuda:13.1.2-cudnn-devel-ubuntu24.04@sha256:08d6facb6a72f918beb3ffe5b57cff72402a203f44428831262e95c7466d4188

ENV DEBIAN_FRONTEND=noninteractive \
    UV_NO_CACHE=1 \
    UV_HTTP_TIMEOUT=180 \
    CUDA_HOME=/usr/local/cuda \
    VIRTUAL_ENV=/opt/venv \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64

RUN apt-get -o Acquire::Retries=3 update \
    && apt-get install -y --no-install-recommends \
       python3 python3-dev python3-venv build-essential ninja-build git patch curl ca-certificates ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir uv

# This is deliberately isolated from the supported LeRobot environment. Install
# the normal GR00T/PI dependencies first, then replace torch with the 2.13 build
# required by the TileKernels revision tested here.
COPY . /workspace/lerobot
RUN uv pip install --python /opt/venv/bin/python \
      -e '/workspace/lerobot[training,pi,groot]' \
    && uv pip install --python /opt/venv/bin/python --reinstall \
      torch==2.13.0 tilelang==0.1.15 \
      'setuptools>=77.0.3' wheel setuptools-scm packaging pytest

ARG TILE_KERNELS_REV=66258df6175d2f630ffecb04c5ab66bff8a2ae6a
RUN git clone https://github.com/deepseek-ai/TileKernels.git /opt/TileKernels \
    && git -C /opt/TileKernels checkout --detach "$TILE_KERNELS_REV"
COPY benchmarks/tilekernels_h200/tilekernels-rope256.patch /tmp/tilekernels-rope256.patch
RUN patch --directory=/opt/TileKernels --strip=1 < /tmp/tilekernels-rope256.patch \
    && uv pip install --python /opt/venv/bin/python --no-deps -e /opt/TileKernels \
    && uv pip freeze --python /opt/venv/bin/python > /opt/tilekernels-requirements.lock

WORKDIR /workspace/lerobot
CMD ["sleep", "infinity"]

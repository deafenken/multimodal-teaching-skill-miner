ARG BASE_IMAGE=nvidia/cuda:12.4.1-base-ubuntu22.04
FROM ${BASE_IMAGE}

ARG BASE_IMAGE_ID
ARG DEBIAN_FRONTEND=noninteractive
ARG UBUNTU_MIRROR=http://archive.ubuntu.com/ubuntu
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PROJECT_WHEEL_SHA256

LABEL org.opencontainers.image.title="Teaching Skill Miner TeachObs ASR worker" \
      org.opencontainers.image.version="1.2.0" \
      org.teaching-skill-miner.asr.base-image-id="${BASE_IMAGE_ID}" \
      org.teaching-skill-miner.asr.ubuntu-mirror="${UBUNTU_MIRROR}" \
      org.teaching-skill-miner.asr.python-package-index="${PIP_INDEX_URL}" \
      org.teaching-skill-miner.asr.private-data-embedded="false"

RUN if [ -f /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list ]; then \
        mv /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list /tmp/cuda.list.disabled; \
    fi \
    && sed -i "s@http://archive.ubuntu.com/ubuntu@${UBUNTU_MIRROR}@g" /etc/apt/sources.list \
    && sed -i "s@http://security.ubuntu.com/ubuntu@${UBUNTU_MIRROR}@g" /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install \
        --index-url "${PIP_INDEX_URL}" \
        --disable-pip-version-check \
        --no-cache-dir \
        --no-compile \
        ctranslate2==4.8.1 \
        faster-whisper==1.2.1 \
        nvidia-cublas-cu12==12.9.2.10 \
        nvidia-cudnn-cu12==9.24.0.43

# Keep the release-wheel binding after the expensive OS/runtime layers.  A
# source-only wheel change must invalidate the project installation and final
# image receipt, but should not force another multi-gigabyte CUDA dependency
# download.
LABEL org.teaching-skill-miner.asr.wheel-sha256="${PROJECT_WHEEL_SHA256}"

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib
ENV HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

ARG PROJECT_WHEEL_FILENAME
COPY ${PROJECT_WHEEL_FILENAME} /tmp/${PROJECT_WHEEL_FILENAME}
RUN python3 -c 'from hashlib import sha256; from pathlib import Path; import sys; actual = sha256(Path(sys.argv[1]).read_bytes()).hexdigest(); expected = sys.argv[2]; raise SystemExit(0 if actual == expected else "release wheel SHA-256 mismatch")' "/tmp/${PROJECT_WHEEL_FILENAME}" "${PROJECT_WHEEL_SHA256}" \
    && python3 -m pip install \
        --disable-pip-version-check \
        --no-cache-dir \
        --no-compile \
        --no-deps \
        "/tmp/${PROJECT_WHEEL_FILENAME}" \
    && rm "/tmp/${PROJECT_WHEEL_FILENAME}"

RUN python3 -c 'from importlib.metadata import version; expected={"ctranslate2":"4.8.1","faster-whisper":"1.2.1","nvidia-cublas-cu12":"12.9.2.10","nvidia-cudnn-cu12":"9.24.0.43"}; actual={name:version(name) for name in expected}; raise SystemExit(0 if actual == expected else f"ASR package mismatch: {actual!r}")' \
    && python3 -m teaching_skill_miner run-teachobs-asr-gpu --help >/dev/null \
    && ffprobe -version >/dev/null

ENTRYPOINT ["python3", "-m", "teaching_skill_miner"]

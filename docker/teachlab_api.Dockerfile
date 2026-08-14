# Production builds must pass digest-pinned images, for example
# node:22-bookworm-slim@sha256:<verified digest>. Mutable tags are rejected.
ARG NODE_BUILD_IMAGE
ARG PYTHON_RUNTIME_IMAGE
FROM ${NODE_BUILD_IMAGE} AS api-build
ARG NODE_BUILD_IMAGE
RUN case "${NODE_BUILD_IMAGE}" in *@sha256:????????????????????????????????????????????????????????????????) ;; *) echo "NODE_BUILD_IMAGE must be digest pinned" >&2; exit 64;; esac
WORKDIR /build/apps/api
COPY apps/api/package.json apps/api/package-lock.json ./
RUN npm ci --ignore-scripts --no-audit --no-fund
COPY apps/api/tsconfig.json apps/api/tsconfig.build.json ./
COPY apps/api/scripts ./scripts
COPY apps/api/src ./src
RUN npm run build \
    && npm prune --omit=dev --ignore-scripts \
    && npm cache clean --force

FROM ${PYTHON_RUNTIME_IMAGE} AS runtime
ARG PYTHON_RUNTIME_IMAGE
ARG PROJECT_WHEEL=dist/teaching_skill_miner-1.2.0-py3-none-any.whl
ARG PROJECT_WHEEL_SHA256
ARG TEACHLAB_RELEASE_VERSION
ARG TEACHLAB_RELEASE_ID
RUN case "${PYTHON_RUNTIME_IMAGE}" in *@sha256:????????????????????????????????????????????????????????????????) ;; *) echo "PYTHON_RUNTIME_IMAGE must be digest pinned" >&2; exit 64;; esac \
    && test "${#PROJECT_WHEEL_SHA256}" = 64 \
    && test "${TEACHLAB_RELEASE_VERSION}" = "1.2.0" \
    && case "${TEACHLAB_RELEASE_ID}" in ????????*) ;; *) echo "TEACHLAB_RELEASE_ID is required" >&2; exit 64;; esac

ENV NODE_ENV=production \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TEACHLAB_RELEASE_VERSION=${TEACHLAB_RELEASE_VERSION} \
    TEACHLAB_RELEASE_ID=${TEACHLAB_RELEASE_ID} \
    HOME=/nonexistent

# PDF and temporal-media extraction are local. Image semantics remain disabled
# unless an explicitly configured provider and consent receipt are present.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates ffmpeg poppler-utils \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 teachlab \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin teachlab

COPY --from=api-build /usr/local/bin/node /usr/local/bin/node
COPY --from=api-build /build/apps/api/dist /opt/teachlab/api/dist
COPY --from=api-build /build/apps/api/node_modules /opt/teachlab/api/node_modules
COPY --from=api-build /build/apps/api/package.json /opt/teachlab/api/package.json
COPY deploy/production/python-runtime.lock /tmp/python-runtime.lock
COPY ${PROJECT_WHEEL} /tmp/teaching_skill_miner-1.2.0-py3-none-any.whl
RUN python - "${PROJECT_WHEEL_SHA256}" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys
actual = sha256(Path('/tmp/teaching_skill_miner-1.2.0-py3-none-any.whl').read_bytes()).hexdigest()
if actual != sys.argv[1]:
    raise SystemExit('release wheel SHA-256 mismatch')
PY
RUN python -m pip install --no-cache-dir --require-hashes --only-binary=:all: --requirement /tmp/python-runtime.lock \
    && python -m pip install --no-cache-dir --no-deps /tmp/teaching_skill_miner-1.2.0-py3-none-any.whl \
    && python -m pip check \
    && python -m teaching_skill_miner.teacher_agent_gateway_worker --self-check >/dev/null \
    && python -m teaching_skill_miner.teacher_agent_safeguarding_supervisor --self-check >/dev/null \
    && rm /tmp/teaching_skill_miner-1.2.0-py3-none-any.whl /tmp/python-runtime.lock \
    && mkdir -p /var/lib/teachlab /run/teachlab /tmp/teachlab \
    && chown -R 10001:10001 /var/lib/teachlab /run/teachlab /tmp/teachlab \
    && chmod 0700 /var/lib/teachlab /run/teachlab /tmp/teachlab

WORKDIR /opt/teachlab/api
USER 10001:10001
EXPOSE 4000
ENTRYPOINT ["node", "dist/main.js"]

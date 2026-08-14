ARG NODE_BUILD_IMAGE
ARG NODE_RUNTIME_IMAGE
FROM ${NODE_BUILD_IMAGE} AS console-build
ARG NODE_BUILD_IMAGE
RUN case "${NODE_BUILD_IMAGE}" in *@sha256:????????????????????????????????????????????????????????????????) ;; *) echo "NODE_BUILD_IMAGE must be digest pinned" >&2; exit 64;; esac
# The authenticated production image must compile the organization-login
# boundary into the client bundle. This is deliberately not a runtime toggle:
# NEXT_PUBLIC_* values are inlined by Next.js during `next build`.
ENV NEXT_TELEMETRY_DISABLED=1 \
    NEXT_PUBLIC_TEACHLAB_AUTH_MODE=oidc
WORKDIR /build/apps/console
COPY apps/console/package.json apps/console/package-lock.json ./
RUN npm ci --ignore-scripts --no-audit --no-fund
COPY apps/console/ ./
RUN npm run build

FROM ${NODE_RUNTIME_IMAGE} AS runtime
ARG NODE_RUNTIME_IMAGE
ARG TEACHLAB_RELEASE_VERSION
ARG TEACHLAB_RELEASE_ID
RUN case "${NODE_RUNTIME_IMAGE}" in *@sha256:????????????????????????????????????????????????????????????????) ;; *) echo "NODE_RUNTIME_IMAGE must be digest pinned" >&2; exit 64;; esac \
    && test "${TEACHLAB_RELEASE_VERSION}" = "1.2.0" \
    && case "${TEACHLAB_RELEASE_ID}" in ????????*) ;; *) echo "TEACHLAB_RELEASE_ID is required" >&2; exit 64;; esac \
    && groupadd --gid 10001 teachlab \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin teachlab
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    NEXT_PUBLIC_TEACHLAB_AUTH_MODE=oidc \
    TEACHLAB_RELEASE_VERSION=${TEACHLAB_RELEASE_VERSION} \
    TEACHLAB_RELEASE_ID=${TEACHLAB_RELEASE_ID} \
    HOSTNAME=0.0.0.0 \
    PORT=3000 \
    HOME=/nonexistent
WORKDIR /opt/teachlab/console
COPY --from=console-build --chown=10001:10001 /build/apps/console/.next/standalone ./
COPY --from=console-build --chown=10001:10001 /build/apps/console/.next/static ./.next/static
COPY --from=console-build --chown=10001:10001 /build/apps/console/public ./public
USER 10001:10001
EXPOSE 3000
ENTRYPOINT ["node", "server.js"]

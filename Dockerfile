# LeRobot Agent Console — a single-page ops UI layered on top of a LeRobot image.
#
# This app is a *consumer* of LeRobot: it FROMs a LeRobot image so the in-pod
# shell and the hermes agent operate on the very same LeRobot checkout/install.
#
# IMPORTANT (VKE): push this image with gzip, not zstd. The VKE node containerd
# (1.6.x) cannot unpack zstd layers (`number of layers and diffIDs don't match`).
#   docker buildx build --output type=image,...,compression=gzip,force-compression=true .
#
# Build:
#   docker build --build-arg BASE_IMAGE=<lerobot-image> -t <registry>/lerobot-console:<tag> .
ARG BASE_IMAGE=iaas-us-cn-beijing.cr.volces.com/physicalai/lerobot:latest
FROM ${BASE_IMAGE}

# --- the chat agent ------------------------------------------------------- #
# hermes drives chat (and only chat). The Volcengine (Ark) api key is supplied
# at runtime via the UI, never baked into the image.
ARG HERMES_PACKAGE=hermes-agent
ARG PIP_INDEX_URL=https://mirrors.volces.com/pypi/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
RUN python -m pip install --no-cache-dir "aiohttp>=3.9" "pyyaml>=6" "${HERMES_PACKAGE}" \
    && (hermes postinstall --yes 2>/dev/null || true)

# --- the console app ------------------------------------------------------ #
WORKDIR /opt/agent-console
COPY server.py ./
COPY static ./static
COPY scripts ./scripts

# --- robot_sft skill (requirement f) -------------------------------------- #
# Installs the LeRobot-SFT skill so the agent knows the training pipeline.
# Non-fatal: if the network/registry is unavailable at build time, the runtime
# entrypoint retries (see docker-entrypoint.sh).
RUN bash scripts/install_skill.sh || echo "WARN: robot_sft skill install deferred to runtime"

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh scripts/install_skill.sh

# The PTY console + agent operate from LEROBOT_HOME (exported to all children).
ENV PORT=8080 \
    LEROBOT_HOME=/workspace/lerobot \
    HERMES_CHAT_SKILL=robot_sft

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "/opt/agent-console/server.py"]

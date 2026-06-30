# LeRobot Agent Console — single-page ops UI on top of a LeRobot image.
#
# One image / one container = lerobot + a SLIM hermes + this console (:8080).
# `kubectl exec` into the pod gives you the same env the console drives.
#
# hermes is installed slim: ACP (stdio) chat + the shell tool only. No Node,
# no browser/Chromium, no ffmpeg, no messaging/matrix/voice/web-search extras,
# no dashboard/TUI frontend. That keeps the image close to the lerobot base.
#
# Build (separate pipeline). Push gzip (compression=gzip) — VKE node containerd
# 1.6.x rejects zstd. The lerobot base images are gzip, so the console image is
# gzip too.
#   docker buildx build \
#     --build-arg BASE_IMAGE=iaas-us-cn-beijing.cr.volces.com/physicalai/lerobot:<lerobot-tag> \
#     --output type=image,name=<registry>/lerobot-console:<tag>,push=true,compression=gzip,oci-mediatypes=true \
#     .
ARG BASE_IMAGE=iaas-us-cn-beijing.cr.volces.com/physicalai/lerobot:00370ca7ffea5b3c8ecb05e098e910b9559ba6e7
FROM ${BASE_IMAGE}

# CN-friendly PyPI mirror (matches the lerobot build). Override with --build-arg.
ARG PIP_INDEX_URL=https://mirrors.volces.com/pypi/simple/
ENV UV_DEFAULT_INDEX=${PIP_INDEX_URL} \
    PIP_INDEX_URL=${PIP_INDEX_URL}

# --- system deps: ripgrep speeds up the agent's file search; git fetches hermes.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ripgrep git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- slim hermes in an isolated venv (its pins would clash with lerobot/torch) ---
# ACP only. No Node, no browser, no messaging/matrix/voice/web-search/mcp extras.
ARG HERMES_REF=main
ENV HERMES_HOME=/opt/data \
    HERMES_DISABLE_LAZY_INSTALLS=1
RUN command -v uv >/dev/null 2>&1 || python -m pip install --no-cache-dir uv \
    && git clone --depth 1 --branch ${HERMES_REF} https://github.com/nousresearch/hermes-agent /opt/hermes \
    && uv venv /opt/hermes/.venv \
    && VIRTUAL_ENV=/opt/hermes/.venv uv pip install --native-tls --no-cache-dir "/opt/hermes[acp]" \
    && ln -sf /opt/hermes/.venv/bin/hermes /usr/local/bin/hermes \
    && hermes acp --check

# --- the console app + its (tiny) deps into the base python ---------------- #
RUN python -m pip install --no-cache-dir "aiohttp>=3.9" "pyyaml>=6"
WORKDIR /opt/agent-console
COPY server.py ./
COPY static ./static
COPY scripts ./scripts

# --- seed hermes config + bake the robot_sft skill INTO the image ----------- #
# Installs the skill at BUILD time (GitHub is reachable then) into $HERMES_HOME,
# then snapshots the whole seeded home to /opt/hermes-seed. At runtime a fresh PVC
# mount shadows $HERMES_HOME, so the entrypoint restores the skill+config from the
# baked snapshot with a local copy — NO runtime GitHub access required.
# (No `|| true`: if the skill can't be baked in, fail the build loudly.)
RUN mkdir -p "${HERMES_HOME}" \
    && hermes config set model.provider custom \
    && hermes config set model.base_url https://ark.cn-beijing.volces.com/api/v3 \
    && hermes config set model.default deepseek-v4-pro-260425 \
    && bash scripts/install_skill.sh \
    && hermes skills list | grep -qi robot_sft \
    && cp -a "${HERMES_HOME}" /opt/hermes-seed

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh scripts/install_skill.sh

# The console shell + agent operate from LEROBOT_HOME (exported to children).
ENV PORT=8080 \
    LEROBOT_HOME=/workspace/lerobot \
    HERMES_CHAT_SKILL=robot_sft

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "/opt/agent-console/server.py"]

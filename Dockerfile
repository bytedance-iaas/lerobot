# LeRobot Agent Console — single-page ops UI on top of a LeRobot image.
#
# One image / one container = lerobot + a SLIM hermes + this console (:8080).
# `kubectl exec` into the pod gives you the same env the console drives.
#
# hermes is installed slim: ACP (stdio) chat + the shell tool only. No Node,
# no browser/Chromium, no ffmpeg, no messaging/matrix/voice/web-search extras,
# no dashboard/TUI frontend. That keeps the image close to the lerobot base.
#
# Build (separate pipeline, FROM a known-good GZIP lerobot tag — VKE node
# containerd 1.6.x rejects zstd):
#   docker buildx build \
#     --build-arg BASE_IMAGE=iaas-us-cn-beijing.cr.volces.com/physicalai/lerobot:<gzip-tag> \
#     --output type=image,name=<registry>/lerobot-console:<tag>,push=true,compression=gzip,force-compression=true,oci-mediatypes=true \
#     .
ARG BASE_IMAGE=iaas-us-cn-beijing.cr.volces.com/physicalai/lerobot:a64719559e6fea31e2f767e296f64fb0d6ef31b7
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

# --- trim hermes built-in tools to what an ops console needs --------------- #
# (best-effort: written to $HERMES_HOME; a runtime PVC mount may shadow it, in
#  which case the entrypoint re-applies — both paths are covered.)
RUN mkdir -p "${HERMES_HOME}" \
    && hermes config set model.provider custom \
    && hermes config set model.base_url https://ark.cn-beijing.volces.com/api/v3 \
    && hermes config set model.default deepseek-v4-pro-260425 \
    && (bash scripts/install_skill.sh || echo "WARN: robot_sft skill install deferred to runtime")

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh scripts/install_skill.sh

# The console shell + agent operate from LEROBOT_HOME (exported to children).
ENV PORT=8080 \
    LEROBOT_HOME=/workspace/lerobot \
    HERMES_CHAT_SKILL=robot_sft

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "/opt/agent-console/server.py"]

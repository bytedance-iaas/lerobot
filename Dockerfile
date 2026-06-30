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

# lerobot's Dockerfile.user ends with `USER user_lerobot` (non-root). The build
# steps below need root (apt, /opt, pip into the venv), and the console runs as
# root at runtime so it can write the HERMES_HOME PVC + drive PTY shells.
USER root

# CN-friendly PyPI mirror (matches the lerobot build). Override with --build-arg.
ARG PIP_INDEX_URL=https://mirrors.volces.com/pypi/simple/
ENV UV_DEFAULT_INDEX=${PIP_INDEX_URL} \
    PIP_INDEX_URL=${PIP_INDEX_URL}

# --- system deps: ripgrep speeds up the agent's file search.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ripgrep ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- slim hermes in an isolated venv (its pins would clash with lerobot/torch) ---
# Installed from PyPI mirrors — NO GitHub, so the CN build pool doesn't need
# github.com. The volces mirror LAGS (only up to 0.16.0), so add aliyun as an extra
# index (it has the latest + is reachable from the CP pool, same host lerobot uses
# for torch wheels) and let uv pick the best version across both indexes.
# ACP extra only; no node/browser/messaging/matrix/voice/web-search/mcp.
ARG HERMES_VERSION=0.17.0
ARG HERMES_EXTRA_INDEX=https://mirrors.aliyun.com/pypi/simple/
ENV HERMES_HOME=/opt/data \
    HERMES_DISABLE_LAZY_INSTALLS=1
RUN command -v uv >/dev/null 2>&1 || python -m pip install --no-cache-dir uv \
    && uv venv /opt/hermes/.venv \
    && VIRTUAL_ENV=/opt/hermes/.venv uv pip install --native-tls --no-cache-dir \
         --extra-index-url "${HERMES_EXTRA_INDEX}" --index-strategy unsafe-best-match \
         "hermes-agent[acp]==${HERMES_VERSION}" \
    && ln -sf /opt/hermes/.venv/bin/hermes /usr/local/bin/hermes \
    && hermes acp --check

# --- the console app + its (tiny) deps into the lerobot venv --------------- #
# The lerobot venv (uv venv) has no `pip`, and `python` resolves to it (on PATH),
# so install with uv targeting that venv. (CMD runs `python server.py` there.)
RUN VIRTUAL_ENV=/lerobot/.venv uv pip install --native-tls --no-cache-dir "aiohttp>=3.9" "pyyaml>=6"
WORKDIR /opt/agent-console
COPY server.py ./
COPY static ./static
COPY scripts ./scripts
# The robot_sft skill is VENDORED into this repo (vendor/robot_sft) so the build
# needs NO GitHub — see vendor/robot_sft.SOURCE for the upstream commit.
COPY vendor ./vendor

# --- seed hermes config + bake the robot_sft skill INTO the image ----------- #
# Drop the vendored skill into $HERMES_HOME/skills (hermes auto-discovers it),
# then snapshot the seeded home to /opt/hermes-seed. At runtime a fresh PVC mount
# shadows $HERMES_HOME, so the entrypoint restores skill+config from the snapshot
# with a local copy — NO GitHub at build OR runtime.
# (No `|| true`: if the skill can't be baked in, fail the build loudly.)
RUN mkdir -p "${HERMES_HOME}/skills" \
    && cp -a /opt/agent-console/vendor/robot_sft "${HERMES_HOME}/skills/robot_sft" \
    && hermes config set model.provider custom \
    && hermes config set model.base_url https://ark.cn-beijing.volces.com/api/v3 \
    && hermes config set model.default deepseek-v4-pro-260425 \
    && hermes skills list | grep -qi robot_sft \
    && cp -a "${HERMES_HOME}" /opt/hermes-seed

# --- assert the runtime wiring at BUILD time (fail fast if a venv is wrong) -- #
# 1) server.py runs in the lerobot venv → must import aiohttp + yaml and parse.
# 2) hermes resolves via the symlink → its own venv (separate process).
RUN python -c "import aiohttp, yaml, ast; ast.parse(open('/opt/agent-console/server.py').read()); print('lerobot venv: console deps + server.py OK')" \
    && test -x /usr/local/bin/hermes \
    && hermes --version \
    && echo "runtime wiring OK (lerobot venv + hermes venv + skill)"

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh scripts/install_skill.sh

# The console shell + agent operate from LEROBOT_HOME (the lerobot checkout in the
# base image is at /lerobot). Exported to children.
ENV PORT=8080 \
    LEROBOT_HOME=/lerobot \
    HERMES_CHAT_SKILL=robot_sft

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "/opt/agent-console/server.py"]

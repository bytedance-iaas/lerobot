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
#     --build-arg CONSOLE_COMMIT=$(git rev-parse HEAD) \
#     --output type=image,name=<registry>/lerobot-agent-console:<tag>,push=true,compression=gzip,oci-mediatypes=true \
#     .
ARG BASE_IMAGE=iaas-us-cn-beijing.cr.volces.com/physicalai/lerobot:18cd48ce718176258aa1a685622f9933f1549a9e
FROM ${BASE_IMAGE}

# lerobot's Dockerfile.user ends with `USER user_lerobot` (non-root). The build
# steps below need root (apt, /opt, pip into the venv), and the console runs as
# root at runtime so it can write the HERMES_HOME PVC + drive PTY shells.
USER root

# CN-friendly PyPI mirror (matches the lerobot build). Override with --build-arg.
ARG PIP_INDEX_URL=https://mirrors.volces.com/pypi/simple/
ENV UV_DEFAULT_INDEX=${PIP_INDEX_URL} \
    PIP_INDEX_URL=${PIP_INDEX_URL}

# --- apt mirror + system deps (ripgrep speeds up the agent's file search) ------ #
# Older lerobot bases point apt at Aliyun; newer ones already point at Volcengine.
# Rewrite either way, so both this build and any `apt update` run inside the
# deployed pod stay on the VKE-local mirror. Host-only rewrite, in the classic
# sources.list and the deb822 sources.list.d/debian.sources, so whatever
# suite/components the base pins still resolve. Debian paths (/debian,
# /debian-security) — NOT ubuntu-*: the base image is Debian, not Ubuntu.
ARG APT_MIRROR=https://mirrors.volces.com
RUN for f in /etc/apt/sources.list /etc/apt/sources.list.d/debian.sources; do \
      [ -f "$f" ] && sed -i -E "s#https?://(mirrors\.aliyun\.com|deb\.debian\.org|security\.debian\.org)#${APT_MIRROR}#g" "$f" || true; \
    done; \
    apt-get update \
    && apt-get install -y --no-install-recommends ripgrep ca-certificates wget jq aria2 \
    && rm -rf /var/lib/apt/lists/*

# --- pod-to-pod ssh (multi-node training: checkpoint scp master->workers) ----- #
# All console pods run this SAME image, so ONE build-time keypair whose pubkey is in the
# image's own authorized_keys makes every pod<->pod ssh passwordless automatically — no
# runtime key exchange, survives pod restarts. Trade-off (accepted for this private
# registry + cluster-internal sshd; 22 is never exposed by APIG/CLB, which forward 8080
# only): anyone with the image has the key. Upgrade path: mount a k8s Secret keypair.
# StrictHostKeyChecking off because pod host keys regenerate on every restart.
# rsync too: for the multi-node checkpoint copy it beats scp — resumable, delta-only on
# re-sync, --partial survives a dropped link, and -P shows progress on multi-GB checkpoints.
RUN apt-get update \
    && apt-get install -y --no-install-recommends openssh-server openssh-client rsync \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd /root/.ssh \
    && ssh-keygen -t ed25519 -N "" -f /root/.ssh/id_ed25519 \
    && cat /root/.ssh/id_ed25519.pub >> /root/.ssh/authorized_keys \
    && printf 'Host *\n  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null\n' > /root/.ssh/config \
    && chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys /root/.ssh/id_ed25519 \
    && sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config

# --- RDMA userspace, so NCCL can actually use the RoCE NICs ------------------- #
# The nodes expose /dev/infiniband/uverbs* into the container, but the image had no
# userspace stack at all — so NCCL logged `Failed to open libibverbs.so[.1]` and fell back
# to TCP *silently*: multi-node training still ran, just over a fraction of the available
# bandwidth, with nothing in the output saying so.
#   libibverbs1       — the verbs library NCCL dlopen()s
#   ibverbs-providers — THE critical one: the userspace device drivers (libmlx5). Without it
#                       libibverbs loads but enumerates ZERO devices, which looks identical
#                       to having no RDMA at all.
#   ibverbs-utils     — ibv_devinfo / ibv_devices, so the M0b check can inspect port state,
#                       link_layer (RoCE vs IB) and GIDs from inside the pod.
# librdmacm1 is not packaged separately on Debian 13 (rdma-core provides it); NCCL does not
# need it for the IB transport. Verify with NCCL_DEBUG=INFO showing NET/IB, never assume.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libibverbs1 ibverbs-providers ibverbs-utils \
    && rm -rf /var/lib/apt/lists/*

# --- oniond: Volcengine TOS model/dataset downloader (onion-ai-data) ---------- #
# The skill prefers oniond for external models/datasets (fast, TOS-local, no AK/SK), then
# falls back to HF (hf-mirror) — see vendor/robot_sft/scripts/fetch.py. Install like
# install_oniond.sh: pull the .deb from the ivolces extra-tools mirror (reachable from the CN
# build pool), extract just the two binaries (skip its aria2-byted hard dep), and symlink
# aria2c_byted -> aria2c. amd64-only, which the CUDA/linux base is.
ARG ONIOND_MIRROR=http://mirrors.ivolces.com/extra-tools/debian
ARG ONIOND_DEB=onion-ai-data_2.4.6_amd64.deb
RUN cd /tmp \
    && wget -q "${ONIOND_MIRROR}/pool/main/o/onion-ai-data/${ONIOND_DEB}" -O "${ONIOND_DEB}" \
    && dpkg-deb -x "${ONIOND_DEB}" /tmp/onion-extract \
    && cp /tmp/onion-extract/usr/bin/hfd /tmp/onion-extract/usr/bin/oniond /usr/local/bin/ \
    && chmod +x /usr/local/bin/hfd /usr/local/bin/oniond \
    && ln -sf /usr/bin/aria2c /usr/local/bin/aria2c_byted \
    && rm -rf /tmp/onion-extract "/tmp/${ONIOND_DEB}" \
    && oniond help >/dev/null 2>&1 || true
# Default bucket for oniond list/download (override at runtime with -e BUCKET=...).
ENV BUCKET=ai-infra

# HF caches on the /opt/data PVC (the deps image defaults them to /home/user_lerobot/.cache,
# which is EPHEMERAL — so models like pi05_base re-download on every pod restart). Point HF_HOME at
# the persistent volume so a downloaded model/checkpoint survives rollouts. Do NOT set HF_HUB_CACHE
# explicitly — let it default to $HF_HOME/hub (the STANDARD models--* layout, where the gated
# paligemma-3b-pt-224 tokenizer lives); an explicit non-/hub override makes HF miss cached repos and
# try to re-download them (gated → 403). HF_LEROBOT_HOME=/lerobot per convention.
ENV HF_HOME=/opt/data/.cache/huggingface \
    HF_LEROBOT_HOME=/lerobot

# --- slim hermes in an isolated venv (its pins would clash with lerobot/torch) ---
# Installed from PyPI mirrors — NO GitHub, so the CN build pool doesn't need
# github.com. The volces mirror LAGS (only up to 0.16.0), so add aliyun as an extra
# index (it has the latest + is reachable from the CP pool, same host lerobot uses
# for torch wheels) and let uv pick the best version across both indexes.
# ACP extra only; no node/browser/messaging/matrix/voice/web-search/mcp.
ARG HERMES_VERSION=0.17.0
ARG HERMES_EXTRA_INDEX=https://mirrors.aliyun.com/pypi/simple/
# HERMES_STREAM_STALE_TIMEOUT: the "no chunks received" streaming watchdog
# (chat_completion_helpers.py). The code default is base=180, scaled to a floor of
# 240s @>50K ctx / 300s @>100K ctx. A REASONING model (deepseek-v4-pro: up to 128K
# thinking budget) can legitimately think for many minutes BEFORE the first output
# chunk once the context is large — Ark buffers that reasoning, so hermes sees zero
# chunks and kills the "stale" connection at 240s → RemoteProtocolError → the web
# chat gets stuck. Raise the base to 600s so the model's thinking phase is never
# reaped as a false stall (a genuinely hung socket still trips it, just later).
ENV HERMES_HOME=/opt/data \
    HERMES_DISABLE_LAZY_INSTALLS=1 \
    HERMES_STREAM_STALE_TIMEOUT=600
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
# `tosfs` registers the tos:// fsspec protocol so StreamingTOSRobotDataset can
# stream a dataset from Volcengine TOS — bake it in so a fresh pod never needs a
# runtime `uv pip install tosfs` (the venv is in the image layer, not the PVC).
RUN VIRTUAL_ENV=/lerobot/.venv uv pip install --native-tls --no-cache-dir "aiohttp>=3.9" "pyyaml>=6" tosfs
WORKDIR /opt/agent-console
COPY server.py ./
COPY static ./static
COPY scripts ./scripts
# (Removed: patch_diffusers_torchao.py. It worked around diffusers <0.36 importing torchao 0.17
# internal paths, but torchao is no longer installed — the fp8 path is TransformerEngine-only —
# so diffusers never touches torchao and the patch is dead weight.)
# hermes' ACP layer (acp/connection.py) catches EVERY request-handler exception, wraps it
# into a generic `RequestError.internal_error(...)` and `raise err from None` — DISCARDING the
# traceback. So a real bug inside a tool handler reaches the web chat as an opaque "Internal
# error" with no file:line, and the session gets stuck undebuggably. Patch ACP to log the real
# traceback FIRST. Runs in the HERMES venv (that is where acp/ lives). Idempotent.
RUN /opt/hermes/.venv/bin/python scripts/patch_acp_logging.py
# The logging patch above unmasked the actual bug: acp_adapter/server.py reads
# `result.get("final_response", "")`, but an INTERRUPTED turn sets that key to None (a
# present-but-None value bypasses the "" default), so `final_response.startswith(...)` raises
# 'NoneType' has no attribute 'startswith' -> "Internal error" -> stuck chat. Coerce to "".
RUN /opt/hermes/.venv/bin/python scripts/patch_acp_interrupt.py
# NOTE: the old `patch_acp_autosave.py` (a 10s `save_session` daemon) is DELETED, not applied.
# It resurrected deleted sessions: `save_session -> _persist` first "ensures the session record
# exists" (acp_adapter/session.py) and so RE-INSERTS a row that was deleted out-of-band (UI or
# `hermes sessions delete`), then re-writes it — the session reappeared ~10s after any delete.
# The autosave was also redundant: (a) `create_session` persists the row immediately, (b) hermes
# appends every message incrementally via `append_message` (wired into the ACP agent through
# `_make_agent(session_db=self._get_db())`), which is finer-grained than a 10s full replace, and
# (c) Stop now does a graceful `session/cancel` instead of killing the acp process mid-turn. So
# durability is covered without the daemon, and dropping it makes deletes actually stick.
# Session listing runs INSIDE the hermes venv (it imports hermes_state), so this one
# file lives next to that venv and is invoked with its interpreter, not ours.
COPY hermes_session_api.py /opt/hermes/session_api.py
# The robot_sft skill is VENDORED into this repo (vendor/robot_sft) so the build
# needs NO GitHub — see vendor/robot_sft.SOURCE for the upstream commit.
COPY vendor ./vendor

# SINGLE SOURCE OF TRUTH for the chat model + endpoint. `ARK_MODELS` (comma-separated) is the ONLY
# place a model name is written: server.py reads it for the chat-header dropdown (+ always adds the
# live hermes model), and the seed below sets `model.default` = ARK_MODELS[0] (the FIRST entry), so
# the default is never a duplicated hardcode — to change models, edit ONLY this env var. base_url is
# seeded too; server.py reads both from config.yaml via /api/status (the frontend never hardcodes).
ENV ARK_MODELS=doubao-seed-evolving,deepseek-v4-pro-260425

# --- seed hermes config + LINK the robot_sft skill from the image ----------- #
# The skill CONTENT stays in the image at /opt/agent-console/vendor/robot_sft; the
# skills dir only holds a SYMLINK to it. hermes follows the symlink and discovers the
# skill as "local/enabled" (verified). This keeps robot_sft OFF the PVC so it tracks
# the image and updates on every rollout — instead of a real copy on the PVC that the
# entrypoint would shadow forever (the entrypoint force-relinks on each boot too).
# Then snapshot the seeded home (config + the symlink) to /opt/hermes-seed for the
# fresh-PVC path. NO GitHub at build OR runtime.
# (No `|| true`: if the skill can't be linked, fail the build loudly.)
RUN mkdir -p "${HERMES_HOME}/skills" \
    && ln -sfn /opt/agent-console/vendor/robot_sft "${HERMES_HOME}/skills/robot_sft" \
    && hermes config set model.provider custom \
    && hermes config set model.base_url https://ark.cn-beijing.volces.com/api/v3 \
    && hermes config set model.default "$(printf '%s' "$ARK_MODELS" | cut -d, -f1)" \
    && hermes skills list | grep -qi robot_sft \
    && cp -a "${HERMES_HOME}" /opt/hermes-seed

# --- assert the runtime wiring at BUILD time (fail fast if a venv is wrong) -- #
# 1) server.py runs in the lerobot venv → must import aiohttp + yaml and parse.
# 2) hermes resolves via the symlink → its own venv (separate process).
# 3) the session bridge runs in the HERMES venv → must import hermes_state and emit JSON.
#    Asserting it here means a hermes upgrade that renames the store API breaks the BUILD,
#    not the session list of a running console.
RUN python -c "import aiohttp, yaml, ast; ast.parse(open('/opt/agent-console/server.py').read()); print('lerobot venv: console deps + server.py OK')" \
    && test -x /usr/local/bin/hermes \
    && hermes --version \
    && /opt/hermes/.venv/bin/python /opt/hermes/session_api.py list --limit 1 \
       | python -c "import json,sys; json.load(sys.stdin); print('hermes venv: session bridge OK')" \
    && echo "runtime wiring OK (lerobot venv + hermes venv + skill + session bridge)"

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh scripts/install_skill.sh

# The console shell + agent operate from CONSOLE_WORKDIR (the lerobot checkout in the
# base image is at /lerobot). Do NOT set LEROBOT_HOME — that is lerobot's own reserved
# cache var and lerobot 0.6.x HARD-ERRORS if it's set; server.py reads CONSOLE_WORKDIR.
# LEROBOT_IMAGE / CONSOLE_COMMIT are surfaced by the UI's "更新说明" button so you can see
# exactly which versions are deployed (LEROBOT_IMAGE's tag = the lerobot commit; pass
# --build-arg CONSOLE_COMMIT=$(git rev-parse HEAD) to fill in this console's commit).
ARG BASE_IMAGE
ARG CONSOLE_COMMIT=""
ENV PORT=8080 \
    CONSOLE_WORKDIR=/lerobot \
    HERMES_CHAT_SKILL=robot_sft \
    LEROBOT_IMAGE=${BASE_IMAGE} \
    CONSOLE_COMMIT=${CONSOLE_COMMIT}

# Auto-activate the lerobot uv venv in interactive login shells. The console terminal spawns
# `bash -l`, whose /etc/profile hardcodes PATH and drops /lerobot/.venv/bin — so `python`
# would resolve to the system python. /etc/profile.d/*.sh is sourced *after* that reset, so
# this re-activates the venv: `python` / `lerobot-*` run in it directly, no `uv run` needed.
RUN printf '%s\n' '[ -f /lerobot/.venv/bin/activate ] && . /lerobot/.venv/bin/activate' \
      > /etc/profile.d/10-lerobot-venv.sh

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "/opt/agent-console/server.py"]

#!/usr/bin/env bash
# Runtime bootstrap. HERMES_HOME is typically a PVC, so anything baked into it at
# build time is shadowed by the (initially empty) volume — re-seed it here.
#   1. point hermes chat at Volcengine Ark (provider/base_url/model; NO api key —
#      the user enters that once in the UI, and it persists on the PVC)
#   2. install the robot_sft skill if missing
# All best-effort: the console still serves even if these fail.
set -e

export HERMES_HOME="${HERMES_HOME:-/opt/data}"
mkdir -p "$HERMES_HOME"

if command -v hermes >/dev/null 2>&1; then
  cfg="$HERMES_HOME/config.yaml"
  if [ ! -s "$cfg" ] || ! grep -q "ark.cn-beijing.volces.com" "$cfg" 2>/dev/null; then
    echo "==> seeding hermes Ark provider config (no api key)"
    hermes config set model.provider custom || true
    hermes config set model.base_url "${ARK_BASE_URL:-https://ark.cn-beijing.volces.com/api/v3}" || true
    hermes config set model.default "${ARK_MODEL:-deepseek-v4-pro-260425}" || true
  fi
  if ! hermes skills list 2>/dev/null | grep -qi "robot_sft"; then
    echo "==> installing robot_sft skill"
    bash /opt/agent-console/scripts/install_skill.sh || echo "WARN: skill install failed; chat still works without it"
  fi
fi

exec "$@"

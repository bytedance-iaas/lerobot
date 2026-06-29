#!/usr/bin/env bash
# Entrypoint: best-effort ensure the robot_sft skill is present (in case the
# build-time install was skipped due to no network), then run the console.
set -e

if command -v hermes >/dev/null 2>&1; then
  if ! hermes skills list 2>/dev/null | grep -qi "robot_sft"; then
    echo "==> robot_sft skill missing, installing at runtime..."
    bash /opt/agent-console/scripts/install_skill.sh || echo "WARN: skill install failed; chat still works without it"
  fi
fi

exec "$@"

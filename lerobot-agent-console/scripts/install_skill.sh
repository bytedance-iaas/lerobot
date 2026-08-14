#!/usr/bin/env bash
# Install the robot_sft hermes skill so the chat agent knows how to drive
# LeRobot SFT (CLAUDE.local.md requirement f). Idempotent; safe to re-run.
#
#   ./scripts/install_skill.sh [skill-source]
#
# Default source is the upstream repo. Override by passing an identifier
# (owner/repo) or a direct SKILL.md URL.
set -euo pipefail

SKILL_SRC="${1:-thesues/robot_sft}"
RAW_URL="https://raw.githubusercontent.com/thesues/robot_sft/main/SKILL.md"

if ! command -v hermes >/dev/null 2>&1; then
  echo "hermes not found on PATH — install the hermes agent first." >&2
  exit 1
fi

echo "==> installing hermes skill: ${SKILL_SRC}"
if hermes skills install "${SKILL_SRC}" --yes; then
  :
else
  echo "==> identifier form failed, retrying with raw SKILL.md URL"
  hermes skills install "${RAW_URL}" --name robot_sft --yes
fi

echo "==> installed skills:"
hermes skills list 2>/dev/null | head -40 || true

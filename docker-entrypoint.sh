#!/usr/bin/env bash
# Runtime bootstrap. HERMES_HOME is typically a PVC, so the config + robot_sft
# skill baked into the image at build time are shadowed by the (initially empty)
# volume. Restore them from the image's baked snapshot with a LOCAL copy — no
# network / GitHub access needed at runtime.
#
# The Ark api key is NOT in the snapshot: the user enters it once in the UI and it
# then persists on the PVC. Chat sessions also accumulate on the PVC.
set -e

export HERMES_HOME="${HERMES_HOME:-/opt/data}"
mkdir -p "$HERMES_HOME"
SEED="/opt/hermes-seed"

# Fresh PVC (no config yet) → seed config + skill from the baked image snapshot.
if [ ! -s "$HERMES_HOME/config.yaml" ] && [ -d "$SEED" ]; then
  echo "==> seeding HERMES_HOME from baked snapshot (offline)"
  cp -a "$SEED/." "$HERMES_HOME/"
fi

# Safety net: ensure the robot_sft skill is present. Prefer offline sources — the
# baked snapshot, then the vendored copy that's always in the image — and only
# fall back to a network install (needs GitHub) as a last resort.
if command -v hermes >/dev/null 2>&1 && ! hermes skills list 2>/dev/null | grep -qi "robot_sft"; then
  mkdir -p "$HERMES_HOME/skills"
  if [ -d "$SEED/skills/robot_sft" ]; then
    echo "==> restoring robot_sft from baked snapshot (offline)"
    cp -a "$SEED/skills/robot_sft" "$HERMES_HOME/skills/robot_sft"
  elif [ -d /opt/agent-console/vendor/robot_sft ]; then
    echo "==> restoring robot_sft from vendored copy (offline)"
    cp -a /opt/agent-console/vendor/robot_sft "$HERMES_HOME/skills/robot_sft"
  else
    echo "==> robot_sft not in image; attempting network install (needs GitHub)"
    bash /opt/agent-console/scripts/install_skill.sh || echo "WARN: skill unavailable; chat still works without it"
  fi
fi

exec "$@"

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

# Enforce console policy on EVERY boot. config.yaml lives on the PVC and SHADOWS the image seed,
# so an image rollout alone never updates an existing PVC — we must re-assert this each start
# (same reason the skill needs a symlink).
#   - delegation.orchestrator_enabled=false : the main agent does ALL work in ONE session; no
#     subagent/task delegation. This is the fix for "switch session shows only the first line":
#     with delegation on, robot_sft spawned a `source=subagent` CHILD session per stage, so the
#     main session the console loads held only the first user message while the real work (25/21/16
#     messages) lived in hidden subagent children. Off → the whole conversation stays in the main
#     session → switch replays full history. Pairs with CHAT_DIRECTIVE's system-prompt steer.
#   NOTE: compression stays ENABLED (default) — it is unrelated to this bug and must stay on.
if command -v hermes >/dev/null 2>&1; then
  hermes config set delegation.orchestrator_enabled false >/dev/null 2>&1 || true
  # The REAL subagent kill-switch is disabling the "delegation" TOOLSET. orchestrator_enabled=false
  # ONLY disables the async orchestrator — the basic `delegate` tool still spawns single subagents
  # (verified: it kept creating source=subagent child sessions). Must be a YAML LIST: `hermes config
  # set` stringifies JSON to '["delegation"]' which frozenset()s into chars and is inert. Edit
  # config.yaml with the hermes interpreter; only rewrites when the entry is missing. Idempotent.
  /opt/hermes/.venv/bin/python - "$HERMES_HOME/config.yaml" <<'PY' 2>/dev/null || echo "WARN: could not disable the delegation toolset"
import sys, yaml
p = sys.argv[1]
c = yaml.safe_load(open(p)) or {}
a = c.setdefault("agent", {})
dt = a.get("disabled_toolsets")
dt = dt if isinstance(dt, list) else []
if "delegation" not in dt:
    dt.append("delegation")
    a["disabled_toolsets"] = dt
    yaml.safe_dump(c, open(p, "w"), default_flow_style=False, allow_unicode=True, sort_keys=False)
PY
  echo "==> enforced: agent.disabled_toolsets includes 'delegation' (subagents OFF); compression left ENABLED"
fi

# robot_sft skill: keep it OFF the PVC so it TRACKS THE IMAGE and updates on every
# rollout. The skill content lives in the image at /opt/agent-console/vendor/robot_sft;
# the PVC's skills/robot_sft is only a SYMLINK to it (hermes follows it — verified).
# Force the symlink on EVERY boot: a new rollout's vendor/robot_sft wins immediately,
# and any stale REAL dir left on an older PVC (from the previous copy-based seeding) is
# replaced. This intentionally means live in-pod edits to the skill do NOT persist — the
# image is now the single source of truth for robot_sft.
VENDOR_SKILL=/opt/agent-console/vendor/robot_sft
if [ -d "$VENDOR_SKILL" ]; then
  mkdir -p "$HERMES_HOME/skills"
  rm -rf "$HERMES_HOME/skills/robot_sft"          # drop stale real dir or old symlink
  ln -sfn "$VENDOR_SKILL" "$HERMES_HOME/skills/robot_sft"
  echo "==> linked robot_sft -> $VENDOR_SKILL (tracks image, not PVC)"
elif command -v hermes >/dev/null 2>&1 && ! hermes skills list 2>/dev/null | grep -qi "robot_sft"; then
  # vendor copy missing (should never happen) → last-resort network install (needs GitHub)
  echo "==> robot_sft vendor copy missing; attempting network install (needs GitHub)"
  bash /opt/agent-console/scripts/install_skill.sh || echo "WARN: skill unavailable; chat still works without it"
fi

exec "$@"

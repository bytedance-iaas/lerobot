# LeRobot Agent Console

A single-page, white-background web console to operate **LeRobot** — a Linux
shell and an AI chat agent side by side, in the browser. It is a standalone app
that **uses** LeRobot (it is not part of the LeRobot source tree); it ships in
the same container as a LeRobot image so the shell and the agent act on the same
checkout.

![reference layout](docs/reference.png)

## What you get

- **Linux console** (left/bottom) — a real PTY into the container. Exactly what
  you'd get from `kubectl exec`, but in the browser. Run any LeRobot command:
  `lerobot-train`, `lerobot-eval`, `lerobot-record`, …
- **Chat** (right) — the [`hermes`](https://hermes.nousresearch.com) agent. Ask
  it to plan an SFT run, evaluate a checkpoint, inspect a dataset, etc. It can
  run safe commands itself; dangerous commands require approval (no `--yolo`).
- **Training monitor** (top-left) — a placeholder panel matching the reference
  layout (wire it to your own metrics source as needed).

## Chat needs a Volcengine (Ark) API key

The agent talks to **Volcengine Ark** (火山方舟). The **first time** you open
chat, a modal asks for your Ark API key. It is written to the hermes config and
**only affects chat** — the terminal and everything else are untouched. Re-enter
it any time via the ⚙ button in the chat header.

The chat **model + endpoint are owned by hermes' `config.yaml`** — the single source of truth,
seeded once in the console `Dockerfile` (`hermes config set model.default / model.base_url …`).
`server.py` reads them (`/api/status`) and the frontend just displays what hermes reports — nothing
in `server.py` / `index.html` / `app.js` hardcodes a model. To change the deployment default, edit
that one Dockerfile block.

- **Header dropdown** offers the preset alternatives from the `ARK_MODELS` env (also set in the
  Dockerfile, comma-separated) plus whatever hermes currently uses; switching writes `model.default`.
- **Add any model from the UI:** the ⚙ modal's *Advanced → 模型* field accepts an arbitrary model
  name (with base URL); it's saved to hermes config (persisted on the PVC) and becomes active. It
  shows in the dropdown while current; add it to `ARK_MODELS` to make it a permanent preset.
- `ARK_BASE_URL` / `ARK_MODEL` env are last-resort fallbacks only (no hardcoded value in code).

## Run locally (dev)

```bash
pip install -r requirements.txt        # aiohttp
hermes --version                       # the agent must be on PATH for chat
python server.py                        # serves http://0.0.0.0:8080
```

Environment knobs:

| var | default | meaning |
|---|---|---|
| `PORT` | `8080` | listen port (plain HTTP — the console never terminates TLS itself) |
| `CONSOLE_USER` / `CONSOLE_PASSWORD` | built-in pair | single-user HTTP Basic auth. **Auth is always on**: if either var is missing the console falls back to a default pair and says so in the UI, so a missing Secret degrades loudly rather than serving unauthenticated. |
| `CONSOLE_WORKDIR` | cwd | shell + agent working dir (the lerobot checkout). Do NOT use `LEROBOT_HOME` — that's lerobot's own deprecated cache var; the console removes it from the env so it never breaks lerobot CLIs. |
| `CONSOLE_SHELL` | `bash` | shell for the PTY console |
| `HERMES_BIN` | `hermes` | hermes executable |
| `HERMES_CHAT_SKILL` | `robot_sft` | skill preloaded into chat (empty to disable) |
| `HERMES_HOME` | `~/.hermes` | hermes config + session store location |
| `ARK_MODELS` | (Dockerfile) | comma-separated preset models for the chat dropdown; the live hermes model is always added |
| `ARK_BASE_URL` / `ARK_MODEL` | unset | last-resort fallbacks for base_url/model when hermes config has none (no hardcoded value in code; real defaults live in hermes, seeded by the Dockerfile) |

## The robot_sft skill (requirement f)

Chat preloads the `robot_sft` skill so the agent knows the LeRobot SFT pipeline.
The skill lives **in this repo** at `vendor/robot_sft/` and is maintained here as a
first-class part of the console — **edit it directly**. (It originated from
[`thesues/robot_sft`](https://github.com/thesues/robot_sft); see
`vendor/robot_sft.SOURCE` for that origin commit.) Neither the build nor a running
pod needs GitHub: the Dockerfile drops it into `$HERMES_HOME/skills/` (hermes
auto-discovers it) and bakes it into the seed; the entrypoint restores it into the
PVC with a local copy.

## Build & deploy on VKE

One image / one container = **lerobot + a slim hermes + this console** (`:8080`).
`kubectl exec` into the pod gives you the same env the console drives.

**hermes is installed slim** (see `Dockerfile`): `pip install "hermes-agent[acp]"`
from the PyPI mirror (no GitHub) into an isolated venv — ACP (stdio) chat + the
shell tool. No Node, no browser/Chromium, no ffmpeg, no messaging/matrix/voice/
web-search/mcp extras, no dashboard/TUI. ~100 MB of Python on top of the lerobot
base (validated:
`hermes acp --check` passes). `HERMES_DISABLE_LAZY_INSTALLS=1` stops it from
pip-installing optional tools at runtime.

```bash
# Separate pipeline. Push gzip (VKE node containerd 1.6.x rejects zstd). The
# lerobot base images are gzip, so the console image is gzip too.
# CONSOLE_COMMIT makes the UI's "更新说明" version banner show this build's commit; tag the
# image by the same commit. (The lerobot commit comes from BASE_IMAGE's tag automatically.)
COMMIT=$(git rev-parse HEAD)
docker buildx build \
  --build-arg BASE_IMAGE=iaas-us-cn-beijing.cr.volces.com/physicalai/lerobot:44effcd03b79f91b0c7ab4d30e757493ca049146 \
  --build-arg CONSOLE_COMMIT=$COMMIT \
  --output type=image,name=ai-containers-cn-beijing.cr.volces.com/physicalai/lerobot-agent-console:$COMMIT,push=true,compression=gzip,oci-mediatypes=true \
  .

# One-time: login secret + a self-signed TLS cert + the hermes PVC
kubectl create secret generic lerobot-console-auth \
  --from-literal=user=lerobot --from-literal=password='<strong-password>'
openssl req -x509 -nodes -newkey rsa:2048 -days 825 \
  -keyout tls.key -out tls.crt -subj "/CN=lerobot-console" \
  -addext "subjectAltName=DNS:lerobot-console"
kubectl create secret tls lerobot-console-tls --cert=tls.crt --key=tls.key
kubectl apply -f k8s/pvc.yaml

# Set your VPC subnet in k8s/service-lb.yaml, then deploy.
kubectl apply -f k8s/deployment.yaml       # console serves HTTPS natively (no sidecar)
kubectl apply -f k8s/service-lb.yaml       # public L4 CLB: 443 -> console:8080

# Get the CLB public IP and open it (self-signed -> accept the browser warning):
kubectl get svc lerobot-console-clb -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
# open https://<that-ip>/   (login = CONSOLE_USER / CONSOLE_PASSWORD)

# Exec into the pod and run anything directly:
kubectl exec -it deploy/lerobot-console -- bash
```

**TLS:** the console does **not** do TLS — it serves plain HTTP on `:8080` and
`server.py` contains no ssl code. **HTTPS is terminated upstream by APIG**
(`k8s/apig-ingress.yaml`), which owns the certificate and hands out a managed
`*.volceapi.com` domain, so there is no cert to mount and no env var to set.
WebSocket upgrades pass through it (verified 101). Each console (dev / test) sits
behind its own APIG instance and domain.

**Persistence & auth:** `HERMES_HOME=/opt/data` is a PVC, so the Ark key (entered
once in the UI), chat sessions, and the `robot_sft` skill survive pod restarts. The
whole console is behind single-user HTTP Basic auth (`CONSOLE_USER` /
`CONSOLE_PASSWORD`, injected from the `lerobot-console-auth` Secret), and enforces a
single open session at a time.

> **zstd ↔ VKE note:** the VKE node here runs containerd 1.6.38, whose zstd
> support is incomplete. Always push this image gzip-compressed. zstd is only
> safe on nodes with containerd ≥ 1.7.

## Remote robot teleop (self-hosted LiveKit)

To drive a SO-100 (or any lerobot robot) on your **home Mac** from the cloud, add a
self-hosted **LiveKit SFU** (`k8s/livekit/`). Both ends dial **out** to the SFU, so
it works behind home/cloud NAT.

```
  HOME Mac (SO-100 + cameras)                 VKE cloud
    mac_daemon ──dial OUT──► livekit-clb (public CLB, 7880/7881 TCP + 7882 UDP)
                                  │  ──► LiveKit SFU ◄── controller (runs IN the console pod,
                                  │                       started from the console terminal)
                                  ▼
    panel served on localhost:8088 in the pod ──► shown in the console viewer via /proxy
```

The **console pod is the controller host** — run `cloud_teleop_so100.py` from the
console terminal; its web panel is shown in the left viewer ("+ 打开" → 8088) over
the console's existing HTTPS, so **no separate web CLB is needed** (only LiveKit's CLB).

```bash
# Set the subnet-id in k8s/livekit/service-clb.yaml, then:
KUBECONFIG=~/Downloads/kube.conf ./scripts/deploy-livekit.sh
# It prints the Mac mac_daemon command + the in-console controller command.
```

The controller dials the **internal** `ws://livekit-clb:7880` (ClusterIP, no public
dep); the Mac dials the **public** `ws://<livekit-clb-ip>:7880`. Needs the webrtc
deps in the image (`aiortc livekit livekit-api`); recent lerobot images bake them via
`uv sync --extra all`. Full walkthrough + verification:
`examples/webrtc_remote_so100/deploy/README.md` in the lerobot repo.

## Endpoints

| route | purpose |
|---|---|
| `GET /` | the single-page UI |
| `GET /healthz` | unauthenticated health check (for LB/k8s probes) |
| `GET /api/status` | `{chat_ready, model, base_url, skill, workdir}` |
| `POST /api/volcano-key` | set the Ark api key for chat (`{api_key, base_url?, model?}`) |
| `GET /api/services` | discovered local services for the viewer |
| `WS /ws/term` | PTY shell bridge |
| `WS /ws/chat` | one hermes turn per message (streaming, ACP) |
| `WS /ws/control` | single-session presence lock |
| `ANY /proxy/{port}/…` | reverse-proxy to an in-pod service (HTTP + WS) |

## License

MIT

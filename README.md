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

Endpoint defaults to `https://ark.cn-beijing.volces.com/api/v3`, model
`deepseek-v4-pro-260425` (both overridable in the modal's *Advanced* section
or via `ARK_BASE_URL` / `ARK_MODEL`).

## Run locally (dev)

```bash
pip install -r requirements.txt        # aiohttp
hermes --version                       # the agent must be on PATH for chat
python server.py                        # serves http://0.0.0.0:8080
```

Environment knobs:

| var | default | meaning |
|---|---|---|
| `PORT` | `8080` | HTTP port |
| `LEROBOT_HOME` | cwd | shell + agent working dir; exported to all child processes (precedence: `LEROBOT_HOME` > `CONSOLE_WORKDIR` > cwd) |
| `CONSOLE_WORKDIR` | cwd | fallback working dir if `LEROBOT_HOME` unset |
| `CONSOLE_SHELL` | `bash` | shell for the PTY console |
| `HERMES_BIN` | `hermes` | hermes executable |
| `HERMES_CHAT_SKILL` | `robot_sft` | skill preloaded into chat (empty to disable) |
| `HERMES_HOME` | `~/.hermes` | hermes config + session store location |
| `ARK_BASE_URL` / `ARK_MODEL` | Ark Beijing / doubao | chat provider defaults |

## The robot_sft skill (requirement f)

Chat preloads the [`robot_sft`](https://github.com/thesues/robot_sft) skill so
the agent knows the LeRobot SFT pipeline. Install it into hermes with:

```bash
./scripts/install_skill.sh           # hermes skills install thesues/robot_sft
```

The Dockerfile installs it at **build time** (when GitHub is reachable) and bakes
the result into the image (`/opt/hermes-seed`). At runtime the entrypoint restores
it into the PVC with a **local copy** — so a running pod never needs GitHub access.

## Build & deploy on VKE

One image / one container = **lerobot + a slim hermes + this console** (`:8080`).
`kubectl exec` into the pod gives you the same env the console drives.

**hermes is installed slim** (see `Dockerfile`): a git checkout in an isolated
venv with only the `acp` extra — ACP (stdio) chat + the shell tool. No Node, no
browser/Chromium, no ffmpeg, no messaging/matrix/voice/web-search/mcp extras, no
dashboard/TUI. ~100 MB of Python on top of the lerobot base (validated:
`hermes acp --check` passes). `HERMES_DISABLE_LAZY_INSTALLS=1` stops it from
pip-installing optional tools at runtime.

```bash
# Separate pipeline: FROM a known-good GZIP lerobot tag (VKE node containerd 1.6.x
# rejects zstd — `number of layers and diffIDs don't match`).
docker buildx build \
  --build-arg BASE_IMAGE=iaas-us-cn-beijing.cr.volces.com/physicalai/lerobot:<gzip-tag> \
  --output type=image,name=<registry>/lerobot-console:<tag>,push=true,compression=gzip,force-compression=true,oci-mediatypes=true \
  .

# One-time: login secret + persistent volumes (Ark key/sessions/skills, and certs)
kubectl create secret generic lerobot-console-auth \
  --from-literal=user=lerobot --from-literal=password='<strong-password>'
kubectl apply -f k8s/pvc.yaml -f k8s/caddy-pvc.yaml

# Set your domain + ACME email in k8s/deployment.yaml (caddy sidecar env:
# CONSOLE_DOMAIN, ACME_EMAIL) and your VPC subnet in k8s/service-lb.yaml.
kubectl apply -f k8s/caddy-config.yaml
kubectl apply -f k8s/service.yaml          # ClusterIP (for port-forward/debug)
kubectl apply -f k8s/deployment.yaml       # console + Caddy auto-HTTPS sidecar
kubectl apply -f k8s/service-lb.yaml       # public L4 LB (443+80 -> Caddy)

# Point your domain's DNS (A record) at the LB's public IP:
kubectl get svc lerobot-console-lb -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
# then open https://<your-domain>  (Caddy auto-issues a Let's Encrypt cert)

# You can still exec into the pod and run anything directly:
kubectl exec -it deploy/lerobot-console -c console -- bash
```

**TLS (auto, option A):** a **Caddy sidecar** terminates HTTPS and
auto-provisions/renews a Let's Encrypt cert for `CONSOLE_DOMAIN` — you just point
a domain at the LB. The LB is plain **L4** (CLB) passing 443+80 through to Caddy;
WebSocket + Basic-auth pass through transparently. ACME needs inbound 80/443
reachable (HTTP-01); on inbound-blocked accounts switch Caddy to a DNS-01 challenge.
For VKE-managed L7 / a 证书中心 cert instead, see `k8s/ingress-alb.yaml` (option B).

**Persistence & auth:** `HERMES_HOME=/opt/data` is a PVC, so the Ark key (entered
once in the UI), chat sessions, and the `robot_sft` skill survive pod restarts;
Caddy's certs persist on their own PVC. The whole console is behind single-user
HTTP Basic auth (`CONSOLE_USER` / `CONSOLE_PASSWORD` via the Secret), and enforces
a single open session at a time.

> **zstd ↔ VKE note:** the VKE node here runs containerd 1.6.38, whose zstd
> support is incomplete. Always push this image gzip-compressed. zstd is only
> safe on nodes with containerd ≥ 1.7.

## Endpoints

| route | purpose |
|---|---|
| `GET /` | the single-page UI |
| `GET /api/status` | `{chat_ready, model, base_url, skill, workdir}` |
| `POST /api/volcano-key` | set the Ark api key for chat (`{api_key, base_url?, model?}`) |
| `WS /ws/term` | PTY shell bridge |
| `WS /ws/chat` | one hermes turn per message, with session continuity |

## License

MIT

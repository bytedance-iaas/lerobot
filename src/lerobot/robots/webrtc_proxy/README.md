# WebRTCProxyRobot

Cloud-side **proxy robot** that presents a Mac-tethered real robot (SO-ARM +
cameras) to LeRobot as if it were local. Control/AI logic runs in the cloud; the
real hardware stays on the user's MacBook; the two are bridged over **WebRTC**.

Full product context (topology, K8s/coturn, paradigm decision): see
[`/webrtc_proxy_robot_context.md`](../../../../webrtc_proxy_robot_context.md).
Feature ledger + status: [`/feature_list.md`](../../../../feature_list.md).

## Why a `Robot` subclass

Every LeRobot policy / record / teleop flow talks to hardware only through
`send_action` and `get_observation`. We implement a fake `Robot` cloud-side and run
the real one on the Mac, transporting **semantic action/obs** (not serial bytes or
USB packets). We subclass + register — never monkey-patch — so LeRobot upgrades
don't break us and the schema metadata (`observation_features` / `action_features`)
is declared correctly.

## Architecture (M1 — loopback)

```
 Mac side (offerer)                         Cloud side (answerer)
 CaptureAgent                               WebRTCProxyRobot  (Robot subclass)
  ├─ capture loop @ capture_fps              ├─ get_observation()  ← AlignmentBuffer
  │   ├─ joints+ts  ─ DataChannel state ───▶ │      (pairs by CAPTURE timestamp)
  │   ├─ {seq,ts}   ─ DataChannel framemeta ▶│
  │   └─ frame      ─ media track (H264/VP8)▶│   _ProxyEndpoint (async, bg loop)
  ├─ action handler ◀ DataChannel action ─── ┤   send_action()  → action DataChannel
  └─ watchdog (P0 safe-stop)                 └─ _EventLoopThread bridges sync↔async
```

- **`protocol.py`** — channel labels, reliability flags, JSON message schemas (incl. RPC).
- **`control.py`** — cloud-driven onboarding (M3): a reliable `control` DataChannel +
  request/response RPC. `DeviceInventory` is the OS seam; `ControlServer` (Mac)
  answers `list_ports` / `list_cameras` / `find_port_begin` / `find_port_result`;
  `ControlClient` (cloud) matches responses by id. Port/camera IDs stay Mac-local.
- **`alignment.py`** — `AlignmentBuffer`: thread-safe nearest-neighbour pairing of
  state↔frame by Mac-side `time.monotonic()` capture timestamp (难点 A). Public-net
  jitter becomes latency, never reordering.
- **`capture_agent.py`** — Mac endpoint (synthetic in M1). Owns the capture clock,
  pushes state/framemeta/video, applies actions, runs the **watchdog** (难点 C).
- **`proxy_robot.py`** — `WebRTCProxyRobot` (sync `Robot` API) + `_ProxyEndpoint`
  (async answerer) + `_EventLoopThread` (sync↔async bridge).
- **`signaling.py`** — `Signaling` protocol + in-process loopback pair + a
  `WebSocketSignaling` client (real relay).
- **`signaling_server.py`** — WebSocket signaling **relay**: pairs a daemon
  (`role=robot`) with a controller (`role=controller`) by session id, buffering SDP
  for late joiners. Standalone: `python -m ...signaling_server --port 8765`.
- **`mac_daemon.py`** — the persistent Mac-side daemon: connect → offer → serve one
  session → safe the arm on drop → loop. Standalone entrypoint with reconnect.
- **`demo_loopback.py`** — runnable single-machine (in-process) demo.

## Install

```bash
uv pip install --native-tls 'aiortc>=1.9.0,<2.0.0'   # or: uv sync --extra webrtc
```

## Manual verification

Run the self-contained loopback demo (synthetic Mac agent ↔ cloud proxy, one
machine, driven through the **synchronous** Robot API):

```bash
uv run python -m lerobot.robots.webrtc_proxy.demo_loopback
```

Expect: `observation_features`/`action_features` printed; ~30 re-assembled
observations (`shoulder_pan.pos` + `front=(120,160,3)uint8`, `skew≈0ms`); the P0
watchdog logging `SAFE STOP` once actions stop and clearing when they resume; a
clean disconnect.

The demo also exercises the **control plane**: `list_ports()`, `list_cameras()`, and
the two-step `find_port_begin()` → (user unplugs the bus) → `find_port_result()`.

### Device onboarding (port + camera IDs)

Physical IDs are Mac-local; the cloud config holds only logical names + resolution.
The cloud discovers them over the control channel instead of storing them:

```python
robot.list_ports()        # serial ports visible on the Mac
robot.list_cameras()      # [{type, index_or_path|serial, name}, ...]
before = robot.find_port_begin()   # snapshot; UI tells the user to unplug the bus
robot.find_port_result()           # the port that disappeared == the motor bus
```

`find_port` is split in two because the human unplugs the bus on the Mac — the
cloud cannot share that stdin, so the sync point moves to the Mac side (vs. the
blocking `input()` in `lerobot-find-port`).

### Real two-process link (Mac daemon ↔ cloud controller)

The cloud runs `WebRTCProxyRobot`; the Mac runs a long-lived **daemon** that outlives
any single cloud session. They meet on a WebSocket signaling relay. On one machine
(same-host, no STUN needed — `ice_servers=[]`):

```bash
# 1) signaling relay (lives cloud-side in prod)
python -m lerobot.robots.webrtc_proxy.signaling_server --port 8765
# 2) Mac daemon (synthetic source in M3; real so_follower + cameras in M2)
python -m lerobot.robots.webrtc_proxy.mac_daemon --signaling-url ws://127.0.0.1:8765/ws
# 3) cloud controller
python - <<'PY'
from lerobot.robots.webrtc_proxy.configuration_webrtc_proxy import WebRTCProxyRobotConfig, WebRTCCameraSpec
from lerobot.robots.webrtc_proxy.proxy_robot import WebRTCProxyRobot
cfg = WebRTCProxyRobotConfig(cameras={"front": WebRTCCameraSpec(480, 640, 30)},
                             signaling_url="ws://127.0.0.1:8765/ws")
r = WebRTCProxyRobot(cfg); r.connect()
print(r.get_observation().keys()); print(r.list_ports()); r.disconnect()
PY
```

Across the public internet the only change is `ice_servers=[...]` (STUN/TURN, M4);
the daemon registers from behind NAT, the relay never sees media.

Tests (suites needing the transport skip automatically without aiortc/aiohttp):

```bash
# NOTE: -p no:hydra_pytest works around an unrelated broken pytest plugin in this env.
uv run pytest tests/robots/test_webrtc_proxy_alignment.py \
              tests/robots/test_webrtc_proxy_loopback.py \
              tests/robots/test_webrtc_proxy_control.py \
              tests/robots/test_webrtc_proxy_daemon.py -p no:hydra_pytest -q
```

## Known limitations (M1 — to fix in later milestones)

- **framemeta 1:1 pop assumes a lossless link.** The cloud tags each decoded video
  frame with the next `framemeta` `{seq,t}` in order. On a real link with frame
  drops this de-syncs. Production must carry `seq` in an RTP header extension or
  in-pixel. (M3)
- **Single camera.** M1 transports one media track. Multi-camera = one track each. (M2)
- **Synthetic source.** `CaptureAgent._capture_sample` / `_apply_action` /
  `_safe_stop` are stubs; M2 wires them to a real `so_follower` + cameras.
- **Synthetic device inventory.** The control plane answers from `SyntheticInventory`;
  a real `LocalDeviceInventory` wrapping `lerobot-find-port` / `lerobot-find-cameras`
  lands with M2/M4 hardware bring-up.
- **Same-host networking only (so far).** WebSocket signaling + the daemon work, but
  with `ice_servers=[]` (host candidates) only same-host / same-LAN peers connect.
  Real public-net NAT traversal needs STUN/TURN(coturn) urls in `ice_servers` and a
  self-managed K8s media path (hostNetwork, announced/external IP) — that's M4 and
  needs real infra, untested here.
- **Daemon reconnect is per-session, single controller.** One `session_id` ↔ one
  daemon ↔ one controller at a time. Multi-tenant routing / auth on the relay is later.
- **send_action returns the optimistic goal** (no real clip/ack from the Mac yet). M2.
- **Paradigm not yet chosen** (real-time per-frame vs intent + local autonomy). M5;
  affects what the action DataChannel actually carries.

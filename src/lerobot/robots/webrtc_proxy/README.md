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

## Architecture

The cloud `WebRTCProxyRobot` is a **pure controller** — it reaches a remote Mac
daemon over a WebSocket signaling relay; it never embeds a Mac agent. (Tests/demo
run the relay + daemon + controller as separate loops in one process — see
`conftest.py` / `demo_loopback.py`.)

```
 Mac daemon (offerer)                       Cloud controller (answerer)
 CaptureAgent                               WebRTCProxyRobot  (Robot subclass)
  ├─ capture loop @ capture_fps              ├─ get_observation()  ← AlignmentBuffer
  │   ├─ joints + seq ─ DataChannel state ─▶ │      (pairs state↔frame by SEQ)
  │   └─ frame(seq in pts) ─ media track ──▶ │   _ProxyEndpoint (async, bg loop)
  ├─ action handler ◀ DataChannel action ─── ┤   send_action()  → action DataChannel
  └─ watchdog (P0 safe-stop)                 └─ _EventLoopThread bridges sync↔async
```

- **`protocol.py`** — channel labels, reliability flags, JSON message schemas (incl. RPC).
- **`control.py`** — cloud-driven onboarding (M3): a reliable `control` DataChannel +
  request/response RPC. `DeviceInventory` is the OS seam; `ControlServer` (Mac)
  answers `list_ports` / `list_cameras` / `grab_camera` / `find_port_*` /
  `set_camera_plan`; `ControlClient` (cloud) matches responses by id. Port/camera IDs
  stay Mac-local.
- **`alignment.py`** — `AlignmentBuffer`: thread-safe pairing of state↔frame by capture
  **seq** (难点 A; joints+frame share a seq, the frame's seq rides its pts). A dropped
  frame/state just skips that seq — no cascade. See `DESIGN.md` §5.1.
- **`transport.py`** — pluggable transport: the `Transport` interface (named data
  channels + a seq-tagged video stream) + `AiortcTransport` (default WebRTC P2P). The
  proxy logic is transport-agnostic, so a different backend (e.g. a LiveKit SFU for
  cross-public-net / scale) can implement `Transport` without touching the rest.
- **`capture_agent.py`** — Mac endpoint. Owns the capture clock, pushes state + video
  (seq in pts) via the transport, applies actions, runs the **watchdog** (难点 C).
- **`proxy_robot.py`** — `WebRTCProxyRobot` (sync `Robot` API) + `_ProxyEndpoint`
  (async answerer) + `_EventLoopThread` (sync↔async bridge).
- **`signaling.py`** — `Signaling` protocol + `WebSocketSignaling` client (real
  relay) + an in-process loopback pair (used internally by direct endpoint tests).
- **`signaling_server.py`** — WebSocket signaling **relay**: pairs a daemon
  (`role=robot`) with a controller (`role=controller`) by session id, buffering SDP
  for late joiners. Standalone: `python -m ...signaling_server --port 8765`.
- **`mac_daemon.py`** — the persistent Mac-side daemon: connect → offer → serve one
  session → safe the arm on drop → loop. Standalone entrypoint with reconnect.
- **`demo_loopback.py`** — runnable single-machine demo (relay + synthetic daemon +
  controller in one process): discovery, obs streaming, watchdog.
- **`sim_remote.py`** — simulate the *remote* control plane on one machine: relay +
  daemon + controller as three loops over localhost, then run one RPC and print the
  result. `python -m ...sim_remote --rpc list_cameras|list_ports|find_port|observe|all`.

## Install

```bash
uv pip install --native-tls 'aiortc>=1.9.0,<2.0.0'   # or: uv sync --extra webrtc
```

## Manual verification

Run the self-contained demo (relay + synthetic daemon + controller in one process,
driven through the **synchronous** Robot API):

```bash
uv run python -m lerobot.robots.webrtc_proxy.demo_loopback
```

Expect: `observation_features`/`action_features` printed; ~30 re-assembled
observations (`shoulder_pan.pos` + `front=(120,160,3)uint8`, `skew≈0ms`); the P0
watchdog logging `SAFE STOP` once actions stop and clearing when they resume; a
clean disconnect.

The demo also exercises the **control plane**: `list_ports()`, `list_cameras()`, and
the two-step `find_port_begin()` → (user unplugs the bus) → `find_port_result()`.

To simulate one such call over the *remote* path (relay + daemon + controller, three
loops, localhost sockets) without three terminals:

```bash
python -m lerobot.robots.webrtc_proxy.sim_remote --rpc list_cameras   # or list_ports / find_port / observe / all
python -m lerobot.robots.webrtc_proxy.sim_remote --rpc list_cameras --real-devices   # this machine's real devices
```

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

By default the daemon answers from `SyntheticInventory` (fake devices). Start it
with `--real-devices` to enumerate the Mac's **actual** ports + cameras via
`LocalDeviceInventory` (wraps lerobot's `find_available_ports` / `find_cameras`),
so the calls above return the same ids the stock `lerobot-find-port` /
`lerobot-find-cameras` CLIs would.

### Real two-process link (Mac daemon ↔ cloud controller)

The cloud runs `WebRTCProxyRobot`; the Mac runs a long-lived **daemon** that outlives
any single cloud session. They meet on a WebSocket signaling relay. On one machine
(same-host, no STUN needed — `ice_servers=[]`):

```bash
# 1) signaling relay (lives cloud-side in prod)
python -m lerobot.robots.webrtc_proxy.signaling_server --port 8765
# 2) Mac daemon. --real-devices => real find_port/list_cameras; --real-camera 0 =>
#    open & stream that opencv camera (index or /dev/videoN) instead of synthetic frames.
python -m lerobot.robots.webrtc_proxy.mac_daemon \
    --signaling-url ws://127.0.0.1:8765/ws --real-devices --real-camera 0 --width 640 --height 480
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

Across an ordinary NAT, add a STUN server to `ice_servers=[...]` so aiortc can form a
direct (srflx) UDP path; the daemon registers from behind NAT, the relay never sees media.
aiortc is **direct UDP only** — if a direct/STUN path can't be formed (symmetric NAT,
UDP blocked, HTTP-proxy-only egress) it gives up by design; for a relay, switch to the
**LiveKit backend** (below) rather than self-hosting TURN/coturn. See `DESIGN.md` §11.1.

Tests (suites needing the transport skip automatically without aiortc/aiohttp):

```bash
# NOTE: -p no:hydra_pytest works around an unrelated broken pytest plugin in this env.
uv run pytest tests/robots/test_webrtc_proxy_*.py -p no:hydra_pytest -q
```

## LiveKit backend (experimental, optional)

`aiortc` (default) is P2P and self-contained. The pluggable transport also has a
**LiveKit (SFU)** backend for cross-public-internet / NAT / scale: both ends dial
*outward* to a LiveKit server (LiveKit Cloud or self-hosted), so neither needs an
inbound path. Verified end-to-end against a local `livekit-server --dev` and LiveKit
Cloud. Install the extra: `uv sync --extra webrtc-livekit`.

Both ends must use `--transport livekit` (an aiortc peer and a LiveKit room don't
interoperate). The daemon needs no `--signaling-url` (LiveKit does its own signaling).
Each process **self-signs its own token** from a shared API key/secret — set the
LiveKit env vars once and you never paste a JWT (the room is `--session`; identities
default to `robot` / `controller`):

```bash
# local dev server (key/secret default to devkey/secret)
livekit-server --dev
export LIVEKIT_URL=ws://127.0.0.1:7880 LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secret

# Mac daemon (publisher) — self-signs identity=robot from the env
uv run python -m lerobot.robots.webrtc_proxy.mac_daemon --session so100 \
    --transport livekit --real-camera 0   # drop --real-camera for synthetic frames

# cloud controller (subscriber) — self-signs identity=controller
uv run python examples/webrtc_remote_so100/cloud_teleop_so100.py --transport livekit
```

For production, don't ship the API secret to the Mac — pass a pre-signed, scoped token
with `--livekit-token` (minted by a cloud token server) instead of the key/secret.

The opt-in e2e test runs the same two-process path:

```bash
LEROBOT_LIVEKIT_URL=ws://127.0.0.1:7880 \
    LEROBOT_LIVEKIT_API_KEY=devkey LEROBOT_LIVEKIT_API_SECRET=secret \
    uv run pytest tests/robots/test_webrtc_proxy_livekit.py -p no:hydra_pytest -q
```

NAT / restrictive-egress reachability (why SFU, not P2P) is covered in `DESIGN.md` §11.1.1.

## Known limitations (M1 — to fix in later milestones)

- **Frame seq rides `pts`, recovered relative to the first received frame.** Robust to
  mid-stream frame loss (a drop just skips a seq), but the receiver re-bases the first
  received frame to pts=0, so if the *initial* frame is lost the seq offset shifts.
  Mitigated by resetting seq per session; production should carry an absolute seq in an
  RTP header extension. See `DESIGN.md` §5.1.
- **Single camera.** M1 transports one media track. Multi-camera = one track each. (M2)
- **Real robot (M2).** Pass a connected lerobot `Robot` (e.g. `SO100Follower`) to the
  daemon (`run_daemon(robot=...)`): joints + camera come from one `robot.get_observation()`
  (shared capture instant), actions call `robot.send_action`, and the watchdog cuts
  torque via `robot.bus.disable_torque()`. All serial-bus access runs on one worker
  thread so the public-net loop never blocks and the bus is never touched concurrently.
  See `examples/webrtc_remote_so100`. Without a robot, the synthetic source (or a bare
  `--real-camera`) still works for transport testing.
- **Camera sizing.** The daemon opens at the requested capture size, falling back to
  native if the camera rejects it; `_fit_frame` + the cloud's defensive re-fit guarantee
  the declared obs shape, and the cloud pushes its spec via `set_camera_plan` at connect.
- **Device inventory: real but read-only.** `--real-devices` enumerates the Mac's
  actual ports + cameras (`LocalDeviceInventory`), so cloud-driven `find_port` /
  `list_cameras` return real ids. Default stays `SyntheticInventory`. Persisting the
  chosen port/camera→role mapping into a daemon config (and using it to open the bus)
  is M2.
- **aiortc reach = direct UDP only.** With `ice_servers=[]` (host candidates) only
  same-host / same-LAN peers connect; add a STUN url for ordinary cone-NAT (srflx) P2P.
  We deliberately do **not** self-host TURN/coturn for aiortc: if a direct/STUN path
  can't be formed, the answer is the **LiveKit backend** (it provides the relay), not
  extra aiortc infra. This drops the old M4 coturn / K8s-`hostNetwork` / announced-IP
  work, which existed only to relay aiortc P2P.
- **Daemon reconnect is per-session, single controller.** One `session_id` ↔ one
  daemon ↔ one controller at a time. Multi-tenant routing / auth on the relay is later.
- **send_action returns the optimistic goal** (no real clip/ack from the Mac yet). M2.
- **Paradigm not yet chosen** (real-time per-frame vs intent + local autonomy). M5;
  affects what the action DataChannel actually carries.

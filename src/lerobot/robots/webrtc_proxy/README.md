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

- **`protocol.py`** — channel labels, reliability flags, JSON message schemas.
- **`alignment.py`** — `AlignmentBuffer`: thread-safe nearest-neighbour pairing of
  state↔frame by Mac-side `time.monotonic()` capture timestamp (难点 A). Public-net
  jitter becomes latency, never reordering.
- **`capture_agent.py`** — Mac endpoint (synthetic in M1). Owns the capture clock,
  pushes state/framemeta/video, applies actions, runs the **watchdog** (难点 C).
- **`proxy_robot.py`** — `WebRTCProxyRobot` (sync `Robot` API) + `_ProxyEndpoint`
  (async answerer) + `_EventLoopThread` (sync↔async bridge).
- **`signaling.py`** — `Signaling` protocol + in-process loopback pair (M3 adds a
  WebSocket impl with the same interface).
- **`demo_loopback.py`** — runnable single-machine demo.

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

Tests (the loopback suite skips automatically without aiortc):

```bash
# NOTE: -p no:hydra_pytest works around an unrelated broken pytest plugin in this env.
uv run pytest tests/robots/test_webrtc_proxy_alignment.py \
              tests/robots/test_webrtc_proxy_loopback.py -p no:hydra_pytest -q
```

## Known limitations (M1 — to fix in later milestones)

- **framemeta 1:1 pop assumes a lossless link.** The cloud tags each decoded video
  frame with the next `framemeta` `{seq,t}` in order. On a real link with frame
  drops this de-syncs. Production must carry `seq` in an RTP header extension or
  in-pixel. (M3)
- **Single camera.** M1 transports one media track. Multi-camera = one track each. (M2)
- **Synthetic source.** `CaptureAgent._capture_sample` / `_apply_action` /
  `_safe_stop` are stubs; M2 wires them to a real `so_follower` + cameras.
- **Loopback signaling only.** No STUN/TURN/NAT traversal; `iceServers=[]`. Real
  public-net peers + self-managed K8s signaling/coturn are M3–M4.
- **send_action returns the optimistic goal** (no real clip/ack from the Mac yet). M2.
- **Paradigm not yet chosen** (real-time per-frame vs intent + local autonomy). M5;
  affects what the action DataChannel actually carries.

# WebRTCProxyRobot — Design Document

Status: living design. Implementation status per section is marked
`[done]` / `[partial]` / `[planned]`. Code lives in
`src/lerobot/robots/webrtc_proxy/`; user-facing setup in `README.md`; original
product handoff in `/webrtc_proxy_robot_context.md`.

---

## 1. Goal & constraints

Let a user run a **real robot (SO-ARM, STS3215 bus) + cameras plugged into their own
Mac**, while the **control/AI logic runs in our cloud**. The two are bridged over
**WebRTC**. This is a product, not a single-user research rig:

- **No GPU at the edge.** We cannot ask each customer to host a GPU box. Control,
  policy inference, recording all run cloud-side. (Rules out "move eval/record local".)
- **The Mac is behind NAT.** No public address; it must register outward.
- **Hardware stays on the Mac.** We transport *semantic* observations/actions, not
  serial bytes or USB packets. (Rules out socat serial-forward, usbip USB-passthrough —
  macOS can't host usbip anyway.)
- **Safety is P0.** A network drop must never leave the arm straining or stuck in a
  dangerous pose.

## 2. Why cut at the `Robot` abstraction (subclass, not monkey-patch)

Every LeRobot policy / record / teleop path talks to hardware only through
`get_observation()` and `send_action()`. So we implement a **fake cloud-side
`Robot`** and run the real one on the Mac. We **subclass + register**
(`@RobotConfig.register_subclass("webrtc_proxy")`) rather than monkey-patch, so:

- LeRobot upgrades don't shatter us;
- `observation_features` / `action_features` are declared correctly (datasets/policies
  read them for shapes);
- existing teleop/record/policy code drives the remote arm **with zero WebRTC-specific
  code** (see `examples/webrtc_remote_so100`).

## 3. Components & topology

```
 USER MAC (NAT'd)                  SIGNALING (FaaS)          CLOUD (K8s)
 ┌────────────────────┐            ┌──────────────┐          ┌────────────────────────┐
 │ mac_daemon.py      │  ws (SDP)  │ relay        │  ws(SDP) │ WebRTCProxyRobot        │
 │  CaptureAgent      │───────────▶│ pair by      │◀─────────│  (used by record /      │
 │  SO100Follower     │            │ session_id   │          │   policy / teleop pod)  │
 │  watchdog (P0)     │            └──────────────┘          │ _ProxyEndpoint          │
 │                    │                                      │ AlignmentBuffer         │
 │   ╞═══════════ WebRTC P2P (media track + DataChannels) ═══════════╡                 │
 │   │  camera→ media (RTP/UDP) ;  joints/action/control → DataChannels               │
 └───┼────────────────┘   (direct, or via coturn TURN relay if NAT blocks P2P)        │
     └──────────────────────────────────────────────────────────────┴────────────────┘
```

Three processes:
- **`mac_daemon.py`** `[done]` — long-lived on the Mac. Connect to the relay, offer,
  serve one cloud session, **safe the arm on drop**, loop for the next. Holds the real
  `SO100Follower` (reused across sessions).
- **`signaling_server.py`** `[done, single-instance]` — pairs a `robot` and a
  `controller` by `session_id`, forwards SDP, buffers for the late joiner, sends `bye`
  on drop. *Carries only signaling — never media.* (FaaS deployment: §11.2.)
- **`WebRTCProxyRobot`** (`proxy_robot.py`) `[done]` — the cloud `Robot`. Pure
  controller; owns a bg asyncio loop running `_ProxyEndpoint` (the WebRTC answerer) and
  bridges the synchronous `Robot` API to it.

## 4. Transport: channels & reliability

WebRTC gives us two kinds of pipe; we use both deliberately (`protocol.py`):

| Carrier | Payload | Reliability | Why |
|---|---|---|---|
| **media track** (RTP/UDP) | camera frames (VP8/H.264) | lossy, no retransmit | raw 640×480×3@30 ≈ 220 Mbps (why we must encode, never raw on a DataChannel); the *encoded* track is ≈1–5 Mbps, adaptive |
| DataChannel `state` | joints + capture `t`,`seq` + applied feedback | **configurable** (default unreliable) | see profiles below |
| DataChannel `action` | goal joints + `seq` + `obs_seq` | **configurable** (default unreliable) | see profiles below |
| DataChannel `control` | onboarding RPC (find_port, list_cameras, grab, plan) | **reliable, ordered** (always) | one-shot commands must arrive |

Never put images on a DataChannel — bandwidth blows up (handoff 难点 A).

**Absolute, not delta, actions.** Goals are absolute `<motor>.pos`. A dropped action
just means the next absolute goal corrects the arm — no accumulating error. Deltas
would be unsafe over a lossy channel.

### 4.1 state/action reliability is per use case `[done]`

`control` is always reliable+ordered. `state`/`action` are configurable
(`reliable_state`/`reliable_action`; `--profile teleop|eval|record` on the daemon),
because the requirements genuinely differ:

| | teleop | eval (policy) | record (dataset) |
|---|---|---|---|
| **state** | unreliable — freshest wins | unreliable — freshest obs → action | **reliable** — never lose an obs |
| **action** | unreliable — stale cmd useless | unreliable | **reliable** — never lose a transition's action |

Realtime closed loops (teleop/eval) prize **freshness**: a lost packet on a *reliable*
channel head-of-line-blocks until retransmitted, delivering a stale sample late — worse
than dropping it and waiting 33 ms for the next. Recording prizes **completeness**: both
state and action are reliable so the dataset has no missing obs or actions. Either way,
total disconnect is still caught by the watchdog (§7), and absolute actions self-correct
after a drop.

Note (current limitation): channels are created by the Mac daemon (the offerer), so the
profile is set **daemon-side** today. Controller-driven selection per session (push the
profile through signaling before the offer) is a later refinement.

## 5. Observation assembly — pairing camera & joints `[done]`

Camera (media track) and joints (state channel) traverse the net with **independent
jitter/loss**. Pairing by *arrival* time would let jitter corrupt temporal ordering of
a dataset. Joints and the camera frame of one cycle come from a single
`robot.get_observation()` on the Mac, so they **share a capture seq**; the cloud
`AlignmentBuffer` pairs `frame.seq == state.seq` — exact, and robust to loss (a dropped
frame/state just skips that seq).

`get_observation` returns the freshest seq present on **both** sides. If the newest seq
is incomplete (its frame or state hasn't arrived / was dropped) it falls back to the
previous complete seq, or holds the last obs on a stall — never a fresh-joints /
stale-frame mismatch.

### 5.1 How a frame carries its capture seq

A decoded video frame arrives **naked**: its RTP `pts` is re-stamped, so it carries no
application-level `seq`. We encode the seq into the frame's `pts`:
`pts = seq * VIDEO_PTS_PER_SEQ`, and the cloud recovers
`seq = round(pts · time_base · clock / VIDEO_PTS_PER_SEQ)`. pts survives VP8/H.264; the
cloud then pairs `frame.seq == state.seq`. A dropped frame just skips a seq — no cascade.

**Caveat:** the receiver re-bases the *first received* frame to `pts=0`, so seq is
recovered relative to the first received frame. The daemon resets seq to 0 per session,
so as long as the first frame lands (true at session start) relative == absolute; loss
of the *initial* frame would shift the offset.

**End state:** carry an absolute seq in an **RTP header extension** (frame
self-describes, no re-basing) once the media stack supports custom header extensions —
aiortc's support is limited today. (Pixel-embedding a seq is rejected: it pollutes the
recorded image.)

## 6. Control loop & RTT — the paradigm decision `[planned, M5]`

`get_observation`/`send_action` look local and instant to callers, but each now costs a
public-net RTT. RTT does not vanish by changing the abstraction; it moves to the RPC
boundary (handoff 难点 B). 50 ms RTT ⇒ ~20 Hz, jittery.

Two product paradigms (must be chosen — it shapes what the action channel carries):
- **Real-time per-frame teleop** — every action crosses the net; hand-feel RTT-locked.
- **Intent + local autonomy** — the tight loop closes on the Mac; the net carries
  high-level intent + monitoring video + occasional takeover. An order of magnitude more
  RTT-tolerant.

Cross-net real-time *synchronization* is physically impossible; we make the loop
**traceable** (§9) and pick a paradigm.

## 7. Safety — watchdog `[done, P0]`

The Mac-side `CaptureAgent` watchdog: if no action arrives within `action_timeout_s`, it
**cuts motor torque** (`robot.bus.disable_torque()`) so the arm goes limp instead of
holding/straining. Torque is re-enabled at session start and when actions resume. All
serial-bus access runs on a single worker thread, so the public-net loop never blocks on
serial and the bus is never touched concurrently.

## 8. Control plane — cloud-driven onboarding `[done]`

Physical IDs (serial port, camera index/serial) are **Mac-local** and never enter the
cloud config (the cloud declares only logical names + resolution). They're discovered
over the reliable `control` DataChannel (`control.py`):

- `list_ports`, `list_cameras` (real via `LocalDeviceInventory`, hushing the noisy
  OpenCV/RealSense probe).
- `find_port` is **event-driven two-step** (`begin` → user unplugs the bus on the Mac →
  `result` diffs) because the human is at the Mac, not sharing the cloud's stdin.
- `grab_camera` returns one JPEG frame of a chosen camera for an onboarding preview
  (over the control channel — distinct from the continuous obs media track).
- `set_camera_plan` lets the cloud push its desired `{w,h,fps}` so the Mac encodes to it
  (bandwidth); correctness doesn't depend on it — `get_observation` re-fits to the spec.

## 9. Traceability — provenance & applied feedback `[done]`

So each transition is reconstructable across the data/track split:
- `ActionMsg.obs_seq` — the cloud stamps each action with the `seq` of the obs it was
  derived from.
- `StateMsg.applied_seq/applied_t` — the Mac piggybacks "last action I applied (seq +
  time)" on the 30 Hz state stream (no extra channel), so the cloud confirms landing and
  measures round-trip / counts dropped actions.

## 10. Packet-loss behaviour (summary)

- **media**: frames drop (UDP); needs seq-keyed re-identification (§5.1) + skew drop as a
  safety net.
- **state/action**: configurable (§4.1); unreliable by default — absolute positions +
  seq pairing + watchdog absorb loss; reliable for record so no obs is lost.
- **control/signaling**: reliable; no loss unless the connection dies.

---

## 11. Deployment

### 11.1 Cloud control plane in **K8s** `[planned, M4]`

What runs in K8s is **whatever consumes `WebRTCProxyRobot`** — a record job, a policy
inference server, or a teleop-session backend pod. Each session = one controller pod ↔
one Mac daemon.

**Decision (relay strategy).** The two backends have a clean, non-overlapping division
of labour and we do **not** build a TURN relay for aiortc:

- **aiortc = direct UDP P2P only.** Host candidates, plus *optional* STUN (`ice_servers`)
  to get a server-reflexive candidate through ordinary cone NATs (STUN is stateless and
  can use a free public server, so it costs nothing). If a direct/STUN path can't be
  formed (symmetric NAT, UDP blocked, HTTP-proxy-only egress), aiortc **gives up** — we
  deliberately do *not* self-host coturn to relay it.
- **Need a relay → switch to the LiveKit backend.** LiveKit's SFU already bundles
  signaling + TURN + scale and is tested; duplicating that with coturn+aiortc is wasted
  effort and the heaviest, most error-prone ops (public IP, port ranges, credential
  rotation, bandwidth). So "relay" is a *backend choice*, not extra aiortc infra.

This deletes the original M4 coturn / `hostNetwork` / announced-IP burden: those existed
only to make aiortc P2P relay across hard NATs. With relay delegated to LiveKit, the
aiortc deployment stays simple (host + optional STUN), and hard-network cases use
`transport_backend="livekit"`.

The **media path is still the constraint** for whichever backend you pick:

- **aiortc:** UDP P2P. A controller pod that must terminate aiortc media needs a routable
  path (e.g. `hostNetwork`/node IP) and correctly **announced external IP** — the classic
  failure is "signaling connects, SDP exchanges, but media never flows" from a wrong
  announced IP. This is exactly the complexity that pushes anything non-trivial to LiveKit.
- **LiveKit:** both peers dial *outward* to the SFU; no inbound path / announced IP / port
  range to manage on our side. `aiortc` (default) gives decoded `ndarray` frames straight
  into the LeRobot pipeline — great for the adapter, but single-connection / weak at scale.
  The transport is pluggable (`transport.py` `Transport` interface; pick via
  `transport_backend`): `AiortcTransport` is the default; `transport_livekit.py` is a
  working (experimental, optional) `LiveKitTransport`, the chosen SFU because it handles
  signaling + TURN + scale AND has a Python SDK to pull frames into LeRobot. It is
  **verified end-to-end** against both a local `livekit-server --dev` and LiveKit Cloud
  (obs + action + control round-trip with fresh aligned observations); see the opt-in
  `tests/robots/test_webrtc_proxy_livekit.py`.

### 11.1.1 NAT / restrictive-egress reachability `[why SFU, not P2P]`

The two ends often sit in asymmetric network conditions; this drives the backend choice:

- **aiortc P2P = direct UDP only.** Host candidates + optional STUN (srflx); it forms a
  path only when one is directly reachable. It has **no TURN relay we operate** and **no
  support for routing media through an HTTP forward proxy**. So symmetric NAT, UDP-blocked,
  or `HTTP(S)_PROXY`-only egress → aiortc simply can't connect. That is by design: the
  answer is not "add coturn", it is "use the LiveKit backend".
- **LiveKit (SFU) = the relay.** **Both** peers *dial outward* to the SFU — neither needs
  an inbound path — so it covers the **NAT side** cleanly (outbound + SFU-side TURN). For a
  **proxy-only side**:
  - *signaling* (wss:443) traverses the HTTP proxy fine (it is just HTTPS/CONNECT);
  - *media* still needs the WebRTC client to tunnel ICE/TURN (TURN/TLS:443) through that
    proxy, which libwebrtc supports only partially and the Python SDK does not expose.
    Typical failure shape: **room connects, no track/data flows** — verify on the real
    proxied host; if media won't traverse, place the controller where it has direct egress
    (the usual cloud deployment, which is the normal topology anyway).

Scaling note: one controller pod per active session (stateful, holds a PeerConnection +
the bg loop). Autoscale on session count; sessions are sticky to their pod for their
lifetime.

### 11.2 Signaling in **FaaS** `[planned]`

Signaling is low-traffic (a handful of SDP messages per session) and bursty — a good
serverless fit. **But a WebSocket relay is awkward on vanilla FaaS**, and this needs
care:

- **The hard part:** the relay must hold *two long-lived WebSocket connections* (robot +
  controller) and forward between them. Plain FaaS is request/response and
  short-lived; the two peers may land on **different ephemeral instances** with no shared
  memory, so in-process forwarding (what `signaling_server.py` does today) doesn't
  translate directly.
- **Recommended FaaS shapes:**
  1. **Cloudflare Durable Objects (cleanest).** One DO instance *per `session_id`* — both
     peers route to the *same* single-threaded stateful actor, which holds both sockets
     and relays in-memory + buffers the early offer. This is essentially the current
     in-process room, but the platform guarantees per-session affinity. Workers stay
     serverless; the DO is the per-session rendezvous.
  2. **AWS API Gateway WebSocket + Lambda + a shared store.** API GW holds the sockets;
     Lambda handles `$connect`/`$message`/`$disconnect`. Map `session_id → {robotConnId,
     controllerConnId}` and buffer the early offer in **DynamoDB/Redis**; forward by
     calling the API GW management API to push to the peer's connection id. (Azure: Web
     PubSub / SignalR Service is the equivalent.)
  3. **火山引擎 veFaaS "Web 应用" (single-instance, multi-concurrency).** veFaaS can host a
     long-running web server (configurable listen port, 单实例并发数, timeout). Run
     `signaling_server.py` **unchanged** as a Web 应用 with **单实例并发数 ≥ 2** and the
     instance count pinned to **1** — both the robot and controller WebSockets land on the
     *same* warm instance, so the in-process `rooms` dict works as-is. Ideal for the
     **single-user** case: no external store, the platform gateway terminates TLS (wss for
     free on a bound domain), and `--port`/`--auth-token` read from `$PORT`/
     `$SIGNALING_AUTH_TOKEN`. Caveats: keep min-instances ≥ 1 (or accept a cold start on
     the daemon's first connect) and don't scale out — a second instance would split the
     two peers. (Same single-instance trick on any FaaS that supports a long-running web
     server + per-instance concurrency.)
- **Porting `signaling_server.py`:** for the multi-instance shapes (1, 2) the in-memory
  `rooms`/`inbox` dicts become external per-session state (DO memory, or DynamoDB/Redis);
  the single-instance shape (3) needs no change at all. The wire protocol
  (`?session=&role=`, `{kind:"sdp"|"bye"}`) is unchanged either way, so `WebSocketSignaling`
  (client) and the daemon/controller need **no changes**.
- **Auth lives here** (§12): the FaaS `$connect`/handshake is the natural place to
  validate a session token before pairing.
- **TURN credentials:** the signaling FaaS is also the natural issuer of short-lived
  coturn credentials (TURN REST API) handed to each peer at session start.

So the split is clean: **signaling = stateless-ish FaaS** (cheap, bursty, per-session
affinity via DO/store); **media = P2P or coturn** (never in FaaS); **control logic =
K8s pods**.

### 11.3 Session lifecycle across the system

1. Mac daemon boots → opens WS to the FaaS relay (`role=robot`, `session_id`) → creates
   the WebRTC offer → relay **buffers** it.
2. A cloud controller pod starts a session → opens WS (`role=controller`, same
   `session_id`) → relay flushes the buffered offer → controller answers → relay forwards.
3. ICE (STUN, else coturn) establishes the P2P media+data path. **Relay drops out of the
   data path.**
4. Stream obs / send actions / run control-plane RPCs.
5. Session ends or drops → relay `bye` to the survivor → daemon **safes the arm**, resets,
   loops for the next session (it outlives any one session).

## 12. Security & multi-tenancy

- **Shared token** `[done]` — `signaling_server --auth-token <str>`; every peer presents
  it (`Authorization: Bearer …`, constant-time compared) or is rejected (401) before
  pairing. Gates the door against scanners. **Limitation:** one token for everyone — it
  does NOT isolate sessions/tenants (anyone holding it can join any `session_id`).
- **Per-session signed token** `[planned]` (FaaS `$connect`): a short-lived JWT binds a
  user to a `session_id` and role; reject mismatched pairings. This is what stops
  cross-tenant hijacking; the shared token is only the first gate.
- **DTLS-SRTP** encrypts media/data end-to-end for free (WebRTC mandatory).
- **TURN credentials** are short-lived, per-session (TURN REST).
- **Daemon identity:** the Mac daemon authenticates to the relay; a stolen `session_id`
  must not let an attacker drive someone's arm. Tokens + per-daemon keys.

## 13. Milestones & status

| M | Scope | Status |
|---|---|---|
| M1 | Loopback transport (channels, alignment, watchdog) | `[done]` |
| M2 | Real `so_follower` (joints/action/torque) + SO-100 example | `[done]` |
| M3 | WS signaling + Mac daemon + control plane (discovery, plan, grab) | `[done]` (same-host) |
| — | Seq-based obs pairing (pts carrier); provenance + applied feedback; configurable channel reliability | `[done]` |
| M4 | Public-net: coturn, K8s media (hostNetwork/announced IP), FaaS signaling, auth | `[planned]` |
| M5 | Paradigm: real-time vs intent+local-autonomy; SFU for scale | `[planned]` |

## 14. Open questions

1. **Frame-seq carrier** (§5.1): move from pts (re-basing caveat) to an RTP header
   extension for an absolute seq. Gating: aiortc header-extension support / SFU choice.
2. **Paradigm** (§6): real-time per-frame vs intent + local autonomy — gates the action
   channel design.
3. **Media plane at scale**: aiortc-per-session vs LiveKit/mediasoup SFU.
4. **FaaS signaling target**: single-user → 火山 veFaaS "Web 应用" single-instance
   (relay runs unchanged, §11.2.3). Multi-tenant → Durable Objects vs API-GW-WS + store
   (externalize the room state).

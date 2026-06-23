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
| **media track** (RTP/UDP) | camera frames (VP8/H.264) | lossy, no retransmit | uncompressed 30 Hz ≈ 220 Mbps — must encode; loss tolerated |
| DataChannel `state` | joints + capture `t`,`seq` + applied feedback | **unreliable** (`ordered:false, maxRetransmits:0`) | stale joints are useless; drop, don't retransmit |
| DataChannel `action` | goal joints + `seq` + `obs_seq` | **unreliable** | absolute goal positions self-correct next tick |
| DataChannel `control` | onboarding RPC (find_port, list_cameras, grab, plan) | **reliable, ordered** | one-shot commands must arrive |

Never put images on a DataChannel — bandwidth blows up (handoff 难点 A).

**Absolute, not delta, actions.** Goals are absolute `<motor>.pos`. A dropped action
just means the next absolute goal corrects the arm — no accumulating error. Deltas
would be unsafe over a lossy channel.

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

### 5.1 How a frame gets its capture timestamp — the hard sub-problem

A decoded video frame arrives **naked**: its RTP `pts` is re-stamped, so it carries
neither the Mac's monotonic `t` nor the capture `seq`. We must re-attach identity.
Options (this is the main open hardening item):

1. **`framemeta` side-channel** `[removed]` — a reliable, ordered DataChannel carrying
   `{seq, t}` per frame, popped 1:1 by order. **Broke on media-frame loss**: framemeta
   (reliable) kept all entries, the track dropped frames → every subsequent frame got the
   wrong timestamp, cascading. Valid only on a lossless link. Replaced by (2).
2. **Encode `seq` in the frame `pts`** `[done]` — `pts = seq * VIDEO_PTS_PER_SEQ`;
   recover `seq = round(pts·time_base·clock / STEP)`. pts survives VP8 cleanly; the cloud
   pairs `frame.seq == state.seq`. A dropped frame just skips a seq — **no cascade**.
   **Caveat:** the receiver re-bases the *first received* frame to `pts=0`, so seq is
   relative to the first received frame; the daemon resets seq to 0 per session, so as
   long as the first frame lands (true at session start) relative == absolute. Initial
   frame loss shifts the offset — fix with (3).
3. **RTP header extension (abs-capture-time / custom `seq`)** `[ideal]` — frame
   self-describes; no re-basing, no side channel, no pixel touch. **Blocked by** limited
   custom-header-extension support in aiortc.
4. **Pixel-embedded seq** `[rejected]` — robust through codec but pollutes the recorded
   image (unless cropped). Not for a dataset product.

**Current state:** seq-keyed pairing via (2) — `framemeta` removed, `AlignmentBuffer`
matches by seq. Move the carrier to (3) when the media stack allows (kills the
re-basing caveat). Nearest-t pairing would only be needed if camera and joints were ever
decoupled into independent different-rate streams (they aren't — one get_observation).

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
- **state/action**: intentionally unreliable; absolute positions + nearest/seq pairing +
  watchdog absorb loss.
- **control/signaling**: reliable; no loss unless the connection dies.

---

## 11. Deployment

### 11.1 Cloud control plane in **K8s** `[planned, M4]`

What runs in K8s is **whatever consumes `WebRTCProxyRobot`** — a record job, a policy
inference server, or a teleop-session backend pod. Each session = one controller pod ↔
one Mac daemon.

The **media path is the constraint**, not the CPU:

- **UDP, large dynamic port range.** Put the media-terminating pod on
  `hostNetwork: true` (or a node with a routable IP) so RTP/UDP isn't mangled by
  Service/NAT port mapping.
- **ICE must advertise the node's public/external IP**, not the pod's cluster IP. The
  classic failure: signaling connects, SDP exchanges, but **media never flows** — almost
  always a mis-set announced/external IP.
- **STUN** (cheap, self-host) for hole-punching; **TURN (coturn)** as the fallback when
  P2P fails — **mandatory in production**, it relays media and eats bandwidth. Deploy
  coturn on `hostNetwork`/dedicated nodes with a public IP and an open relay port range;
  inject its URLs into `ice_servers` on both peers (`RTCConfiguration`).
- **Media framework choice.** `aiortc` (current) gives decoded `ndarray` frames straight
  into the LeRobot pipeline — perfect for the adapter, but single-connection / weak at
  scale. For many concurrent users, move the media plane to **LiveKit / mediasoup** (SFU)
  and demote aiortc to the "stream → LeRobot obs" adapter.

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
- **Porting `signaling_server.py`:** the in-memory `rooms`/`inbox` dicts become the
  external per-session state (DO memory, or DynamoDB/Redis). The wire protocol
  (`?session=&role=`, `{kind:"sdp"|"bye"}`) is unchanged, so `WebSocketSignaling` (client)
  and the daemon/controller need **no changes**.
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

## 12. Security & multi-tenancy `[planned]`

- **AuthN/Z at signaling** (FaaS `$connect`): a session token binds a user to a
  `session_id` and to *their* daemon; reject mismatched pairings. One daemon ↔ one
  controller per `session_id` (no cross-tenant routing today).
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
| — | Provenance + applied feedback; skew-drop; seq carrier investigation | `[done]` / `[partial]` |
| M4 | Public-net: coturn, K8s media (hostNetwork/announced IP), FaaS signaling, auth | `[planned]` |
| M5 | Paradigm: real-time vs intent+local-autonomy; SFU for scale | `[planned]` |

## 14. Open questions

1. **Frame-seq carrier** (§5.1): RTP header extension vs pts-with-anchor vs keep
   framemeta-by-seq. Gating: aiortc header-extension support / SFU choice.
2. **Paradigm** (§6): real-time per-frame vs intent + local autonomy — gates the action
   channel design.
3. **Media plane at scale**: aiortc-per-session vs LiveKit/mediasoup SFU.
4. **FaaS signaling target**: Durable Objects vs API-GW-WS + store — drives how
   `signaling_server.py`'s room state is externalized.

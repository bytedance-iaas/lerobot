# WebRTCProxyRobot — Feature Ledger (需求账本)

> 任务开始后，本文件的需求描述/验收步骤/测试标准**不可改写**；只允许更新「状态」字段。
> 背景上下文见 `webrtc_proxy_robot_context.md`。Robot 接口契约见 `src/lerobot/robots/robot.py`
> （当前版本：`get_observation`，`observation_features`/`action_features` 为 property）。

## 里程碑划分

- **M1（本会话范围）**：最小可跑 aiortc 回环链路（handoff §5）。单机/localhost 打通
  transport + 时间戳重组，paradigm-agnostic（先做裸 obs/action RPC 管子）。
- M2：接真实 so_follower + 真实相机（Mac 端）。
- M3：独立信令 WebSocket + STUN/TURN（coturn），跨公网。
- M4：自管 K8s 部署（hostNetwork、announced IP、coturn）。
- M5：paradigm 落地（intent + 本地自主执行 vs 实时遥操作）。

---

## M1 Features

### F1 — Transport 引导（回环对等连接）
- **目标**：用 aiortc 建立两个 `RTCPeerConnection`，回环（无 STUN）交换 SDP，开三条逻辑通道：
  `state`(Mac→cloud, unreliable)、`action`(cloud→Mac, unreliable)、`framemeta`(Mac→cloud, ordered)，
  外加一路 video media track(Mac→cloud)。
- **验收**：回环 `connect()` 后双方 connectionState=connected，三条 DataChannel open，track 收到帧。
- **状态**：completed

### F2 — Mac 采集端（假 obs push）
- **目标**：`CaptureAgent` 定频（默认 30Hz）生成「假」一帧：关节角(6 motor)+`time.monotonic()`
  时间戳走 `state` DataChannel；一路合成相机帧走 media track；每帧 `{seq,t}` 走 `framemeta`。
- **验收**：cloud 端能收到 state 消息（含单调时间戳）与 video 帧，速率 ≈ 配置频率。
- **状态**：completed

### F3 — 时间戳对齐 + 云端重组
- **目标**：`AlignmentBuffer` 按采集时间戳做最近邻配对，把 state(关节) + 最近邻 video 帧
  重组成 LeRobot 期望的 obs dict：`{"<motor>.pos": float, "<cam>": HxWx3 ndarray}`。
  云端按**采集时间戳**配对，不按到达时刻。
- **验收**：`get_observation()` 返回的 dict 键与 `observation_features` 一致；
  打印的 state/frame 时间戳偏差（skew）在容差内。
- **状态**：completed
- **prototype 简化（明确标注）**：M1 中 video 帧的采集时间戳通过 `framemeta` DataChannel
  以「有序 1:1 弹出」方式贴回（回环无丢帧成立）。生产需用 RTP header extension / 像素内嵌
  携带 seq，见 README「已知局限」。

### F4 — send_action 回路
- **目标**：cloud `send_action(action)` 把 action(+seq+t) 经 `action` DataChannel 发到 Mac；
  Mac 端「应用」后回 ack（applied action）；`send_action` 返回实际下发的 action（契约要求）。
- **验收**：cloud 调 `send_action({"<motor>.pos": x})`，Mac 端收到对应 action；返回值非 None。
- **状态**：completed

### F5 — 看门狗（P0 安全）
- **目标**：Mac 端 `CaptureAgent` 维护 watchdog：超过 `action_timeout_s` 未收到新 action →
  调用 `on_safe_stop` 回调（M1 用日志/标志位代替停力/回安全位）。
- **验收**：停止发 action 后，watchdog 在 timeout 内触发一次 safe-stop；恢复发 action 后解除。
- **状态**：completed

### F6 — WebRTCProxyRobot(Robot) 子类 + 注册
- **目标**：实现并 `@RobotConfig.register_subclass("webrtc_proxy")`，正确代理
  `connect/disconnect/get_observation/send_action/observation_features/action_features/is_connected`
  /`is_calibrated`/`calibrate`/`configure`，schema 由配置（motor 名 + 相机 h/w）声明，镜像 so_follower。
- **验收**：实例化后 `observation_features`/`action_features` 形状正确；`get_observation` 走 F3，
  `send_action` 走 F4。
- **状态**：completed

### F7 — 可跑回环 demo + 测试
- **目标**：`demo_loopback.py` 单进程跑通 Mac 端↔cloud 端，打印 ~30Hz 重组 obs + watchdog 演示；
  `tests/` 下加 alignment 单测 + 回环集成 smoke 测试（aiortc 缺失时 skip）。
- **验收**：`uv run python -m lerobot.robots.webrtc_proxy.demo_loopback` 正常打印并干净退出；
  `uv run pytest tests/robots/test_webrtc_proxy*.py` 通过。
- **状态**：completed

---

## 不在 M1 范围（不要回头做）
- 真实串口/相机（M2）、信令服务/STUN/TURN（M3）、K8s/coturn（M4）、paradigm 落地（M5）。
- 已否决方案：socat 串口转发、usbip USB 透传、把 record/eval 挪回本地（见 context §6）。

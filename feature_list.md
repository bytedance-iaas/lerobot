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

## M3 Features —— 控制面 / 设备开通（cloud-driven onboarding）

> 决策（2026-06-22）：设备发现（串口 / 相机 ID）走**云端驱动的控制面 RPC**，不做
> Mac 本地指纹方案（`resolve_cameras.py` 只是用户个人便利脚本，非标准做法，不并入产品）。
> 拆成两半：**控制面（可回环测）** 与 **信令/STUN/TURN 基础设施（需真网络，本机不可测）**。

### F8 — 控制 DataChannel + RPC（可回环测）
- **目标**：新增 `control` DataChannel（ordered+reliable，区别于实时 state/action）；
  其上跑 request/response RPC（`RpcRequest{id,method,params}` / `RpcResponse{id,ok,result,error}`）。
  Mac 端 `ControlServer` 派发到 `DeviceInventory`；云端 `ControlClient` 按 id 配对 future。
- **验收**：云端经控制面调一个 RPC，Mac 端处理并返回；按 id 正确配对、超时可控。
- **状态**：completed

### F9 — 设备发现 RPC（event-driven，可回环测）
- **目标**：把 `lerobot-find-port` 的阻塞 `input()` 改成事件驱动两段式：
  `find_port_begin`(快照 ports_before) → 人拔线 → `find_port_result`(差集出消失的口)。
  外加 `list_ports` / `list_cameras`（枚举 + metadata，对应 `lerobot-find-cameras`）。
  云端 `WebRTCProxyRobot` 暴露 `list_ports()/list_cameras()/find_port_begin()/find_port_result()`。
- **验收**：回环下 `find_port_begin`→模拟拔线→`find_port_result` 返回正确串口；
  `list_cameras` 返回带稳定标识（opencv index_or_path / realsense serial）的清单。
- **状态**：completed（SyntheticInventory + 真 `LocalDeviceInventory`，daemon `--real-devices`
  开关枚举 Mac 真实串口/相机；真实串口已经控制面回环验证。把选定的 port/camera→role 持久化并据此
  开真总线属 M2。）
- **设备绑定原则**：物理 ID（port / camera index|serial）只存在 Mac 端 CaptureAgent，
  云端 config 只有逻辑名 + 分辨率；onboarding 由云端 UI 驱动、人在 Mac 旁确认拔插/选相机。

### F10 — WebSocket 信令 + Mac daemon（同机两进程可测）
- **目标**：`Signaling` 的 WebSocket 实现（aiohttp client）+ 信令 relay server（按 session/role
  转发 SDP，缓冲晚到方）+ `mac_daemon.py` 常驻入口（连接→offer→服务一次 session→断连停力→重连循环）+
  云端 `WebRTCProxyRobot` 支持 `signaling_url=ws://`（controller 模式，不起本地 agent）+
  `ice_servers` 配置入口（[]=host-only；STUN/TURN urls 留给 M4）。
- **验收**：同机三进程/三 loop（relay+daemon+cloud）经真 localhost WS 信令 + WebRTC host candidate
  跑通 obs/control/action；daemon 在一次 session 结束后能继续服务下一次 session（活得比 session 久）。
- **状态**：completed（localhost/同机两进程；真跨公网 STUN/TURN/coturn/K8s 属 M4）

---

## M2 Features —— 接真实 so_follower（完成）

### F11 — CaptureAgent 接真实 Robot
- **目标**：CaptureAgent 接受 `robot`（连接好的 lerobot Robot，如 SO100Follower）：
  关节+相机经一次 `robot.get_observation()` 取（同采集时刻，利于对齐）；`_apply_action` 调
  `robot.send_action`；看门狗 `_safe_stop` 调 `robot.bus.disable_torque()` 切扭矩（P0）；
  session 开始/动作恢复时 `enable_torque`。所有串口访问走**单线程 executor**（不阻塞公网 event loop、
  且总线不并发访问）。run_daemon 加 `robot=` 透传。
- **验收**：回环（FakeRobot）下 obs 关节来自 robot、action 到达 robot、断流后 watchdog 切扭矩；
  真实硬件经 examples/webrtc_remote_so100 跑通（需真臂，本机不可全测）。
- **状态**：completed（FakeRobot 24 测试通过；真硬件验证靠 example）

### F12 — examples/webrtc_remote_so100
- **目标**：端到端示例：Mac 端 `mac_daemon_so100.py`（真 SO100Follower + run_daemon），
  云端 `cloud_teleop_so100.py`（WebRTCProxyRobot + 复用 web_so100 jog 面板远程遥操作），README。
- **验收**：示例 import/symbol 全解析；文档含 onboarding(find-port/find-cameras)+三件套启动步骤。
- **状态**：completed

## 不在 M3 范围（不要回头做）
- Mac 本地相机指纹方案（resolve_cameras.py 路线，已否决，非标准）。
- 真跨公网 coturn / K8s hostNetwork / announced IP（M4）。
- paradigm 落地（M5）。

## M4 待办（需真网络/真硬件，本机不可测）
- STUN/TURN（coturn）：`ice_servers` 填真 url；NAT 穿透。
- K8s：媒体 Pod hostNetwork、announced/external IP、信令 relay 普通 Deployment、coturn。
- relay 多租户路由 + 鉴权（当前单 session 单 controller）。

> **决策（2026-06-23，relay 策略）**：aiortc 定位为**纯 UDP 直连**（host + 可选 STUN srflx），
> **不自建 TURN/coturn**；直连/STUN 打不通（对称 NAT / UDP 被封 / 只能走 http_proxy）即放弃，
> 改用 **LiveKit backend 做 relay**（SFU 自带 signaling+TURN+扩展，已端到端验证）。
> 因此上面 M4 的 `coturn` 与 `K8s hostNetwork/announced IP` 两项**作废**（它们本就只为给
> aiortc P2P 兜底中继）。仍保留：aiortc 的可选 STUN、relay 多租户路由+鉴权。细节见 DESIGN §11.1。

## 不在 M1 范围（不要回头做）
- 真实串口/相机（M2）、信令服务/STUN/TURN（M3）、K8s/coturn（M4）、paradigm 落地（M5）。
- 已否决方案：socat 串口转发、usbip USB 透传、把 record/eval 挪回本地（见 context §6）。

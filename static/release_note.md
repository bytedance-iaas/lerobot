# LeRobot Agent Console — 更新说明

本控制台在标准 [LeRobot](https://github.com/huggingface/lerobot) 之上，把「云端 GPU × 远端机器人 × 对象存储数据 × AI Agent 运维」这套组合搬进一个浏览器页面：既增强了 LeRobot 本身，也让训练 / 评估 / 遥操作的日常操作可以对着 Agent 聊天完成。

**增强后的 LeRobot 开源代码**（含下述所有增强）：[bytedance-iaas/lerobot @ dev](https://github.com/bytedance-iaas/lerobot/tree/dev)。

---

## 一、对 LeRobot 的能力增强

### 1. 直接访问 TOS 上的数据集 · `StreamingTOSRobotDataset`

传统流程里，训练前要把整个数据集从对象存储**下载到本地磁盘**——大数据集又慢又占空间。我们新增了 `StreamingTOSRobotDataset`（`StreamingLeRobotDataset` 的子类），**无需下载，直接以流式方式**读取火山 **TOS** 上的 LeRobot 数据集。

- **面向 v3 格式**：适配 LeRobot **v3.0** 数据集布局（`meta/` + `data/*/*.parquet` + `videos/`）。
- **凭证走环境变量**：TOS AK/SK 从环境读取（`TOS_ACCESS_KEY` / `TOS_SECRET_KEY`，可选 `TOS_ENDPOINT` / `TOS_REGION`），只给 `tos://bucket/prefix` URL 即可，无需再手写 `storage_options`。
- **按需读取，不落盘**：元数据(meta) 只镜像几 MB 到本地；低维数据(parquet) 流式拉取；视频经 fsspec 直接解码（`tosfs` 已内置于镜像），全程不写本地磁盘。
- **多视频文件对齐修复**：修复了 `StreamingLeRobotDataset` 的一个上游 bug——视频时间戳按「全局 index/fps」计算，数据集拆成多个 `.mp4` 后靠后的 episode 会解码到越界/冻结的帧；现改为按视频文件相对时间戳，已逐帧比对验证与非流式读取**完全一致**。
- **即插即用**：它是 `StreamingLeRobotDataset` 的子类，可喂给自定义训练循环 / 数据探查 / 离线评估。

```python
from lerobot.datasets import StreamingTOSRobotDataset   # 凭证从环境变量读取

ds = StreamingTOSRobotDataset(
    "tos://my-bucket/lerobot-datasets/finish_sandwich",
    episodes=[0, 3, 17],   # 可选：只取部分 episode，做 train/eval 切分
)
for item in ds:            # IterableDataset：迭代取，不支持 ds[i]
    item["observation.images.front"]   # (C, H, W)，帧直接来自 TOS
    item["observation.state"]; item["action"]
    break
```

### 2. 云端直连机器人 · LiveKit 传输

云端 GPU 与机器人往往**不在同一张网里**，也很难直接互联（内网隔离、家用 NAT）。我们让 `WebRTCProxyRobot` 支持 **LiveKit（SFU）** 传输：机器人与云端各自**主动拨出**连到 LiveKit，借此穿透 NAT，**无需机器人侧暴露任何公网入站**。云端拿到的就是一个普通的 lerobot `Robot`——`get_observation()` 取远端关节 + 摄像头，`send_action()` 驱动远端电机，record / teleop / eval 全部无改动即可用；机器人侧内置安全看门狗，链路中断自动 safe-stop。

**用法举例：**

> 本仓库机器人侧的模块名是 **`mac_daemon`**（不是 `robot_daemon`）：`python -m lerobot.robots.webrtc_proxy.mac_daemon ...`。

- **先做一次 LiveKit 连通性测试**（不接机械臂，用合成关节 + 一路真实摄像头 `--real-camera 0` 验证能连上 LiveKit、视频能推流）：

```bash
python -m lerobot.robots.webrtc_proxy.mac_daemon \
  --transport livekit --session so100 \
  --livekit-url ws://<你的 LiveKit 地址>:7880 \
  --livekit-api-key devkey --livekit-api-secret lerobotlivekitsecret0123456789abcd \
  --real-camera 0
```

- **机器人侧**（接着 SO-100 的那台机器）跑采集守护进程，拨出连到 LiveKit（`--robot.*` 会被 draccus 解析成真实机器人，取代 `--real-camera`）：

```bash
python -m lerobot.robots.webrtc_proxy.mac_daemon \
  --transport livekit --session so100 \
  --livekit-url ws://<你的 LiveKit 地址>:7880 \
  --livekit-api-key devkey --livekit-api-secret lerobotlivekitsecret0123456789abcd \
  --robot.type=so100_follower --robot.port=/dev/tty.usbmodemXXXX \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 0, width: 640, height: 480} }"
```

- **云端 / 控制侧**跑控制脚本，连同一个 LiveKit，就能看到远端摄像头并遥操作（在本控制台里，启动后它的 web 操作面板会作为一个新标签页直接在这里打开）。控制台与 LiveKit 在**同一个集群**里，直接用集群内服务名 `ws://livekit-clb:7880` 连接，无需走公网：

```bash
python examples/webrtc_remote_so100/cloud_teleop_so100.py \
  --mode web --transport livekit --session so100 --cameras "front,wrist" --web-port 8080 \
  --livekit-url ws://livekit-clb:7880 \
  --livekit-api-key devkey --livekit-api-secret lerobotlivekitsecret0123456789abcd
```

**机器人侧守护进程参数（`mac_daemon`）：**

| 参数 | 示例值 | 说明 |
|------|--------|------|
| `--transport` | `livekit` | 传输后端，`livekit` 或 `aiortc`（默认 `aiortc`） |
| `--session` | `so100` | 会话 id == LiveKit room；**必须与控制侧一致** |
| `--livekit-url` | `ws://<你的 LiveKit 地址>:7880` | LiveKit 信令地址（机器人主动拨出，填真实地址，别留 `{LK}` 占位） |
| `--livekit-api-key` | `devkey` | LiveKit API key（需与服务端一致） |
| `--livekit-api-secret` | `lerobotlivekitsecret0123456789abcd` | LiveKit API secret（需与服务端一致） |
| `--real-camera` | `0` | 快速测试用：不接机械臂，只开这一路 opencv 摄像头（索引如 `0` 或 `/dev/videoN`）+ 合成关节；与 `--robot.*` 互斥 |
| `--camera-name` / `--width` / `--height` / `--fps` | `front` / `640` / `480` / `30` | `--real-camera` 那路的名字与分辨率/帧率 |
| `--robot.type` | `so100_follower` | 接真实机器人时用；任意 lerobot 机器人（draccus `--robot.*`，会取代 `--real-camera`） |
| `--robot.port` | `/dev/tty.usbmodemXXXX` | 串口，用 `uv run lerobot-find-port` 找 |
| `--robot.id` | `my_awesome_follower_arm` | 机器人 id |
| `--robot.cameras` | draccus dict | 摄像头**名字**（`front`/`wrist`）要与控制侧 `--cameras` 对齐；索引用 `uv run lerobot-find-cameras` 找；某摄像头达不到指定 fps 会报错，可省略 `fps` 用原生帧率 |

**控制侧参数（cloud_teleop_so100.py）：**

| 参数 | 默认 | 说明 |
|------|------|------|
| `--mode` | `web` | `web`（网页面板）或 `console`（终端） |
| `--session` | `so100` | 会话 id == LiveKit room；**必须与机器人侧一致** |
| `--web-port` | `8080` | web 面板端口 |
| `--transport` | `aiortc` | 传输后端，此处用 `livekit` |
| `--cameras` | `front` | 逗号分隔的摄像头名，**必须与机器人侧 `--robot.cameras` 的 key 对齐**；每项写 `name` 或 `name:WxH` |
| `--livekit-url` | `$LIVEKIT_URL` | LiveKit 信令地址；控制台在集群内，用服务名 `ws://livekit-clb:7880`（不走公网） |
| `--livekit-api-key` | `$LIVEKIT_API_KEY` | 同机器人侧，`devkey` |
| `--livekit-api-secret` | `$LIVEKIT_API_SECRET` | 同机器人侧 |
| `--livekit-token` | 自签发 | 预签发 JWT；不填则用 key/secret 自签 |
| `--livekit-identity` | `controller` | 控制侧在 LiveKit 里的 identity |
| `--signaling-url` | `$SIGNALING_URL` | aiortc 中继 WS 地址（仅 `aiortc` 后端） |
| `--auth-token` | `$SIGNALING_AUTH_TOKEN` | 带鉴权的中继共享 token（仅 `aiortc` 后端） |

**两侧必须对齐的参数：** `--session`（== LiveKit room）、`--livekit-api-key` / `--livekit-api-secret`、摄像头名字与数量（`--robot.cameras` ↔ `--cameras`）。
> 常见报错：`cv2.error: !ssize.empty()` 通常是两侧摄像头数量不一致（1 路发 640x480，2 路发 640x960）。
>
> 控制侧要在**机器人守护进程开始推流之后**再启动——它的 `connect()` 会阻塞等待第一帧视频，收不到就退出。

LiveKit 服务端配置：信令 `7880/TCP`、rtc `7881/TCP`、媒体 `7882/UDP`；`keys` 里定义 `<api_key>: <api_secret>`（示例 `devkey: lerobotlivekitsecret...` 是弱口令占位，生产请用 `livekit-server generate-keys` 轮换）。

更详细的配置、传输后端与设计说明，见 [lerobot webrtc_proxy README](doc:webrtc)（直接读镜像里 `/lerobot` 下的本地文档）。

### 2.1 远程本体 × 远程推理 · WebRTC 接入 async_inference（零改代码）

上面的 LiveKit 远程把**机器人本体**搬到了远端；LeRobot 的 `async_inference` 则把**策略推理**搬到远端（机器人只发观测、收回成块的动作 `action chunk`，推理延迟由动作队列吸收，不打断执行）。这两件事是**正交的**，可以叠加——而且**不用改任何代码**就能把「被 WebRTC 代理的远端机器人」交给一个 policy server 驱动。

原因：`RobotClient` 用标准工厂 `make_robot_from_config(config.robot)` 构造机器人，而工厂**原生支持** `--robot.type=webrtc_proxy`（`WebRTCProxyRobot` 实现了完整 `Robot` 接口，其 observation/action schema 与 `so_follower` 完全一致：`<motor>.pos` 浮点 + 每路相机 `HxWx3`）。于是拓扑变成：

```
真机(mac_daemon) ⟷ LiveKit ⟷ [云: RobotClient + WebRTCProxyRobot] ⟷ gRPC ⟷ PolicyServer(GPU)
```

**① 云端 GPU：起 policy server**（策略由 client 下发，server 只需 host/port/fps）：

```bash
python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080 --fps=30
```

**② 机器人侧（你的 Mac）：照常起 mac_daemon**（和遥操作用的完全一样，`--session` 即 LiveKit room）：

```bash
python -m lerobot.robots.webrtc_proxy.mac_daemon \
  --transport livekit --session so100 \
  --livekit-url ws://<你的 LiveKit 地址>:7880 \
  --livekit-api-key devkey --livekit-api-secret lerobotlivekitsecret0123456789abcd \
  --robot.type=so100_follower --robot.port=/dev/tty.usbmodemXXXX \
  --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 0, width: 640, height: 480} }"
```

**③ 先预签一个 LiveKit JWT**——云侧的 `WebRTCProxyRobotConfig` 只有 `livekit_token` 字段、**不能像 mac_daemon 那样用 api-key/secret 自签**，room 编码在 token 里，所以要先生成（identity 用 `controller`，room 必须 == mac 侧的 `--session`）：

```bash
python -c "from lerobot.robots.webrtc_proxy.transport_livekit import make_livekit_token; \
print(make_livekit_token(api_key='devkey', api_secret='lerobotlivekitsecret0123456789abcd', identity='controller', room='so100'))"
```

**④ 云端：起 robot_client，`--robot.type=webrtc_proxy` 驱动远端机器人跑策略**（在控制台终端里跑，与 policy server 同集群）：

```bash
python src/lerobot/async_inference/robot_client.py \
  --robot.type=webrtc_proxy \
  --robot.transport_backend=livekit \
  --robot.livekit_url=ws://livekit-clb:7880 \
  --robot.livekit_token='<上一步生成的 JWT>' \
  --robot.motors='[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]' \
  --robot.cameras='{front: {height: 480, width: 640, fps: 30}, wrist: {height: 480, width: 640, fps: 30}}' \
  --server_address=127.0.0.1:8080 \
  --policy_type=act --pretrained_name_or_path=<你的模型> \
  --policy_device=cuda --client_device=cpu \
  --task="fold the towel" \
  --actions_per_chunk=50 --chunk_size_threshold=0.5 --fps=30 \
  --aggregate_fn_name=weighted_average
```

**字段对齐（成败在这，不是代码）：**

| 要素 | 必须满足 | 出错表现 |
|------|----------|----------|
| `--robot.motors` | == 策略训练时的电机集合（默认 6 个 SO-100 电机 → `.pos` 键；SO-100/101 策略开箱即用） | 动作维度不匹配报错 |
| `--robot.cameras` **名字** | 每路相机名 == 策略的 `observation.images.<name>`，也 == mac 侧 `--robot.cameras` 的 key。**默认只有 `front` 一路，多数策略是 `front`+`wrist`，必须显式补齐** | 缺相机键 / 观测维度错 |
| `--robot.cameras` **形状** | `HxW` 尽量对齐训练分辨率（否则依赖 policy processor 缩放） | 精度下降 |
| 动作键名 | 策略输出的动作以 `.pos` 结尾（`send_action` 只保留 `.pos` 键） | 动作被静默丢弃、机器人不动 |
| LiveKit room | 预签 token 的 `room` == mac 侧 `--session` | 连不上、收不到观测 |

**延迟注意：** PolicyServer 的推理延迟被 async_inference 的动作队列吸收（这正是它和逐帧遥操作相比的优势），但 **LiveKit 取观测那一跳仍在环里**（`WebRTCProxyRobot.send_action` 内含 `timeout=2.0`，链路卡超过 2s 会抛）。网络差时调大 `--chunk_size_threshold` 留足缓冲。

> 一句话选型：要**人在环的遥操作 / 数据采集 / 实时视频**，用上面的 §2 遥操作路径；要**纯策略的远程推理部署**，用本节的 async_inference 路径——动作块机制天生比逐帧 WebRTC 更抗延迟。

### 3. fp8（float8）训练 · NVIDIA TransformerEngine

在 Hopper / Ada 架构 GPU（H20 / H100 / L40S）上，**pi0 / pi05 现在支持 fp8（float8）训练**：每个 VLM（PaliGemma）层的 `post_attention_layernorm` + gate/up/down MLP 被融合成一个 **`te.LayerNormMLP`**（RMSNorm + FC1 geglu + FC2 全程 fp8），动作专家（action expert）保持 bf16。主权重仍是 bf16，fp8 与 bf16 autocast 叠加——既省显存、在 VLA 上还更快。

- **一键开启**：`robot_sft` 的 `plan_training.py` 加 `--float8` 即可（**仅 pi0 / pi05**，其它策略会直接报错拒绝）；等价于给 `lerobot-train` 传 `--policy.vlm_mlp_fp8_enable=true --policy.dtype=bfloat16`。
- **两种 recipe**（HYBRID：E4M3 前向 / E5M2 反向）：`delayed_scaling`（默认，逐张量延迟缩放，16 步 amax 历史）与 `float8_block_scaling`（块级——激活/梯度 1D 行缩放、权重 2D tile，数值更稳、开销略高）。
- **实测收益**（pi05 / 单卡 H20）：相比 bf16 约 **1.37× 提速、显存少 ~2.9 GB**。
- **从 bf16 checkpoint 起训**：直接加载 `pi05_base` 等 bf16 权重即可，TE 的 fp8 缩放元数据（`_extra_state`）自动新初始化并在训练中标定——权重照常加载，无需预转换。
- **依赖**：需要 TransformerEngine（已内置于 lerobot 镜像）+ Hopper/Ada 的 fp8 tensor core；旧卡会在运行时报错，所以 `plan_training.py` 只在检测到 H20/Hopper 时才允许 `--float8`。

---

## 二、控制台自带的能力

### 4. 内置 AI Agent · 豆包驱动

右侧就是一个 AI Agent 对话框，接入 **豆包 / 火山方舟（Volcengine Ark）** 大模型。用自然语言即可让它探索数据集、规划并启动 SFT 训练、评估 checkpoint，或直接在下方控制台里执行命令。首次使用只需填入火山方舟 API Key（**仅用于 chat，不影响终端与其他功能**）。底层由 hermes agent 驱动。

### 5. 配套的 robot_sft 技能

Agent 预装了 **`robot_sft`** 技能：把一次机器人模仿学习 / VLA 策略的 SFT 训练，拆成一串小的、可独立验证、文件存档的阶段——数据集探查 → train/eval 切分 → 计划 + 预检（含冒烟测试）→ 训练（自愈看门狗 + 定期离线评估 + 监控面板）。崩溃或上下文重置也不丢进度：重读会话状态即可继续。上面「直接访问 TOS 数据集」的能力也由它串起来。

### 6. 自动发现并打开控制台里的服务

在下方 Linux 终端里启动的 web 服务（如 webrtc 远程遥操作面板、训练监控面板），控制台会**自动发现**；点右上角「＋ 打开」即可把它作为一个标签页在这里打开，也可手动输入端口 / 网址。Agent 输出的 HTML 同样会在这里打开——**终端、Agent、内嵌浏览器三者在同一个页面里协同**。

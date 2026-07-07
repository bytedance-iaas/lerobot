# LeRobot Agent Console — 更新说明

本控制台在标准 [LeRobot](https://github.com/huggingface/lerobot) 之上，把「云端 GPU × 远端机器人 × 对象存储数据 × AI Agent 运维」这套组合搬进一个浏览器页面：既增强了 LeRobot 本身，也让训练 / 评估 / 遥操作的日常操作可以对着 Agent 聊天完成。

**增强后的 LeRobot 开源代码**（含下述所有增强）：[bytedance-iaas/lerobot @ dev](https://github.com/bytedance-iaas/lerobot/tree/dev)。

---

## 一、对 LeRobot 的能力增强

### 1. 直接访问 TOS 上的数据集 · `fsspec_dataset.py`

传统流程里，训练前要把整个数据集从对象存储**下载到本地磁盘**——大数据集又慢又占空间。我们新增了 `FsspecLeRobotDataset`，**无需下载，直接以流式方式**读取对象存储上的 LeRobot 数据集。

- **面向 v3 格式**：适配 LeRobot **v3.0** 数据集布局（`meta/` + `data/*/*.parquet` + `videos/`）。
- **任意 fsspec 后端**：火山 **TOS**、S3、GCS 等都支持；只要给出 `tos://bucket/prefix` 这样的 URL 和访问凭证。
- **按需读取，不落盘**：元数据(meta) 只镜像几 MB 到本地；低维数据(parquet) 流式拉取；视频经 fsspec 直接解码，全程不写本地磁盘。
- **即插即用**：它是 `StreamingLeRobotDataset` 的子类，可直接喂给 `lerobot-train` 与评估流程。

```python
from lerobot.datasets import FsspecLeRobotDataset

ds = FsspecLeRobotDataset(
    "tos://my-bucket/lerobot-datasets/finish_sandwich",
    storage_options={"key": ..., "secret": ..., "endpoint": "https://tos-cn-beijing.volces.com", "region": "cn-beijing"},
    episodes=[0, 3, 17],   # 可选：只取部分 episode，做 train/eval 切分
)
for item in ds:
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

---

## 二、控制台自带的能力

### 3. 内置 AI Agent · 豆包驱动

右侧就是一个 AI Agent 对话框，接入 **豆包 / 火山方舟（Volcengine Ark）** 大模型。用自然语言即可让它探索数据集、规划并启动 SFT 训练、评估 checkpoint，或直接在下方控制台里执行命令。首次使用只需填入火山方舟 API Key（**仅用于 chat，不影响终端与其他功能**）。底层由 hermes agent 驱动。

### 4. 配套的 robot_sft 技能

Agent 预装了 **`robot_sft`** 技能：把一次机器人模仿学习 / VLA 策略的 SFT 训练，拆成一串小的、可独立验证、文件存档的阶段——数据集探查 → train/eval 切分 → 计划 + 预检（含冒烟测试）→ 训练（自愈看门狗 + 定期离线评估 + 监控面板）。崩溃或上下文重置也不丢进度：重读会话状态即可继续。上面「直接访问 TOS 数据集」的能力也由它串起来。

### 5. 自动发现并打开控制台里的服务

在下方 Linux 终端里启动的 web 服务（如 webrtc 远程遥操作面板、训练监控面板），控制台会**自动发现**；点右上角「＋ 打开」即可把它作为一个标签页在这里打开，也可手动输入端口 / 网址。Agent 输出的 HTML 同样会在这里打开——**终端、Agent、内嵌浏览器三者在同一个页面里协同**。

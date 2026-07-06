# LeRobot Agent Console — 更新说明

本控制台在标准 [LeRobot](https://github.com/huggingface/lerobot) 之上，把「云端 GPU × 远端机器人 × 对象存储数据 × AI Agent 运维」这套组合搬进一个浏览器页面：既增强了 LeRobot 本身，也让训练 / 评估 / 遥操作的日常操作可以对着 Agent 聊天完成。

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

**大概怎么用：**

- 机器人侧（接着 SO-100 的那台机器）跑采集守护进程，拨出连到 LiveKit：

```bash
python examples/webrtc_remote_so100/robot_daemon_so100.py
```

- 云端 / 控制侧跑控制脚本，连同一个 LiveKit，就能看到远端摄像头并遥操作：

```bash
python examples/webrtc_remote_so100/cloud_teleop_so100.py
```

- 用 `--transport livekit` 选择该传输后端，并配置 `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET`。在本控制台里，启动后它的 web 操作面板会作为一个新标签页直接在这里打开。

更详细的配置、传输后端与设计说明，见 [lerobot webrtc_proxy README](https://github.com/thesues/lerobot/blob/remote_robot/src/lerobot/robots/webrtc_proxy/README.md)。

---

## 二、控制台自带的能力

### 3. 内置 AI Agent · 豆包驱动

右侧就是一个 AI Agent 对话框，接入 **豆包 / 火山方舟（Volcengine Ark）** 大模型。用自然语言即可让它探索数据集、规划并启动 SFT 训练、评估 checkpoint，或直接在下方控制台里执行命令。首次使用只需填入火山方舟 API Key（**仅用于 chat，不影响终端与其他功能**）。底层由 hermes agent 驱动。

### 4. 配套的 robot_sft 技能

Agent 预装了 **`robot_sft`** 技能：把一次机器人模仿学习 / VLA 策略的 SFT 训练，拆成一串小的、可独立验证、文件存档的阶段——数据集探查 → train/eval 切分 → 计划 + 预检（含冒烟测试）→ 训练（自愈看门狗 + 定期离线评估 + 监控面板）。崩溃或上下文重置也不丢进度：重读会话状态即可继续。上面「直接访问 TOS 数据集」的能力也由它串起来。

### 5. 自动发现并打开控制台里的服务

在下方 Linux 终端里启动的 web 服务（如 webrtc 远程遥操作面板、训练监控面板），控制台会**自动发现**；点右上角「＋ 打开」即可把它作为一个标签页在这里打开，也可手动输入端口 / 网址。Agent 输出的 HTML 同样会在这里打开——**终端、Agent、内嵌浏览器三者在同一个页面里协同**。

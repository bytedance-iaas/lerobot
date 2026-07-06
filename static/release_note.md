# LeRobot Agent Console — 更新说明

在标准 [LeRobot](https://github.com/huggingface/lerobot) 之上，本控制台集成了两项能力增强，让「云端 GPU × 远端机器人 × 对象存储数据」这套组合真正能跑通。

---

## 1. 直接访问 TOS 上的数据集 · `fsspec_dataset.py`

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

---

## 2. 云端直连机器人 · LiveKit 传输

云端 GPU 与机器人往往**不在同一张网里**，也很难直接互联（内网隔离、家用 NAT）。我们让 `WebRTCProxyRobot` 支持 **LiveKit（SFU）** 传输：机器人与云端各自**主动拨出**连到 LiveKit，借此穿透 NAT，**无需机器人侧暴露任何公网入站**。

- **云端就是一个普通 `Robot`**：`get_observation()` 拿到远端关节 + 摄像头，`send_action()` 直接驱动远端电机——record / teleop / eval 全部无改动即可用。
- **内网推荐 `aiortc`**：点对点直连，延迟最低，自带信令中继，无需任何外部服务。
- **公网推荐 `livekit`**：经 SFU 中转，自动处理 NAT 穿透 / TURN，适合跨公网、多并发、规模化。
- **实时且安全**：状态与动作走 WebRTC DataChannel，摄像头走媒体轨（每帧带采集序号做对齐）；机器人侧内置安全看门狗，链路中断自动 safe-stop，绝不让机械臂卡在危险姿态。

在本控制台里，启动 `webrtc_remote_so100` 远程遥操作后，它的 web 操作面板会作为一个新标签页直接在这里打开。

---

_点击右上角「＋ 打开」可以打开控制台里启动的 web 服务或任意网址；关闭全部标签页后仍可从这里重新打开。_

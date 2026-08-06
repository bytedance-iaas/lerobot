# DreamZero → LeRobot 移植计划

日期:2026-07-21 · 决策:树内 policy / LoRA 后训练优先 / eval = 离线开环 + lerobot 原生(真机 gRPC) + DROID sim-evals

**2026-07-21 范围修正(用户确认)**:
1. 计划整体同意,从 M1 开工。
2. **不用 `convert_dataset_v21_to_v30.py` 转换数据**——离线评估直接读 GEAR/v2.1 原始布局(`data/chunk-*/episode_*.parquet` + `videos/chunk-*/.../episode_*.mp4` + `meta/*`),写轻量 episode 读取器,不依赖 LeRobotDataset v3。
3. **M3(v3 数据管线 + LoRA 训练)本期不做**;eval 先做离线开环(M2)+ gRPC PolicyServer(M4 前半);`droid_sim` EnvConfig(IsaacLab sim-evals)顺延。
4. **推理不走 vllm 方式(用户强调)**:inference engine 从 upstream DreamZero 原生 PyTorch 路径移植——`WANPolicyHead.lazy_joint_video_action`(KV-cache 因果自回归)+ `GrootSimPolicy` 归一化/unapply,包进 lerobot `PreTrainedPolicy.select_action`/`predict_action_chunk`,远程走 lerobot 自带 gRPC `PolicyServer`。**不移植** vllm-omni 的 `pipeline_dreamzero.py`/stage/deploy 那套。vllm-omni 仅作两处参考:(a) parity 测试方法论,(b) transform 语义(视角拼接/语言模板),不作为推理引擎。

## 背景

DreamZero(`/Users/dongmao.zhang/upstream/dreamzero`,NVIDIA GEAR)是基于 **Wan 视频扩散 backbone** 的 World Action Model:一个因果 DiT(`CausalWanModel`)在同一 token 序列里联合去噪**视频 latent + 动作寄存器 + 状态寄存器**,blockwise causal attention,推理时 KV-cache 逐 block 自回归出 action chunk。上游技术栈与 lerobot 冲突较大:

| 维度 | dreamzero 上游 | lerobot 现状 |
|---|---|---|
| 训练 | Hydra + HF `Trainer`(transformers 4.51)+ DeepSpeed ZeRO | draccus + 自有 loop + accelerate(DDP/FSDP,无 DeepSpeed) |
| 数据 | GEAR = LeRobot **v2.1** 布局 + `meta/{modality,stats,relative_stats}.json` | LeRobot **v3.0**(stats 已含 q01/q99) |
| 推理 | torchrun 分布式 websocket 服务器(OpenPI 协议,CFG 双卡) | async_inference **gRPC** PolicyServer |
| 依赖 | torch 2.8 pin、transformers 4.51、diffusers 0.30、numpy 1.26、flash-attn | torch>=2.7、transformers 5.4+、diffusers<0.36、numpy 2.x |

**可参考的先例**:
- `../vllm-omni` 已移植 DreamZero 推理:`vllm_omni/diffusion/models/dreamzero/`(单文件重实现 `causal_wan_model.py` ~36KB + pipeline + transform),直接从 HF `GEAR-Dreams/DreamZero-DROID` 加载,并用**上游服务器 e2e 数值 parity 测试**锁定正确性(`tests/dreamzero/upstream/test_openpi_e2e_source_parity.py`:同 checkpoint、`NUM_DIT_STEPS=16`、no-compile、逐 chunk 比对)。证明了脱离 transformers 4.51/旧 diffusers 是可行的。
- lerobot 树内 `src/lerobot/policies/groot/`:embodiment tag、q99 归一化、relative-action stats(`processor_groot.py:712` 起)、内部归一化不走通用 Normalizer——与 dreamzero 语义几乎一一对应。
- lerobot 树内 `src/lerobot/policies/fastwam/`:Wan 视频世界模型已在树内(`fastwam/wan/` ~4.4k 行),FSDP 可用的先例。

**"符合 dreamzero 的要求"** 的落点:q99 + relative action + embodiment 语义逐位对齐;训练超参对齐上游脚本;用 parity 测试锁定推理数值一致;GEAR 数据可转换进入。

## 目标形态

新目录 `src/lerobot/policies/dreamzero/`,注册名 `dreamzero`,照 `docs/source/bring_your_own_policies.mdx` 接入:

```
src/lerobot/policies/dreamzero/
├── __init__.py
├── configuration_dreamzero.py   # @PreTrainedConfig.register_subclass("dreamzero")
├── modeling_dreamzero.py        # DreamZeroPolicy(PreTrainedPolicy)
├── processor_dreamzero.py       # make_dreamzero_pre_post_processors
├── wan/                         # 移植的模型核心(自包含,不依赖 groot.* 命名空间)
│   ├── causal_wan_model.py      # ← wan_video_dit_action_casual_chunk.py (2245行)
│   ├── vae.py                   # ← wan_video_vae.py (WanVideoVAE 16ch / VAE38 48ch)
│   ├── text_encoder.py          # ← wan_video_text_encoder.py (umt5-xxl, 自实现无 transformers 依赖)
│   ├── image_encoder.py         # ← wan_video_image_encoder.py (open-clip XLM-R ViT-H)
│   ├── attention.py             # ← wan2_1_attention.py,加 SDPA fallback(macOS/无 flash-attn 可跑)
│   ├── schedulers.py            # ← flow_match_scheduler.py + flow_unipc_multistep_scheduler.py
│   └── kv_cache.py              # KV cache + DiT step-skip mask(should_run_model 逻辑)
├── loading.py                   # ← base_vla.py 的定制加载(分片 safetensors、PEFT key 清洗、组件权重 hf_hub_download)
└── README.md
```

工程接入点(小改动):
- `src/lerobot/policies/factory.py`:`get_policy_class` / `make_policy_config` / `make_pre_post_processors` 各加一个 `dreamzero` 分支(lazy import)。
- `pyproject.toml`:新 extra `dreamzero = ["lerobot[transformers-dep]", "lerobot[diffusers-dep]", "lerobot[peft-dep]", "lerobot[dataset]", ...]`(照 `groot`/`fastwam` 模式);flash-attn 不进 extra,运行时探测。
- 上游的 `IdentityBackbone`/`VLA` 包装、HF Trainer、Hydra、tianshou/ray/gear/multi-storage-client 依赖全部**不移植**;上游 yaml 里引用了不存在的 `n1_5.modules.cross_attention_dit`(`vl_self_attention_cfg`),移植时剪掉。

## 分阶段计划

### M1 — 模型核心 + 单卡离线推理跑通

1. 移植 `wan/` 各模块(来源见上表;`cudnn_attention.py`/TensorRT/TransformerEngine 路径先不移植,保留 flash-attn + SDPA 两档)。
2. `configuration_dreamzero.py`:把 Hydra 配置翻译成 draccus dataclass——**只做 `wan21_14b`**(dim 5120/176×320/VAE 16ch),关键字段 `num_frames=33`、`action_horizon=24`、`num_views=3`、`num_frame_per_block=2`、`num_action_per_block`、`max_state_dim`/`max_action_dim`、embodiment 表(`oxe_droid`/`agibot`/`yam` + projector index)、`cfg_scale=5.0`、`num_dit_steps`。
3. `loading.py` + `DreamZeroPolicy.from_pretrained`:支持直接加载 HF `GEAR-Dreams/DreamZero-DROID` / `DreamZero-AgiBot`(vllm-omni 已验证此路线);LoRA 检查点走 lerobot 自带 PEFT 机制加载。
4. `modeling_dreamzero.py`:
   - `predict_action_chunk` / `select_action`:移植 `lazy_joint_video_action`(KV-cache 因果推理)+ `GrootSimPolicy.unapply` 的反归一化/relative→absolute 语义;会话状态(`current_start_frame`、KV cache、帧累积)挂在 policy 实例上,`reset()` 清空。**首版限 num_envs=1**(KV cache 无 batch 语义)。
   - 单卡即可(cond+uncond 串行,vllm-omni 实测 ≥74GB → H20 96GB OK);CFG 双卡并行为后续优化。
   - DiT step-skip(`NUM_DIT_STEPS` 5/6/7/8 的静态 mask)一并移植——纯算法、收益大(16 步→6 步)。
5. 验收:复现上游 `test_client_AR.py` 场景——用 `debug_image/*.mp4` 的 DROID 观测序列,本地单卡出 action chunk + 预测视频 MP4。

### M2 — 离线开环 parity(移植正确性锁定)

- 脚本 `src/lerobot/policies/dreamzero/scripts/offline_eval.py`(或 examples/):held-out episode 逐 chunk 闭环重放,输出 action MSE(对齐上游 `compare_loss.py`/`open_looop_yam.py` 语义)+ 视频预测导出。
- parity 方法论照 vllm-omni:`DREAMZERO_REPO` 指向上游 checkout,同 checkpoint、同 obs 序列、`NUM_DIT_STEPS=16`、no-compile,逐 chunk 数值比对(阈值放宽到 bf16 容差)。作为可选测试(需 GPU + 上游 repo)进 `tests/policies/test_dreamzero_parity.py`。
- 另配 tiny-config 单测(小 dim/两层)跑 CPU:forward loss、select_action、processor 往返、q99/relative 数学对照离线 golden fixtures。

### M3 — 数据管线 + lerobot-train LoRA

1. **采样窗口语义是正确性核心,先做 golden 对照**:用上游 dataloader(`DreamTransform.apply_single`,`transform/dreamzero_cotrain.py:504`)在小数据集上 dump 样本,逐字段(images/text/state/action/mask/embodiment_id)对照移植实现。
2. 原生消费 LeRobotDataset v3(不移植上游 sharded dataset/decord):
   - `observation_delta_indices` = 未来 `num_frames` 帧窗口(视频预测 GT),`action_delta_indices` = 对齐 block 结构的未来 action 窗口——具体索引以 golden 对照为准。
   - `processor_dreamzero.py`:多视角帧 stack + resize(176×320 / 160×320)、文本 tokenize(umt5 tokenizer + 固定 negative prompt)、状态 pad 到 `max_state_dim`、**q99 归一化到 [-1,1]**、relative action(对齐 `relative_action_keys`)、`embodiment_id` 注入;复用 groot 的 stats 机制(v3 stats 自带 q01/q99;relative stats 照 `processor_groot.py` 临时扫描模式)。
   - GEAR/上游预处理数据(v2.1 布局)→ 现成 `src/lerobot/scripts/convert_dataset_v21_to_v30.py` 转换即可;`meta/modality.json` 的 key→[start,end] 切片映射翻译进 config 字段。新 embodiment(如 SO-101)按 `docs/DATASET_TO_GEAR_AND_TRAIN.md` 的语义在 config 里注册(enum + modality + projector index)。
3. 训练(对齐上游 `scripts/train/*_training_lora.sh`):
   - lerobot-train + PEFT(`use_peft`,target `q,k,v,o,ffn.0,ffn.2`,rank=alpha=4);冻结 VAE/text/image encoder(`get_optim_params` 只回 LoRA 参数);bf16 + grad checkpointing;`get_optimizer_preset` = AdamW lr 1e-4 + warmup。
   - 分布式:8×H20 先 DDP(LoRA 参数量小,ZeRO 非必需);显存不够再上 accelerate FSDP FULL_SHARD(`docs/source/multi_gpu_training.mdx`,wrap `WanAttentionBlock`)。全参微调(上游 ZeRO-2+offload)明确**不在本期范围**。
   - 注意:lerobot-train 无梯度累积(`update_policy` 无 `accelerator.accumulate`),等效 batch 靠多卡;如需要再加。
4. 验收:AgiBot checkpoint + 小数据(上游 agibot 配方或 SO-101 自采 ~30min play data)LoRA 后训练 loss 收敛,离线开环合理。

### M4 — 在线 eval:真机 gRPC + DROID sim-evals

1. **lerobot 原生 serving(免费获得)**:`predict_action_chunk` 实现后,`python -m lerobot.async_inference.policy_server` 直接可服 DreamZero;SO-101 真机走既有 webrtc proxy + RobotClient 闭环。
2. **DROID sim-evals**:新 `EnvConfig` 子类 `droid_sim`(`src/lerobot/envs/` 或跟随 policy 目录),包 IsaacLab + `sim_evals` 的 DROID env(Linux GPU 服务器 only,依赖照 vllm-omni README 的隔离方式处理);`features_map` 把 `external_cam/external_cam_2/wrist_cam/arm_joint_pos/gripper_pos` 映射到 policy 输入 keys;open-loop horizon=8 由 `n_action_steps` 控制;gripper 二值化进 postprocessor。跑通后与上游 `run_sim_eval.py` 的成功率对表。

### M5 — 收尾

- `pre-commit run --all-files`;lazy import 守护(`require_package("...", extra="dreamzero")`);README(GEAR→v3 转换、训练命令、eval 命令、显存要求)。
- 可选优化(不阻塞):CFG 双卡并行(上游 `parallelize`/P2P 交换)、DYNAMIC_CACHE_SCHEDULE、torch.compile encoders、TensorRT。

## 风险清单

1. **采样窗口/block 对齐语义**(33 帧 ↔ latent 9 帧 ↔ block ↔ action chunk)最易错——M3 第 1 步 golden 对照不可省。
2. KV-cache 推理与 lerobot 向量化 env 的冲突 → 首版 num_envs=1,文档标明。
3. transformers 5.x:text/image encoder 是自实现,理论无依赖;但 tokenizer(umt5)加载路径需在 5.x 下验证。
4. flash-attn 在 H20(Hopper)可用;macOS 仅 SDPA + tiny config 单测,不做完整推理。
5. 显存:14B 推理单卡 ≥74GB(H20 96GB OK);LoRA 训练 8×H20 预计可行(上游 LoRA 配方即 ZeRO-2 无 offload),若爆则 FSDP。
6. lerobot v3 数据的 fps 与上游 GEAR `fps` 假设(droid 15fps 等)需逐数据集核对,视频 delta 窗口按 `i/fps` 换算。

## 里程碑

| 里程碑 | 交付 | 验收 |
|---|---|---|
| M1 | 模型核心 + from_pretrained + 单卡推理 | 复现 test_client_AR 输出(DROID ckpt) |
| M2 | 离线开环脚本 + parity 测试 | 与上游逐 chunk 数值对齐 |
| M3 | ~~v3 数据管线 + LoRA 训练~~(本期不做) | — |
| M4 | gRPC serve(+ droid_sim EnvConfig 顺延) | SO-101 闭环;sim-evals 成功率对表 |
| M5 | 测试/文档/lint | pre-commit 全绿;README 可复现 |

## 进度(分支 webrtc-daemon)

### 已完成 — 模型核心

`src/lerobot/policies/dreamzero/wan/` 下 12 个文件(near-verbatim,仅头部/import 改写):
- `action_encoder.py` `attention.py` `causal_attention.py` `submodule.py`
- `causal_wan_model.py`(2245 行联合视频+动作因果 DiT) `vae.py`(WanVideoVAE+VAE38)
- `text_encoder.py`(umt5) `image_encoder.py` `schedulers.py`(FlowMatch+FlowUniPC) `vram.py`
- `action_head.py` = `WANPolicyHead`+`WANPolicyHeadConfig`
- `base_action_head.py` = 极简 `ActionHead` 基类

移植期新增的两处 surgical 修改(均有注释说明):
- `action_head.py.__init__` 补 `self.trt_engine = None`。上游只在 `post_initialize()` 里赋值,
  而 `_run_diffusion_steps` 每步都读它 —— 不调那个(顺带 torch.compile 的)钩子就 AttributeError。
- `schedulers.py` 的两个 UniPC 更新步改为 `@_maybe_compile`(默认 eager,
  `DREAMZERO_COMPILE_SCHEDULER=1` 开启)。上游用 `fullgraph=True, dynamic=False`,而
  `order`/`step_index` 是 python int,Dynamo 逐值特化,16 步必然撞穿重编译上限。

### 已完成 — 直接加载 released checkpoint(**不做转换**)

`gear_checkpoint.py`。原先的 `convert_dreamzero_checkpoint.py` 已删除:实测那次"转换"里
**0 个 key 被丢弃、0 个被改名、dtype 不变**(上游 2146 个 key 全是 `action_head.*`,且已是 bf16),
产出的 43 GB `model.safetensors` 是 10 个分片的逐字节重排,纯浪费。现在三样东西全部在内存里推导:

| 需要的信息 | 从 checkpoint 的哪个文件推导 |
|---|---|
| model variant / DiT 维度 / block 尺寸 / `train_architecture` | `config.json` 的 `action_head_cfg.config` |
| state/action **concat 顺序**、per-view resize、crop、`relative_action_keys` | `experiment_cfg/conf.yaml` 的 `ConcatTransform` |
| q01/q99 统计 | `experiment_cfg/metadata.json` |
| 权重 | `model-*.safetensors` + `index.json`,按分片直读 |

`DreamZeroPolicy.from_pretrained` 用 `config.json` 的内容(有 `action_head_cfg` 且无 `type`)
区分 GEAR 与 lerobot checkpoint,两种都吃。

### 对着 DreamZero-DROID 实测,抓到并修掉的 6 个错

这些错的共同点是**都不会报错,只会安静地算错**,所以现在全部改成加载期硬失败:

1. **统计读不到** — 原 converter 找 `meta/stats.json` + `meta/relative_stats_dreamzero.json`,
   released checkpoint 里两个都不存在(实际在 `experiment_cfg/metadata.json`)。原实现只
   `logger.warning`,于是产出空统计 → `_QuantileLayout.dim == 0` → 归一化被跳过、decode 退化成
   直通。现在缺任何一个 key 都抛 `KeyError`。
2. **`per_view_height` 160 应为 176** — `frame_seqlen=880` 反推:352x640 → latent 44x80 →
   patch/2 → 22x40 = 880。160 只能给出 800。`__post_init__` 现在强制校验这个恒等式。
3. **variant 猜错** — DROID release 是 `wan21_14b`(dim 5120 / 40 层 / in_dim 36 / WanVideoVAE),
   而 converter 默认 `wan22_5b`。现在从 checkpoint 的 DiT 维度反查,对不上就抛错。
4. **`train_architecture`** 默认 `"lora"`,checkpoint 是 `"full"` → 会给 DiT 套一层 PEFT,
   与 state dict 里每个 q/k/v/o/ffn key 对不上。现在从 checkpoint 读。
5. **`backbone_embedding_dim`** 默认 1536,checkpoint 是 **0**(IdentityBackbone)。同上。
6. **layout 取错来源** — `metadata.json` 的 `modalities` 是**按字母序列出全部可用 key**
   (DROID 下还含 `cartesian_position`),不是这次训练的 concat 顺序。用它会把 state 从
   `joint(7)+gripper(1)=8` 变成 `cartesian(6)+gripper(1)+joint(7)=14`,整个向量错位。
   现在只从 `conf.yaml` 的 `ConcatTransform` 取,取不到就抛错。

另外确认:**checkpoint 已是纯 bf16**(22.924 G params x 2 B = 45.85 GB,与 index 里的
`total_size` 精确吻合),无需任何 dtype 转换。`prepare_for_inference()` 把模型保持在 bf16
(对齐上游 `post_initialize`),否则模块以 fp32 构造 → 92 GB 且第一个 bf16 激活撞上 fp32 conv 就报错。

### 已完成 — 离线开环 eval 打通,并跑出有效信号

`offline_eval.py` 在 H200 上跑通 `GEAR-Dreams/DreamZero-DROID` x `lerobot/droid_1.0.1`。
拼接画布实测 `[1, 3, 1, 352, 640]`,与推导一致。~6 s/次查询,单卡 53 GB。

数据集选择有讲究:`lerobot/droid_100` **不能用** —— 它的 state/action 是 cartesian(7 维)。
`lerobot/droid_1.0.1` 的 `observation.state` 与 `action` 都是
`joint_position(7) + gripper_position(1)`,与 checkpoint 的 concat 顺序逐位对齐。

裸 MSE 没有可解释性(动作空间是绝对关节位置,大部分数值就是当前状态本身),所以 eval 同时报
"保持锚点状态不动"基线、**位移相关系数**(剥离状态本身,只看模型真正预测的那部分)和
**位移回归斜率**(相关系数对缩放不敏感,斜率才反映幅度是否标定)。

**结果(64 帧/ep,`n_action_steps=8`)**:

| ep | action_mse | hold_state_mse | delta_corr | joint_slope |
|---|---|---|---|---|
| 3 | 0.00346 | 0.00414 | 0.573 | 0.651 |
| 1 | 0.00333 | 0.00266 | 0.349 | — |
| 7 | 0.01353 | 0.00721 | 0.733 | 1.417 |

解读:位移相关性稳定且显著(0.35–0.73),说明链路确实在产出与演示相关的动作;但幅度时高时低。
**这恰恰排除了系统性缩放缺陷** —— 归一化跨度写错、guidance scale 接错这类问题会让斜率在所有 episode 上
朝同一方向偏,而实测斜率跨在 1.0 两侧。3 个 episode 里只有 1 个胜过基线,结果算不上好,
但剩余差距是模型在 held-out DROID 上的开环行为,不是明显的接线错误。
要真正定论,仍需对上游做同 checkpoint、同观测的逐 chunk 数值 parity —— 这是仍未关闭的 M2 gate。

### 决策:只支持 14B backbone

`wan21_14b` 是唯一支持的 variant,checkpoint 的 DiT 维度对不上就按名字拒绝加载。

理由不是"图省事",而是**只有 14B 有已发布的 DreamZero 权重**:`GEAR-Dreams/DreamZero-DROID` 是 14B,
上游连 LIBERO 也是从它起步(`libero_sft_dreamzero_14b.yaml` 的 `model_path` 指向 DreamZero-DROID)。
Wan2.2-TI2V-5B 那档在上游只以**冷启动**形式存在 —— `libero_sft_dreamzero_5b.yaml` 是 `model_path: null`,
DiT/VAE/T5 取自 `Wan-AI/Wan2.2-TI2V-5B`,CLIP 图像编码器取自 `Wan-AI/Wan2.1-I2V-14B-480P`
(2.2 的 TI2V 模型不自带那个 open-clip XLM-R ViT-H)。没有公开 checkpoint 就无法验证移植正确性,
留一条验证不了的代码路径比不留更糟。

顺带修掉两个死配置:`cfg_scale` 与 `num_inference_steps` 在上游是 `WANPolicyHead.__init__` 里硬编码的,
不从 config 读;我们的 config 有同名字段但从未生效。现在在 policy 构造后显式赋值(默认值与上游一致)。
另外 `num_inference_timesteps`(checkpoint 为 4)是**训练**参数,与推理循环用的 `num_inference_steps`(16)
是两回事,原先被混为一谈,现已分开并从 checkpoint 读取。

### 调试过程中额外抓到的两个错(对照 RLinf 定位)

7. **腕部相机展宽用错 API**。上游是 `np.repeat(wrist, 2, axis=-1)`,即**逐像素横向 2x 拉伸**
   (`[a,b,c] -> [a,a,b,b,c,c]`);我们用了 `Tensor.repeat`,是**平铺**(`[a,b,c] -> [a,b,c,a,b,c]`),
   等于把腕部画面并排放了两份。改用 `repeat_interleave`。
8. **相对动作解码锚点用错帧**。模型的 chunk 是相对**预测发起帧**的位移,整块必须用同一个锚点解码;
   而 postprocessor 每帧都用当前帧的 state 当锚点,把已经走过的位移又加了一遍,误差随 chunk 增长到
   与信号同量级。实测影响巨大:逐步锚点时 `action_mse` 0.00279 **劣于**基线 0.00126;
   改成整块解码后才有了上面的正结果。

这条也暴露了**线上路径的真实缺陷**:`select_action` 逐步弹出动作时锚点必然错,gRPC PolicyServer
走的正是这条路。已在 README 标注为已知问题 —— 修它需要 processor 能观察到 policy 何时重填队列。

### 逐项核对结果(vs RLinf)

视图顺序、per-view crop/resize(中心裁剪 0.95 -> 176x320)、eval 模式关闭随机裁剪与色彩抖动、
像素格式(uint8 `(B,T,H,W,C)`,`/255` + Normalize(0.5,0.5) 在模型内部)、state 拆分与 q99 pad、
prompt 模板(**逐字节一致**)、negative prompt(**逐字节一致**)、统计来源、相对解码语义 —— 全部对齐。

另确认两点:
- RLinf 推理**也是每次只喂 1 帧**(`main_images: [B,H,W,C]` 单时刻),所以
  `videos.shape[2] == 1 -> reset current_start_frame` 是**正常路径,不是 bug**。
- RLinf 对 DROID 做 `binarize_gripper`,但那是映射到仿真环境动作空间(`>0 -> +1 else -1`),
  对数据集 MSE 不适用,我们不做是对的。

### eval 入口:改用 lerobot 标准形态,删掉 dreamzero 专用脚本

原先的 `src/lerobot/policies/dreamzero/scripts/offline_eval.py` 已删除,替换为
**`examples/eval_open_loop.py`** —— policy 无关,参照 `examples/rtc/eval_dataset.py` 的既有形态
(draccus 配置、`--policy.path` + `--dataset.repo_id`、`__get_path_fields__`),走
`make_policy` / `make_pre_post_processors` 标准工厂。

**为什么不是 `lerobot-eval`**:`lerobot-eval` 的 `env` 是必填字段,核心是 `rollout()` 调
`env.step(action)` 让仿真器演进状态、统计成功率。DreamZero 的环境是 NVIDIA 的 `sim-evals`,
依赖 `isaaclab[all,isaacsim]==2.2.0`;而 sim-evals 官方跑法是**两个进程**——IsaacSim 一个 venv,
策略作为 websocket 服务在另一个进程(`openpi_client.websocket_client_policy`)。这是硬约束,
IsaacSim 自带的 torch/numpy 约束与 lerobot 装不进同一环境。所以进程内的 `droid_sim` EnvConfig
不成立,闭环 DROID 评估等于自建那套双进程环境。

**接入方式(两处,都遵循已有模式)**:
- `DreamZeroConfig.from_foreign_checkpoint()` —— 让通用工具能认领 `config.json` 不带 `type` 的
  GEAR 目录。通用脚本用鸭子类型探测这个类方法,不需要改共享的 `PreTrainedConfig`。
- `factory.make_pre_post_processors` 的 `pretrained_path` 分支补了 `DreamZeroConfig` 一项,
  照 `GrootConfig` 的先例——两者都是把归一化统计放在 checkpoint 里,而不是序列化的 processor 管线,
  所以没有 `policy_preprocessor.json` 可加载。

**相机命名差异改用 `--rename_map`**(lerobot 标准机制),不再用 dreamzero 专用的
`--image-view-order`。policy config 里保留模型训练时的相机名,由 rename_map 适配数据集。

顺带修掉一个静默失败:`make_dreamzero_pre_post_processors` 原先收 `**kwargs` 但**从不使用**
`preprocessor_overrides`,rename_map 和 device 覆盖会被悄悄丢弃。现在照 GR00T 的
`_apply_groot_step_overrides` 实现了等价逻辑,**匹配不到的 override key 直接报错**。

## 训练管线(进行中)

### 采样窗口:用 lerobot 原生 delta indices,不需要移植上游采样器

上游的多锚点采样器双向扩展只改变窗口的**采样权重**,不扩大**取值集合**;同样的窗口用原生
delta indices + `drop_n_last_frames` 就能表达,也不需要拒绝重采(短窗口本来就不会被采到)。

窗口参数是**从几何推导**的,不是硬编码——换机器人改 `action_horizon`/`num_frames` 会自动跟随:

```
frames_per_block   = (num_frames - 1) / max_chunk_size = 32/4 = 8
video_frame_stride = action_horizon / frames_per_block = 24/8 = 3
observation_delta_indices = range(0, 97, 3)   # 33 帧
action_delta_indices      = range(96)          # 24 x 4
drop_n_last_frames        = 96
```

### 相对动作:按宏块锚定,已用发布数据 golden 验证

块 c 的 24 个动作减去**该块首行**的 raw state,然后才做 q99 归一化。用
`GEAR-Dreams/DreamZero-DROID-Data` 发布的 `meta/relative_stats_dreamzero.json` 反推约定:

| 约定 | q99(关节 0-2) | 相对误差 |
|---|---|---|
| 发布值 | [0.3715, 0.7104, 0.3510] | — |
| 逐步 `action[i]-state[i]` | [0.058, 0.158, 0.082] | 74% |
| **按宏块(本实现)** | [0.3797, 0.7107, 0.3679] | **6.3%** |
| 整段锚定 `action[i]-state[0]` | [0.5479, 1.353, 0.458] | 122% |

误差随样本量单调收敛(10 ep → 18.3%,50 ep → 6.3%,200 ep → 5.2%);剩余部分来自我们只有
200/95658 个 episode,且发布统计算在过滤子集 `droid_101_success_idlefiltered6` 上。
两个替代约定分别小 6 倍、大 2 倍,不可能是同一个量。**计划里的头号风险(采样/相对动作语义)据此关闭。**

用单一锚点为什么会毁掉训练信号:块 3 相对块 0 的位移可达 +3.0,而相对动作的 q99 区间是 ±2.0
—— q99 归一化会把它 clamp 到 1.0,信息直接丢失。已有专门的测试锁住这点。

### 视频:训练随机裁剪 + 色彩抖动,偏移跨视图共享

顺序 crop → resize → jitter → stitch,与上游 `oxe_droid` transform 链一致。关键细节:上游把
**所有视图和所有帧拼成一个 batch 只做一次变换**,所以裁剪偏移和抖动参数在视图间**共享**。
按视图独立随机裁剪会让拼接画布错位——画面看着正常,几何却不是模型训练时见过的。

### 已放弃的两个做法

- **不做 checkpoint 转换**:RLinf 也不转换。它把"配置来源"和"权重来源"分开——配置来自框架
  自己的配置系统,checkpoint 只提供权重和统计。lerobot 等价写法是
  `--policy.type=dreamzero --policy.pretrained_path=<GEAR 目录>`,共享代码一行不用改。
- **不用 `--policy.path`**:那条路会走 `PreTrainedConfig.from_pretrained`,撞上 GEAR config.json
  没有 `type` 键的问题。

### 优化器预设已更正

原先是旧 LoRA 脚本的值(lr 1e-4 / betas 0.9,0.95 / wd 1e-4 / warmup 0.05),现改为官方 DROID
全参配方:**lr 1e-5 / betas (0.95, 0.999) / eps 1e-8 / wd 1e-5 / warmup 0.01**
(`scripts/train/droid_training_full_finetune_wan21.sh`,RLinf 的 yaml 一致)。

### 训练已跑通(M3 单卡 / M4 八卡全参)

| | 单卡 lora_rank=0 | 8xH200 全参 FSDP |
|---|---|---|
| 可训练参数 | 89.4 M(0.39%) | 16,484 M |
| 稳态单步 | 5.0 s | **6.9 s** |
| 显存 | 57.7 GB | 57.0 GB/卡 |
| 梯度范数 | 0.08–0.12 | 0.23–0.38 |

吞吐与 RLinf 公开数字吻合(8xH100 上纯 FSDP2 为 9.0 s/步、带其编译优化为 6.7 s/步;我们未移植那套优化,在更快的硬件上落在 6.9 s)。这是有意义的交叉验证——实现若有严重问题不会恰好落进参考区间。

**精度方案与 RLinf 及官方一致:fp32 主权重 + bf16 计算。** 官方的 DeepSpeed 配置里看不到 "fp32",
是因为 ZeRO-2 默认把 fp32 副本放在被分片的"优化器状态"里,不需要声明。

bf16 主权重在参考 lr 下不可行,用 checkpoint 真实权重实测(`|w|` 中位数 1.2e-2):
lr=1e-5 时仅 **16%** 的 Adam 更新能改变权重,lr=1e-4 时 98%。所以它是超参选择问题,
但换 lr 就丢掉了上游验证过的配方,而在 143 GB 的卡上省下的显存换不到什么。

### 训练路径上修复的缺陷(全部为原移植遗留,除最后一条)

| 缺陷 | 暴露点 |
|---|---|
| `get_scheduler_preset` 引用不存在的 `self.max_steps` | 配置校验 |
| 缺 `normalization_mapping` | 构造 processor override |
| `get_optim_params` 返回 dict 而非列表(优化器会迭代出字符串键) | 优化器构造 |
| 统计 schema 不匹配 → 动作分支被 `if` 静默跳过 | 训练 forward,报错在数百行之外 |
| 对带梯度张量直接 `float()` | 运行时警告 |
| 0 维参数提升放在 `__init__`,先于加载 → 形状冲突(**本轮引入**) | FSDP 包装 |

前五条能潜伏至今,只因训练路径从未被执行过一次;28 个单测全绿也一个都发现不了。
其中第 4 条最典型:`and self._action_layout.dim > 0` 这个条件把配置错误变成了静默降级,
现在改为有动作但无统计即报错。

### 调用方式:type + pretrained_path,不是 path

`--policy.path=<GEAR 目录>` 会走 `PreTrainedConfig.from_pretrained`,撞上 GEAR config.json
没有 `type` 键的问题。正确写法是 `--policy.type=dreamzero --policy.pretrained_path=<目录>`
——配置来自 lerobot 自己的注册表,checkpoint 只提供权重和统计,与 RLinf 的分离方式相同。

相机命名差异有两种走法,**`--rename_map` 在 `lerobot-train` 上同样可用**
(它是 `TrainPipelineConfig` 的字段,见 `configs/train.py:134`,要求配 `--policy.pretrained_path`)。
但它替代不了 `video_modality_keys`:rename_map 只改名不定序,而拼图要的是**哪路相机落在哪个象限**,
提示词也由同一个列表生成。`video_modality_keys` 为空时回退到键名排序,DROID 恰好排对了,
那是字母序的巧合,换成 `top`/`wrist` 的机器人就不成立。

### 统计量的保存与再生成

**微调 checkpoint 现在会带上自己的统计量。** 之前 `statistics.json` 只有读、没有写:
微调产物重新加载时找不到量化常数,q99 反归一化和相对→绝对解码**双双退化成恒等变换**——
动作看着合理,实际全错。现在 `DreamZeroPolicy._save_pretrained` 会写出来,
读写两侧共用 `gear_checkpoint.load/save_checkpoint_statistics`,不会各自漂移。

已在真机验证:单卡训练 2 步存档 → 重新加载 → 与原始 GEAR checkpoint 的 q01/q99 逐位相同
(max|diff| = 0.000e+00,state/action/decode 三处)。

> 顺带:GR00T 有同样的缺口(它也只读不写 `statistics.json`)。未改动,不在本次范围内。

**新数据集的统计量**由 `scripts/compute_statistics.py` 生成。它直接读表里的 state/action 列
(不解码视频),并且把位移计算**路由回 processor 自己的 `_encode_relative_actions`**,
所以生成的统计量不可能描述一套和训练时不同的约定。

用它重算 DROID 自己的统计量、与 NVIDIA 发布值对照(16 段):

| 约定 | 与发布分位数的平均相对误差 |
|---|---|
| 绝对动作 | 91.2% |
| 逐步 delta(减当前状态) | 69.8% |
| **按宏块 delta(本实现)** | **12.2%** |

12.2% 是样本量而非约定问题:同一比较随段数收敛(10 段 18.3% → 16 段 12.2% → 50 段 6.3% →
200 段 5.2%)。错误约定不收敛到任何地方——它们差一个倍数,而用它们训练出的策略照样能跑,
只是每个位移都被缩放了那个倍数。

### LoRA:注入顺序是关键

PEFT 会重命名它包装的每一个参数
(`blocks.0.self_attn.q.weight` → `base_model.model.blocks.0.self_attn.q.base_layer.weight`),
所以适配器该在加载权重之前还是之后注入,**取决于读的是哪种 checkpoint**:

* GEAR 发布版没有适配器、键名未包装 → 必须**先加载再注入**(`defer_lora_injection=True`);
* LoRA 微调自己产出的 checkpoint 已含包装后的键 → 必须**先注入再加载**(构造时注入)。

搞反不会报错:`load_state_dict(strict=False)` 只会把每个 DiT 权重报为 missing,
然后把随机初始化的值原样留下。`DreamZeroPolicy._finalize_lora` 处理这个方向差异。

两个方向都用真实的 2 步 LoRA 训练验证过:构造时包装能**完全**复现保存的键集
(2946/2946,无 missing 无 unexpected),而在同一路径上误用延迟注入会让 **2117 个键**被报为
missing —— 且 `strict=False` 会让训练照常继续。

### `save_lora_only`:208 MB 而不是 46 GB

LoRA 一轮只更新 22,924 M 里的 108.6 M,把其余 99.5% 再写一遍每个 checkpoint 要多花 ~46 GB。
现在 `training_mode=lora` 默认开启 `save_lora_only`(与上游 `droid_training_lora.sh` 一致):

| | 全量权重 | `save_lora_only` |
|---|---|---|
| 大小 | 45.8 GB | **208 MB** |
| 张量数 | 2946 | 814 |
| 自包含 | 是 | 否,需要基座 checkpoint |

和上游一样,存的是**所有 `requires_grad` 参数**而不只是 `lora_*`:投影层也在训练
(108.6 M 里有 89.4 M 是投影层),丢掉它们会静默扔掉大部分微调结果。

因为不自包含,checkpoint 会写一个 `lora_adapter.json` 记下基座路径,`from_pretrained`
先加载基座再叠加增量;基座不在就报错而不是猜。要自包含就传 `--policy.save_lora_only=false`。
`full` 永远写完整权重——它重写整个 DiT,没有值得省下的冻结部分。

真机端到端验证:814 个张量逐位复原;400 个 `lora_B` 全部非零
(PEFT 把它们初始化为 0,所以非零正是"微调确实落上了、不是被重新初始化"的判据);
冻结的 DiT 权重与基座逐位相同。

> 我先前按自注意力估算的 12.6 M 是错的:目标模块名 `q,k,v,o` 同时匹配了自注意力**和交叉注意力**,
> 每个 block 实际包装 10 个模块(不是 6 个)。19.2 M 是从 checkpoint 实测的。

### 命令行必须显式写 `--policy.training_mode`

默认值是 `lora`,但**文档和实际命令都应显式写出来**。两个模式差着两个数量级
(可训练参数 0.47% vs 71.9%,存档 208 MB vs 46 GB),留给默认值意味着这条命令
脱离"当时默认值是什么"就读不懂、也复现不了 —— 而这个默认值今天才从 `full` 改成 `lora`。

启动时那一行是权威记录:

```
DreamZero training_mode=lora (lora_rank=4): 108.6 M trainable of 22943.3 M (0.47%)
```

两个间接信号也可用于交叉确认:显存(LoRA ~42.5 GB/卡 vs 全参 ~57.0)和存档大小。

### 训练模式合并为两个

`projector_only` 并入 `lora`:两者本来是嵌套的(LoRA 训投影层**加**适配器),
差别只在有没有适配器,也就是 `lora_rank`。

| training_mode | lora_rank | 可训练 | checkpoint |
|---|---|---|---|
| `full` | — | 16,484 M | 45.8 GB(完整) |
| `lora` | 0 | 89.4 M | **171 MB** |
| `lora` | 4 | 108.6 M | **208 MB** |

原 `projector_only` 现在也享受增量保存(45.8 GB → 171 MB)。
`--policy.training_mode=projector_only` 仍可用,会映射成 `lora` + `lora_rank=0` 并打日志。

> 我先前在配置注释里写 "`full` updates everything, so there is no base to defer to" 是**错的**:
> text/image/VAE 编码器(6,440 M)在三个模式下都冻结。`full` 理论上也能只存 16,484 M
> (45.8 GB → 33.0 GB,无损)。按你的决定,`full` 保持原样写完整权重。

### M5:新本体(SO-100/SO-101)

**拼图**走上游对 agibot/yam/xdof 用的同一条通用路径:2×2 网格,依次填左上、左下、右上,
空的象限留黑。DROID 保留它自己的三视图布局。画布恒为 352×640(由 `frame_seqlen` 钉死),
相机少不等于画布小——少的部分是黑边,这正是模型训练时见过的填充。

**提示词**按象限生成,因为这是模型唯一能知道哪个格子是哪个相机的渠道。措辞照上游
AGIBOT/YAM("...split into four views: The top-left view shows..., and the bottom-right view
is a black screen (inactive view)")。相机名取自 `view_descriptions`,未设则从
`video_modality_keys` 的键后缀推。没有名字时**报错**而不是猜。

**帧率**是个容易漏的语义问题。窗口契约数的是**步**不是**秒**:96 个动作、33 帧步长 3。
DreamZero 在 DROID 的 15 fps 上训练,所以那 96 步 = 6.4 秒。30 fps 录的机器人会喂给模型
3.2 秒的窗口,而它学的是 6.4 秒的动力学。`--policy.source_fps` 会按 `source_fps/15` 拉伸
delta 索引:

| source_fps | 倍率 | 观测行 | 动作 | 原始跨度 | 实际时长 |
|---|---|---|---|---|---|
| 15 | 1 | 33 @ 步长 3 | 96 @ 步长 1 | 96 | 6.4 s |
| 30 | 2 | 33 @ 步长 6 | 96 @ 步长 2 | 192 | 6.4 s |
| 60 | 4 | 33 @ 步长 12 | 96 @ 步长 4 | 384 | 6.4 s |

动作是绝对位置目标而非增量,所以隔帧取样改的是控制频率不是语义。低于 15 fps 直接拒绝
(没有任何步长能把短窗口拉长)。块锚点仍是观测行 [0,8,16,24]——倍率在分子分母上抵消了。

**SO-100 具体映射**:6 个电机(5 关节 + 夹爪)→ `joint_position:5, gripper_position:1`。
模型侧零改动,6 维填充进 max_state_dim 64。关节是归一化单位、夹爪 0-100,与 DROID 的弧度
和 0-1 完全不同,但 q99 归一化本来就吸收量纲,只要统计量来自 SO-100 数据。

**必须说清楚的局限**:能吸收形态差异的按本体投影层是被**绕过**的
(DiT 硬编码 `embodiment_id = 0`,编码器按 `max_num_embodiments = 1` 构建)。
7 自由度 Franka(弧度)→ 5 自由度舵机臂(归一化单位),**没有任何专用适配层**,
全部要挤进共享权重。预期是需要实打实的微调量,不是少样本迁移。
合成 SO-100 窗口已端到端验证(33 帧、352×640、4 锚点、96 动作),但**没有跑过 SO-100 微调**,
所以这里不对策略效果做任何声明。

## 与 RLinf 的对照(`/data/dongmao_dev/RLinf`)

RLinf 做了 DreamZero 的 SFT(DROID/LIBERO/Franka),但**不 vendor 模型代码** —— `WANPolicyHead`/
`CausalWanModel`/loss 仍是外部 `groot` 依赖。它自有的是数据管线、embodiment 注册表、FSDP2 训练循环。

**逐条印证了本移植的推理侧实现**:per-view 176x320;wrist 顶部横向复制 2x、exterior_1 左下、
exterior_2 右下;canvas 352x640;state/action 全部 q99;统计按 embodiment tag 从 `metadata.json` 取;
DiT forward 里 `embodiment_id` 被硬置 0。

**解答了本计划的头号风险(采样窗口语义)**。原风险清单第 1 条"33 帧 ↔ latent ↔ block ↔ action chunk
最易错"——RLinf 的答案是:`delta_indices` 是死配置,没人读。真正的契约是

```
video  = 8 帧/宏块 x max_chunk_size + 1 个边界帧   = 8*4+1 = 33
action = action_horizon x max_chunk_size          = 24*4  = 96
state  = 1/宏块                                    = 4
```

宏锚点间隔 `action_horizon`,块内视频帧 stride 3,受语言标注边界约束双向扩展;窗口不足**直接丢弃重采**,
不 pad。相对动作**按宏块**减去该块首行的 state,且在归一化**之前**做。

**与本计划原假设的两处出入,需要修正计划**:
- 原计划 M3 定的是 **LoRA 后训练**;RLinf 实际用的是**全参微调** + FSDP2 full_shard
  (14B 在 8xH100 上 `micro_batch_size=1`,6.7 s/step)。LoRA 脚手架在,但所有已发布配置都是 `full`。
- 原计划设想复用 groot 的 relative stats 机制;实际统计由
  `toolkits/lerobot/generate_dreamzero_metadata.py` 从数据集现算 q01/q99。

## 下一步

| # | 事项 | 说明 |
|---|---|---|
| 1 | eval 结果对基线的解读 | MSE 必须显著优于 hold-state 基线才说明推理链路真的对 |
| 2 | 数值 parity | 对上游/RLinf 同 checkpoint 同观测逐 chunk 比对,`num_inference_steps=16`、不 compile |
| 3 | gRPC PolicyServer | `predict_action_chunk` 已就绪,理论上直接可服;未验证 |
| 4 | 新机器人 SFT | 见下 |

### 新机器人(如 SO-101)要做什么

模型侧**不需要改维度** —— state <= 64、action <= 32 都是零 pad,再按统计里的 per-key 宽度切回来。
需要做的是:

1. 一个 embodiment transform:相机 key、拼接布局、**与该布局文字对应的 prompt 模板**、q99 归一化。
   现在 `processor_dreamzero.py` 的拼接与模板对 oxe_droid 硬编码,其余 embodiment 直接
   `NotImplementedError` —— 这是有意的,套用未验证的语义比报错更糟。
2. embodiment tag + projector index(注意:训练时 projector 被 `embodiment_id=0` 旁路,
   该 index 只在 `action_loss_embodiment_ids` 掩码和推理时起作用)。
3. 用数据集现算 q01/q99 统计。
4. 训练数据管线 —— 即上面那套 `8*max_chunk_size+1` 采样契约,目前**未移植**。

坦白讲,4 是真正的工作量,且 14B 全参微调需要 8xH100/H200 级别的资源。
如果目标只是"先能在自己机器人上跑起来",更务实的路径是先不碰 SFT,
把 DROID checkpoint 的推理链路 + gRPC 真机闭环走通。

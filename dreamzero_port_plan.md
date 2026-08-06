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
2. `configuration_dreamzero.py`:把 Hydra 配置翻译成 draccus dataclass——两档模型(`wan21_14b`:dim 5120/176×320/VAE 16ch;`wan22_5b`:dim 3072/160×320/VAE38 48ch),关键字段 `num_frames=33`、`action_horizon=24`、`num_views=3`、`num_frame_per_block=2`、`num_action_per_block`、`max_state_dim`/`max_action_dim`、embodiment 表(`oxe_droid`/`agibot`/`yam` + projector index)、`cfg_scale=5.0`、`num_dit_steps`。
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

## 进度(分支 port-dreamzero)

**已完成 — 模型核心全部移植,diff 逐字节校验 + py_compile 通过**
`src/lerobot/policies/dreamzero/wan/` 下 12 个文件(near-verbatim,仅头部/import 改写):
- `action_encoder.py` `attention.py` `causal_attention.py`(inline `is_hopper_gpu`) `submodule.py`
- `causal_wan_model.py`(2245 行联合视频+动作因果 DiT) `vae.py`(WanVideoVAE+VAE38)
- `text_encoder.py`(umt5) `image_encoder.py`(inline `flash_attention`) `schedulers.py`(FlowMatch+FlowUniPC) `vram.py`(inline `init_weights_on_device`)
- `action_head.py` = `WANPolicyHead`+`WANPolicyHeadConfig`,surgical 8 处(diff 已验证方法体逐字节一致):去 hydra `instantiate`/accelerate;config→普通 dataclass;4 个 `instantiate()`→直接构造(`_component_kwargs` 剥 hydra meta + `vae_class` 选 VAE/VAE38);groot→相对 import;TensorRT→raise。
- `base_action_head.py` = 极简 `ActionHead` 基类。

**已完成 — lerobot wrapper 骨架(import-verified,ruff 全绿)**
- `configuration_dreamzero.py`:`DreamZeroConfig(PreTrainedConfig)` 注册 `dreamzero`(draccus choice 已验证),两档 variant(14B/5B DiT dim/in_dim/layers/VAE 类经 build 验证正确),`build_action_head_config()` 组装内部 `WANPolicyHeadConfig`,实现全部抽象方法(observation/action_delta_indices、AdamW/cosine preset、validate_features)。embodiment→projector 索引 + 固定 negative prompt 常量已就位。
- `modeling_dreamzero.py`:`DreamZeroPolicy(PreTrainedPolicy)` name=`dreamzero`;`predict_action_chunk`/`select_action` 调 `lazy_joint_video_action`(原生 KV-cache 推理),`reset()` 清会话态(`current_start_frame`/kv_cache/`clip_feas`/`ys`/`language`)+ action queue;`self.action_head` 命名对齐释放 checkpoint `action_head.*` key;首版单环境(num_envs=1)。
- `processor_dreamzero.py`:pipeline 骨架已 build(rename→add batch→pack→device;decode→device);两个 DreamZero-specific step(`DreamZeroPackInputsStep`/`DreamZeroActionDecodeStep`)**故意 raise NotImplementedError**,不静默套用未验证的 normalization/tokenizer——等 golden 验证(M2)。
- `__init__.py` lazy 导出 + `factory.py` 三处分支(get_policy_class/make_policy_config/make_pre_post_processors 均解析 `dreamzero`)+ `pyproject` `dreamzero` extra(+ 进 `all`)。
- 验证:`get_policy_class('dreamzero')→DreamZeroPolicy`;两档 config build 出正确 DiT 维度;processor 双 pipeline build 成功;ruff 全绿。(本地装了 peft 完成 import check。)

**未完成 — 下一阶段(需 checkpoint + GPU 验证)**
1. `processor_dreamzero.py` 两个 step 的真实实现(q99→[-1,1] + relative action + state pad + umt5 tokenize + 多视角拼接 + embodiment_id / 反归一化 + relative→absolute + gripper),对 upstream dataloader golden 样本逐字段校验。**最大正确性风险。**
2. checkpoint 转换器:读 NVIDIA `GEAR-Dreams/DreamZero-DROID`(DreamZero 格式 config + `model.safetensors` VLA 结构)→ 写 lerobot 格式 checkpoint(lerobot `config.json` + `action_head.*` key + processor config)。lerobot `from_pretrained`(strict=False)才能加载。
3. 离线开环脚本 + parity 测试(对 upstream 逐 chunk 比对,`DREAMZERO_REPO` + `NUM_DIT_STEPS=16`)。
4. gRPC PolicyServer 服真机(predict_action_chunk 就绪即可用)。
5. tiny-config CPU 单测(forward/select_action/processor 往返)。

**本地限制**:macOS/CPU 无法验证 GPU 推理与释放 checkpoint 的精确 config/normalization(VAE 构造硬编码 `device='cuda'`);需 H20 + `GEAR-Dreams/DreamZero-DROID` checkpoint。

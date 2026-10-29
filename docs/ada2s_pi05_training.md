# Ada2S：UMD 修复与 Pi0.5 单卡训练

核对日期：2026-10-29（本机时区）。目标节点 `root@10.249.186.251`。
环境搭建和代码改动见 [Lark 文档](https://bytedance.us.larkoffice.com/docx/XHwadb4hKo8zhGx9tBvuMYlcsMc)。

已连接目标机器并完成基础计算对照：容器加载宿主机 UMD 后，加法、逐元素乘法和矩阵乘法
均通过数值回读验证。`ada2s` 分支的 Pi0.5 单卡微调也已完成 20 step，退出码 0。

## Pi0.5 单卡训练结果

| 项目 | 本次配置 / 结果 |
| --- | --- |
| 容器 / 源码 | `ada2s_compute` / `/workspace/lerobot-ada2s`，基线 commit `900e6f4c5f29b4487beb5adefd68cd81287db931` 加本地 Ada 适配 |
| 设备 | 物理卡 0，逻辑设备 `ada:0`，单进程 |
| 预训练权重 | `lerobot/pi05_base`，revision `22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0`，全部 key 成功加载 |
| 数据 | `lerobot/pusht` v3.0，206 episode、25,650 frame |
| 训练 | BF16，batch 1，20 step，开启 gradient checkpointing，关闭 compile |
| 参数 | 日志记录约 4.14B learnable parameter，保持完整模型微调配置 |
| 归一化 | 状态和动作使用 `MEAN_STD`，图像 `IDENTITY`；此数据集未提供 quantile stats |
| CPU 亲和性 | `taskset -c 0-15`，`OMP_NUM_THREADS=8` |
| 稳定 step 时间 | 约 1.03 秒，排除初始化、首轮预热和 checkpoint 保存 |
| 设备内存 | peak allocated 36.93 GiB，reserved peak 42.67 GiB，allocator retries 0 |
| loss | 20 个 step 均有限；最后一个 step 0.733，未将短程 loss 当作策略效果指标 |
| checkpoint | 第 10、20 step 保存成功；第 20 step 模型和 7/3 步前后处理器重新加载通过 |
| 参数更新 | action projection 对初始权重的最大绝对变化 `0.0002141036093235016` |
| optimizer 状态 | 805 个参数状态的 step 均为 20；保存的 batch size / process count 均为 1 |

容器内运行目录为 `/workspace/ada2s-training-20261030/`，宿主机对应
`/data/workspace/ada2s-training-20261030/`：

- 日志：`train.log`，退出状态：`train.exit`（0）。
- 最新模型：`pi05-single/checkpoints/000020/pretrained_model/`。
- 训练状态：`pi05-single/checkpoints/000020/training_state/`。
- 重载和参数更新证据：`checkpoint-check.json`。

启动脚本为 [train_pi05_ada2s.sh](../examples/training/train_pi05_ada2s.sh)。在目标宿主机上复跑到
新目录，避免与已完成的输出目录冲突：

```bash
docker exec -it \
  -e OUTPUT_DIR=/workspace/ada2s-training-20261030/pi05-single-next \
  -e STEPS=20 \
  -w /workspace/lerobot-ada2s \
  ada2s_compute taskset -c 0-15 bash examples/training/train_pi05_ada2s.sh
```

脚本默认保存中途和最终 checkpoint，间隔随 `STEPS` 缩放；也可用 `SAVE_FREQ` 或原 CLI 参数
覆盖。当前权重和数据已预取，脚本使用离线 HF 缓存。20 step 验证了训练和保存链路，
尚未评估策略任务成功率。

## 现场结果

宿主机 `ada-info` 报告 suite `1.0.1`，32 张卡均被 `bdaatool -l` 标为 `avail`。
实际宿主机 UMD 和原容器 UMD 不一致：

| 组件 | 宿主机 | 原容器 ada2s_smoke2 |
| --- | --- | --- |
| ucdrv | 4.1.1 | 4.0.16 |
| bdaa-sdk / libbdaa_plat | 3.8.1 | 3.7.17 |

通过 `/proc/self/maps` 确认原容器实际加载了旧库。保持 torch `2.9.0+cpu`、
torch_adas `1.2.6+torch2.9.0.gitebc58df`、adax-rt `1.1.13`、adadnn 等工具链不变，
仅将宿主机 `libucdrv`、`libucdrvhal`、`libbdaa_plat` 复制到独立目录并为测试进程设置
`LD_LIBRARY_PATH`，确认加载路径改变后执行同一脚本：

| 环境 / 卡号 / 矩阵尺寸 | FP32 vector | FP32 非 vector | BF16 |
| --- | --- | --- | --- |
| 原容器 / 0 / 4×4 | H2D/D2H 阶段超时 45 秒 | 未测 | 未测 |
| 宿主机 UMD / 0 / 4×4 | 通过 | 通过 | 通过 |
| 宿主机 UMD / 0 / 256×256 | 通过 | 通过 | 通过 |
| 宿主机 UMD / 15 / 256×256 | 通过 | 通过 | 通过 |
| 宿主机 UMD / 31 / 256×256 | 通过 | 通过 | 通过 |

每个“通过”都包含 H2D/D2H、加法、逐元素乘法和矩阵乘法的 CPU 数值对照；
使用真卡模式、同步 launch、禁用 CPU fallback 和 fake kernel。
基础计算测试覆盖 Box 0、1、3，并非全 32 卡验收。后续的反向传播和 AdamW 验证
在物理卡 0 的完整 Pi0.5 训练中完成，结果见上方训练记录。

这些结果支持“原容器使用与宿主机不匹配的 UMD 导致当前计算阻塞”的结论，
也否定了仅凭原环境失败就直接认定整机计算硬件损坏的判断。
本次同时替换两个组件的库，没有进一步区分具体是哪一个库引发阻塞。
`no coredump ... callback` 警告在通过的环境仍会出现，因此它不能单独作为计算失败判据。

日志和宿主机库对照副本位于目标机器 `/data/workspace/ada2s-mul-20261030/`。
原容器里可复现已通过的测试：

```bash
# 在目标宿主机执行。
docker exec \
  -e LD_LIBRARY_PATH=/workspace/ada2s-mul-20261030/host-umd \
  ada2s_smoke2 python3 -u /workspace/ada2s-mul-20261030/smoke.py \
  --card 0 --size 256 --timeout 45
```

该目录中的库副本对应本次宿主机版本，宿主机升级后应重新核对。
已创建并启动 `ada2s_compute` 容器：从安装好工具链的原容器保存镜像
`local/ada2s-toolkit:pi05-20261030`，只读挂载宿主机 `/usr/local/ucdrv/lib` 和
`/etc/sysop/bdaa/lib`，并将这两个目录放到 `LD_LIBRARY_PATH` 前面。
默认设置真卡模式、Ada2S 架构和物理卡 0；脚本的 `--card` 可以按测试覆盖需要重新选择卡。

通过 `/proc/self/maps` 确认新容器加载 `libucdrv.so.4.1.1`、`libucdrvhal.so.4.1.1` 和
`libbdaa_plat.so.3.8.1`。在新容器中，原始 `torch.randn(4,4,device='ada') → x @ x`
以及逐元素乘法都通过 CPU 数值对照；FP32 vector、FP32 非 vector、BF16 的 4×4
完整基础计算校验也全部通过。

```bash
# 以下命令在目标宿主机执行。
docker exec -it ada2s_compute bash

# 或直接复跑带超时和数值校验的测试。
docker exec ada2s_compute python3 -u /workspace/ada2s-mul-20261030/smoke.py \
  --card 0 --size 256 --timeout 45

# 容器停止后重新启动。
docker start ada2s_compute
```

新容器保留现有工具链依赖，没有升级宿主机驱动或固件。

## 实验记录能证明什么

[实验报告](https://bytedance.us.larkoffice.com/docx/U5dYdqVpnoJl3IxwGVSu15DusfP)
revision 3 记录：32 张卡中抽样 12 张，`torch.randn(4,4,device='ada')` 返回后，
首次 `x @ x` 挂起；重启和补挂 `virtualTopology.xml` 没有改善。

这证明报告中的环境存在跨多卡的计算阻塞，但不能据此确认整机硬件损坏。
张量创建返回不代表异步计算完成；矩阵乘挂起也可能来自共享的运行时、库版本、
特定 dtype 或 kernel 路径。报告里的 XID `0x40d0307` 被查询工具描述为事件完成通知，
不能单独作为硬件损坏证据。

## 优先核对的差异

| 项目 | 文档证据 | 应核对的内容 |
| --- | --- | --- |
| 真卡模式 | Torch Ada 运行时表格 revision 439：`ADAX_RT_USE_TSIM_MODE` 默认 1；0=ASIC，1=性能模拟器，2=lite 模拟器 | 必须在测试进程启动、导入 torch_ada 前显式设置 0；安装脚本的 export 不代表后续 shell 继承了它 |
| FP32 矩阵乘 | 同一表格：`TORCH_ADA_ALLOW_FP32_BY_VECTOR` 默认 1 | 分别测试 FP32 vector、FP32 非 vector、BF16；报告未提供 BF16 对照 |
| 工具链版本 | Torch Ada 手册环境准备章节当前稳定版 `0.1.12`；报告安装 `0.1.11` | 先记录现有版本与实际加载库，再选择厂商确认配套的组合；不要只升级一个 wheel |
| fullstack | fullstack 手册当前业务基线 `1.1.0.3-1`（2026-10-19），生产基线 `1.0.5`；报告镜像 tag 为 `0.1.7_0.9.17` | 镜像 tag 不能证明宿主机当前驱动与固件版本，须现场运行 `ada-info` |
| driver/UMD 注入 | fullstack 手册 §2.2：从 `1.0.4.3` 开始，启动容器时动态注入宿主机 driver 文件 | 使用这一代 fullstack 时核对 ada-container-toolkit 配置、容器挂载和实际加载的 UMD；旧版环境需按其配套规则判断 |
| 拓扑文件 | 容器指南 §6.2：`virtualTopology.xml` 供 PDCCL/AdaCCL/M2N 为 rank 选择网卡 | 补挂该文件不能证明单卡 GEMM 的运行时正确；身份 token 和训练数据权限也不能直接解释小矩阵乘挂起 |

新增的 driver 注入指南和 Torch Ada release note 链接当前账号无读取权限。
本次只确认了上级手册的要求，尚未获得注入指南里的具体配置命令。

后续现场检查先保存 `ada-info`、`ada-smi -L`、`bdaatool -l`、容器的镜像/挂载/设备清单，
再检查进程环境和实际加载的 `.so` 路径。环境变量应只输出与 Ada 相关的白名单，
不要打印身份 token 内容或完整环境中的凭据。若需要切换软件栈，创建独立对照容器，
保留原环境用于对比。

## 带超时的单卡计算复测

脚本 [ada2s_smoke.py](../examples/training/ada2s_smoke.py) 使用容器内已经安装
vendor torch/torch_ada 的解释器。它不依赖 LeRobot 安装，所以也能在官方 Python 3.11
容器中运行。每种计算路径单独启动子进程；默认单项超时 60 秒，超时只清理本次创建
的进程组。

```bash
# 在目标容器的仓库目录执行，使用 vendor 软件栈所在的解释器。
python3 examples/training/ada2s_smoke.py --card 0 --size 4 --timeout 60

# 首轮通过或发现小尺寸特例后，再对照较大的矩阵。
python3 examples/training/ada2s_smoke.py --card 0 --size 256 --timeout 60

# 也可以仅复测 BF16。
python3 examples/training/ada2s_smoke.py --card 0 --case bf16 --timeout 60
```

脚本将指定物理卡通过 `ADAX_VISIBLE_DEVICES` 暴露为 `ada:0`，强制真卡模式、同步 launch，
关闭 fake-kernel 模式和未注册/运行失败算子的 CPU fallback。每项依次验证：

1. 导入 torch 和 torch_ada。
2. CPU 创建确定性输入，H2D 后 D2H 回读并核对数值。
3. 加法、逐元素乘法和矩阵乘，分别回读并与 CPU 结果对比。

额外传入 `--training` 才继续检查反向传播和 AdamW。该脚本的现场复测使用基础计算模式；
完整训练另由上方 Pi0.5 启动脚本验证。

日志中最后一条没有对应 `PASS` 的 `START` 定位阻塞阶段。CPU fallback 导致的成功
不能作为 Ada 算子通过的证据；这里禁用 fallback 也意味着不支持的算子会显式报错。

| 对照结果 | 下一步 |
| --- | --- |
| 回读或加法已挂起 | 优先查 runtime/UMD/设备状态，而非 GEMM 或 Pi0.5 |
| 仅 FP32 vector 挂起，BF16 正常 | 查 FP32 vector kernel；可继续验证 BF16 模型，但仍要覆盖模型中的 FP32 算子 |
| 仅 4×4 挂起，大矩阵正常 | 查小尺寸 kernel/调度，不宜判为所有计算都失效 |
| 原环境失败，配套官方对照环境通过 | 查安装、库加载和容器注入差异 |
| 配套官方环境也失败 | 带版本、加载库、分阶段日志和 XID 请平台/厂商诊断；此时才有更强依据申请置换 |

## Pi0.5 单卡适配与环境

`ada2s` 分支现已支持 Python 3.11：将 Python 3.12 专用的泛型和类型别名语法改为
`TypeVar` / `Generic` / `TypeAlias`，同步修改 `pyproject.toml` 和 `uv.lock`。
535 个 Python 源文件在目标 Python 3.11 解释器下通过语法检查。

| 组件 | 实际安装版本 |
| --- | --- |
| Python | 3.11.2 |
| torch | 2.9.0+cpu |
| torch_adas（导入名 torch_ada） | 1.2.6+torch2.9.0.gitebc58df |
| torchvision / torchaudio | 0.24.0+cpu / 2.9.0+cpu |
| transformers / accelerate | 5.5.4 / 1.15.0 |
| datasets / PyAV | 4.8.5 / 15.1.0 |

Ada 使用厂商 torch 和匹配的插件。Linux 的 uv sources 仍指向 CUDA wheel，
准备 Ada 容器时使用容器解释器和 pip 约束文件，保护预装 torch。
`/workspace/ada2s-training-20261030/torch-constraints.txt` 锁住 torch、torchvision 和 torch_adas；
CPU torchaudio 单独固定为 2.9.0+cpu。最终 `pip check` 返回 `No broken requirements found`。
训练依赖安装在 `ada2s_compute` 的容器层；重建容器时需重新安装，源码和训练文件在宿主机挂载目录。

实现包含以下调整：

- 设备工具懒加载 `torch_ada`，识别 `ada` / `ada:0`，查询实际可用设备及 BF16 支持。
- [AdaSingleDeviceAccelerator](../src/lerobot/utils/ada.py) 明确指定 `ada:0`，要求单进程，
  通过厂商 `torch.autocast("ada", dtype=torch.bfloat16)` 执行 BF16 前向；继续复用 Accelerate 的
  数据准备、反向和优化器管理。
- Pi0.5 时间编码原来先生成 FP64 再转换为 FP32，实际触发
  `Double --> Float has not been supported in Ada cast`。Ada 路径现在直接计算 FP32，
  在 64 个时间点、1024 维编码上的 CPU FP64 对照最大绝对误差约 `1.83e-4`。
- 请求预训练权重时，加载失败会抛出 `StateDictLoadError`，避免继续使用未加载的随机权重。

`torch` 提供 Tensor、自动求导、模型和优化器；`torch_ada` 接入 Ada2S，设备类型是 `ada`；
`torch_npu` 接入昇腾，设备类型是 `npu`。现有 `npu_fused_*` 分支继续用于昇腾，
本次 Ada 训练使用通用模型算子路径。

已有处理器流水线和 IO 测试共 86 项通过。本机旧 Hydra 自动插件与 Python 3.12 不兼容，
针对性检查使用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 执行；目标容器的真实训练和 checkpoint
加载则使用 Python 3.11。多卡通信、图编译和策略任务效果留待后续任务验证。

## 参考来源

- [实验故障报告](https://bytedance.us.larkoffice.com/docx/U5dYdqVpnoJl3IxwGVSu15DusfP)，revision 3。
- [Torch Ada 使用手册](https://picoheart.feishu.cn/wiki/D3KAwCLIbiWFbAkDjwhcRSssndh)，revision 9858；
  环境准备、Debug；嵌入运行时表格 revision 439。
- [fullstack 手册](https://picoheart.feishu.cn/wiki/FWvUwlOFTiyndWkgfHscv48bnfd)，revision 11536；
  当前基线和 §2.2 容器部署。
- [Ada2S 规格与适配文档](https://bytedance.larkoffice.com/docx/DLhMd70P1ovDooxh1gocBbflnif)，
  revision 3115；§2 软件版本、§6 容器。
- [套餐适配跟踪](https://bytedance.larkoffice.com/wiki/GKkUwa7vvirYpFklcVncKFJtnJf)，revision 2509；
  已读取目录并检索目标 IP / M12 / Torch Ada，没有取得目标节点配套版本的独立证明。
- [官方模型训练 demo](https://picoheart.feishu.cn/wiki/GBArw5WvqiIGNfkdO6VcToXbn7c)，revision 473；
  使用 AdaAlign 的 M8 FSDP demo，与本仓库 Pi0.5 和 transformers 版本不同，不可直接混用环境。

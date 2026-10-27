# DreamZero (NVIDIA GEAR) — in-tree LeRobot policy

DreamZero is a Wan-video **World Action Model**: a causal DiT (`CausalWanModel`) that jointly
denoises video latents + action/state registers in one token sequence, blockwise-causal, and
autoregresses an action chunk with a KV cache at inference. This port wraps DreamZero's own native
PyTorch inference/training (NOT vLLM) behind the LeRobot `PreTrainedPolicy` interface.

Registered as `dreamzero`. Install extra: `pip install -e ".[dreamzero]"`.

## Status

| Piece | State |
|---|---|
| Model core (`wan/`: causal DiT, VAE, umt5/clip encoders, schedulers, `WANPolicyHead`) | ported — **249/254 functions byte-identical to upstream** |
| Config + policy wrapper (`select_action`/`predict_action_chunk` via `lazy_joint_video_action`) | ported (M1) |
| Processor numerics (oxe_droid): q99, view stitch, umt5, relative↔absolute | done + CPU-tested (M2) |
| Direct load of released NVIDIA checkpoints (`gear_checkpoint.py`) | done + CPU-tested (M2) |
| Offline open-loop eval | **runs end-to-end on `DreamZero-DROID` + `lerobot/droid_1.0.1`** (M2) |
| Continued SFT on DROID (`lerobot-train`) | **runs — 1-GPU and 8-GPU LoRA, and 8-GPU FSDP full fine-tune** (M3/M4) |
| Statistics for a new dataset (`scripts/compute_statistics.py`) | done + CPU-tested, cross-checked against the released quantiles (M5) |
| New embodiments (generic canvas + prompt + frame-rate stretch) | **8-GPU SO-101 fine-tune runs**; policy quality unevaluated — see "Training a new robot" |

**num_envs == 1** only (the KV cache has no batch semantics).

## Port fidelity

The model core was diff-verified against the upstream DreamZero source
(`github.com/RLinf/dreamzero`, `groot/vla/model/dreamzero/`) at function granularity: **249 of 254
functions are byte-identical**, with no upstream function silently dropped. The five that differ
are all deliberate and documented:

| Function | Why it differs |
|---|---|
| `WANPolicyHeadConfig.__init__` | plain dataclass instead of HF `PretrainedConfig` (no `super().__init__(**kwargs)`) |
| `WANPolicyHead.__init__` | hydra `instantiate` replaced by direct construction; `trt_engine`/`dit_step_mask` initialised here |
| `WANPolicyHead.post_initialize` | TensorRT engine loading raises instead of importing `groot.control.tensorrt_utils` |
| `flash_attention` (x2) | a different upstream revision; both dispatch to the same `flash_attn_varlen_func` with an equivalent `cu_seqlens` computation |

Notably `causal_wan_model.py` (the 2245-line joint video+action causal DiT, including the
blockwise-causal attention) is 35/35 identical, and `vae.py` is 56/56.

`DreamZeroPolicy.predict_action_chunk` calls `action_head.lazy_joint_video_action`, which is the
same method upstream's `VLA.lazy_joint_video_action_causal` wraps — the `_causal` variant only
threads a `latent_video` argument through, and callers pass `None`.

## Only the 14B backbone

`wan21_14b` (Wan2.1-I2V-14B DiT, 16-channel `WanVideoVAE`, 176x320 per view) is the only supported
variant, and a checkpoint whose DiT dims say otherwise is rejected by name at load time.

That is not a simplification for its own sake — it is the only backbone with released DreamZero
weights. `GEAR-Dreams/DreamZero-DROID` is 14B, and upstream's LIBERO configs also start from it
(`libero_sft_dreamzero_14b.yaml` sets `model_path: .../DreamZero-DROID`). The Wan2.2-TI2V-5B
variant exists upstream only as a **cold start** from base Wan components —
`libero_sft_dreamzero_5b.yaml` has `model_path: null` and pulls the DiT/VAE/T5 from
`Wan-AI/Wan2.2-TI2V-5B` and the CLIP image encoder from `Wan-AI/Wan2.1-I2V-14B-480P` — so there is
no published checkpoint to validate a 5B port against. Carrying an unverifiable second code path
is worse than not carrying it.

## Eval — no conversion step

A released checkpoint is loaded as-is. `gear_checkpoint.py` reads the geometry from the
checkpoint's own `config.json`, the state/action concat order from `experiment_cfg/conf.yaml`, and
the q01/q99 statistics from `experiment_cfg/metadata.json`.

`lerobot-eval` is closed-loop: it rolls the policy out in a gym environment and reports task
success. DreamZero's environment is NVIDIA's `sim-evals`, which requires IsaacSim and — by its own
design — runs the policy in a *separate process* behind a websocket, because IsaacSim cannot share
a Python environment with the policy stack. So there is no in-process `droid_sim` EnvConfig to
register; closed-loop DROID evaluation means standing up that two-process setup yourself.

Open-loop evaluation needs no simulator, and is what "does the port reproduce the reference
behaviour" actually calls for:

```bash
hf download GEAR-Dreams/DreamZero-DROID --local-dir /data/models/DreamZero-DROID \
    --exclude 'tensorrt/*'          # 19 GB of Blackwell-only assets, see below

uv run python examples/eval_open_loop.py \
    --policy.path=/data/models/DreamZero-DROID \
    --dataset.repo_id=lerobot/droid_1.0.1 \
    --dataset.root=/data/datasets/droid_1.0.1 \
    --episodes='[1,2,3,7,8,9]' \
    --max_frames_per_episode=96 \
    --device=cuda \
    --rename_map='{"observation.images.exterior_1_left":"observation.images.exterior_image_1_left",
                   "observation.images.exterior_2_left":"observation.images.exterior_image_2_left",
                   "observation.images.wrist_left":"observation.images.wrist_image_left"}'
```

`examples/eval_open_loop.py` is policy-agnostic — it goes through `make_policy` /
`make_pre_post_processors` like any other LeRobot entry point. Two things make a released NVIDIA
checkpoint work through it:

* `DreamZeroConfig.from_foreign_checkpoint` lets the script adopt a directory whose `config.json`
  is a HuggingFace `VLA` config rather than a draccus one.
* `--rename_map` maps the held-out dataset's camera keys onto the ones the checkpoint was trained
  on (`exterior_1_left` -> `exterior_image_1_left`). This is LeRobot's standard mechanism; the
  policy config keeps the names the model actually saw.

**The dataset must expose DROID's joint action space.** `lerobot/droid_100` does not — its
`observation.state`/`action` are cartesian (7-dim). `lerobot/droid_1.0.1` does: `observation.state`
and `action` are both `joint_position(7) + gripper_position(1)`, matching the checkpoint's concat
order exactly.

### Geometry, and why it is checked

Per view: center-crop 0.95 → resize to **176x320**. The three views stitch into a 2H x 2W canvas
(wrist across the whole top row, exterior_1 bottom-left, exterior_2 bottom-right) = **352x640**.
That canvas must tokenize to the DiT's `frame_seqlen`:

```
352/8/2 * 640/8/2 = 22 * 40 = 880 == frame_seqlen (wan21_14b)
```

`DreamZeroConfig.__post_init__` enforces this. It is worth enforcing because the wrong per-view
size still runs to completion — `frame_seqlen` only sizes the blockwise-causal attention mask, so a
mismatch silently mis-aligns every video block against the action registers.

## Interpreting the eval numbers

`offline_eval` reports four things, because the raw MSE on its own is misleading — the action
space is *absolute* joint positions, so most of the target is just the current state:

* `action_mse` — mean squared error against the recorded action.
* `hold_state_mse` — the same error for a policy that outputs the anchor state unchanged. The
  policy has to beat this to be doing anything at all.
* `delta_corr` — correlation between the predicted and the true *displacement* from the anchor
  state. This isolates the part the policy actually predicts; ~0 means no signal no matter how
  small the MSE looks.
* `delta_slope` — least-squares gain of predicted displacement over true displacement (reported
  for joints and gripper separately, since radians and a 0-1 opening are not comparable).
  Correlation is scale-free, so slope is what tells you whether the magnitude is calibrated.

Measured on `DreamZero-DROID` x `lerobot/droid_1.0.1`, 96 frames each, `n_action_steps=8`,
`num_dit_steps=8`:

| episode | action_mse | hold_state_mse | ratio | delta_corr | joint_slope |
|---|---|---|---|---|---|
| 1 | 0.00349 | 0.00367 | 0.95 | 0.450 | 0.450 |
| 2 | 0.02449 | 0.02883 | 0.85 | 0.515 | 0.356 |
| 3 | 0.00308 | 0.00749 | 0.41 | 0.757 | 0.669 |
| 7 | 0.01080 | 0.01117 | 0.97 | 0.749 | 1.290 |
| 8 | 0.00503 | 0.01211 | 0.42 | 0.730 | 0.492 |
| 9 | 0.00721 | 0.00747 | 0.97 | 0.467 | 0.523 |

All six beat the hold-state baseline, with displacement correlation 0.45-0.76.

**Use a long enough window.** At 64 frames the same episodes looked far worse (only 1 of 3 beat
the baseline). Every window starts at the episode's first frame, where a DROID robot is often
nearly stationary — and the checkpoint was trained on `droid_101_success_idlefiltered6`, an
idle-filtered subset, so near-static segments are out of distribution. Short windows are dominated
by exactly the regime the model never saw.

Inference is deterministic (the head fixes `seed = 1140`): re-running an episode with the same
config reproduces the metrics bit-for-bit, which is what makes A/B knob comparisons meaningful.

`num_dit_steps` 16 vs 8, same episodes: 16 is marginally *worse* (ep3 corr 0.546 vs 0.573, ep7
0.696 vs 0.733). The skip schedule is not costing accuracy here.

## The relative-action anchor

The model's action chunk is expressed *relative to the state at the frame it was predicted from*,
so every action in it must be decoded against that one anchor. `DreamZeroActionDecodeStep`
therefore **latches** the anchor when a chunk starts and holds it until the chunk is consumed,
rather than reading the pack step's latest cached state on each call.

That distinction is not cosmetic. The pack step caches the current state on every
`preprocessor()` call, which is right for the frame that triggered the prediction and wrong for
the frames that follow while the chunk is still being popped — decoding against those re-adds the
motion that already happened, with an error that grows across the chunk to roughly the magnitude
of the signal. Before the latch, per-step decoding scored *worse* than the hold-state baseline
(`action_mse` 0.00279 vs 0.00126).

Chunk length comes from `config.n_action_steps`, which is exactly what `select_action` queues per
prediction (`deque(maxlen=n_action_steps)`, refilled when empty); `reset()` drops the latch so a
new episode re-anchors immediately, and `PolicyProcessorPipeline.reset()` propagates it (the sync
and RTC rollout paths both call it).

Verified on DROID: per-frame `select_action` and a whole-chunk `predict_action_chunk` +
single postprocessor pass now agree to `0.000e+00` over 16 actions.


## Memory and speed

The released weights are bf16 (23B params over DiT 16.5B + umt5 5.7B + CLIP 0.6B + VAE 0.13B) and
`prepare_for_inference` keeps them there: ~46 GB, vs ~92 GB if left in the fp32 the modules are
constructed in.

All 16 diffusion steps evaluate the DiT. Upstream ships hand-tuned schedules that skip 8-11 of
them (reusing the previous prediction) behind a `NUM_DIT_STEPS` env var, defaulting to 8; this
port drops that knob. Skipping measured slightly *worse* on DROID rather than free, and an
approximation left on by default silently taints every number taken with it — vllm-omni's parity
test asserts on the log that upstream ran 16 steps for exactly this reason.

| | per policy query | GPU (single H200) |
|---|---|---|
| 16 steps (this port) | 9.9 s (8.8 s diffusion) | ~89 GB |
| 8 steps (upstream default) | 5.7 s (4.5 s diffusion) | ~53 GB |

### Loading

A released checkpoint loads in ~15 s (H200, 46 GB of bf16 shards). Two things get it there:

* The modules are built under `init_weights_on_device()` and the checkpoint is applied with
  `load_state_dict(assign=True)`, so the weights are materialised exactly once. Constructing 23B
  parameters for real first would allocate and initialise ~92 GB of fp32 host memory that the
  checkpoint immediately overwrites. Anything the checkpoint does not supply stays a meta tensor
  and would fail much later with an unrelated-looking error, so `_materialize_meta_tensors` names
  it instead.
* `skip_component_loading` is honoured for the text/image/VAE encoders, not just the DiT. Upstream
  fetches the base Wan weights for those unconditionally — 11 GB (umt5) + 4.5 GB (CLIP) + 485 MB
  (VAE) — which a full DreamZero checkpoint then overwrites wholesale.

The remaining ~14 s is 2146 individual tensor copies to the GPU (~3.2 GB/s). The shards are
mmap'd and the page cache holds them, so this is transfer overhead rather than disk I/O; batching
into pinned buffers would help if it ever matters.

### Single-frame, not autoregressive

`WANPolicyHead` supports AR rollout: the first call anchors the i2v conditioning on the original
image (`clip_feas`/`ys`) and builds the KV cache, and each later call advances 2 latent frames
while reusing both. `max_chunk_size=4` such blocks fill the 33-frame (9 latent frame) window, and
that shared image conditioning is what training looks like — a training sample *is* one 33-frame
window with one starting image. Upstream's `test_client_AR.py` drives it with 4 frames of history
per call (`RELATIVE_OFFSETS = [-23, -16, -8, 0]`, advancing by `action_horizon`); note the head
maps `T == 4` to `image = videos[:, :, -1:]`, so the *last* frame conditions, not the first.

This port uses the single-frame path instead, where `videos.shape[2] == 1` resets
`current_start_frame` on every call and each query is an independent i2v prediction. Measured on
episode 3 (2 sequences x 4 blocks, 192 scored actions):

| | action_mse | hold_state_mse | ratio | delta_corr |
|---|---|---|---|---|
| single-frame | 0.01252 | 0.02131 | 0.59 | 0.690 |
| AR (4-frame history) | 0.01352 | 0.02131 | 0.63 | 0.630 |

AR is not better, and it costs a history window plus cross-call session state (upstream's server
carries a `session_id` for precisely that). Single-frame also suits the gRPC `PolicyServer`, whose
requests carry one observation each, and re-anchors every chunk on the freshest observation rather
than letting the image conditioning age across a sequence. RLinf's inference is single-frame too.

Caveat on how strong that conclusion is: one episode, two sequences. AR *is* the regime the model
was trained in, so its not helping is mildly surprising and could still mean a protocol detail is
off. Worth revisiting for long-horizon closed-loop work, where video continuity may matter in a
way open-loop action scoring cannot see.

Do not conflate the two anchors involved:

| | granularity | shared across the 4 blocks |
|---|---|---|
| image conditioning (`clip_feas`/`ys`) | one per 33-frame window | yes (AR only) |
| relative-action state | one per block | no — each block subtracts its own first-row state |


## Attention: do not install `flash-attn` on Hopper

`flash-attn` is deliberately absent from the `dreamzero` extra. With it uninstalled,
`AttentionModule` asks for the `FA2` backend, `_gpu_supports_flash_attention()` reports False, and
the code falls through to `torch.nn.functional.scaled_dot_product_attention`.

That fallback is **not** a degraded path, for two separate reasons:

* **It is numerically equivalent here.** The DiT's blockwise-causal attention slices q/k/v into
  blocks in Python and calls `self.attn(q_block, k_context, v_context)` per block — causality comes
  from *which* k/v slice each q block sees, not from a kernel-side mask or varlen packing.
  `self.attn` is never passed `q_lens`/`k_lens`, so the fallback's "padding mask is disabled"
  branch never runs and `attn_mask=None, is_causal=False` over a dense slice is exactly what
  FlashAttention would compute. Only floating-point accumulation order differs (bf16 tolerance).
* **SDPA already dispatches to a FlashAttention kernel.** On an H200 the selected kernel is
  `cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16` — cuDNN's Hopper flash
  implementation. Measured at the production shape (1x40x7920x128, bf16, 50 iters):

  | backend | time |
  |---|---|
  | SDPA default (auto -> cuDNN) | 2.16 ms |
  | forced `CUDNN_ATTENTION` | 2.27 ms |
  | forced `FLASH_ATTENTION` (PyTorch's bundled kernel) | 3.67 ms |

  Dao-AILab's `flash-attn` belongs to the same family as that last row, so installing it makes this
  path **slower**, not faster.

Note also that `AttentionModule.__init__` sets `backend = "torch"` and then unconditionally
overwrites it with `"FA2"` in an `else` with no condition — so its `backend` constructor argument
is dead, and only the `ATTENTION_BACKEND` environment variable has any effect. That code is
byte-identical to upstream; it is inherited, not introduced here. To pin the backend explicitly,
use `ATTENTION_BACKEND=torch`.

## TensorRT is not ported, and the shipped engine is Blackwell-only

`GEAR-Dreams/DreamZero-DROID` ships ~19 GB of TensorRT assets (an NVFP4-quantized
`CausalWanModel.onnx` + `.onnx_data`, a prebuilt `WanModel_nvfp4.trt`, and the build log). None of
it is usable outside Blackwell:

* The engine was built on an **NVIDIA GB200, compute capability 10.0**, TensorRT 10.13.2
  (`WanModel_nvfp4_build.log`). TensorRT engines cannot be deserialized across compute
  capabilities, so an H100/H200 (SM 9.0) cannot load it.
* Rebuilding from the shipped ONNX does not help either: the graph carries `TRT_FP4DynamicQuantize`
  / `FP4E2M1` nodes, and Hopper has no FP4 tensor cores. Getting a Hopper engine means
  re-quantizing from the original weights to FP8.

(The `nvfp4` in the filename refers to the pre-quantized ONNX, not the build flags — trtexec was
invoked with `--fp8 --fp16 --bf16`.)

Upstream uses the engine only for diffusion steps that do *not* update the KV cache. Its build log
reports 38.3 ms mean GPU compute per DiT forward on GB200, against ~530 ms/step here in bf16 on an
H200 — an order of magnitude, though that conflates the hardware generation with the precision.

`LOAD_TRT_ENGINE` therefore raises rather than silently falling back, so a run cannot quietly
believe it is accelerated. **When downloading the checkpoint, `--exclude 'tensorrt/*'` skips 19 GB
of the 61 GB** unless you are on Blackwell.

## Training

`lerobot-train` fine-tunes from the released checkpoint. Note the invocation: `--policy.type`
plus `--policy.pretrained_path`, *not* `--policy.path` — the latter routes through
`PreTrainedConfig.from_pretrained`, which cannot parse a GEAR `config.json` (no `type` key). This
also mirrors how RLinf does it: the config comes from the framework's own config system and the
checkpoint supplies only weights and statistics.

```bash
lerobot-train \
  --policy.type=dreamzero \
  --policy.pretrained_path=/data/models/DreamZero-DROID \
  --policy.training_mode=lora \
  --policy.device=cuda --policy.push_to_hub=false \
  --rename_map='{"observation.images.exterior_1_left":"observation.images.exterior_image_1_left","observation.images.exterior_2_left":"observation.images.exterior_image_2_left","observation.images.wrist_left":"observation.images.wrist_image_left"}' \
  --dataset.repo_id=lerobot/droid_1.0.1 \
  --dataset.root=/data/datasets/droid_1.0.1 \
  --dataset.episodes='[0,1,2,3]' \
  --batch_size=1 --steps=2 --output_dir=outputs/train/dreamzero_smoke
```

`--rename_map` works here too (it is a `TrainPipelineConfig` field, and requires
`--policy.pretrained_path`), so a dataset whose cameras are named differently can be handled either
way — see "rename_map vs video_modality_keys" below. Restrict `--dataset.episodes`
unless you really have all of `droid_1.0.1` locally — otherwise LeRobot loads metadata for all
95,658 episodes.

**Always pass `--policy.training_mode` explicitly**, even though `lora` is the default. The two
modes differ by more than two orders of magnitude in what they train (0.47% vs 71.9% of the
parameters) and in what they write (208 MB vs 46 GB), so a command that leaves it implicit cannot
be read — or reproduced — without also knowing which version of the default was in effect. The
resolved mode is echoed at startup and is the authoritative record:

```
DreamZero training_mode=lora (lora_rank=4): 108.6 M trainable of 22943.3 M (0.47%)
```

Measured on one H200, batch 1:

| | |
|---|---|
| step time | 5.6 s (one steady-state sample from a 2-step run; the first step is ~90 s of cuDNN/lazy-init warmup) |
| GPU memory | 59.2 GB |
| trainable | 108.6 M of 22,943 M (0.47%) |
| checkpoint | 208 MB |
| training window | `images (1, 3, 33, 352, 640)`, `state (1, 4, 64)`, `action (1, 96, 32)` |

#### `rename_map` vs `video_modality_keys`

They solve different halves of the problem and neither replaces the other:

| | what it does |
|---|---|
| `--rename_map` | renames observation keys (`observation.images.wrist_left` -> `...wrist_image_left`). A dict; it carries no order — `RenameObservationsProcessorStep` iterates the *observation's* keys, not the map's. |
| `--policy.video_modality_keys` | picks **which** keys go on the canvas and **in what order**, i.e. which quadrant each camera lands in — and the generated prompt is built from the same list, so the two cannot disagree. Normally not passed: it is resolved from the checkpoint. |

Renaming alone therefore cannot say that the wrist camera belongs on DROID's top row. So
`video_modality_keys` is resolved from the checkpoint's own `experiment_cfg/conf.yaml`
(`video_concat_order`) when it is left empty — the same way GR00T resolves its identically-named
list from its checkpoint, and via the same `_ordered_image_keys` seam. A fine-tune therefore does
not normally pass it; use `--rename_map` when the dataset's camera names differ.

The remaining fallback, sorted camera keys, is a last resort. It gives DROID's order
alphabetically, which is luck rather than a guarantee. **The dataset's own feature order is not
usable**: `lerobot/droid_1.0.1` lists the cameras as `wrist_left, exterior_1_left, exterior_2_left`
— wrist *first*, where the checkpoint stitches it *last*. Deriving the order from dataset metadata,
as policies like pi0 do for `image_features`, would put the wrist camera in an exterior quadrant
while the prompt still announced it on the top row: same shapes, same token count, no error.

That difference is architectural, not incidental. pi0/pi05 encode each camera independently and
concatenate the resulting token blocks, so view order carries little meaning. DreamZero composes
one spatial canvas *and* names the quadrants in the prompt, so the camera-to-quadrant mapping is
part of what the weights learned — as it is for GR00T, which stacks views along a dedicated axis
and likewise takes its order from the checkpoint.

### `training_mode`

| mode | trainable | cost |
|---|---|---|
| **`lora`** (default) | 108.6 M of 22,943 M (0.47%) | one GPU at 42.5 GB, 0.2 GB checkpoints, 7.7 s/step |
| `full` | 16,484 M of 22,924 M (71.9%) | ~8 GPUs at 57 GB each, 66 GB checkpoints, 7.1 s/step. Upstream's and RLinf's published DROID recipe. |

**On a new embodiment, `full` is the one that works.** Fine-tuning the released DROID checkpoint
on 60 episodes of SO-101 cube pick-and-place and scoring 10 held-out episodes open-loop:

| | action MSE | vs. hold-state (126.7) | episodes won | delta corr |
|---|---|---|---|---|
| base checkpoint | 197.1 | worse | 0/10 | -0.088 |
| `lora`, rank 4 | 256.4 | worse | 0/10 | -0.239 |
| `lora`, rank 32 | 158.8 | worse | 3/10 | 0.320 |
| `lora`, rank 128 | 176.1 | worse | 1/10 | 0.410 |
| **`full`** | **79.7** | **37% better** | **10/10** | **0.633** |

Raising the rank moves the direction of the predicted motion monotonically the right way, but no
rank tested ever beat "output the current state unchanged" — the change a new robot needs is not
in a low-rank subspace. Note also that more steps did not help: 3000 steps (1.11 epoch) scored
*worse* than 1000 (0.38 epoch) on every metric, so the ceiling here is data, not optimisation.

That result is specific to a change of embodiment. `lora` on the *same* robot — upstream's own
use for it, e.g. their LIBERO configs starting from `DreamZero-DROID` — is untested here.

Neither mode touches the text/image/VAE encoders — `WANPolicyHead` freezes those itself, which is
why even `full` leaves 6,440 M parameters untouched.

#### Where the adapters go

`lora` trains two things: the **projectors** (89.4 M — `state_encoder`, `action_encoder`,
`action_decoder`, which live *inside* the DiT module) and **rank-4 adapters on 10 linear modules in
every one of the 40 DiT blocks** (19.2 M). The original weights are frozen; LoRA adds a parallel
path `W·x + (α/r)·B(A·x)`, and at the upstream defaults `α = r = 4` the scaling is exactly 1.

| module (x40 blocks) | shape | adapter params | what it feeds |
|---|---|---|---|
| `self_attn.{q,k,v,o}` | 5120x5120 | 4 x 1.638 M | attention across the video/action/state token sequence — the dynamics |
| `cross_attn.{q,k,v,o}` | 5120x5120 | 4 x 1.638 M | attention into the umt5 text embedding — language conditioning |
| `ffn.0` | 5120x13824 | 3.031 M | the block MLP |
| `ffn.2` | 13824x5120 | 3.031 M | |
| **total** | | **19.169 M** | |

Target modules are `q,k,v,o,ffn.0,ffn.2`, straight from upstream's `droid_training_lora.sh`. PEFT
matches by suffix, so `q,k,v,o` catches **both** the self- and cross-attention projections — 10
wrapped modules per block, not 6.

What is *not* wrapped, and is worth knowing before assuming full coverage:

* `cross_attn.{k_img,v_img}` — **2,098 M parameters** of image-conditioning projections, the path
  that reads the CLIP reference-image embedding. Upstream's default target list omits them; add
  `k_img,v_img` to `--policy.lora_target_modules` to include them (22.5 M adapters instead of 19.2 M).
* `time_embedding`, `time_projection` (157 M), `text_embedding`, `img_emb`, `head.head` — 240 M of
  linear layers outside the blocks.
* All norms, registers and positional embeddings (92.8 M) — not linear, so PEFT cannot wrap them.

**LoRA injection order.** PEFT renames every parameter it wraps
(`blocks.0.self_attn.q.weight` → `base_model.model.blocks.0.self_attn.q.base_layer.weight`), so
whether the adapters go on before or after the weights are loaded depends on which checkpoint is
being read: a released GEAR checkpoint has unwrapped keys and must be **loaded first**, while a
checkpoint written by a LoRA fine-tune already has wrapped keys and must be **wrapped first**.
`DreamZeroPolicy.from_pretrained` defers injection on the GEAR path and `_finalize_lora` completes
it after loading. Getting this backwards does not raise — `load_state_dict(strict=False)` reports
every DiT weight as missing and leaves the freshly-initialised ones in place.

Both directions are checked against a real 2-step LoRA run: wrapping at construction reproduces the
saved key set exactly (2946 of 2946, nothing missing or unexpected), while deferring on that same
path would report 2117 checkpoint keys as missing — and `strict=False` would let the run continue.

### `save_lora_only`

A LoRA run updates 108.6 M of 22,924 M parameters, so writing the other 99.5% into every checkpoint
costs ~46 GB apiece for nothing. With `save_lora_only` (default **on** for `training_mode=lora`,
matching upstream's `droid_training_lora.sh`) a checkpoint holds only what training changed:

| | full weights | `save_lora_only` |
|---|---|---|
| checkpoint size | 45.8 GB | **208 MB** |
| tensors | 2946 | 814 |
| self-contained | yes | no — needs the base checkpoint |

Like upstream, this saves every `requires_grad` parameter rather than just the `lora_*` ones: the
state/action projectors are trained too (89.4 M of the 108.6 M), and dropping them would silently
discard most of the fine-tune.

The checkpoint is **not self-contained**, so it records where its frozen weights live in a
`lora_adapter.json` manifest, and `from_pretrained` loads the base first and applies the delta on
top. If the base is gone the load raises rather than guessing. Set
`--policy.save_lora_only=false` for a self-contained checkpoint. `full` always writes complete
weights: it rewrites the whole DiT, so there is no frozen remainder worth deferring.

Verified end to end on a real run: all 814 saved tensors reload bit-identically, all 400 `lora_B`
tensors are non-zero (PEFT initialises them to zero, so this is what shows the fine-tune actually
landed rather than being re-initialised), and a frozen DiT weight matches the base checkpoint
exactly.

### Multi-GPU full fine-tune

`src/lerobot/policies/dreamzero/fsdp.yaml` is an accelerate FSDP config mirroring RLinf's
`droid_sft_dreamzero_14b.yaml`: FULL_SHARD, bf16 mixed precision, no grad scaler, per-block
wrapping of `CausalWanAttentionBlock,T5SelfAttention,AttentionBlock`.

```bash
accelerate launch --config_file src/lerobot/policies/dreamzero/fsdp.yaml \
  $(which lerobot-train) \
  --policy.type=dreamzero \
  --policy.pretrained_path=/data/models/DreamZero-DROID \
  --policy.training_mode=full \
  --policy.compute_dtype=float32 \
  --policy.device=cuda --policy.push_to_hub=false \
  --rename_map='{"observation.images.exterior_1_left":"observation.images.exterior_image_1_left","observation.images.exterior_2_left":"observation.images.exterior_image_2_left","observation.images.wrist_left":"observation.images.wrist_image_left"}' \
  --dataset.repo_id=lerobot/droid_1.0.1 \
  --dataset.root=/data/datasets/droid_1.0.1 \
  --dataset.episodes='[0,1,2,3]' \
  --policy.gradient_checkpointing=28 \
  --batch_size=1 --steps=... --output_dir=outputs/train/dreamzero_fsdp
```

28 rather than the default `True` because micro-batch 1 leaves ~78 GiB of a 141 GB card unused,
and rather than `False` because all 40 blocks of activations need ~217 GiB. Both numbers are
measured; see the dial below, including why an earlier version of this file said `False`.

Measured on 8xH200, `training_mode=full`, micro-batch 1:

| | |
|---|---|
| step time | 6.9 s steady state (first step ~95 s of warmup) |
| GPU memory | 57.0 GB per rank |
| trainable | 16,484 M |
| effective batch | 8 |
| gradient norm | 0.23-0.38 |
| model FLOPs | 1461.3 TFLOP per sample |
| **MFU** | **32.6%** of 650.2 TFLOP/s (sustained bf16) at the measured step |
| **HFU** | **42.8%** — the gap is recompute |

The 6.9 s was measured before the modulation change described under the dial below, which took
17.5 TFLOP per sample and ~1.2 GiB per block out of the step; expect ~6.84 s and ~32.9% MFU
now, and re-measure before quoting either.

For reference, RLinf reports 9.0 s/step on 8xH100 with plain FSDP2 and 6.7 s/step with their
compile/CUDA-graph stack, which this port does not carry — landing at 6.9 s on faster hardware
without those optimizations is the expected place to be.

### Reading those two numbers

```bash
python benchmarks/mfu.py dreamzero --training-mode=full --ranks 8 --step-s 6.9 --peak-tflops 650.2
```

MFU counts only the arithmetic the model's math requires; HFU counts what the card actually ran.
They differ here by **24% of the work executed**, which is exactly the DiT blocks being
recomputed in the backward pass — so the 10 points between them are the price of
`gradient_checkpointing`, not a property of the kernels. Chasing MFU up to HFU means finding
that memory somewhere else, which is what the dial below does.

### `gradient_checkpointing` is a dial

All-or-nothing is the wrong granularity for this model. All 40 blocks of activations need
~217 GiB at batch 1, so recompute cannot be turned off on any card that exists — but an 8xH200
full fine-tune sits at 57.0 GB of 141 GB, and **leaving most of that headroom unused costs a
quarter of every step**. So the field takes a count as well as a bool: `True` checkpoints all
40 blocks, `False` none, and an int checkpoints that many.

```bash
--policy.gradient_checkpointing=28     # 8xH200 at micro-batch 1: spends ~61 of ~78 GiB
```

Which blocks are chosen is immaterial — every block has the same activation footprint and the
same recompute cost, and the peak is reached at the end of the forward with every
un-checkpointed block live at once, so only *how many* matters.
`CausalWanModel.checkpointed_blocks` spreads them evenly for readability.

Pricing the ladder with the tool (`--checkpoint-blocks N`). **Only the first row is measured**;
the step times below it are that 6.9 s rescaled by hardware FLOPs, which assumes the step stays
compute-bound — so they are what to expect, not what was observed:

| checkpointed | recompute | step (predicted) | MFU | samples/s | extra peak | fits 78 GiB? |
|---|---|---|---|---|---|---|
| 40 (`True`) | 457.1 TFLOP | 6.84 s | 32.9% | 1.17 | — | yes |
| 32 | 365.7 | 6.51 s | 34.5% | 1.23 | 38.8 GiB | yes |
| **28** | 320.0 | 6.35 s | **35.4%** | **1.26** | 61.1 GiB | yes |
| 25 | 285.7 | 6.23 s | 36.1% | 1.28 | 77.7 GiB | at the line |
| 20 | 228.5 | 6.02 s | 37.3% | 1.33 | 105.5 GiB | **no** |
| 0 (`False`) | 0 | 5.21 s | 43.2% | 1.54 | 216.6 GiB | **no** |

The peak column is **measured**, which it was not before. A sweep on one 950PR card
(`training_mode=lora`, batch 1, `--log_freq=1`, six steps, perfectly flat from step two) reads:

| checkpointed | un-recomputed blocks | peak | step over previous |
|---|---|---|---|
| 40 | 0 | 56.68 GiB | — |
| 38 | 2 | 62.20 GiB | +5.52 |
| 36 | 4 | 73.31 GiB | +11.11 |
| 34 | 6 | 84.41 GiB | +11.10 |
| 32 | 8 | 95.52 GiB | +11.11 |

Three consecutive increments of 11.11 GiB over two blocks: **5.56 GiB per block left
unrecomputed**. The first pair costs half that because at `True` the peak is not at the end of
the forward — it is reached during recompute — so the first two blocks fit inside the slack
before the forward's live set becomes the peak. A clean crossover, not noise.

**An earlier version of this file said `False` was reachable and recommended it. That was
wrong, and the way it was wrong is worth keeping.** The figure it rested on, "40 blocks of
activations reach 121.7 GiB", was read off the allocator's message when a GC-off run OOMed —
*121.74 GiB already allocated* of a 123.16 GiB card. That is where the run ran out, not what it
needed. Both the pre-change run and the post-change one fail at ~122 GiB, because both want
more than the card has and both fill it before the next allocation fails; the number carries no
information about the requirement. Divided by 40 it gave 3.04 GiB per block, a figure that then
survived two rounds of argument about what to subtract from it. The measured answer is 5.56,
and all 40 blocks want ~217 GiB — over twice what the OOM point suggested.

So on 8xH200 at micro-batch 1, ~78 GiB of headroom buys 15 blocks (5.52 + 13 x 5.555 = 77.7),
i.e. `gradient_checkpointing=25` at the line and **28 as the value to start from** — it leaves
~17 GiB for fragmentation and FSDP's all-gather transients, and `max_memory_allocated` does not
count either. The dial is worth ~2.5 points of MFU and ~8% of throughput here, not the ten
points to HFU.

One reason not to carry 5.56 GiB/block straight to an H200: `rope_dtypes` returns complex64 on
NPU and **complex128 on CUDA**, and each block ropes a q and a k of shape [1, 15940, 40, 64] —
326 MB apiece in complex64, 653 MB in complex128. If those are retained, CUDA pays ~0.65 GiB
per block more, so expect ~6.2 GiB there and measure it. That also makes the rope dtype a
memory lever on CUDA, which it is not on NPU, against the accuracy it is there to buy.

On a 950PR, `True` remains the only value that fits under FSDP: a single card runs with a few
blocks unrecomputed (9.54 vs 9.89 s/step at 5 of them) but 8-card FSDP shards parameters and
gradients and not activations, so the per-card activation bill is unchanged by adding cards.

#### `EI0006` at the first barrier, on Ascend

An 8-rank run on a 950PR fails before the model exists, with a traceback that points at
`HcclAllreduce` and bottoms out in `accelerator.wait_for_everyone()`. Three things make that
hard to read, and it is worth knowing all three:

* HCCL errors surface asynchronously, so the operator named is usually the next synchronisation
  point rather than the one that failed.
* torch_npu implements `dist.barrier()` **as an all-reduce**, so the failure reads like a
  gradient reduction when it is a barrier.
* the real code is `EI0006`, an HCCL connect timeout, and it only appears in the log as a bare
  error code.

The first barrier is where HCCL actually builds the communicator, and with 8 processes each
importing torch_npu and initialising a device under a container CPU quota, the slowest ranks
arrive past `HCCL_CONNECT_TIMEOUT`'s 120 s default. Four ranks fit under it, which is why
smaller runs on the same box are fine. Raise it:

```bash
export HCCL_CONNECT_TIMEOUT=1800 HCCL_EXEC_TIMEOUT=1800
```

Nothing about this is specific to sequence parallelism — it reproduced identically with
`sp_degree=1`, which is how it was ruled out as a cause. When a parallel change and an
environment failure land in the same run, the control is what separates them.

### Ulysses sequence parallelism

The dial above tops out at 28 because activations do not shard. `gradient_checkpointing=False`
wants ~217 GiB and no card has it — H200 has 131, a GB200 179 — and FSDP shards parameters and
gradients, not activations, so adding cards does nothing. Sequence parallelism is what divides
that bill, and it is the only route to the ten points between MFU and HFU.

```bash
--policy.sp_degree=4     # with 8 ranks: 4-way Ulysses x 2-way data parallelism
```

**Ulysses rather than ring/context parallelism, because of what this model's attention is.**
The all-to-all moves the shard from tokens to heads *inside* the attention, so every rank then
holds **all** tokens for N/P heads — and the teacher-forcing pattern, 640 attention calls built
by slicing token ranges in Python rather than with a mask, is correct with no changes at all.
Ring parallelism keeps the shard on tokens, which would mean re-deriving the block-causal
pattern per shard across five functions, and recombining softmax across ranks. The cost of the
choice is that the degree is bounded by the head count; `wan/ulysses.py` has the details.

Measured on one 950PR, peak memory per card against the blocks left unrecomputed:

| checkpointed | degree 1 | degree 4 |
|---|---|---|
| 40 | 56.68 GiB | 47.16 GiB |
| 24 | — | 68.64 |
| 8 | — | 91.47 |
| **per block** | **5.56 GiB** | **1.385 GiB** |

5.56 / 4 = 1.39 against a measured 1.385, so the activation bill divides by the degree to
within half a percent — and the 5.56 that the whole dial rests on is now two independent slopes
rather than one derivation.

**Both divisibility constraints bind, and they do not agree.** The heads must divide, because
each rank runs `num_heads / degree` of them: 40 admits 1, 2, 4, 5, 8, 10, 20, 40. The sequence
must divide, because the all-to-all splits it evenly: L = 15,940 = 2² x 5 x 797 admits 1, 2, 4,
5, 10, 20. **8 is in one set and not the other**, so eight cards means degree 4 against 2-way
data parallelism — which is enough, since 217/4 = 54 GiB fits the ~78 GiB micro-batch 1 leaves
free. A degree of 8 would need the sequence padded, which is not implemented.

Three things have to be true alongside it, and **none of them raises when it is false** — the
forward is exact either way:

* **The loss is multiplied by the degree** (`DreamZeroPolicy.forward`). Every weight's gradient
  is a partial sum over the rank's own tokens and the partials add up to the whole; FSDP
  averages across ranks rather than summing, which leaves everything short by exactly the
  degree. The symptom of dropping it is a run that converges as if the learning rate were
  `degree` times smaller. **It also makes the logged loss `degree` times larger than a
  degree-1 run's** — divide before comparing curves. Correcting the gradients after the
  backward instead would keep the number honest and cost a pass over all 16.5 B gradients each
  step, which is not worth a log line.
* **The dataloader shards over the data-parallel axis only.** The ranks of one Ulysses group
  hold shards of the *same sample*, so `accelerator.prepare`'s `world_size`-way sharding would
  hand each a different episode and then exchange shards between them.
  `PreTrainedPolicy.replicated_data_ranks()` is the hook; the training script shards
  `world_size / degree` ways.
* **The random draws are broadcast within the group** (`ulysses.sync_in_group`). Equal seeds
  are not enough: the states start equal and drift, because a DataLoader's worker seeds come
  out of the main process's generator and the ranks do not consume the same amount of it.
  Drifting noise has each rank denoising its shard towards a different target.

That the head runs **on the shard**, with only its 64-wide output gathered rather than its
5120-wide input, is what makes the first of those a single uniform factor: run it redundantly
on a gathered input instead and its weights get a whole gradient where every other weight has a
partial one, and no one scale factor corrects both.

Verified: a whole `_forward_train` at degree 2 against one rank, prediction **bitwise**
identical and the summed per-rank weight gradients within a bf16 ulp (`LEROBOT_TEST_SP=1`), and
the all-to-alls over hccl on four 950PR cards as well as gloo. Not yet verified: a long enough
run to compare loss curves against a degree-1 control, which is the only thing that can catch
the three items above.

#### Why the default stays `True`, and why batch is not the other answer

Activations are per sample, so the dial and the micro-batch compete for the same ~78 GiB. At
5.56 GiB a block the default has to stay `True`: `batch_size` defaults to 8, which at any
setting below 40 is an immediate OOM, and a 950PR has no headroom at batch 1 at all. The recipe
that can afford to open the dial says so on its own command line.

The competition is not a throughput trade, though, because model FLOPs per sample do not depend
on the batch: step time grows with N and `samples/s = 8N / step(N)` cancels it. What the two
knobs actually trade is throughput against *effective* batch:

| | checkpointed | micro-batch | samples/s | effective batch |
|---|---|---|---|---|
| **the recipe above** | **28** | **1** | **1.26** | **8** |
| | 36 | 2 | 1.20 | 16 |
| | 40 | 6 | 1.17 | 48 |

Raising the micro-batch at a *fixed* dial setting would be free, since samples/s does not depend
on N — the cost of a larger effective batch is entirely in having to turn the dial up to make
room, and at 5.56 GiB a block that is expensive: a second sample needs another ~57 GiB of the
~78, which is most of the dial's range. There is no gradient accumulation to get it cheaper:
`lerobot_train` steps the optimizer every micro-batch and never calls `accelerator.accumulate`,
so accelerate's `gradient_accumulation_steps` would only rescale the loss. Upstream's and
RLinf's published DROID recipe is 8 ranks at micro-batch 1, effective batch 8, which is the
first row.

Where the per-sample FLOPs go, and what each part is billed:

| part | TFLOP | multiplier | why |
|---|---|---|---|
| DiT blocks x40 | 1152.9 | 3x | fwd + dgrad + wgrad; all 16.5 B parameters train |
| DiT attention (640 calls) | 218.3 | 3x | 14 self-attention calls per layer + both halves of i2v cross-attention |
| VAE encode (tiled) | 84.6 | 1x | frozen, and nothing precedes it, so no gradient reaches it |
| DiT embed/head/projectors | 0.4 | 3x | patch embedding, time embedding/projection, head |
| text encoder (umt5) | 4.7 | 1x | frozen; 512 tokens is cheap even at 5.7 B parameters |
| image encoder (CLIP) | 0.3 | 1x | frozen; one 224x224 frame |

That 0.4 is the whole point of the modulation change seen from the FLOPs side: the time
embedding and projection used to run on 15,940 rows instead of 118, which was 17.5 of the
17.9 TFLOP the row used to read — 1.2% of the step, spent computing 135 copies of each row.

Two things that table makes visible. **`tiled=True` costs 47.6 TFLOP per sample** — 84.6
against the 37.0 `--untiled-vae` prices the same encode at, about 3% of the step — because the
352x640 canvas is cut into **741 tiles of 34x34** at the default geometry, each a full pass
through the VAE encoder. That is not 47.6 TFLOP for nothing, though, and an earlier version of
this file said so wrongly: the tiles overlap and `build_mask` feathers them together, so the
two paths give different latents. Measured at the real geometry, 67.8% of latent elements
differ, by 0.18% of full scale on average and 6.0% at worst. **`tiled=False` is a change to
what the DiT trains on, not a free switch** — score it against the released checkpoint first.
What it is worth in time is also unmeasured on a GPU; `benchmarks/vae_tiling.py` settles both
on the card, and note that on CPU the two paths come out the same speed despite the 3.8x pixel
area, so the FLOPs ratio is not a time estimate.

And **the attention is 640 separate kernel launches per
forward**, because the teacher-forcing pattern is built by slicing q and k/v in Python rather than
with a mask. It was 800 until the 160 single-token state self-attention calls came out: at
`num_state_per_block=1` each one is softmax over a single key, which is exactly 1, so the call
returned v. That is why this row's TFLOP did not move — those launches were ~20 kFLOP apiece,
and what they cost was launch overhead, an `empty_like` and four slice-assigns per layer whose
backward is a chain of `CopySlices`.

That is also why DreamZero's `estimate_flops` works differently from pi05's and GR00T's: it runs
the training forward on the meta device with hooks on every Linear/Conv and stubs over both
attention entry points, and adds up what was called. Nothing is allocated and nothing is computed
— meta tensors only propagate shapes — but a hand-written formula would have to restate five
block-slicing functions and the convolutional VAE's per-layer resolutions, and would drift from
them silently.

**`--policy.compute_dtype=float32` is required for a full fine-tune**, and pairs with the
`mixed_precision: bf16` in the accelerate config: fp32 master weights, bf16 forward/backward and
gradient reduction. This is what RLinf does (`actor.model.precision: fp32` plus
`mixed_precision.param_dtype: bf16`) and, less visibly, what upstream does too — its DeepSpeed
ZeRO-2 config never says "fp32" because the fp32 master copies live in the partitioned optimizer
state by default.

bf16 master weights are not a safe substitute at the reference lr. Measured on the released
checkpoint's own DiT weights (`|w|` median 1.2e-2), an Adam step of ~lr lands as follows:

| lr | fraction of updates that change the weight in bf16 |
|---|---|
| 1e-5 (the reference lr) | 16% |
| 1e-4 | 98% |
| 1e-3 | 100% |

so at 1e-5 most of the update rounds away. It becomes viable if the learning rate is raised, but
that discards the upstream-validated hyperparameters, and on 143 GB cards the memory it would save
buys nothing.


## Training a new robot

### The generic canvas

`oxe_droid` keeps its bespoke layout — wrist stretched across the top row, two exteriors below.
Every other embodiment goes through the layout upstream uses for agibot / yam / xdof: a 2x2 grid
filled top-left, bottom-left, top-right, with any spare quadrant left black.

```
[view 0 | view 2]      SO-101/SO-100 with two cameras:   [front | black]
[view 1 | black  ]                                       [wrist | black]
```

The canvas stays 352x640 whatever the view count, because `frame_seqlen` pins it (see "Geometry"
above) — fewer cameras means black padding, not a smaller canvas. The generated prompt names each
quadrant, since that is the only thing telling the model which camera it is looking at:

> A multi-view video shows that a robot pick up the red cube The video is split into four views:
> The top-left view shows the camera view from the front, the bottom-left view shows the camera
> view from the wrist, and the top-right and bottom-right views are black screens (inactive
> views). The robot pick up the red cube

Names come from `video_modality_keys` (key suffix) or, better, from `view_descriptions`.

### Frame rate

The window contract counts **steps, not seconds**: 96 actions and 33 frames at stride 3. DreamZero
trained at DROID's 15 fps, so those 96 steps are 6.4 s. A robot recorded at 30 fps would hand the
model a 3.2 s window of a dynamics model that learned 6.4 s ones. Set `--policy.source_fps` and the
delta indices stretch by `source_fps / 15`:

| `source_fps` | multiplier | observation rows | actions | raw span | wall clock |
|---|---|---|---|---|---|
| 15 (default) | 1 | 33 @ stride 3 | 96 @ stride 1 | 96 | 6.4 s |
| 30 | 2 | 33 @ stride 6 | 96 @ stride 2 | 192 | 6.4 s |
| 60 | 4 | 33 @ stride 12 | 96 @ stride 4 | 384 | 6.4 s |

Striding the actions is sound because they are absolute position targets, not increments —
dropping intermediate ones changes the control rate, not the meaning. A frame rate *below* 15 is
rejected rather than stretched, since no stride makes a short window longer.

### Worked example: SO-100 / SO-101

Six motors — `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll` plus `gripper` —
so the layout is `joint_position:5, gripper_position:1`. Nothing model-side changes: 6 dims pad
into `max_state_dim` 64 and `max_action_dim` 32. The arm joints are normalized units and the
gripper is 0-100 rather than DROID's radians and 0-1, which the q99 normalization absorbs as long
as the statistics come from the SO-100 data:

```bash
python -m lerobot.policies.dreamzero.scripts.compute_statistics \
  --repo_id=<your/so100_dataset> --output=<checkpoint dir> \
  --embodiment_tag=so100 \
  --state_layout='joint_position:5,gripper_position:1' \
  --action_layout='joint_position:5,gripper_position:1'

accelerate launch --config_file src/lerobot/policies/dreamzero/fsdp.yaml $(which lerobot-train) \
  --policy.type=dreamzero --policy.pretrained_path=<checkpoint dir> \
  --policy.embodiment_tag=so100 --policy.source_fps=30 \
  --policy.state_modality_keys='[joint_position,gripper_position]' \
  --policy.action_modality_keys='[joint_position,gripper_position]' \
  --policy.video_modality_keys='[observation.images.front,observation.images.wrist]' \
  --policy.view_descriptions='[front camera,robot wrist]' \
  --policy.training_mode=full --policy.compute_dtype=float32 \
  --dataset.repo_id=<your/so100_dataset> ...
```

**What this does not buy you.** The per-embodiment projector that would normally absorb a
morphology change is bypassed — the DiT hardcodes `embodiment_id = 0` and builds its encoders with
`max_num_embodiments = 1` (see the last bullet below). So going from a 7-DOF Franka in radians to a
5-DOF servo arm in normalized units has *no* dedicated adapter: all of it has to land in the shared
weights. Expect this to need a real fine-tune, not few-shot transfer. The pipeline is verified
end to end on a synthetic SO-100 window (33 frames, 352x640 canvas, 4 block anchors, 96 actions);
**no SO-100 fine-tune has been run**, so there is no claim here about resulting policy quality.

### Statistics for a new dataset

DreamZero normalizes with q01/q99 quantiles rather than mean/std, and for `relative_action_keys`
those quantiles describe the **per-block displacement**, not the raw action. Recomputing them for a
new dataset is what `compute_statistics` is for:

```bash
python -m lerobot.policies.dreamzero.scripts.compute_statistics \
  --repo_id=<your/dataset> --root=<path> --output=<checkpoint dir>
```

It writes the `statistics.json` that `from_pretrained` reads, so dropping it into a checkpoint
directory is enough to fine-tune against it. It reads the state/action columns straight off the
dataset table (no video decoding) and routes the displacement computation through the processor's
own `_encode_relative_actions`, so the statistics cannot end up describing a different convention
from the one training applies.

Getting the convention wrong is the failure this guards against, and it is silent. Recomputing
DROID's own action quantiles from 16 episodes and comparing the resulting range against the
released checkpoint's:

| convention | mean relative error vs. the shipped quantiles |
|---|---|
| absolute actions | 91.2% |
| per-step delta (action minus the *current* state) | 69.8% |
| **per-block delta (what this generates)** | **12.2%** |

The residual 12.2% is sample size, not convention: the same comparison run over more episodes
converges (18.3% at 10 episodes, 12.2% at 16, 6.3% at 50, 5.2% at 200). The wrong conventions do
not converge anywhere near — they are off by a factor, and a policy trained against them would
still run, predicting displacements scaled by roughly that factor.

### What a genuinely new embodiment still needs

The reference implementation to follow is **RLinf** (`examples/sft/`, `rlinf/data/datasets/dreamzero/`),
which does DreamZero SFT on DROID / LIBERO / Franka. What it establishes:

* **The sampling contract is not the delta-index list** *(ported)*. Upstream's `oxe_droid` conf
  declares `video delta_indices 0..24` while the model config says `num_frames: 33`; RLinf reads
  neither. It samples `max_chunk_size` macro-anchors spaced `action_horizon` apart, takes 8 video
  frames per anchor at stride 3, and appends one boundary frame — so the video length is *derived*
  as `8 * max_chunk_size + 1 = 33`, actions are `action_horizon * max_chunk_size = 96`, and states
  are one per anchor. Partial windows are rejected and resampled, never padded. LeRobot's native
  delta indices express exactly this window support, so no bespoke sampler was needed: see the
  `observation_delta_indices` / `action_delta_indices` / `drop_n_last_frames` properties on the
  config, and `_assert_window_not_padded` for the "never padded" half.
* **Relative actions are per macro chunk** *(ported)*, subtracting the state at the *first* row of
  each block, and applied *before* q99 normalization. Inference latches one anchor per chunk in
  `DreamZeroActionDecodeStep` rather than re-reading the current state per step, which is the same
  thing for the single chunk inference predicts at a time.
* **Full fine-tune, not LoRA** *(both supported here; `full` is the reference recipe)*
  (`train_architecture: full`, `tune_projector`/`tune_diffusion_model`
  true, gradient checkpointing on). FSDP2 `full_shard` with bf16 mixed precision, wrapping
  `CausalWanModel` as a unit; 8xH100 with `micro_batch_size: 1` for the 14B.
* **A new embodiment needs no model-side dimension change** as long as state <= `max_state_dim` (64)
  and action <= `max_action_dim` (32) — everything is zero-padded to those widths and sliced back
  using the per-key widths from the statistics. What a new robot does need: a transform module
  (camera keys, stitch layout, prompt text matching that layout, q99 normalization), an embodiment
  tag + projector index, and generated q01/q99 statistics.
* **The projector index is unused at run time.** The DiT's forward hardcodes `embodiment_id = 0`
  before the action/state encoders (`causal_wan_model.py:1773`; RLinf's patched train forward does
  the same) and the encoders are built with `max_num_embodiments = 1`, so the per-embodiment
  projector is bypassed. Nor does the index select a loss mask: `action_loss_embodiment_ids`
  appears only in upstream's YAML and is read by no code — the action loss is masked by
  `has_real_action` and `action_mask` alone (`action_head.py:829-830`). Registering a tag is
  bookkeeping, not behaviour.

## Validated locally vs on GPU

CPU tests lock the deterministic parts:

| file | covers |
|---|---|
| `test_dreamzero_processor.py` | q99 round-trip, the DROID stitch geometry (including that the wrist view is *stretched* 2x, not tiled), crop/resize, the relative->absolute decode, the training-pack <-> decode round-trip, per-chunk anchor latching |
| `test_dreamzero_gear_checkpoint.py` | checkpoint discrimination by content, geometry + concat-order derivation, variant inference from the DiT dims, statistics slicing, the `statistics.json` save/reload round-trip |
| `test_dreamzero_config.py` | which parameters each `training_mode` trains, LoRA injection ordering, the optimizer preset, and the training-window contract |
| `test_dreamzero_statistics.py` | that the generated action quantiles describe displacement rather than absolute position, and leave non-relative keys alone |
| `test_dreamzero_embodiments.py` | the generic 2x2 stitch (slot order, black quadrants), the generated prompt matching the canvas (and refusing to guess or misdescribe it), the frame-rate stretch |

GPU work is exercised by the offline eval above, the single-GPU and 8-GPU training runs, and the
checkpoint save/reload check. Numerics parity against the upstream server is still open — the
function-level byte-diff above is stronger evidence for the model core than a tolerance check would
be, but it says nothing about the surrounding pipeline.

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
  --batch_size=1 --steps=... --output_dir=outputs/train/dreamzero_fsdp
```

Measured on 8xH200, `training_mode=full`, micro-batch 1:

| | |
|---|---|
| step time | 6.9 s steady state (first step ~95 s of warmup) |
| GPU memory | 57.0 GB per rank |
| trainable | 16,484 M |
| effective batch | 8 |
| gradient norm | 0.23-0.38 |

For reference, RLinf reports 9.0 s/step on 8xH100 with plain FSDP2 and 6.7 s/step with their
compile/CUDA-graph stack, which this port does not carry — landing at 6.9 s on faster hardware
without those optimizations is the expected place to be.

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

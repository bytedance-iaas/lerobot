# FSDP / fp8 / torch.compile — when and how

Read this only when someone is considering FSDP, fp8, or torch.compile. The three verdicts that
matter are already in SKILL.md; this file is the reasoning, the measurements, and the how-to.

- **torch.compile is OFF by default** (it costs a multi-minute first-step warm-up). Whether it
  speeds a given run up has not been established here — don't claim either way without measuring
  that setup. Opt in with **`--compile`** for a VLA policy that supports it
  (pi0/pi05/pi0_fast/smolvla/diffusion; skipped with a note for ACT etc.), which adds
  `--policy.compile_model=true --policy.compile_mode=reduce-overhead`. **Default mode is
  `reduce-overhead`, NOT the policies' own max-autotune default**: on a 4B VLA max-autotune's
  kernel search warms up 10+ min (and uses more GPU memory), while reduce-overhead compiles in
  ~a minute. Use `--compile-mode max-autotune` on a long run. **preflight strips compile** so the
  2-step smoke isn't swamped by warm-up.
- **FSDP and fp8 do not currently work together — pick one.** A pi05 run with both reported a
  Linear shape mismatch (7744 vs 524288) at model build. **The mechanism is NOT established**, so
  do not repeat a cause for it; in particular `--fp8_enable_fsdp_float8_all_gather` is NOT it —
  that flag lives on accelerate's `AORecipeKwargs` (**torchao**, FSDP2-only) and never touches
  TE's state, and torchao is not even installed in this image (fp8 here is TE-only). The one
  solid structural fact: pi05's fp8 replaces each of the 18 Gemma-2B VLM MLPs with a
  `te.LayerNormMLP` whose weights and fp8 `_extra_state` belong together, so any wrap policy that
  can split a module mid-way (notably `SIZE_BASED_WRAP`) is the wrong tool — use
  `TRANSFORMER_BASED_WRAP`. Until someone reproduces it with a traceback: **run FSDP without
  `--float8`, or fp8 without FSDP.**
- **pi0/pi05 CANNOT be trained with FSDP — use DDP.** Every wrap policy fails the same way, so
  don't spend a day re-deriving it. `compute_layer_complete` never calls the decoder layer's
  `forward` — it reaches into `layer.input_layernorm` / `layer.self_attn.q_proj` directly,
  because the VLM and action-expert towers interleave through one shared attention. FSDP
  unshards parameters in that forward hook, so the parameters stay sharded and you get a size-0
  tensor. (Not the adarms branches and not `find_unused_parameters` — fixing those changes
  nothing.) It is also not worth fixing: pi05 is 3B and fits, and at b12 the 83.5 GB is mostly
  ACTIVATIONS, which FSDP does not shard.
  **Scale it with DDP + a bigger per-rank batch** — measured 2-node b8→b12: 10.5 → 13.3
  samples/s (+27%), no code change.
- **How to train with FSDP:** lerobot supports it through accelerate and the repo documents it —
  **`docs/source/multi_gpu_training.mdx` → "Training Large Models with FSDP"** is the reference
  (launch command, minimal `fsdp.yaml`, and the checkpoint/resume semantics). Do not re-derive it.
  What that page does not say, and matters here:
  - **Use FSDP1** (`fsdp_version: 1`). It is what the doc targets, what accelerate 1.13 still
    defaults to, and the only version with a run that worked here. FSDP2 exists (torch 2.11 has
    `fully_shard`) but nothing in this repo has been proven on it.
  - **Pick the sharding strategy by whether the PARAMETERS fit:**
    - **`SHARD_GRAD_OP`** (ZeRO-2) — shards gradients + optimizer state. **Enough for pi05**
      (3B). One all-gather per unit per step instead of two, and each TE module's weights stay
      whole through forward/backward.
    - **`FULL_SHARD`** (ZeRO-3) — also shards parameters. For models whose weights alone do not
      fit; `src/lerobot/policies/dreamzero/fsdp.yaml` uses it for DreamZero (16.5B).
    Note it is **not** textbook ZeRO-2: parameters are still sharded at rest and merely stay
    gathered between forward and backward, so it costs less memory than DeepSpeed ZeRO-2 would.
    ⚠️ `sharding_strategy` is the **FSDP1 spelling**. FSDP2 spells the same choice
    `reshard_after_forward` (`false` = ZeRO-2, `true` = ZeRO-3) and only *warns* about the old
    key — so a `SHARD_GRAD_OP` line carried into an FSDP2 config silently does nothing and you
    get ZeRO-3. FSDP2 also rejects `fsdp_backward_prefetch`, which `dreamzero/fsdp.yaml` sets.
  - **`fsdp_use_orig_params: true` is REQUIRED** (lerobot builds the optimizer from
    `get_optim_params()` before `accelerator.prepare()`, so the parameter objects must survive).
  - **`fsdp_transformer_layer_cls_to_wrap` is per-model, and for pi0/pi05 it is
    `_PiGemmaDecoderLayerBase` — NOT `GemmaDecoderLayer`.** lerobot subclasses the block
    (`pi_gemma.py`), and both stacks (`paligemma.model.language_model.layers` and
    `gemma_expert.model.layers`) are built from `PiGemmaModel`, so one entry covers both. The
    class is defined INSIDE a factory function, so it is not importable and `dir(module)` will
    not show it — accelerate matches on `type(m).__name__`, so the string still works. Confirm
    against a live model rather than trusting any name (including this one):
    `type(policy.model.paligemma_with_expert.paligemma.model.language_model.layers[0]).__name__`.
    A working reference config for a different policy lives at
    `src/lerobot/policies/dreamzero/fsdp.yaml`.
  - **`--policy.dtype=bfloat16` breaks FSDP1 wrapping: "Must flatten tensors with uniform
    dtype".** `to_bfloat16_for_selected_params` casts the model to bf16 and then puts a few
    names back to fp32 — and two of them, `input_layernorm` and `post_attention_layernorm`, sit
    INSIDE each decoder layer. So the wrap unit holds bf16 attention/MLP weights next to fp32
    norms, which FlatParameter refuses. `use_orig_params: true` does not help; the flatten
    happens regardless. **Drop `--policy.dtype=bfloat16`** — the `float32` branch returns before
    the fp32 override list is applied, so every parameter is uniformly fp32 — and let
    `mixed_precision: bf16` in the accelerate config provide bf16 compute. That is the normal
    FSDP arrangement anyway (fp32 master weights, bf16 compute).
  - Resume works and can change world size, but lerobot loads the FSDP optimizer state AFTER
    `prepare()` (`load_fsdp_optimizer_state`) — so a resume that skips that path silently starts
    with a fresh optimizer.
- **fp8 training — NEVER enable it silently; ASK THE USER first.** fp8 is a *numerics* change,
  not a free speedup, and it has a downstream consequence that has already bitten us: the saved
  `config.json` carries `vlm_mlp_fp8_*` fields, so the checkpoint **fails to load on a lerobot
  that lacks them** ("The fields vlm_mlp_fp8_… are not valid for PI05Config") — e.g. an eval or
  PolicyServer box on upstream/older lerobot. Present the trade-off (memory saved vs. numerics
  change + checkpoint portability + needs the TE-enabled image on every machine that will load
  it) and use the user's answer. Default to **off** unless they say otherwise. Details:
- **fp8 training (pi0/pi05 via TransformerEngine):** add **`--float8`** to `plan_training.py` for
  a **pi0/pi05** policy on a **Hopper/Ada GPU (H20/H100, sm_89/90+)** — it appends
  `--policy.vlm_mlp_fp8_enable=true --policy.dtype=bfloat16 --policy.vlm_mlp_fp8_recipe_kind=delayed_scaling`,
  fusing each VLM MLP into a fp8 `te.LayerNormMLP`. **Errors for non-pi0/pi05** policies and needs
  the **TE-enabled lerobot image**. `--float8-recipe delayed_scaling|float8_block_scaling`. NOT on
  older GPUs (TE errors at runtime), so only pass it when `check_hardware` reports an H20/Hopper.
  preflight inherits it from the plan (`--session`). fp8 is mainly a **memory** lever — re-benchmark
  before assuming a speedup. See `references/policy_selection.md`.

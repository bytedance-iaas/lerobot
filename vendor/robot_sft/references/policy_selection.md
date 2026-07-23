# Policy selection for lerobot SFT

Quick decision tree for picking a policy when training on robot demonstration data.

## Decision tree

```
User wants to train on a robot dataset
├── Has working HF token + gated repo access?
│   ├── Yes → pi05 (gemma_300m for ≤24GB GPU, gemma_2b for ≥40GB)
│   └── No → from-scratch policy (below)
│
├── Dataset size?
│   ├── ≤200 episodes → ACT (proven, fast, no gated deps)
│   ├── 200-1000 episodes → ACT or Diffusion Policy
│   └── >1000 episodes → ACT still works; Diffusion/VQ-BeT may generalize better
│
├── Task complexity?
│   ├── Single-mode (simple pick-place) → ACT or Gaussian Actor
│   ├── Multi-modal (multiple strategies) → Diffusion Policy or VQ-BeT
│   └── Long-horizon → TD-MPC (model-based, needs more compute)
│
└── GPU memory?
    ├── <12 GB → ACT (batch=8-16), Gaussian Actor
    ├── 12-24 GB → ACT, Diffusion, VQ-BeT (batch=16-64)
    └── ≥40 GB → pi05 gemma_2b, SmolVLA, full VLA policies
```

## Policy reference card

### From-scratch (no gated model dependencies)

| Policy | `--policy.type` | GPU fit | Typical batch | Steps/epoch (70k frames) | Checkpoint size |
|--------|----------------|---------|---------------|--------------------------|-----------------|
| ACT | `act` | 8-24 GB | 16-64 | ~1.1k-4.4k | ~0.5-2 GB |
| Diffusion Policy | `diffusion` | 12-24 GB | 32-128 | ~550-2.2k | ~0.3-1 GB |
| VQ-BeT | `vqbet` | 12-24 GB | 32-64 | ~1.1k-2.2k | ~0.5-1 GB |
| TD-MPC | `tdmpc` | 8-16 GB | 16-32 | ~2.2k-4.4k | ~1-3 GB |
| Gaussian Actor | `gaussian_actor` | 4-12 GB | 32-128 | ~550-2.2k | ~0.1-0.5 GB |
| Multi-Task DiT | `multi_task_dit` | 12-24 GB | 16-32 | ~2.2k-4.4k | ~1-2 GB |

### VLA (need HF token + gated license)

| Policy | `--policy.type` | Gated dep | GPU fit | Batch |
|--------|----------------|-----------|---------|-------|
| pi05 (300m) | `pi05` + `gemma_300m` | PaliGemma tokenizer | ≥12 GB | 4-8 |
| pi05 (2b) | `pi05` + `gemma_2b` | PaliGemma tokenizer | ≥40 GB | 1-4 |
| pi0 | `pi0` | PaliGemma backbone | ≥40 GB | 1-2 |
| pi0_fast | `pi0_fast` | PaliGemma backbone | ≥40 GB | 1-2 |
| SmolVLA | `smolvla` | SmolVLM base | ≥24 GB | 1-4 |

## Recommendation by dataset scale

| Episodes | Frames | Recommended | Fallback |
|----------|--------|-------------|----------|
| 10-50 | <50k | **ACT** (batch=8-16, 8 epochs) | Gaussian Actor |
| 50-200 | 50k-200k | **ACT** (batch=16-32, 5-8 epochs) | Diffusion Policy |
| 200-1k | 200k-1M | ACT or Diffusion (batch=32-64, 3-5 epochs) | pi05 (300m) |
| 1k+ | >1M | pi05/VLA or ACT (batch=64+, 2-3 epochs) | Diffusion Policy |

## pi05-specific notes

- `gemma_300m` variant: ~8-10 GB VRAM, batch=4-8 on A30 24GB
- `gemma_2b` variant: ~23 GB VRAM at batch=1, OOM on 24GB GPUs
- Always run preflight to verify memory before launching
- PaliGemma tokenizer (`google/paligemma-3b-pt-224`) is gated — must accept license on HF
- `hf auth login --token` has shell glob issues with `***` — use `export HF_TOKEN=...` instead

### fp8 (float8) training — pi0/pi05 via TransformerEngine

fp8 uses **NVIDIA TransformerEngine**: each VLM (PaliGemma) layer's
`post_attention_layernorm` + gate/up/down MLP is fused into one `te.LayerNormMLP`
(RMSNorm + FC1 geglu + FC2 in fp8). **pi0/pi05 only** (the only policies with the
config fields); the action expert stays bf16. Enable via `plan_training.py --float8`,
which appends:

```
--policy.vlm_mlp_fp8_enable=true --policy.dtype=bfloat16 \
--policy.vlm_mlp_fp8_recipe_kind=delayed_scaling
```

- **Recipe** (`--float8-recipe`): `delayed_scaling` (default, per-tensor 16-step amax
  history, HYBRID E4M3 fwd/E5M2 bwd) or `float8_block_scaling` (block-wise).
- **Hopper/Ada GPU only** — needs fp8 tensor cores: H20 / H100 / L40S (sm_89/90+).
  On older cards TE errors at runtime, so only pass `--float8` when `check_hardware`
  reports an H20/Hopper. plan_training also **errors if the policy isn't pi0/pi05**.
- **Needs the TE-enabled lerobot image** (TransformerEngine baked into the deps image).
- fp8 composes with bf16 autocast — master weights stay bf16 (`--policy.dtype=bfloat16`).
- The old **torchao** path (`--use_float8`) was removed; don't use it.
- Note: an earlier pi05/H20 benchmark found plain bf16-no-compile was fastest for the
  torchao path — re-benchmark this TE path before assuming a speedup; treat fp8 as
  primarily a **memory** lever.

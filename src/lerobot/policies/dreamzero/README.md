# DreamZero (NVIDIA GEAR) — in-tree LeRobot policy

DreamZero is a Wan-video **World Action Model**: a causal DiT (`CausalWanModel`) that jointly
denoises video latents + action/state registers in one token sequence, blockwise-causal, and
autoregresses an action chunk with a KV cache at inference. This port wraps DreamZero's own native
PyTorch inference/training (NOT vLLM) behind the LeRobot `PreTrainedPolicy` interface.

Registered as `dreamzero`. Install extra: `pip install -e ".[dreamzero]"`.

## Status

| Piece | State |
|---|---|
| Model core (`wan/`: causal DiT, VAE38, umt5/clip encoders, schedulers, `WANPolicyHead`) | ported (M1) |
| Config + policy wrapper (`select_action`/`predict_action_chunk` via `lazy_joint_video_action`) | ported (M1) |
| Processor numerics (oxe_droid): q99, view stitch, umt5, relative↔absolute | done + CPU-tested (M2) |
| Checkpoint converter (GEAR `VLA` → LeRobot) | done + structure-tested (M2) |
| Offline open-loop eval | done — needs H20 + checkpoint to run (M2) |
| SFT/LoRA: LoRA + freeze wired (via `WANPolicyHead`), training action packing | done + CPU-tested; training run/data-window tuning on H20 (M3) |

First version targets **oxe_droid / DROID** (`wan22_5b`) and **num_envs == 1** (the KV cache has no
batch semantics). Other embodiments raise `NotImplementedError` in the processor (view stitch +
language template are embodiment-specific). See `dreamzero_port_plan.md` for the full plan.

## Eval (convert → offline open-loop)

```bash
# 1) Convert the released NVIDIA checkpoint to a LeRobot checkpoint dir.
python -m lerobot.policies.dreamzero.scripts.convert_dreamzero_checkpoint \
    --src /path/to/GEAR-Dreams/DreamZero-DROID \
    --dst /path/to/lerobot-dreamzero-droid \
    --embodiment-tag oxe_droid --model-variant wan22_5b

# 2) Offline open-loop eval on a held-out episode (needs an H20-class GPU).
python -m lerobot.policies.dreamzero.scripts.offline_eval \
    --checkpoint /path/to/lerobot-dreamzero-droid \
    --repo-id <held-out LeRobotDataset> --episode 0 --device cuda
```

The converter emits `config.json` + `model.safetensors` (action_head.* only) + `statistics.json`
(GEAR q01/q99; joint uses relative stats, gripper absolute). `from_pretrained` loads it and the
processor reads `statistics.json` for q99 normalization + the relative→absolute decode.

## SFT (LoRA)

LoRA + component freezing is applied inside `WANPolicyHead` from the config
(`train_architecture="lora"`, `lora_target_modules="q,k,v,o,ffn.0,ffn.2"`, VAE/text/image encoders
frozen); `get_optim_params` returns only the trainable (LoRA + projector) params. The processor
also packs training actions (relative-encode joints + q99 with relative stats, pad, `action_mask`,
`has_real_action`). Wiring `lerobot-train` end-to-end (the video sampling window + the exact
training batch flattening into `forward`) is validated on the first 8×H20 run — see M3.

## Validated locally vs on H20

CPU tests (`tests/policies/test_dreamzero_processor.py`, `test_dreamzero_converter.py`) lock the
deterministic numerics: q99 round-trip, the DROID stitch geometry, crop/resize, the
relative→absolute decode, and the training-pack ↔ decode round-trip. The umt5 tokenizer, GPU
inference, checkpoint numerics parity, and LoRA training require an H20 + the released checkpoint
(the VAE hardcodes `device='cuda'`); these are the M2/M3 validation gates.

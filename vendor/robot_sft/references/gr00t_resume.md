# GR00T / HF-Trainer resume & checkpoint anatomy

Read this before writing the watchdog's resume logic or trusting any checkpoint. Verified
against Isaac GR00T N1.7 (`gr00t/experiment/`). Concepts generalize to any HF-Trainer SFT.

## Two different "checkpoints" — don't conflate

| Concept | What it is | How set |
|---|---|---|
| `start_from_checkpoint` | the **base/pretrained weights to fine-tune FROM** (foreign HF model) | `--base-model-path` → `launch_finetune.py:95`. NOT an HF-Trainer resume. |
| `resume_from_checkpoint` | HF-Trainer **resume THIS run** (optimizer/scheduler/RNG/step) | hardcoded `True` at `experiment.py:305/307`. |

## Resume is automatic

`Gr00tTrainer.train()` (`gr00t/experiment/trainer.py:259-282`) does, in effect:

```python
def train(self, resume_from_checkpoint=None, **kwargs):
    if resume_from_checkpoint is False: resume_from_checkpoint = None
    if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
        resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)  # latest checkpoint-N
        if resume_from_checkpoint is None:
            logging.warning(f"No valid checkpoint found in output directory ({self.args.output_dir})")
    if resume_from_checkpoint is not None:
        self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, "trainer_state.json"))
    return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)
```

**Implication for the watchdog:** to resume, just **re-invoke the identical launch command
with the same `--output-dir`**. No `--resume` flag exists or is needed. If a resumable
`checkpoint-N` is present it continues from there; otherwise it starts fresh and logs a
warning. So the watchdog's "resume" = relaunch-same-command, *after* confirming a resumable
checkpoint exists (else you silently restart from zero).

Caveat: resume restores step/optimizer/RNG but the code sets `ignore_data_skip=True` and
reseeds the dataloader rather than fast-forwarding it — data order is not byte-identical
after resume ("non-reproducible"). Loss continuity is fine; exact reproducibility is not.

## What a checkpoint-N/ contains

**Resumable** (must have ALL of these — written by HF Trainer save):
- `trainer_state.json`  ← written **last**; its presence ⇒ the save finished
- `optimizer.pt`
- `scheduler.pt`
- `rng_state.pth` (or `rng_state_<rank>.pth` per rank under distributed)
- model shards: `model-0000X-of-0000Y.safetensors` + `model.safetensors.index.json`
- `training_args.bin`, `config.json`

**Inference-only additions** (copied by a save callback): `processor/` (→
`processor_config.json` + tokenizer), `experiment_cfg/` (configs + normalization),
`statistics.json`, `wandb_config.json`.

## Two rules the watchdog must enforce

1. **`--save_only_model` must be OFF** for any resumable run. With it on, only model
   shards/config are saved → no optimizer/scheduler/RNG → HF reloads weights but restarts
   at step 0. (`finetune_config.py:157`, `experiment.py:223`.)
2. **A complete checkpoint is signalled by `trainer_state.json`**, because it is written
   after the model shards. Use it as the completion sentinel both when (a) deciding a
   checkpoint is safe to resume from, and (b) waiting before SIGTERM if stopping at a
   target step (see lessons_learned #4).

## `is_resumable(checkpoint_dir)` — the predicate to implement

```
return all file exists in checkpoint_dir for file in
    ["trainer_state.json", "optimizer.pt", "scheduler.pt"]
  and any(name.startswith("rng_state") for name in listdir(checkpoint_dir))
  and any(name.endswith(".safetensors") for name in listdir(checkpoint_dir))
```
`scripts/watchdog.py` implements exactly this; `latest_resumable_checkpoint(output_dir)`
returns the highest-step `checkpoint-N` that passes it (or None ⇒ a restart would be from
scratch, which the watchdog must flag, not hide).

## Key file references
- `gr00t/experiment/launch_finetune.py:95,115,116`
- `gr00t/experiment/experiment.py:209-240,305,307`
- `gr00t/experiment/trainer.py:221-235,259-282`
- `gr00t/experiment/utils.py:24-71` (inference-file copy callback), `74-133` (best-metric)
- `gr00t/configs/finetune_config.py:157-163`

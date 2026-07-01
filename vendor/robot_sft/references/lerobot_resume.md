# lerobot resume & checkpoint anatomy

How `lerobot-train` checkpointing and resume actually work (verified against the lerobot
checkout this skill ships with — `src/lerobot/common/train_utils.py`,
`src/lerobot/configs/train.py`, `src/lerobot/scripts/lerobot_train.py`). Read this before
touching the watchdog's resume logic or trusting any checkpoint.

## Checkpoint layout

Every `--save_freq` steps (and at the final step) lerobot writes, under
`<output_dir>/checkpoints/`:

```
<output_dir>/checkpoints/
├── 005000/                        # zero-padded training step
│   ├── pretrained_model/          # everything needed for INFERENCE / deployment
│   │   ├── config.json            #   policy config
│   │   ├── model.safetensors      #   policy weights
│   │   ├── train_config.json      #   full TrainPipelineConfig (resume entry point)
│   │   ├── processor.json         #   processor pipeline config (if any)
│   │   └── step_*.safetensors     #   processor state files (if any)
│   └── training_state/            # everything needed to RESUME training
│       ├── optimizer_state.safetensors
│       ├── optimizer_param_groups.json
│       ├── rng_state.safetensors
│       ├── scheduler_state.json
│       └── training_step.json     # {"step": N, "num_processes": ..., "batch_size": ...}
└── last -> 005000                 # relative symlink, updated after each successful save
```

Key facts:

- **`pretrained_model/` is written before `training_state/`.** A kill in between leaves an
  inference-loadable but NOT resumable checkpoint. `training_state/training_step.json`
  arriving is the practical "save finished" marker (what the watchdog and eval_watcher key on).
- **The `last` symlink is updated after the save completes** — following
  `checkpoints/last` always lands on a complete checkpoint.
- **There is no checkpoint rotation** (no `save_total_limit`): every saved step stays on
  disk until you delete it. Budget disk accordingly (lessons #5) and prune old step dirs
  manually on long runs — keep the best-eval and the latest.

## How resume works

Resume is **explicit**, not automatic:

```bash
cd /lerobot && uv run lerobot-train \
  --resume=true \
  --config_path=<output_dir>/checkpoints/last/pretrained_model/train_config.json
```

- `--config_path` must point at a **local** `train_config.json` (Hub resume unsupported).
  The full original config is loaded from it, so you don't repeat the other flags; CLI
  overrides on top are possible but keep them minimal.
- From `config_path`, lerobot derives the checkpoint dir and reloads policy weights,
  optimizer, scheduler, and RNG state, and continues from the recorded step.
- **Data order is sample-exact on resume** for the same `(num_processes, batch_size)`;
  the values recorded in `training_step.json` are compared and a warning explains the
  data-order implications if you changed them. Don't change batch size mid-run and expect
  a clean continuation (lessons #16).

Two deliberate guardrails to know:

- Running the ORIGINAL command again does **not** resume — an existing `--output_dir`
  without `--resume=true` is a hard `FileExistsError` (protects against silently
  overwriting a run). This is why the watchdog relaunches with the resume command, and
  why a from-scratch retry must first clear an empty output_dir.
- `--resume=true` **without** `--config_path` is a hard error
  ("A config_path is expected when resuming").

## Resumable vs inference-only

| state | pretrained_model/ | training_state/ | use |
|---|---|---|---|
| complete | ✓ | ✓ | resume **and** eval/deploy |
| truncated save | ✓ | ✗ / partial | eval/deploy only — resume from the previous step |
| corrupted | ✗ | — | nothing; delete |

`is_resumable()` in `watchdog.py` encodes exactly this: it requires
`pretrained_model/{config.json,model.safetensors,train_config.json}` AND
`training_state/{optimizer_state.safetensors,training_step.json,rng_state*}`.
`verify_run.py` reuses it, and separately checks `inference_ready`
(config + weights only), so a truncated final save is reported as "evaluable but resume
from step N-1" rather than a blanket failure.

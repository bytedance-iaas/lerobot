# Running LeRobot on Ascend NPU

Verified 2026-09-04 on an `Ascend950PR` host (16 devices) inside the
`quay.io/ascend/vllm-ascend:v0.23.0-a5` image, `torch 2.10.0+cpu` + `torch_npu 2.10.0.post4`.
ACT trained end to end on `lerobot/pusht`.

`dev_v2` supports CUDA and Ascend from the same tree — the device branches are additive and CUDA
keeps its priority in `auto_select_torch_device`. Nothing here changes behaviour on a CUDA host.

---

## Install: use pip, do not use uv

```bash
pip install -e . --no-build-isolation
pip install -e '.[dataset]' --no-build-isolation
pip install 'accelerate>=1.14.0,<2.0.0' 'wandb>=0.24.0,<0.28.0'
```

Three constraints. Each one has a concrete failure behind it.

### 1. Do not use uv

`pyproject.toml` pins torch to a CUDA index:

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cu128", marker = "sys_platform == 'linux'" }]
torchvision = [{ index = "pytorch-cu128", marker = "sys_platform == 'linux'" }]
```

uv honours that and installs a CUDA build, replacing the CPU build that `torch_npu` is compiled
against — NPU support disappears. pip ignores `[tool.uv.sources]` entirely, and the preinstalled
torch already satisfies `torch>=2.7,<2.12`, so pip leaves it alone. Confirmed with
`pip install -e . --dry-run`: torch and torchvision report `already satisfied`.

Whether `uv pip install` reads `[tool.uv.sources]` was not tested, so the rule is **avoid uv**,
not "avoid `uv sync`". Note this contradicts `CLAUDE.md`'s "prefer uv run" — that guidance is for
CUDA hosts.

### 2. Do not create a virtualenv

`torch_npu` ships **in the image**, not in the project. A fresh venv does not have it, and
installing torch into that venv gets you a plain build with no Ascend plugin. Install into the
image's interpreter (`/usr/local/python3.12.13` in the vllm-ascend image). pip's
"running as root" warning is expected inside a container.

### 3. Do not install the `[training]` extra directly

It is self-referential:

```toml
training = ["lerobot[dataset]", "wandb>=0.24.0,<0.28.0", "lerobot[accelerate-dep]"]
```

pip resolves `lerobot` from PyPI and installs the published wheel **over the editable install**
(a dry run showed it pulling `lerobot-0.6.1`). Install what the extra actually needs —
`accelerate` and `wandb` — by name instead.

### After switching branches, regenerate the editable metadata

`pip install -e .` records dependency metadata from the pyproject **as it was at that commit**.
Switch branches without reinstalling and `pip check` reports conflicts that do not exist — an
older checkout claimed `draccus>=0.11.6` while the current branch pins `==0.10.0`.

```bash
pip install -e . --no-deps --no-build-isolation
```

### Known dependency conflict

`triton-ascend 3.2.2` requires `numpy==1.26.4`; LeRobot requires `numpy>=2.0.0`. numpy 2.2.6 is
what gets installed, and `torch_npu` keeps working with it (verified: `torch.npu.is_available()`
is True, 16 devices). Anything that goes through Triton — `torch.compile` — needs this resolved
first.

---

## Run

```bash
HF_ENDPOINT=https://hf-mirror.com ASCEND_RT_VISIBLE_DEVICES=15 \
lerobot-train \
  --policy.type=act --policy.device=npu --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/pusht \
  --batch_size=2 --steps=5 --num_workers=0 \
  --save_checkpoint=false --wandb.enable=false \
  --output_dir=/tmp/act-smoke
```

Three of these are load-bearing:

| Flag | Without it |
|---|---|
| `--policy.device=npu` | defaults to cuda; `torch.cuda.is_available()` is False, so it falls back to CPU |
| `--policy.push_to_hub=false` | `ValueError: 'repo_id' argument missing` |
| `HF_ENDPOINT` | huggingface.co is unreachable from CN hosts (HTTP 000); the mirror answers 200 |

### The video backend needs no flag (since 627c0ce77)

`get_safe_default_video_backend` used to accept `find_spec("torchcodec")` as proof the decoder
worked, so it selected torchcodec here and training died in the data loader minutes later. It now
imports it and falls back on failure, naming the real error:

```
WARNING lerobot.utils.import_utils: 'torchcodec' cannot be loaded
  (RuntimeError: Could not load libtorchcodec. Likely causes:), falling back to 'pyav'
```

Verified on this host: ACT trains to completion with no `--dataset.video_backend` flag.

> Do not read torchcodec's **own** "Falling back to 'pyav' as a default decoder" line as proof
> the fallback happened. It prints that while loading its libraries, the wording is nearly
> identical to LeRobot's, and before the fix the run went on using torchcodec anyway.

### Installing ffmpeg does not fix torchcodec here — don't bother

The first error names ffmpeg (`libavutil.so.56: cannot open shared object file`), so installing
it is the obvious move. It was tried: `apt-get install -y ffmpeg` on Ubuntu 22.04 gives ffmpeg
4.4.2 and `libavutil.so.56`, which is what torchcodec's `core4` variant wants. The remaining
errors then are:

```
libavutil.so.57 / .58 / .59 / .60: cannot open      # core5..core8 want ffmpeg 5/6/7/8
libtorchcodec_core4.so: undefined symbol: torch_dtype_float4_e2m1fn_x2
```

The second one is the wall. That symbol comes from **torch**, not ffmpeg: torchcodec 0.11.1 was
built against a different torch than the `2.10.0+cpu` this image ships. Fixing it means finding a
torchcodec built for exactly this torch, and redoing that search on every image bump — while the
torch itself cannot be changed without breaking torch_npu.

**pyav does not have this problem, for the reason it looked redundant:** it bundles its own
ffmpeg inside the wheel (`site-packages/av.libs/libavcodec-*.so.61`, ffmpeg 7.x, name-mangled)
and links no torch symbols at all. It sits out the ABI negotiation entirely. torchcodec is
faster, but it couples ffmpeg's ABI to torch's, which is the wrong trade when a vendor image
pins torch.

---

## Why `--policy.device=npu` needed a code change

`is_torch_device_available` knew four devices and raised on anything else:

```python
else:
    raise ValueError(f"Unknown device {try_device}. Supported devices are: cuda, mps, xpu or cpu.")
```

draccus reports that as `Couldn't instantiate class ACTConfig using the given arguments`, which
names neither the device nor the check — the message points nowhere near the cause.

`torch_npu` is an **out-of-tree backend**: `torch.npu` does not exist until `import torch_npu`
registers it. So the probe imports it lazily and returns False if it is absent, which is what
keeps CUDA hosts working unchanged.

## torch is a `+cpu` build, and that is correct

`torch 2.10.0+cpu` with `torch_npu 2.10.0.post4` reports 16 devices. Ascend kernels come from the
plugin, not from the torch build. There is no "GPU build" of torch to switch to here, and
replacing torch will break the ABI `torch_npu` was compiled against.

`npu-smi info` working is **not** the same as `torch.npu.is_available()` being True — a container
can have the driver mounted and still fail the second check, or vice versa. Verify both.

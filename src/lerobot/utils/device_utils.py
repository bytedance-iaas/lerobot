#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch


def _npu_available() -> bool:
    """True when torch_npu is installed and an Ascend device is visible.

    torch has no `npu` attribute until `torch_npu` is imported — it is an out-of-tree backend
    that registers itself on import. Importing it here rather than at module top keeps this file
    usable on machines without Ascend, where the import raises.
    """
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return torch.npu.is_available()


def auto_select_torch_device() -> torch.device:
    """Tries to select automatically a torch device."""
    if torch.cuda.is_available():
        logging.info("Cuda backend detected, using cuda.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        logging.info("Metal backend detected, using mps.")
        return torch.device("mps")
    elif torch.xpu.is_available():
        logging.info("Intel XPU backend detected, using xpu.")
        return torch.device("xpu")
    elif _npu_available():
        logging.info("Ascend NPU backend detected, using npu.")
        return torch.device("npu")
    else:
        logging.warning("No accelerated backend detected. Using default cpu, this will be slow.")
        return torch.device("cpu")


# TODO(Steven): Remove log. log shouldn't be an argument, this should be handled by the logger level
def get_safe_torch_device(try_device: str, log: bool = False) -> torch.device:
    """Given a string, return a torch.device with checks on whether the device is available."""
    try_device = str(try_device)
    if try_device.startswith("cuda"):
        assert torch.cuda.is_available()
        device = torch.device(try_device)
    elif try_device == "mps":
        assert torch.backends.mps.is_available()
        device = torch.device("mps")
    elif try_device == "xpu":
        assert torch.xpu.is_available()
        device = torch.device("xpu")
    elif try_device == "cpu":
        device = torch.device("cpu")
        if log:
            logging.warning("Using CPU, this will be slow.")
    else:
        device = torch.device(try_device)
        if log:
            logging.warning(f"Using custom {try_device} device.")
    return device


def get_safe_dtype(dtype: torch.dtype, device: str | torch.device):
    """
    mps is currently not compatible with float64
    """
    if isinstance(device, torch.device):
        device = device.type
    if device == "mps" and dtype == torch.float64:
        return torch.float32
    if device == "xpu" and dtype == torch.float64:
        if hasattr(torch.xpu, "get_device_capability"):
            device_capability = torch.xpu.get_device_capability()
            # NOTE: Some Intel XPU devices do not support double precision (FP64).
            # The `has_fp64` flag is returned by `torch.xpu.get_device_capability()`
            # when available; if False, we fall back to float32 for compatibility.
            if not device_capability.get("has_fp64", False):
                logging.warning(f"Device {device} does not support float64, using float32 instead.")
                return torch.float32
        else:
            logging.warning(
                f"Device {device} capability check failed. Assuming no support for float64, using float32 instead."
            )
            return torch.float32
        return dtype
    else:
        return dtype


def is_torch_device_available(try_device: str) -> bool:
    try_device = str(try_device)  # Ensure try_device is a string
    if try_device.startswith("cuda"):
        return torch.cuda.is_available()
    elif try_device == "mps":
        return torch.backends.mps.is_available()
    elif try_device == "xpu":
        return torch.xpu.is_available()
    elif try_device.startswith("npu"):
        return _npu_available()
    elif try_device == "cpu":
        return True
    else:
        raise ValueError(
            f"Unknown device {try_device}. Supported devices are: cuda, mps, xpu, npu or cpu."
        )


def is_amp_available(device: str):
    if device.startswith("npu"):
        return _npu_available()
    if device in ["cuda", "xpu", "cpu"]:
        return True
    elif device == "mps":
        return False
    else:
        raise ValueError(f"Unknown device '{device}.")


def vision_preprocess_device(device: str | torch.device) -> torch.device:
    """Where to run Hugging Face image preprocessing for a model living on ``device``.

    The Qwen2/3-VL image processors patchify with a ten-dimensional view → permute → reshape.
    CANN caps tensors at eight dimensions ("The self tensor cannot be larger than 8
    dimensions"), and the reshape of a permuted view materialises a copy, so on NPU that
    surfaces as ``aclnnInplaceCopy failed, error code is 161002`` -- an error that names the
    copy rather than the rank that caused it. Keeping the vision preprocessing on the host
    sidesteps it; the processor's outputs are moved to the device by the pipeline's device
    step anyway.

    Every other backend keeps preprocessing on-device, which is the point of the
    torchvision-backed fast processors: it avoids a device→host→device roundtrip per step.
    """
    dev = device if isinstance(device, torch.device) else torch.device(device)
    return torch.device("cpu") if dev.type == "npu" else dev


def accelerator_module(device: str | torch.device):
    """The ``torch.<backend>`` module for ``device`` (``torch.cuda``, ``torch.npu``, ...).

    Returns None on CPU, and for any backend torch does not expose a module for. Code that
    reaches for ``torch.cuda.empty_cache`` / ``Event`` / ``mem_get_info`` can go through this
    instead of hardcoding CUDA -- the NPU equivalents live under ``torch.npu`` with the same
    names, but only exist once torch_npu has been imported, so this looks the module up by
    name rather than importing anything.
    """
    dev = device if isinstance(device, torch.device) else torch.device(device)
    if dev.type == "cpu":
        return None
    return getattr(torch, dev.type, None)


def rope_dtypes(device: str | torch.device) -> tuple[torch.dtype, torch.dtype]:
    """Real/complex dtypes for a complex-valued rotary embedding on ``device``.

    CANN implements polar/cat/mul for DT_FLOAT only ("not implemented for DT_COMPLEX128"),
    so NPU runs the rotation in single precision; every other backend keeps float64/complex128.
    Measured on the Wan rotary embedding: max relative error 1.0e-07 against the float64 path,
    just under float32's eps of 1.2e-07. It is that small because the angles are still computed
    in float64 at construction time and only the resulting unit-modulus values are rounded, the
    activations arrive as float32 anyway, and these rope_apply functions return float32. Much
    longer sequences would stress the angle computation instead, which this does not touch.
    """
    dev = device if isinstance(device, torch.device) else torch.device(device)
    if dev.type == "npu":
        return torch.float32, torch.complex64
    return torch.float64, torch.complex128

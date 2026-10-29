# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Single-device Accelerate integration for the out-of-tree Torch Ada backend.

Import only when Ada is requested: Accelerate remains an optional training dependency.
"""

import os
from contextlib import nullcontext

import torch
from accelerate import Accelerator


class AdaSingleDeviceAccelerator(Accelerator):
    """Use Accelerate's training helpers with Ada's native BF16 autocast.

    Accelerate supports an explicit torch device but does not recognize Ada in its
    mixed-precision backend selection. Disable that selection and use the vendor's
    torch.autocast implementation; BF16 does not require a gradient scaler.
    """

    def __init__(self, *, device: str, mixed_precision: str | None = None, **kwargs):
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("Ada training currently supports one process on one device")
        if mixed_precision not in (None, "no", "bf16"):
            raise ValueError("Ada training supports float32 or bfloat16 precision")
        self.ada_mixed_precision = mixed_precision
        previous_device = os.environ.get("ACCELERATE_TORCH_DEVICE")
        os.environ["ACCELERATE_TORCH_DEVICE"] = device
        try:
            super().__init__(mixed_precision="no", **kwargs)
        finally:
            if previous_device is None:
                os.environ.pop("ACCELERATE_TORCH_DEVICE", None)
            else:
                os.environ["ACCELERATE_TORCH_DEVICE"] = previous_device
        if self.num_processes != 1 or self.device != torch.device(device):
            raise ValueError(f"Expected single-device training on {device}, got {self.device}")

    def autocast(self, autocast_handler=None):
        if self.ada_mixed_precision != "bf16":
            return nullcontext()
        handler = autocast_handler or self.autocast_handler
        kwargs = handler.to_kwargs() if handler is not None else {}
        return torch.autocast(device_type="ada", dtype=torch.bfloat16, **kwargs)

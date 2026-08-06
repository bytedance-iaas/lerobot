#!/usr/bin/env python

# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""DreamZero policy wrapper for LeRobot.

Wraps DreamZero's `WANPolicyHead` (Wan-video joint video+action causal DiT) behind the
LeRobot `PreTrainedPolicy` interface. Inference uses DreamZero's own native KV-cache
autoregressive path (`WANPolicyHead.lazy_joint_video_action`) — NOT any vLLM serving
pipeline. The stateful cache (current_start_frame / kv caches / clip_feas / ys / language)
lives on the action head and is cleared by `reset()`.

First version targets single-environment rollout (num_envs == 1): the KV cache has no batch
semantics, so batched vector-env eval is out of scope until the cache is made batch-aware.
"""

import logging
from collections import deque
from pathlib import Path

import torch
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import log_model_loading_keys
from lerobot.utils.import_utils import require_package

from .configuration_dreamzero import TRAINING_MODES, DreamZeroConfig

logger = logging.getLogger(__name__)


class DreamZeroPolicy(PreTrainedPolicy):
    """DreamZero World Action Model policy."""

    config_class = DreamZeroConfig
    name = "dreamzero"

    def __init__(
        self,
        config: DreamZeroConfig,
        dataset_stats: dict | None = None,
        dataset_meta=None,
        defer_lora_injection: bool = False,
    ):
        super().__init__(config)
        require_package("diffusers", extra="dreamzero")
        config.validate_features()
        self.config = config

        # Imported lazily so `import lerobot.policies.dreamzero` stays cheap and the heavy
        # torch/diffusers deps only load when a policy is actually constructed.
        from transformers.feature_extraction_utils import BatchFeature

        from .wan.action_head import WANPolicyHead

        self._BatchFeature = BatchFeature
        # Named `action_head` so released DreamZero checkpoint keys (action_head.*) map directly.
        self.action_head = WANPolicyHead(config.build_action_head_config(defer_lora_injection))
        self._lora_injection_deferred = bool(defer_lora_injection)
        # Upstream fixes these in `WANPolicyHead.__init__` rather than taking them from its config,
        # so apply them here — otherwise the corresponding LeRobot config fields silently do
        # nothing. The defaults reproduce upstream's values.
        self.action_head.num_inference_steps = config.num_inference_steps
        # The step mask is indexed by diffusion step, so it has to track `num_inference_steps`.
        # All-True: this port never uses upstream's skip schedule (see WANPolicyHead.__init__).
        self.action_head.dit_step_mask = [True] * config.num_inference_steps
        self.action_head.cfg_scale = config.cfg_scale

        self.apply_training_mode()
        # FSDP cannot shard a 0-d parameter and this model has one. Doing it here rather than at
        # the training entry point keeps it out of the caller's hands: it is idempotent and a
        # no-op on a single GPU, so there is no reason to make it conditional.
        self.promote_scalar_params_for_fsdp()

        # q01/q99 quantiles carried over from the checkpoint this policy was loaded from, so
        # `_save_pretrained` can write them back out. The processor normalizes with these, and a
        # fine-tuned checkpoint that lost them would decode actions as if unnormalized — wrong, but
        # not obviously so. Set by `from_pretrained`; stays None for a freshly-initialised policy.
        self.checkpoint_statistics: dict | None = None
        # Where the frozen weights came from. Recorded so a `save_lora_only` checkpoint, which
        # stores only the trainable delta, can say what it needs to be loaded against.
        self.base_checkpoint_path: str | None = None

        # select_action's rolling chunk buffer.
        self._action_queue: deque[Tensor] = deque([], maxlen=config.n_action_steps)
        self.reset()

    # ------------------------------------------------------------------
    # PreTrainedPolicy surface
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, config=None, strict: bool = False, **kwargs):
        """Load a LeRobot checkpoint, or a released NVIDIA one (GEAR ``VLA`` layout) as-is.

        ``GEAR-Dreams/DreamZero-*`` repos ship everything needed — geometry in ``config.json``, the
        run's modality layout in ``experiment_cfg/conf.yaml``, statistics in
        ``experiment_cfg/metadata.json``, and bf16 ``action_head.*`` shards — so they are read
        directly, with no conversion step. Their ``config.json`` is a HuggingFace ``VLA`` config
        rather than a draccus dump, which is why the base implementation cannot parse it and this
        override builds the :class:`DreamZeroConfig` from the GEAR files instead.

        Anything else (including what ``save_pretrained`` writes after fine-tuning) goes through
        the normal LeRobot path.
        """
        from . import gear_checkpoint

        manifest = gear_checkpoint.read_adapter_manifest(pretrained_name_or_path)
        if manifest is not None:
            return cls._from_adapter_checkpoint(pretrained_name_or_path, manifest, config=config, **kwargs)

        if not gear_checkpoint.is_gear_checkpoint(pretrained_name_or_path):
            policy = super().from_pretrained(pretrained_name_or_path, config=config, strict=strict, **kwargs)
            policy._attach_checkpoint_statistics(pretrained_name_or_path)
            policy.base_checkpoint_path = str(pretrained_name_or_path)
            # Same landing state as the GEAR path below: on `config.device` at `compute_dtype`,
            # in eval mode (training re-enters train() through the trainer). Without this, a full
            # fine-tuned checkpoint reloaded for inference stays in the fp32 the modules were
            # constructed in — ~92 GB instead of ~46 GB, and the head's bf16 activations then hit
            # fp32 convs.
            policy.prepare_for_inference()
            policy.eval()
            return policy

        device = kwargs.pop("device", None) or (config.device if config is not None else None)
        if config is None:
            config = gear_checkpoint.build_config(
                pretrained_name_or_path,
                embodiment_tag=kwargs.pop("embodiment_tag", "oxe_droid"),
                overrides=kwargs.pop("config_overrides", None),
            )
        if device is not None:
            config.device = device

        # Build on the meta device: constructing 23B parameters for real would allocate and
        # initialise ~92 GB of fp32 host memory that the checkpoint immediately overwrites with
        # bf16. `assign=True` then hands the loaded tensors straight to the modules, so the
        # weights are materialised exactly once, in the dtype they are stored in.
        from .wan.vram import init_weights_on_device

        with init_weights_on_device():
            # A GEAR checkpoint stores unwrapped DiT keys, so LoRA adapters can only be attached
            # after the weights are in (see `_finalize_lora`).
            policy = cls(config, defer_lora_injection=True)

        state_dict = gear_checkpoint.load_state_dict(pretrained_name_or_path, device="cpu")
        missing_keys, unexpected_keys = policy.load_state_dict(state_dict, strict=strict, assign=True)
        log_model_loading_keys(missing_keys, unexpected_keys)

        # Anything the checkpoint did not supply is still a meta tensor, which fails much later
        # with an unrelated-looking error. Materialise buffers/params that were left behind, and
        # refuse to continue if a real parameter is missing.
        policy._materialize_meta_tensors()
        policy._finalize_lora()
        policy._attach_checkpoint_statistics(pretrained_name_or_path)
        policy.base_checkpoint_path = str(pretrained_name_or_path)
        policy.prepare_for_inference()
        policy.eval()
        return policy

    @classmethod
    def _from_adapter_checkpoint(cls, path, manifest: dict, *, config=None, **kwargs):
        """Load a checkpoint that stores only the trainable delta, on top of its base checkpoint.

        The base supplies the 99.5% of weights the run never touched; this directory supplies the
        LoRA adapters and the projectors, plus its own config and statistics. Both halves are
        required — every key stored here must land on a parameter, or the fine-tune is partly lost,
        so the load is strict in the direction that matters even though the state dict is by design
        an incomplete one.
        """
        from safetensors.torch import load_file

        from lerobot.configs.policies import PreTrainedConfig

        path = Path(path)
        base = manifest.get("base_checkpoint")
        if not base or not Path(base).exists():
            raise FileNotFoundError(
                f"{path} stores only the trainable delta and needs its base checkpoint "
                f"{base!r}, which is not present. Restore it, or re-run the fine-tune with "
                f"--policy.save_lora_only=false to get a self-contained checkpoint."
            )
        if config is None:
            config = PreTrainedConfig.from_pretrained(path)

        logger.info("Loading frozen weights from the base checkpoint %s", base)
        policy = cls.from_pretrained(base, config=config, **kwargs)

        delta = load_file(str(path / "model.safetensors"))
        own = dict(policy.named_parameters())
        stranded = sorted(k for k in delta if k not in own)
        if stranded:
            raise RuntimeError(
                f"{len(stranded)} tensor(s) in {path} match no parameter of the rebuilt policy, "
                f"e.g. {stranded[:5]}. The base checkpoint or the LoRA settings differ from the "
                f"ones the fine-tune used."
            )
        with torch.no_grad():
            for name, tensor in delta.items():
                own[name].copy_(tensor.to(own[name].device, own[name].dtype))
        logger.info("Applied %d fine-tuned tensors from %s", len(delta), path)

        # The adapter directory carries the statistics the fine-tune trained against, which are the
        # ones to normalize with — not the base checkpoint's, which `from_pretrained` just set.
        policy._attach_checkpoint_statistics(path)
        policy.base_checkpoint_path = str(base)
        policy.eval()
        return policy

    def _attach_checkpoint_statistics(self, pretrained_name_or_path) -> None:
        """Remember the checkpoint's q01/q99 so a fine-tune can write them back out."""
        from . import gear_checkpoint

        self.checkpoint_statistics = gear_checkpoint.load_checkpoint_statistics(
            pretrained_name_or_path,
            self.config.embodiment_tag,
            self.config.state_modality_keys,
            self.config.action_modality_keys,
            override_path=self.config.statistics_path,
        )

    @staticmethod
    def _saves_adapter_only(config: DreamZeroConfig) -> bool:
        """Whether `_save_pretrained` writes only the trainable delta.

        Static so it can be checked against a config without constructing 23B parameters.
        """
        return config.training_mode == "lora" and config.save_lora_only

    def _save_adapter(self, save_directory: Path, state_dict: dict[str, Tensor] | None) -> None:
        """Write only the parameters training updated, plus a manifest naming the base checkpoint.

        A LoRA run changes 108.6 M of 22,924 M parameters, so storing the other 99.5% again per
        checkpoint costs ~46 GB each for nothing. Upstream's LoRA recipe does the same
        (`save_lora_only=true`), and like upstream this saves every `requires_grad` parameter, not
        just the `lora_*` ones — the state/action projectors are trained too, and dropping them
        would silently discard most of the fine-tune.
        """
        from huggingface_hub import save_torch_state_dict

        from . import gear_checkpoint

        if self.base_checkpoint_path is None:
            raise ValueError(
                "save_lora_only=True needs the base checkpoint this run started from, but this "
                "policy was not loaded from one. Load through `from_pretrained`, or set "
                "`--policy.save_lora_only=false` to write complete weights."
            )
        self.config._save_pretrained(save_directory)
        trainable = {name for name, p in self.named_parameters() if p.requires_grad}
        full = self.state_dict() if state_dict is None else state_dict
        delta = {k: v for k, v in full.items() if k in trainable}
        if not delta:
            raise ValueError("save_lora_only=True but no parameter requires grad — nothing to save.")
        total = sum(t.numel() * t.element_size() for t in delta.values())
        save_torch_state_dict(delta, str(save_directory), max_shard_size=max(total, 1))
        gear_checkpoint.write_adapter_manifest(
            save_directory,
            self.base_checkpoint_path,
            saved_keys=len(delta),
            trainable_params=sum(t.numel() for t in delta.values()),
        )
        logger.info(
            "Saved %d trainable tensors (%.1f M params, %.2f GB) to %s; frozen weights stay in %s.",
            len(delta),
            sum(t.numel() for t in delta.values()) / 1e6,
            total / 1e9,
            save_directory,
            self.base_checkpoint_path,
        )

    def _save_pretrained(self, save_directory: Path, state_dict: dict[str, Tensor] | None = None) -> None:
        """Write the checkpoint, plus the ``statistics.json`` the processor needs to reload it.

        The base implementation saves weights and config only. DreamZero's normalization constants
        live neither in the weights nor in the config — they come from the checkpoint the run
        started from — so without this the fine-tuned checkpoint reloads with no statistics at all
        and every action it decodes is silently left unnormalized.
        """
        if self._saves_adapter_only(self.config):
            self._save_adapter(save_directory, state_dict)
        else:
            super()._save_pretrained(save_directory, state_dict=state_dict)
        if not self.checkpoint_statistics:
            logger.warning(
                "No checkpoint statistics to save alongside %s; reloading it will have no q01/q99 "
                "for the DreamZero processor. This is expected only for a policy that was built "
                "from scratch rather than loaded from a checkpoint.",
                save_directory,
            )
            return
        from . import gear_checkpoint

        gear_checkpoint.save_checkpoint_statistics(
            save_directory, self.config.embodiment_tag, self.checkpoint_statistics
        )

    def _materialize_meta_tensors(self) -> None:
        """Fail loudly on parameters the checkpoint left on the meta device; zero-fill buffers.

        `init_weights_on_device` leaves every parameter on `meta` until `load_state_dict` assigns
        a real tensor. A parameter with no checkpoint entry therefore stays meta and only explodes
        deep inside the forward pass — so name it here instead. Buffers are not covered by
        `include_buffers=False`, but any that did end up on meta are safe to zero-fill.
        """
        stranded = [name for name, p in self.named_parameters() if p.is_meta]
        if stranded:
            raise RuntimeError(
                f"{len(stranded)} parameter(s) were not present in the checkpoint and remain "
                f"uninitialised on the meta device, e.g. {stranded[:5]}."
            )
        for name, buf in list(self.named_buffers()):
            if not buf.is_meta:
                continue
            module_path, _, attr = name.rpartition(".")
            module = self.get_submodule(module_path) if module_path else self
            setattr(module, attr, torch.zeros_like(buf, device="cpu"))

    def prepare_for_inference(self) -> None:
        """Place the model on its device at the runtime dtype (upstream ``post_initialize``).

        The released weights are bf16 and every component of the inference path assumes bf16
        activations, so leaving the modules in the fp32 they were constructed in both doubles the
        footprint (~92 GB vs ~46 GB for 23B params) and raises a dtype mismatch as soon as the
        first bf16 activation reaches an fp32 conv. Kept separate from ``__init__`` because
        training builds the same modules but is driven by LeRobot's own AMP/dtype handling.
        """
        dtype = getattr(torch, self.config.compute_dtype)
        # `_device` is what the action head's internal helpers use to place freshly-made tensors.
        self.action_head._device = self.config.device
        self.action_head.to(device=self.config.device, dtype=dtype)
        if self.config.compile_encoders:
            self._compile_encoders()

    def _compile_encoders(self) -> None:
        """torch.compile the text/image/VAE encoders, as upstream ``post_initialize`` does."""
        head = self.action_head
        compile_kwargs = {"mode": "reduce-overhead", "fullgraph": True, "dynamic": False}
        head.text_encoder.forward = torch.compile(**compile_kwargs)(head.text_encoder.forward)
        head.image_encoder.model.visual.forward = torch.compile(**compile_kwargs)(
            head.image_encoder.model.visual.forward
        )
        head.vae.model.encode = torch.compile(**compile_kwargs)(head.vae.model.encode)

    @classmethod
    def _load_as_safetensor(cls, model, model_file: str, map_location: str, strict: bool):
        """Load LeRobot-format weights, additionally accepting HF's SHARDED safetensors layout.

        LeRobot's base loader only knows the single-file ``model.safetensors``. DreamZero is a
        ~23B-parameter model (46 GB in bf16), so a fine-tuned checkpoint may well be sharded via
        ``model.safetensors.index.json``. When the single file is absent but an index is present,
        load the shards directly; otherwise defer to the base implementation.
        """
        from . import gear_checkpoint

        path = Path(model_file)
        if path.exists() or not path.with_name("model.safetensors.index.json").exists():
            return super()._load_as_safetensor(model, model_file, map_location, strict)

        state_dict = gear_checkpoint.load_state_dict(path.parent, device=map_location)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
        log_model_loading_keys(missing_keys, unexpected_keys)
        return model

    def apply_training_mode(self) -> dict[str, int]:
        """Set `requires_grad` according to `config.training_mode`; returns a param-count summary.

        `full` unfreezes the DiT here; `lora` is set up by the head when it injects the adapters
        (which also unfreezes the projectors sitting inside the DiT). The text/image/VAE encoders
        stay frozen in both modes — the head freezes them itself — which is why even `full` leaves
        6.44 B of the 22.92 B parameters untouched.
        """
        mode = self.config.training_mode
        if mode not in TRAINING_MODES:
            raise ValueError(f"Unknown training_mode {mode!r}; expected one of {TRAINING_MODES}.")
        if (
            self.config.training_mode == "lora"
            and self._lora_injection_deferred
            and not self._lora_injected()
        ):
            # The adapters do not exist yet, so every parameter still reads as trainable and any
            # count taken here would be wrong. `_finalize_lora` redoes this after injection.
            logger.info("DreamZero training_mode=lora: adapters not injected yet, deferring.")
            return {"trainable": 0, "total": sum(p.numel() for p in self.parameters())}

        if mode == "full":
            self.action_head.model.requires_grad_(True)
        # else "lora": `inject_lora_after_loading` already set requires_grad on the adapters and
        # the projectors.

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            "DreamZero training_mode=%s (lora_rank=%d): %.1f M trainable of %.1f M (%.2f%%)",
            mode,
            self.config.lora_rank if mode == "lora" else 0,
            trainable / 1e6,
            total / 1e6,
            100 * trainable / total if total else 0.0,
        )
        if not trainable:
            raise ValueError(f"training_mode={mode!r} left no trainable parameters.")
        return {"trainable": trainable, "total": total}

    def _lora_injected(self) -> bool:
        """True once PEFT has wrapped the DiT (``get_peft_model`` attaches ``peft_config``)."""
        return getattr(self.action_head.model, "peft_config", None) is not None

    def _finalize_lora(self) -> None:
        """Inject the LoRA adapters that construction deferred, then redo the trainable accounting.

        LoRA injection has to happen on the opposite side of weight loading depending on which
        checkpoint is being read, because PEFT renames every wrapped parameter
        (``blocks.0.self_attn.q.weight`` -> ``base_model.model.blocks.0.self_attn.q.base_layer.weight``):

        * a released GEAR checkpoint has no adapters and unwrapped keys, so it must be **loaded
          first** and wrapped after — hence `defer_lora_injection=True` on that path;
        * a checkpoint written by a LoRA fine-tune already contains adapter keys under the wrapped
          names, so it must be **wrapped first** — the default construction-time injection.

        Getting this backwards does not fail loudly: `load_state_dict(strict=False)` would report
        every DiT weight as missing and quietly leave the randomly-initialised ones in place.
        """
        if self.config.training_mode != "lora" or self._lora_injected():
            return
        self.action_head.inject_lora_after_loading()
        self._lora_injection_deferred = False
        # New parameters appeared, so both the freeze decision and the FSDP scalar promotion have
        # to be redone over the wrapped module.
        self.apply_training_mode()
        self.promote_scalar_params_for_fsdp()

    def promote_scalar_params_for_fsdp(self) -> list[str]:
        """Reshape 0-d parameters to shape [1]; returns the names that were promoted.

        FSDP cannot shard a 0-dimensional parameter. This model has one — the CLIP image encoder's
        `log_scale` — which is frozen in every training mode but still enters the FSDP graph, so
        multi-GPU training needs it promoted first. Mirrors RLinf's `_promote_scalar_params_to_1d`.

        A no-op on the single-GPU path, so it is safe to call unconditionally before wrapping.
        """
        import torch.nn as nn

        promoted = [name for name, p in self.named_parameters() if p.ndim == 0]
        for full_name in promoted:
            module_path, _, param_name = full_name.rpartition(".")
            module = self.get_submodule(module_path) if module_path else self
            old = getattr(module, param_name)
            setattr(
                module, param_name, nn.Parameter(old.detach().reshape(1), requires_grad=old.requires_grad)
            )
        if promoted:
            logger.info("Promoted %d scalar parameter(s) to 1-D for FSDP: %s", len(promoted), promoted)
        return promoted

    def get_optim_params(self) -> list[Tensor]:
        """Only the parameters `training_mode` left trainable.

        Returns a flat list, not a `{"params": ...}` dict: `make_optimizer_and_scheduler` hands
        this straight to the torch optimizer, which would iterate a bare dict as its *keys* and
        reject them as strings.
        """
        return [p for p in self.parameters() if p.requires_grad]

    def reset(self):
        """Clear the AR session state and the action-selection queue."""
        self._action_queue.clear()
        head = self.action_head
        head.current_start_frame = 0
        head.language = None
        head.clip_feas = None
        head.ys = None
        head.kv_cache1 = None
        head.kv_cache_neg = None
        head.crossattn_cache = None
        head.crossattn_cache_neg = None

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """Training forward — flow-matching loss over video + action.

        Training is not the focus of the initial port (LoRA post-training is staged separately),
        but the wiring matches upstream: `WANPolicyHead.forward` returns a BatchFeature carrying
        the combined loss.
        """
        action_input = self._BatchFeature(data=dict(batch))
        backbone_output = self._empty_backbone_output(action_input)
        outputs = self.action_head(backbone_output, action_input)
        loss = outputs["loss"]
        # detach before float(): these are the reported component losses (dynamics/action), which
        # still carry grad_fn — converting them straight to Python scalars warns and needlessly
        # keeps the graph alive for logging.
        loss_dict = {
            k: float(v.detach()) if isinstance(v, Tensor) else float(v)
            for k, v in outputs.items()
            if k != "loss" and _is_scalar(v)
        }
        return loss, loss_dict or None

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Run one AR step of the joint video+action model → (B, action_horizon, action_dim).

        Returns the still-normalized action chunk; the postprocessor unnormalizes and converts
        relative joint deltas back to absolute.
        """
        action_input = self._BatchFeature(data=dict(batch))
        backbone_output = self._empty_backbone_output(action_input)
        outputs = self.action_head.lazy_joint_video_action(backbone_output, action_input, latent_video=None)
        return outputs["action_pred"]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Return a single action, refilling the chunk queue from the model when empty."""
        if len(self._action_queue) == 0:
            chunk = self.predict_action_chunk(batch, **kwargs)  # (B, horizon, action_dim)
            chunk = chunk[:, : self.config.n_action_steps]
            # queue holds per-timestep (B, action_dim) tensors, popped left-to-right.
            self._action_queue.extend(chunk.transpose(0, 1))
        return self._action_queue.popleft()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _empty_backbone_output(self, action_input):
        """DreamZero's backbone is an identity that contributes no real features.

        The action head only reads `action_input`; it never consumes backbone features, so an
        empty BatchFeature carrying the expected `backbone_features` key is sufficient.
        """
        return self._BatchFeature(data={"backbone_features": torch.empty(0)})


def _is_scalar(v) -> bool:
    return isinstance(v, (int, float)) or (torch.is_tensor(v) and v.numel() == 1)

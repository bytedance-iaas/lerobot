from collections.abc import Callable

import torch

try:  # optional: Ascend NPU fused kernels
    import torch_npu
except ImportError:
    torch_npu = None


def is_npu_fused_adamw_available() -> bool:
    """True when ``npu_apply_adam_w`` can actually be dispatched on this host."""
    if torch_npu is None or not hasattr(torch_npu, "npu_apply_adam_w"):
        return False
    npu = getattr(torch, "npu", None)
    return npu is not None and npu.is_available()


class NpuFusedAdamW(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        if torch_npu is None or not hasattr(torch_npu, "npu_apply_adam_w"):
            raise RuntimeError(
                "NpuFusedAdamW requires torch_npu with npu_apply_adam_w; use torch.optim.AdamW instead."
            )
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("NpuFusedAdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    # ``step`` is a tensor, not an int: lerobot serialises optimizer state
                    # with safetensors, which rejects plain Python scalars.
                    state["step"] = torch.zeros((), dtype=torch.float32)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1

                grad = p.grad if p.grad.is_contiguous() else p.grad.contiguous()
                step = int(state["step"].item())
                torch_npu.npu_apply_adam_w(
                    beta1 ** (step - 1),  # beta1_power: powers from before this step
                    beta2 ** (step - 1),  # beta2_power
                    lr,
                    weight_decay,
                    beta1,
                    beta2,
                    eps,
                    grad,
                    None,  # max_grad_norm: clipping is done by the training loop
                    False,  # amsgrad
                    False,  # maximize
                    out=(p, state["exp_avg"], state["exp_avg_sq"]),
                )

        return loss

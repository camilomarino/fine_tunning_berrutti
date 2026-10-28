"""Keep Accelerate routing while avoiding its compiler-only frame decorators."""

from __future__ import annotations

import inspect
import os

POLICY = "eager Accelerate routing; compiler decorators bypassed; device hooks retained"


def original_hook(function):
    """Fail closed unless this is Accelerate's exact lazy compiler wrapper."""
    original = getattr(function, "__wrapped__", None)
    if (
        original is None
        or function.__module__ != "accelerate.hooks"
        or function.__code__.co_name != "wrapper"
        or set(function.__code__.co_freevars) != {"fn", "wrapper"}
        or original.__module__ != "accelerate.hooks"
    ):
        raise ValueError("Unknown Accelerate compiler wrapper")
    closure = dict(
        zip(function.__code__.co_freevars, function.__closure__, strict=True)
    )
    if closure["fn"].cell_contents is not original:
        raise ValueError("Accelerate wrapper no longer calls its declared original")
    return original


def install():
    if os.environ.get("TORCHDYNAMO_DISABLE") != "1":
        raise RuntimeError("Eager routing requires TORCHDYNAMO_DISABLE=1 before import")
    import accelerate.hooks as hooks

    changed = []
    for name, cls in vars(hooks).items():
        if not inspect.isclass(cls) or cls.__module__ != hooks.__name__:
            continue
        for method in ("pre_forward", "post_forward"):
            function = vars(cls).get(method)
            if function is None or not hasattr(function, "__wrapped__"):
                continue
            replacement = original_hook(function)
            setattr(cls, method, replacement)
            changed.append(f"{name}.{method}")
    if not {"AlignDevicesHook.pre_forward", "AlignDevicesHook.post_forward"} <= set(
        changed
    ):
        raise ValueError("Expected routing wrappers were not found")
    return {"policy": POLICY, "methods": sorted(changed)}


def verify_routing():
    """A real two-GPU routed model must match explicit routing and gradients."""
    import torch
    from accelerate import dispatch_model
    from copy import deepcopy
    from torch.utils.checkpoint import checkpoint

    class Routed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Linear(16, 16)
            self.last = torch.nn.Linear(16, 8)

        def forward(self, value):
            value = checkpoint(self.first, value, use_reentrant=False).tanh()
            return checkpoint(self.last, value, use_reentrant=False)

    # Preserve the experiment's RNG; this diagnostic must not change LoRA initialization.
    with torch.random.fork_rng(devices=[0, 1]):
        torch.manual_seed(20261007)
        model = Routed().to(dtype=torch.bfloat16)
        reference = deepcopy(model)
        model = dispatch_model(
            model, device_map={"first": 0, "last": 1}, force_hooks=True
        )
        reference.first.to("cuda:0")
        reference.last.to("cuda:1")
        if not all(hasattr(module, "_hf_hook") for module in (model.first, model.last)):
            raise RuntimeError("Routing smoke did not attach real device hooks")
        for index in range(128):
            count = (1, 7, 31, 127)[index % 4]
            value = torch.randn(count, 16, device="cuda:0", dtype=torch.bfloat16)
            model.eval()
            with torch.no_grad():
                actual = model(value)
                expected = reference.last(reference.first(value).tanh().to("cuda:1"))
            if not torch.equal(actual, expected.to(actual.device)):
                raise RuntimeError("Eager routing changed forward output")
            model.train()
            actual = model(value)
            # Accelerate returns the routed output to the input device. Compare
            # the same objective on that device, including its backward path;
            # reducing the reference loss on GPU 1 is a different CUDA graph.
            expected = reference.last(reference.first(value).tanh().to("cuda:1")).to(
                actual.device
            )
            actual.float().square().mean().backward()
            expected.float().square().mean().backward()
            for (name, left), (other, right) in zip(
                model.named_parameters(), reference.named_parameters(), strict=True
            ):
                if (
                    name != other
                    or left.grad is None
                    or right.grad is None
                    or not torch.isfinite(left.grad).all()
                    or not torch.isfinite(right.grad).all()
                    or not torch.equal(left.grad, right.grad)
                ):
                    difference = (
                        float((left.grad.float() - right.grad.float()).abs().max())
                        if left.grad is not None and right.grad is not None
                        else None
                    )
                    raise RuntimeError(
                        "Eager routing changed or invalidated gradients: "
                        f"sample={index}, batch={count}, parameter={name}, "
                        f"reference_parameter={other}, max_abs_difference={difference}"
                    )
            model.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
        for device in range(2):
            torch.cuda.synchronize(device)
    return {
        "success": True,
        "cuda_device_count": 2,
        "samples_completed": 128,
        "optimizer_steps": 0,
        "reference_loss_device": "cuda:0",
    }

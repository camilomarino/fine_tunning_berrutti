"""Frozen visual Conv3d in FP32 without cuDNN; decoder and output remain BF16."""

from __future__ import annotations

from types import MethodType

from finetune.recipes.berrutti_qwen38.comparison import check_native_forward

POLICY = "frozen native Conv3d FP32; cuDNN disabled only here; TF32 off; BF16 output"


def float_convolution(module, value):
    import torch

    with (
        torch.autocast("cuda", enabled=False),
        torch.backends.cudnn.flags(enabled=False),
    ):
        return torch.nn.functional.conv3d(
            value.float(),
            module.weight.float(),
            module.bias.float() if module.bias is not None else None,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
        )


def stable_forward(module, value):
    return float_convolution(module, value).to(value.dtype)


def install(model):
    import torch

    projection = model.base_model.model.model.visual.patch_embed.proj
    check_native_forward(projection, torch.nn.Conv3d.forward)
    if any(parameter.requires_grad for parameter in projection.parameters()):
        raise ValueError("FP32 fallback is restricted to the frozen visual projection")
    torch.backends.cuda.matmul.allow_tf32 = False
    replacement = MethodType(stable_forward, projection)
    if hasattr(projection, "_hf_hook"):
        projection._old_forward = replacement
    else:
        projection.forward = replacement
    check_native_forward(projection, stable_forward)
    return projection


def verify(projection, pixels):
    import torch

    if not (
        projection.kernel_size == projection.stride
        and projection.padding == (0, 0, 0)
        and projection.dilation == (1, 1, 1)
        and projection.groups == 1
    ):
        raise ValueError("Projection cannot be checked against an exact patch matrix")
    value = pixels.to(projection.weight.device, dtype=projection.weight.dtype).view(
        -1, projection.in_channels, *projection.kernel_size
    )
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        convolution = float_convolution(projection, value)
        reference = torch.nn.functional.linear(
            value.flatten(1).float(),
            projection.weight.flatten(1).float(),
            projection.bias.float() if projection.bias is not None else None,
        ).view_as(convolution)
        relative = float(
            (convolution - reference).norm() / reference.norm().clamp_min(1e-12)
        )
    if not torch.isfinite(convolution).all() or relative > 1e-6:
        raise RuntimeError(
            "FP32 native projection failed its strict FP32 reference check"
        )
    return {
        "policy": POLICY,
        "patches": value.shape[0],
        "projection_relative_l2": relative,
        "reference_tolerance": 1e-6,
        "output_dtype": str(value.dtype),
    }

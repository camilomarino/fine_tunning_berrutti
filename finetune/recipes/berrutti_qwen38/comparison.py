"""Native visual-projection interface guard; annotations live in data.annotations."""

from functools import partial

from data.annotations import aligned_regions, canonical_text, reference_order  # noqa: F401


def check_native_forward(module, native_function) -> None:
    """Check the implementation beneath Accelerate routing, preserving the hook."""
    forward = module.forward
    if isinstance(forward, partial):
        if (
            forward.func.__module__ != "accelerate.hooks"
            or forward.func.__name__ != "new_forward"
            or forward.args != (module,)
            or not hasattr(module, "_hf_hook")
        ):
            raise RuntimeError("Unrecognized wrapper on visual projection")
        forward = module._old_forward
    if getattr(forward, "__func__", None) is not native_function:
        raise RuntimeError("Visual projection is not the native implementation")

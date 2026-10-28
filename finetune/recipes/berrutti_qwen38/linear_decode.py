"""Inference-only Qwen prefill candidate justified by sealed kernel replays."""

from contextlib import contextmanager
import hashlib
import inspect
import math
from pathlib import Path
from types import SimpleNamespace

DECODE_LINEAR_POLICY = "native-pytorch-chunk-fp32-fixed-bf16-output-v1"
NATIVE_QWEN_SHA256 = "aac2a1bcca88829afc6dbdf83f97155ebf64d800ee9d703af3802a9aebcfcd18"
REFERENCE_LIMIT = 1e-5


def native_chunk_body(modeling):
    """Bypass only the installed hub dispatcher, retaining its exact fallback."""
    body = inspect.unwrap(modeling.torch_chunk_gated_delta_rule)
    if (
        body.__name__ != "torch_chunk_gated_delta_rule"
        or body.__module__ != modeling.__name__
        or hashlib.sha256(Path(inspect.getfile(body)).read_bytes()).hexdigest()
        != NATIVE_QWEN_SHA256
    ):
        raise ValueError("Unverified native Qwen chunk body")
    return body


def linear_reference(torch, q, k, v, g, beta, initial_state=None):
    """Independent FP64 token recurrence; used only for bounded numerical checks."""
    q, k, v, g, beta = [t.double() for t in (q, k, v, g, beta)]
    q = q / ((q * q).sum(-1, keepdim=True) + 1e-6).sqrt()
    k = k / ((k * k).sum(-1, keepdim=True) + 1e-6).sqrt()
    q = q / math.sqrt(q.shape[-1])
    state = (
        torch.zeros(
            (q.shape[0], q.shape[2], q.shape[-1], v.shape[-1]),
            dtype=torch.float64,
            device=q.device,
        )
        if initial_state is None
        else initial_state.double().clone()
    )
    outputs = []
    for t in range(q.shape[1]):
        state = state * g[:, t].exp()[..., None, None]
        prediction = (state * k[:, t, :, :, None]).sum(-2)
        delta = (v[:, t] - prediction) * beta[:, t, :, None]
        state = state + k[:, t, :, :, None] * delta[:, :, None, :]
        outputs.append((state * q[:, t, :, :, None]).sum(-2))
    return torch.stack(outputs, dim=1), state


def relative_error(actual, reference):
    return float(
        (actual.double() - reference.double()).norm()
        / reference.double().norm().clamp_min(1e-12)
    )


def tensor_identity(tensor):
    try:
        version = tensor._version
    except RuntimeError as error:
        if "Inference tensors do not track version counter" not in str(error):
            raise
        version = None
    return (id(tensor), tensor.data_ptr(), version, tensor.dtype, tensor.device)


@contextmanager
def precise_decode_linear(model, torch, modeling, emit):
    """Use FP32 native PyTorch prefill; leave the cached FLA recurrent untouched."""
    original = modeling.torch_chunk_gated_delta_rule
    recurrent = modeling.torch_recurrent_gated_delta_rule
    body = native_chunk_body(modeling)
    parameters = tuple(model.parameters())
    before = tuple(tensor_identity(p) for p in parameters)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    control = {
        "policy": DECODE_LINEAR_POLICY,
        "native_source_sha256": NATIVE_QWEN_SHA256,
        "calls": 0,
        "prefill_calls": 0,
        "reference_checks": 0,
        "maximum_reference_relative_l2": 0.0,
        "finite": True,
        "reference_close": True,
        "inputs_unchanged": True,
        "cache_format_preserved": True,
        "output_cast_exact": True,
        "cached_kernel_unchanged": True,
        "parameters_unchanged": False,
        "interfaces_restored": False,
    }

    def prefill(
        query,
        key,
        value,
        g,
        beta,
        chunk_size=64,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        **kwargs,
    ):
        if torch.is_grad_enabled():
            raise ValueError("Linear prefill precision is inference-only")
        values = (query, key, value, g, beta)
        if (
            query.ndim != 4
            or query.dtype != torch.bfloat16
            or query.device.type != "cuda"
            or query.shape[1] < 1
            or chunk_size != 64
            or use_qk_l2norm_in_kernel is not True
            or kwargs.get("cu_seqlens") is not None
            or any(t.device != query.device for t in values)
            or key.shape != query.shape
            or value.shape[:3] != query.shape[:3]
            or g.shape != query.shape[:3]
            or beta.shape != g.shape
            or (
                initial_state is not None
                and (
                    initial_state.dtype != torch.float32
                    or initial_state.device != query.device
                    or initial_state.shape
                    != (
                        query.shape[0],
                        query.shape[2],
                        query.shape[-1],
                        value.shape[-1],
                    )
                )
            )
        ):
            raise ValueError("Unexpected Qwen linear prefill contract")
        observed = values + (() if initial_state is None else (initial_state,))
        identities = tuple(tensor_identity(t) for t in observed)
        # Inference tensors have no version counters. Compare their full values
        # against private copies rather than silently omitting mutation checks.
        snapshots = [(t, t.clone()) for t in observed if t.is_inference()]
        output, state = body(
            *(t.float() for t in values),
            chunk_size=chunk_size,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=True,
            **kwargs,
        )
        if identities != tuple(tensor_identity(t) for t in observed) or any(
            not torch.equal(t, saved) for t, saved in snapshots
        ):
            control["inputs_unchanged"] = False
            raise RuntimeError("Linear prefill mutated its inputs/cache")
        if not torch.isfinite(output).all() or (
            state is not None and not torch.isfinite(state).all()
        ):
            control["finite"] = False
            raise RuntimeError("Non-finite native PyTorch linear prefill")
        expected_state_shape = (
            query.shape[0],
            query.shape[2],
            query.shape[-1],
            value.shape[-1],
        )
        if (
            output.dtype != torch.float32
            or output.shape != value.shape
            or (
                output_final_state
                and (
                    state is None
                    or state.dtype != torch.float32
                    or state.shape != expected_state_shape
                )
            )
            or (not output_final_state and state is not None)
        ):
            control["cache_format_preserved"] = False
            raise RuntimeError("Linear prefill changed native output/cache format")
        # Check a causal prefix of one head before the required BF16 output cast.
        # The full returned FP32 state is checked separately in the two-GPU control.
        if control["calls"] % 256 == 0:
            length = min(8, query.shape[1])
            args = [t[:, :length, :1] for t in values]
            initial = None if initial_state is None else initial_state[:, :1]
            reference, _ = linear_reference(torch, *args, initial)
            error = relative_error(output[:, :length, :1], reference)
            control["reference_checks"] += 1
            control["maximum_reference_relative_l2"] = max(
                control["maximum_reference_relative_l2"], error
            )
            if not math.isfinite(error) or not 0 <= error <= REFERENCE_LIMIT:
                control["reference_close"] = False
                raise RuntimeError("Native PyTorch prefill failed FP64 prefix check")
        control["calls"] += 1
        control["prefill_calls"] += 1
        return output.to(query.dtype), state

    torch.backends.cuda.matmul.allow_tf32 = False
    modeling.torch_chunk_gated_delta_rule = prefill
    try:
        yield control
    finally:
        unchanged = modeling.torch_recurrent_gated_delta_rule is recurrent
        modeling.torch_chunk_gated_delta_rule = original
        modeling.torch_recurrent_gated_delta_rule = recurrent
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        control["cached_kernel_unchanged"] = unchanged
        control["interfaces_restored"] = (
            modeling.torch_chunk_gated_delta_rule is original
            and modeling.torch_recurrent_gated_delta_rule is recurrent
        )
        control["parameters_unchanged"] = before == tuple(
            tensor_identity(p) for p in parameters
        )
        emit("decode_linear_precision", **control)
        if not all(
            control[k]
            for k in (
                "cached_kernel_unchanged",
                "interfaces_restored",
                "parameters_unchanged",
            )
        ):
            raise RuntimeError("Linear decode changed parameters or native interfaces")


def verify_decode_linear(torch, modeling):
    """Four bounded cases per GPU; never a full Qwen or generation assay."""
    body = native_chunk_body(modeling)
    rows = []
    with torch.random.fork_rng(devices=[0, 1]), torch.inference_mode():
        for device in range(2):
            torch.manual_seed(20261007)
            for case, length in enumerate((1, 9, 9, 65)):
                shape = (1, length, 3, 8)
                q, k = [
                    torch.randn(shape, device=f"cuda:{device}", dtype=torch.bfloat16)
                    for _ in range(2)
                ]
                v = torch.randn((1, length, 3, 4), device=q.device, dtype=q.dtype)
                g = -torch.rand(shape[:3], device=q.device, dtype=torch.float32)
                beta = torch.rand(shape[:3], device=q.device, dtype=q.dtype)
                if case == 0:
                    q, k = [
                        torch.ones((1, 1, 3, 1), device=q.device, dtype=q.dtype)
                        for _ in range(2)
                    ]
                    v = torch.full_like(q, 2)
                    g.zero_()
                    beta.fill_(0.5)
                initial = (
                    None if case < 2 else torch.randn((1, 3, 8, 4), device=q.device)
                )
                values = (q, k, v, g, beta)
                reference, ref_state = linear_reference(torch, *values, initial)
                model = SimpleNamespace(parameters=lambda: [])
                with precise_decode_linear(
                    model, torch, modeling, lambda *a, **k: None
                ) as control:
                    output, state = modeling.torch_chunk_gated_delta_rule(
                        *values,
                        initial_state=initial,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=True,
                        cu_seqlens=None,
                    )
                    raw, raw_state = body(
                        *(t.float() for t in values),
                        initial_state=initial,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=True,
                    )
                errors = [
                    relative_error(raw, reference),
                    relative_error(state, ref_state),
                ]
                row = {
                    **control,
                    "device": device,
                    "case": case,
                    "output_reference_relative_l2": errors[0],
                    "state_reference_relative_l2": errors[1],
                    "output_cast_exact": torch.equal(output, raw.to(q.dtype)),
                    "state_exact": torch.equal(state, raw_state),
                }
                rows.append(row)
                if (
                    not row["output_cast_exact"]
                    or not row["state_exact"]
                    or any(
                        not math.isfinite(e) or not 0 <= e <= REFERENCE_LIMIT
                        for e in errors
                    )
                ):
                    raise RuntimeError("Two-GPU native PyTorch prefill control failed")
    return {
        "policy": DECODE_LINEAR_POLICY,
        "cuda_device_count": 2,
        "cases_per_device": 4,
        "success": True,
        "rows": rows,
    }


def validate_linear_controls(receipt):
    identity = receipt.get("decode_linear_control", {})
    rows = identity.get("rows", [])
    controls = receipt.get("decode_linear_controls", [])
    if (
        receipt.get("decode_linear_checked") is not True
        or identity.get("policy") != DECODE_LINEAR_POLICY
        or identity.get("success") is not True
        or identity.get("cuda_device_count") != 2
        or identity.get("cases_per_device") != 4
        or [(r.get("device"), r.get("case")) for r in rows]
        != [(d, c) for d in range(2) for c in range(4)]
        or len(controls) != 6
    ):
        raise ValueError("Missing complete linear prefill GPU controls")
    for row in [*rows, *controls]:
        if (
            row.get("policy") != DECODE_LINEAR_POLICY
            or row.get("native_source_sha256") != NATIVE_QWEN_SHA256
            or any(
                row.get(k, 0) < 1
                for k in ("calls", "prefill_calls", "reference_checks")
            )
            or any(
                row.get(k) is not True
                for k in (
                    "finite",
                    "reference_close",
                    "inputs_unchanged",
                    "cache_format_preserved",
                    "output_cast_exact",
                    "cached_kernel_unchanged",
                    "parameters_unchanged",
                    "interfaces_restored",
                )
            )
            or not math.isfinite(row.get("maximum_reference_relative_l2", float("inf")))
            or not 0 <= row["maximum_reference_relative_l2"] <= REFERENCE_LIMIT
        ):
            raise ValueError("Unverified inference-only linear prefill control")
    for row in rows:
        if row.get("state_exact") is not True or any(
            not math.isfinite(row.get(k, float("inf")))
            or not 0 <= row[k] <= REFERENCE_LIMIT
            for k in ("output_reference_relative_l2", "state_reference_relative_l2")
        ):
            raise ValueError("Linear prefill output/state reference check failed")

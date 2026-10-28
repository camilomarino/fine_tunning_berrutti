"""Exact host-staged CUDA transfers for a separately gated ClusterUY candidate."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path

POLICY = "host-staged-cross-cuda-exact-v1"


@contextmanager
def host_staged_transfers(torch):
    """Preserve Tensor.to overloads/autograd; stage only cross-CUDA copies."""
    original = torch.Tensor.to
    if getattr(original, "_berrutti_transport", False):
        raise RuntimeError("A transport override is already installed")
    control = dict(
        policy=POLICY,
        calls=0,
        bytes=0,
        exact=True,
        source_unchanged=True,
        dtype_shape_preserved=True,
        interfaces_restored=False,
    )

    def bytes_of(value):
        # Singleton dimensions may be "contiguous" with a non-unit last stride.
        # An explicit flat contiguous clone also handles scalars before byte view.
        return (
            value.detach()
            .reshape(-1)
            .clone(memory_format=torch.contiguous_format)
            .view(torch.uint8)
        )

    def transfer(tensor, *args, **kwargs):
        # The native parser resolves all public overloads, including .to(other).
        # `copy` belongs to Tensor.to but is not accepted by this parser.
        parsed = {key: value for key, value in kwargs.items() if key != "copy"}
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **parsed)
        if device is None or not tensor.is_cuda or device.type != "cuda":
            return original(tensor, *args, **kwargs)
        destination = (
            torch.cuda.current_device() if device.index is None else device.index
        )
        if destination == tensor.device.index:
            return original(tensor, *args, **kwargs)
        # Do not detach: backward must follow the same CPU intermediate.
        staged = original(tensor, device="cpu", non_blocking=False)
        before = staged.detach().clone()
        result = original(staged, *args, **kwargs)
        returned = original(result.detach(), device="cpu", non_blocking=False)
        expected = original(before, dtype=result.dtype)
        byte_equal = torch.equal(bytes_of(returned), bytes_of(expected))
        source_equal = torch.equal(bytes_of(staged), bytes_of(before))
        format_equal = (
            result.shape == tensor.shape
            and result.dtype == (dtype or tensor.dtype)
            and result.device.index == destination
        )
        control["calls"] += 1
        control["bytes"] += tensor.numel() * tensor.element_size()
        control["exact"] &= byte_equal
        control["source_unchanged"] &= source_equal
        control["dtype_shape_preserved"] &= format_equal
        if not byte_equal or not source_equal or not format_equal:
            raise RuntimeError("Host-staged CUDA transfer failed exact identity")
        return result

    transfer._berrutti_transport = True
    torch.Tensor.to = transfer
    try:
        yield control
    finally:
        torch.Tensor.to = original
        control["interfaces_restored"] = torch.Tensor.to is original


def validate_transport_receipt(receipt):
    """A bounded transport control cannot approve the full Qwen assay."""
    if receipt.get("policy") != POLICY or receipt.get("success") is not True:
        raise ValueError("Missing exact transport receipt")
    rows = receipt.get("transfer_cases", [])
    expected = {
        (a, b, dtype, count)
        for a, b in [(0, 1), (1, 0)]
        for dtype in ("torch.int64", "torch.float32", "torch.bfloat16")
        for count in (1, 127)
    }
    keys = {
        (x.get("source"), x.get("destination"), x.get("dtype"), x.get("count"))
        for x in rows
    }
    if (
        len(rows) != 12
        or keys != expected
        or any(x.get("exact") is not True or x.get("overloads") != 5 for x in rows)
    ):
        raise ValueError("Transport control coverage or exact copies failed")
    routing = receipt.get("routing_control", {})
    runtime = receipt.get("runtime_control", {})
    if (
        routing.get("success") is not True
        or routing.get("cuda_device_count") != 2
        or routing.get("samples_completed") != 128
    ):
        raise ValueError("Full original routing/gradient identity control is required")
    if (
        runtime.get("policy") != POLICY
        or runtime.get("calls", 0) < 1
        or any(
            runtime.get(k) is not True
            for k in (
                "exact",
                "source_unchanged",
                "dtype_shape_preserved",
                "interfaces_restored",
            )
        )
    ):
        raise ValueError("Runtime transport identity or restoration failed")


def verify_transport(torch):
    """Twelve cases and five native overloads; small control, never Qwen proof."""
    rows = []
    with torch.random.fork_rng(devices=[0, 1]):
        for dtype in (torch.int64, torch.float32, torch.bfloat16):
            for source, destination in [(0, 1), (1, 0)]:
                for count in (1, 127):
                    cpu = (torch.arange(count * 16).reshape(count, 16) % 17 - 8).to(
                        dtype
                    )
                    value = cpu.to(f"cuda:{source}")
                    target = torch.empty(0, dtype=dtype, device=f"cuda:{destination}")
                    outputs = [
                        value.to(f"cuda:{destination}"),
                        value.to(device=f"cuda:{destination}"),
                        value.to(target),
                        value.to(f"cuda:{destination}", copy=True),
                        value.T.to(
                            f"cuda:{destination}", memory_format=torch.contiguous_format
                        ).T,
                    ]
                    if not all(torch.equal(x.detach().cpu(), cpu) for x in outputs):
                        raise RuntimeError(
                            "Cross-CUDA Tensor.to overload changed values"
                        )
                    rows.append(
                        dict(
                            source=source,
                            destination=destination,
                            dtype=str(dtype),
                            count=count,
                            exact=True,
                            overloads=5,
                        )
                    )
    return rows


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    args, _ = parser.parse_known_args()
    config = json.loads(args.config.read_text())
    if config.get("cuda_transfer_policy") != POLICY or config.get("gpu_type") != "A40":
        raise ValueError("Host-staged transport requires its explicit A40 candidate")
    from finetune.recipes.berrutti_qwen38 import runner

    if args.mode == "preflight":
        return runner.main()
    if args.mode != "smoke":
        raise ValueError("This transport candidate authorizes only a full assay")
    import torch

    runner.check_gpu_names(
        [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "A40",
    )
    receipt_path = args.root / "transport-completed.json"
    if receipt_path.exists() or (args.root / "completed.json").exists():
        raise RuntimeError("Never overwrite a transport receipt")
    args.root.mkdir(parents=True, exist_ok=True)
    receipt = dict(
        policy=POLICY, success=False, transfer_cases=[], full_qwen_gate=False
    )
    control = {}
    try:
        with host_staged_transfers(torch) as control:
            receipt["transfer_cases"] = verify_transport(torch)
            # Runtime installs the original hooks and runs its unchanged 128-sample
            # gradient check before loading Qwen. Installing hooks twice is invalid.
            result = runner.main()
        events = [
            json.loads(line)
            for line in (args.root / "events.jsonl").read_text().splitlines()
        ]
        routing = [row for row in events if row.get("event") == "runtime_smoke"]
        if len(routing) != 1:
            raise RuntimeError("Exactly one original routing control is required")
        receipt.update(
            success=result == 0, runtime_control=control, routing_control=routing[0]
        )
        validate_transport_receipt(receipt)
    except BaseException as error:
        receipt.update(
            success=False,
            runtime_control=control,
            error_type=type(error).__name__,
            error=str(error),
        )
        with (args.root / "transport-diagnosis.json").open("x") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
        raise
    # This separate closed receipt supplements, and never edits, runner evidence.
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    raise SystemExit(main())

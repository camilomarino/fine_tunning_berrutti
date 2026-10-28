"""CPU-only supplementary check of optimizer metadata and adapter structure."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def optimizer_contract(groups):
    if not groups:
        raise ValueError("Empty optimizer parameter groups")
    canonical = []
    for group in groups:
        if (
            tuple(group["betas"]) != (0.9, 0.999)
            or group["eps"] != 1e-8
            or group["weight_decay"] != 0
            or any(
                group.get(k) is not False
                for k in ("amsgrad", "maximize", "capturable", "differentiable")
            )
        ):
            raise ValueError("AdamW hyperparameters differ from the frozen recipe")
        canonical.append({k: v for k, v in group.items() if k != "lr"})
    indices = [p for group in groups for p in group["params"]]
    if len(indices) != len(set(indices)):
        raise ValueError("Duplicate optimizer parameter membership")
    return canonical


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def metadata(payload):
    contract = optimizer_contract(payload["optimizer"]["param_groups"])
    structure = [
        (name, list(tensor.shape), str(tensor.dtype))
        for name, tensor in sorted(payload["adapter"].items())
    ]
    if not structure:
        raise ValueError("Empty adapter metadata")
    return fingerprint(contract), fingerprint(structure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--definition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Run only on a Slurm CPU allocation")
    import torch
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    torch.set_num_threads(1)
    definition = json.loads(args.definition.read_text())
    records = []
    calibrated = False
    progress = args.output.with_suffix(".progress.jsonl")
    for case in definition["cases"]:
        for root in case["event_roots"]:
            for receipt_path in sorted(
                (Path(root) / "checkpoints").glob("step-*/closed.json")
            ):
                receipt = json.loads(receipt_path.read_text())
                path = receipt_path.parent / "state.pt"
                with FakeTensorMode() as mode:
                    payload = torch.load(
                        path, map_location="cpu", weights_only=False, mmap=True
                    )
                if any(
                    not isinstance(tensor, FakeTensor) or tensor.fake_mode is not mode
                    for tensor in payload["adapter"].values()
                ):
                    raise ValueError("Metadata load did not skip tensor storage")
                contract_hash, structure_hash = metadata(payload)
                del payload
                if not calibrated:
                    native = torch.load(
                        path, map_location="cpu", weights_only=False, mmap=True
                    )
                    if metadata(native) != (contract_hash, structure_hash):
                        raise ValueError("Metadata differs from native deserialization")
                    del native
                    calibrated = True
                records.append(
                    {
                        "case": case["name"],
                        "step": receipt["global_step"],
                        "optimizer_contract_sha256": contract_hash,
                        "adapter_structure_sha256": structure_hash,
                    }
                )
                with progress.open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "case": case["name"],
                                "step": receipt["global_step"],
                                "checkpoints": len(records),
                            }
                        )
                        + "\n"
                    )
    if len(records) != 208 or len({(r["case"], r["step"]) for r in records}) != 208:
        raise ValueError("Metadata checkpoint coverage differs")
    if len({r["optimizer_contract_sha256"] for r in records}) != 1:
        raise ValueError("Optimizer metadata changed across checkpoints/models")
    if len({r["adapter_structure_sha256"] for r in records}) != 1:
        raise ValueError("Adapter names, shapes or dtypes changed")
    result = {
        "success": True,
        "scope": "metadata_only_with_one_native_calibration",
        "metadata_storage_skipped": True,
        "native_calibration_checkpoints": int(calibrated),
        "checkpoints": 208,
        "adamw_hyperparameters_common_and_exact": True,
        "adapter_structure_common": True,
        "optimizer_updates_performed": 0,
        "generations_performed": 0,
        "records": records,
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()

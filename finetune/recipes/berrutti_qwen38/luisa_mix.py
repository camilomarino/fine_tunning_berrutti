"""Experiment four: complete mixed epochs followed by Berrutti consolidation."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from itertools import zip_longest
from pathlib import Path
from types import MethodType

from finetune.recipes.berrutti_qwen38.qwen_sampling import (
    InstructSamplingPolicy,
    seeded_sampling,
    page_seed,
)
from data.integrity import file_sha256
from finetune.recipes.berrutti_qwen38.linear_decode import (
    DECODE_LINEAR_POLICY,
    precise_decode_linear,
    validate_linear_controls,
    verify_decode_linear,
)
from data.preparation.luisa import EXTENSION
from finetune.recipes.berrutti_qwen38.performance import cache_key
from finetune.recipes.berrutti_qwen38.runner import (
    Runtime,
    closed_json,
    learning_rate,
    verify_inputs,
)

STAGE = "combined_luisa_synthetic"
ASSAY_PROTOCOL = "mixed-twice-consolidation-once-v2"
OPTIMIZER_MEMORY_POLICY = "adamw-moments-on-cpu-between-updates-v1"
DECODE_PRECISION_POLICY = "fp32-output-projection-fixed-bf16-weights-v1"
DECODE_ATTENTION_POLICY = "fp32-decoder-sdpa-math-fixed-bf16-weights-v1"
DECODE_CONVOLUTION_POLICY = "fp32-causal-depthwise-conv-fixed-bf16-state-v1"


@contextmanager
def precise_decode_convolution(model, torch, modeling, emit):
    """Accumulate causal depthwise convolution in FP32; keep native BF16 state."""
    originals = (modeling.causal_conv1d_fn, modeling.causal_conv1d_update)
    parameters = tuple(model.parameters())

    def versions():
        return tuple(
            (id(p), p.data_ptr(), p._version, p.dtype, p.device) for p in parameters
        )

    before = versions()
    control = {
        "policy": DECODE_CONVOLUTION_POLICY,
        "calls": 0,
        "prefill_calls": 0,
        "cached_calls": 0,
        "reference_checks": 0,
        "maximum_reference_relative_l2": 0.0,
        "finite": True,
        "reference_close": True,
        "cache_checks": 0,
        "cache_exact": True,
        "parameters_unchanged": False,
    }

    def convolve(hidden, weight, bias, activation, *, padding, output_tokens):
        if (
            torch.is_grad_enabled()
            or hidden.ndim != 3
            or weight.ndim != 2
            or hidden.dtype != torch.bfloat16
            or weight.dtype != torch.bfloat16
            or hidden.device.type != "cuda"
            or hidden.device != weight.device
            or hidden.shape[1] != weight.shape[0]
            or activation not in (None, "silu")
            or (
                bias is not None
                and (bias.dtype != weight.dtype or bias.device != weight.device)
            )
        ):
            raise ValueError("Unexpected inference causal convolution contract")
        batch, channels, _ = hidden.shape
        kernel = weight.shape[-1]
        result = torch.empty(
            (batch, channels, output_tokens), dtype=hidden.dtype, device=hidden.device
        )
        # Depthwise channels are independent. Bound temporary prefill storage;
        # cached single-token decode can process all channels together.
        block_size = 512 if output_tokens > 1 else channels
        check_reference = control["calls"] % 256 == 0 or control["calls"] == 1
        for start in range(0, channels, block_size):
            end = min(start + block_size, channels)
            source = hidden[:, start:end].float()
            if padding:
                source = torch.nn.functional.pad(source, (padding, 0))
            else:
                source = source[..., -(output_tokens + kernel - 1) :]
            coefficients = weight[start:end].float()
            actual = torch.zeros(
                (batch, end - start, output_tokens),
                dtype=torch.float32,
                device=hidden.device,
            )
            # Explicit FP32 products/sums avoid backend-dependent TF32 use.
            for tap in range(kernel):
                actual.add_(
                    source[..., tap : tap + output_tokens]
                    * coefficients[None, :, tap, None]
                )
            if bias is not None:
                actual.add_(bias[None, start:end, None].float())
            if activation is not None:
                actual = modeling.ACT2FN[activation](actual)
            if not torch.isfinite(actual).all():
                control["finite"] = False
                raise RuntimeError("Non-finite FP32 causal convolution")
            if check_reference and start == 0:
                count = min(8, channels)
                # Independent FP64 dot product of the last causal window.
                reference = (
                    source[:, :count, -kernel:].double()
                    * weight[None, :count, :].double()
                ).sum(dim=-1)
                if bias is not None:
                    reference = reference + bias[None, :count].double()
                if activation is not None:
                    reference = modeling.ACT2FN[activation](reference)
                relative = float(
                    (actual[:, :count, -1].double() - reference).norm()
                    / reference.norm().clamp_min(1e-12)
                )
                control["reference_checks"] += 1
                control["maximum_reference_relative_l2"] = max(
                    control["maximum_reference_relative_l2"], relative
                )
                if not math.isfinite(relative) or relative > 1e-5:
                    control["reference_close"] = False
                    raise RuntimeError("FP32 convolution failed its FP64 reference")
            result[:, start:end] = actual.to(hidden.dtype)
        if not torch.isfinite(result).all():
            control["finite"] = False
            raise RuntimeError("Non-finite BF16 causal convolution output")
        control["calls"] += 1
        return result

    def prefill(hidden_states, weight, bias=None, activation=None, **kwargs):
        result = convolve(
            hidden_states,
            weight,
            bias,
            activation,
            padding=weight.shape[-1] - 1,
            output_tokens=hidden_states.shape[-1],
        )
        control["prefill_calls"] += 1
        return result

    def cached(hidden_states, conv_state, weight, bias=None, activation=None):
        if torch.is_grad_enabled():
            raise ValueError("Causal convolution precision is inference-only")
        state_identity = (conv_state.data_ptr(), conv_state.dtype, conv_state.device)
        if conv_state.dtype != weight.dtype or conv_state.device != weight.device:
            raise ValueError("Native BF16 convolution cache is required")
        combined = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
        expected = combined[..., -conv_state.shape[-1] :].clone()
        conv_state.copy_(expected)
        if (
            conv_state.data_ptr(),
            conv_state.dtype,
            conv_state.device,
        ) != state_identity or not torch.equal(conv_state, expected):
            control["cache_exact"] = False
            raise RuntimeError("Causal convolution changed the native cache update")
        control["cache_checks"] += 1
        result = convolve(
            combined,
            weight,
            bias,
            activation,
            padding=0,
            output_tokens=hidden_states.shape[-1],
        )
        control["cached_calls"] += 1
        return result

    modeling.causal_conv1d_fn, modeling.causal_conv1d_update = prefill, cached
    try:
        yield control
    finally:
        modeling.causal_conv1d_fn, modeling.causal_conv1d_update = originals
        control["parameters_unchanged"] = before == versions()
        emit("decode_convolution_precision", **control)
        if not control["parameters_unchanged"]:
            raise RuntimeError("Convolution decode changed stored parameters")


def verify_decode_convolution(torch, modeling):
    """Small GPU numerical/cache control; never a substitute for the Qwen assay."""
    from types import SimpleNamespace

    control = {
        "policy": DECODE_CONVOLUTION_POLICY,
        "cuda_device_count": 2,
        "cases_per_device": 4,
        "finite": True,
        "reference_close": True,
        "cache_exact": True,
        "parameters_unchanged": True,
    }
    with torch.random.fork_rng(devices=[0, 1]), torch.inference_mode():
        for device in range(2):
            torch.manual_seed(20261007)
            weight = torch.randn(16, 4, device=f"cuda:{device}", dtype=torch.bfloat16)
            stored_weight = weight.clone()
            weight_identity = (weight.data_ptr(), weight.dtype, weight.device)
            native_prefill, native_cached = (
                modeling.causal_conv1d_fn,
                modeling.causal_conv1d_update,
            )
            with precise_decode_convolution(
                SimpleNamespace(parameters=lambda: []),
                torch,
                modeling,
                lambda *a, **k: None,
            ) as proof:
                for length in (3, 17):
                    value = torch.randn(
                        1, 16, length, device=weight.device, dtype=weight.dtype
                    )
                    actual = modeling.causal_conv1d_fn(value, weight, activation="silu")
                    reference = torch.nn.functional.silu(
                        torch.nn.functional.conv1d(
                            value.double(),
                            weight.double().unsqueeze(1),
                            padding=3,
                            groups=16,
                        )[..., :length]
                    ).to(value.dtype)
                    if not torch.allclose(actual, reference, rtol=0.01, atol=0.002):
                        raise RuntimeError("GPU convolution prefill control failed")
                state = torch.randn(1, 16, 4, device=weight.device, dtype=weight.dtype)
                for _ in range(2):
                    value = torch.randn(
                        1, 16, 1, device=weight.device, dtype=weight.dtype
                    )
                    expected = torch.cat([state, value], dim=-1)
                    reference = torch.nn.functional.silu(
                        torch.nn.functional.conv1d(
                            expected.double(),
                            weight.double().unsqueeze(1),
                            groups=16,
                        )[..., -1:]
                    ).to(value.dtype)
                    actual = modeling.causal_conv1d_update(
                        value, state, weight, activation="silu"
                    )
                    if not torch.equal(state, expected[..., -4:]) or not torch.allclose(
                        actual, reference, rtol=0.01, atol=0.002
                    ):
                        raise RuntimeError("GPU convolution cache control failed")
            if (
                proof["calls"] != 4
                or proof["prefill_calls"] != 2
                or proof["cached_calls"] != 2
                or proof["cache_checks"] != 2
                or not all(
                    proof[k]
                    for k in (
                        "finite",
                        "reference_close",
                        "cache_exact",
                        "parameters_unchanged",
                    )
                )
                or modeling.causal_conv1d_fn is not native_prefill
                or modeling.causal_conv1d_update is not native_cached
                or not torch.equal(weight, stored_weight)
                or (weight.data_ptr(), weight.dtype, weight.device) != weight_identity
            ):
                raise RuntimeError("GPU convolution precision scope control failed")
    return control


@contextmanager
def precise_decode_attention(model, torch, registry, emit):
    """Use FP32 math SDPA in decoder attention, retaining cache and weight dtypes."""
    original = registry.get_interface("sdpa", None)
    if original is None:
        raise ValueError("The native SDPA interface is required")
    parameters = tuple(model.parameters())

    def versions():
        return tuple(
            (id(p), p.data_ptr(), p._version, p.dtype, p.device) for p in parameters
        )

    before = versions()
    control = {
        "policy": DECODE_ATTENTION_POLICY,
        "calls": 0,
        "prefill_calls": 0,
        "cached_calls": 0,
        "reference_checks": 0,
        "maximum_reference_relative_l2": 0.0,
        "finite": True,
        "reference_close": True,
        "parameters_unchanged": False,
    }

    def attention(module, query, key, value, attention_mask=None, **kwargs):
        # Vision attention and all training keep their original implementations.
        if type(module).__name__ != "Qwen3_5Attention":
            return original(module, query, key, value, attention_mask, **kwargs)
        if (
            module.training
            or query.dtype != torch.bfloat16
            or kwargs.get("dropout", 0.0) != 0.0
            or kwargs.get("cache") is not None
            or kwargs.get("position_bias") is not None
        ):
            raise ValueError("Unexpected decoder SDPA contract")
        mask = attention_mask
        float_mask = (
            mask.float() if mask is not None and mask.dtype != torch.bool else mask
        )
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            output, weights = original(
                module, query.float(), key.float(), value.float(), float_mask, **kwargs
            )
        if not torch.isfinite(output).all():
            control["finite"] = False
            raise RuntimeError("Non-finite FP32 decoder attention")
        # Check a bounded row against FP64 using the same keys, mask and scale.
        # No extra generation, reference text or retry policy enters this check.
        if control["calls"] % 256 == 0 or control["calls"] == 1:
            reference_key, reference_value = key, value
            reference_mask = mask
            if mask is not None and mask.shape[-2] != 1:
                reference_mask = mask[..., -1:, :]
            causal = kwargs.get("is_causal", getattr(module, "is_causal", True))
            if query.shape[2] > 1 and mask is None and causal:
                reference_key = key[:, :, : query.shape[2], :]
                reference_value = value[:, :, : query.shape[2], :]
            if reference_mask is not None and reference_mask.dtype != torch.bool:
                reference_mask = reference_mask.double()
            reference_kwargs = {**kwargs, "is_causal": False}
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                reference, _ = original(
                    module,
                    query[:, :, -1:, :].double(),
                    reference_key.double(),
                    reference_value.double(),
                    reference_mask,
                    **reference_kwargs,
                )
            actual = output[:, -1:, :, :].double()
            relative = float(
                (actual - reference).norm() / reference.norm().clamp_min(1e-12)
            )
            control["maximum_reference_relative_l2"] = max(
                control["maximum_reference_relative_l2"], relative
            )
            control["reference_checks"] += 1
            if not math.isfinite(relative) or relative > 1e-5:
                control["reference_close"] = False
                raise RuntimeError(
                    "FP32 decoder attention failed its FP64 row reference"
                )
        control["calls"] += 1
        control["prefill_calls" if query.shape[2] > 1 else "cached_calls"] += 1
        return output.to(query.dtype), weights

    registry.register("sdpa", attention)
    try:
        yield control
    finally:
        registry.register("sdpa", original)
        control["parameters_unchanged"] = before == versions()
        emit("decode_attention_precision", **control)
        if not control["parameters_unchanged"]:
            raise RuntimeError("Decode changed stored parameters")


@contextmanager
def precise_decode_head(model, torch, emit):
    """Keep greedy scores in FP32; preserve stored weights and routing hooks."""
    head = model.get_output_embeddings()
    if (
        not isinstance(head, torch.nn.Linear)
        or head.weight.requires_grad
        or head.weight.dtype != torch.bfloat16
        or head.weight.device.type != "cuda"
        or head.bias is not None
    ):
        raise ValueError("Decode precision requires the frozen native BF16 linear head")
    slot = "_old_forward" if hasattr(head, "_hf_hook") else "forward"
    original = getattr(head, slot)
    weight = head.weight.detach().float()
    for start in range(0, head.weight.shape[0], 4096):
        if not torch.equal(
            weight[start : start + 4096].to(head.weight.dtype),
            head.weight[start : start + 4096],
        ):
            raise RuntimeError("FP32 decode copy changed the stored output weights")
    control = {
        "policy": DECODE_PRECISION_POLICY,
        "calls": 0,
        "finite": True,
        "native_bf16_ties": 0,
        "native_argmax_changes": 0,
        "reference_exact": True,
        "stored_weights_exact": True,
    }

    def forward(module, hidden):
        scores = torch.nn.functional.linear(hidden.float(), weight)
        last = scores[:, -1:, :]
        if not torch.isfinite(last).all():
            raise RuntimeError("Non-finite full-precision generation scores")
        # Observe native scores without generating or selecting another attempt.
        # Fixed intervals do not inspect references, output text or CER.
        if control["calls"] % 16 == 0:
            native = original(hidden[:, -1:, :])
            reference = torch.nn.functional.linear(hidden[:, -1:, :].float(), weight)
            if not torch.equal(last, reference):
                control["reference_exact"] = False
                raise RuntimeError(
                    "Decode output differs from the FP32 linear reference"
                )
            control["native_bf16_ties"] += int(
                (native == native.max(dim=-1, keepdim=True).values).sum() > 1
            )
            control["native_argmax_changes"] += int(
                (native.argmax(dim=-1) != last.argmax(dim=-1)).any()
            )
        control["calls"] += 1
        return scores

    setattr(head, slot, MethodType(forward, head))
    try:
        yield control
    finally:
        setattr(head, slot, original)
        emit("decode_precision", **control, stored_weight_dtype=str(head.weight.dtype))
        weight = None
        torch.cuda.empty_cache()


def move_adamw_moments(optimizer, *, to_cpu):
    """Move only idle moments, keeping dtype, step counters and AdamW unchanged."""
    moved_bytes = 0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            if state.keys() - {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}:
                raise ValueError("Unknown AdamW state cannot be relocated safely")
            for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if name in state:
                    moment = state[name]
                    target = "cpu" if to_cpu else parameter.device
                    state[name] = moment.to(device=target)
                    moved_bytes += moment.numel() * moment.element_size()
    return moved_bytes


def verify_optimizer_parking(torch):
    """Compare four AdamW steps and every moment against GPU-resident AdamW."""
    for device in range(2):
        initial = torch.linspace(-1, 1, 256, device=f"cuda:{device}")
        actual = torch.nn.Parameter(initial.clone())
        reference = torch.nn.Parameter(initial.clone())
        optimizer = torch.optim.AdamW([actual], lr=5e-5, weight_decay=0.0)
        control = torch.optim.AdamW([reference], lr=5e-5, weight_decay=0.0)
        for step in range(4):
            gradient = initial.cos() * (step + 1)
            actual.grad, reference.grad = gradient.clone(), gradient.clone()
            move_adamw_moments(optimizer, to_cpu=False)
            optimizer.step()
            control.step()
            optimizer.zero_grad(set_to_none=True)
            control.zero_grad(set_to_none=True)
            before = exact_state_digest(optimizer.state_dict(), torch)
            move_adamw_moments(optimizer, to_cpu=True)
            if (
                not torch.isfinite(actual).all()
                or not torch.equal(actual, reference)
                or exact_state_digest(optimizer.state_dict(), torch) != before
                or before != exact_state_digest(control.state_dict(), torch)
            ):
                raise RuntimeError("Relocating AdamW moments changed the update/state")
    return {
        "policy": OPTIMIZER_MEMORY_POLICY,
        "cuda_device_count": 2,
        "updates_per_device": 4,
        "parameters_exact": True,
        "moments_exact": True,
        "finite": True,
    }


def unit_groups(units, seed, epoch):
    if (
        epoch not in (1, 2, 3, 4)
        or len(units) != 125
        or len({u["page_id"] for u in units}) != 125
    ):
        raise ValueError("Four epochs over every Berrutti source page are required")
    ordered = sorted(units, key=lambda u: u["page_id"])
    random.Random(seed + epoch).shuffle(ordered)
    return [ordered[start : start + 8] for start in range(0, 125, 8)]


def select_epoch(validations):
    if {row["epoch"] for row in validations} != {1, 2, 3, 4} or len(validations) != 4:
        return None
    eligible = [
        row
        for row in validations
        if row["split"] == "validation"
        and row["pages"] == 25
        and row["all_pages_completed"] is True
        and row["failed_pages"] == 0
        and math.isfinite(row["cer"])
        and row["cer"] >= 0
    ]
    return min(eligible, key=lambda r: (r["cer"], r["epoch"]), default=None)


def verify_mixed_image_hashes(manifest):
    """Read every sealed image in full, with four bounded concurrent readers."""
    expected = {}
    for entry in manifest["epochs"]:
        for unit in entry["units"]:
            for example in unit["luisa"] + unit["synthetic"]:
                path, digest = str(Path(example["image"])), example["image_sha256"]
                if path in expected and expected[path] != digest:
                    raise ValueError("Conflicting hashes for the same mixed image")
                expected[path] = digest

    def check_image(row):
        path, digest = row
        if file_sha256(Path(path)) != digest:
            raise ValueError("Mixed input image changed")

    # This is the same complete SHA256 operation as the serial verifier. Only
    # independent file reads overlap; no corpus, byte or digest is skipped.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for _ in pool.map(check_image, expected.items()):
            pass
    return len(expected)


def verify_manifest(config, emit=None):
    if emit is not None:
        emit("phase", phase="sealed_input_verification")
    cohort, pages = verify_inputs(config)
    root = Path(config["mix_root"])
    receipt = json.loads((root / "completed.json").read_text())
    if receipt.get("success") is not True or receipt["manifest_sha256"] != file_sha256(
        root / "manifest.json"
    ):
        raise ValueError("Mixed corpus has no valid closed preparation receipt")
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest["protocol"] != EXTENSION
        or manifest["source_cohort_sha256"] != config["cohort_sha256"]
        or manifest["total_optimizer_steps"] != config["total_optimizer_steps"]
    ):
        raise ValueError("Mixed curriculum differs from the frozen configuration")
    if (
        manifest["seed"] != config["seed"]
        or [e["epoch"] for e in manifest["epochs"]] != [1, 2, 3, 4]
        or config["total_optimizer_steps"] != 64
    ):
        raise ValueError("The complete four-epoch mixed schedule is required")
    if manifest["records_sha256"] != file_sha256(
        Path(config["luisa_root"]) / "records.jsonl"
    ):
        raise ValueError("Original LUISA references/index changed")
    if manifest["audit"]["selection"] != config["luisa_selection"]:
        raise ValueError("LUISA inclusion decision changed")
    expected_ids = {p["page_id"] for p in pages["train"]}
    expected_luisa = None
    for entry in manifest["epochs"]:
        if {u["page_id"] for u in entry["units"]} != expected_ids:
            raise ValueError("Mixed curriculum lost a Berrutti training page")
        unit_groups(entry["units"], config["seed"], entry["epoch"])
        luisa_ids = [row["line_id"] for unit in entry["units"] for row in unit["luisa"]]
        if entry["epoch"] <= 3:
            if (
                len(luisa_ids) != len(set(luisa_ids))
                or len(luisa_ids) != manifest["audit"]["luisa_selected"]
            ):
                raise ValueError("Missing or repeated LUISA line in a mixed epoch")
            if expected_luisa is None:
                expected_luisa = set(luisa_ids)
            if set(luisa_ids) != expected_luisa:
                raise ValueError("LUISA coverage differs across epochs")
        elif luisa_ids or any(unit["synthetic"] for unit in entry["units"]):
            raise ValueError("Consolidation must contain only Berrutti")
    if emit is not None:
        emit("phase", phase="mixed_input_hashes")
    checked = verify_mixed_image_hashes(manifest)
    if emit is not None:
        emit(
            "mixed_input_hashes_verified",
            unique_images=checked,
            readers=4,
            all_bytes_hashed=True,
            expected_hashes_exact=True,
        )
    return cohort, pages, manifest


class MixedRuntime(Runtime):
    def __init__(self, config, root, emit, manifest):
        super().__init__(config, root, emit)
        self.policy = InstructSamplingPolicy()
        self.decode_sampling_controls = []
        self.identity = cache_key(
            {
                "base_runtime": self.identity,
                "generation_policy": self.policy.metadata(),
                "mixed_manifest": file_sha256(
                    Path(config["mix_root"]) / "manifest.json"
                ),
                "steps": config["total_optimizer_steps"],
                "extension": EXTENSION,
            }
        )
        self.prepared = OrderedDict()
        self.manifest = manifest
        emit(
            "extension_runtime",
            runtime_identity=self.identity,
            initial_adapter_sha256=self.initial_adapter_sha256,
            total_optimizer_steps=64,
        )
        self.decode_precision_controls = []
        self.decode_attention_controls = []
        self.decode_convolution_controls = []
        self.decode_linear_controls = []

    def decode(self, page, root):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.qwen3_5 import modeling_qwen3_5

        from finetune.recipes.berrutti_qwen38.runner import resumed_attempts

        # Consumed attempts retain their original page/attempt seed on resume.
        previous = resumed_attempts(root, self.policy)
        with (
            seeded_sampling(
                self.model,
                self.torch,
                self.policy,
                page,
                self.config["seed"],
                self.emit,
                completed_attempts=len(previous),
            ) as sampling_control,
            precise_decode_linear(
                self.model, self.torch, modeling_qwen3_5, self.emit
            ) as linear_control,
            precise_decode_convolution(
                self.model, self.torch, modeling_qwen3_5, self.emit
            ) as convolution_control,
            precise_decode_attention(
                self.model, self.torch, ALL_ATTENTION_FUNCTIONS, self.emit
            ) as attention_control,
            precise_decode_head(self.model, self.torch, self.emit) as control,
        ):
            result = super().decode(page, root)
        self.decode_precision_controls.append(dict(control))
        self.decode_attention_controls.append(dict(attention_control))
        self.decode_convolution_controls.append(dict(convolution_control))
        self.decode_linear_controls.append(dict(linear_control))
        self.decode_sampling_controls.append(dict(sampling_control))
        return result

    def prepare(self, page, line=None, *, training=False):
        result = super().prepare(page, line, training=training)
        key = (
            page["page_id"],
            None if line is None else line["region_index"],
            training,
        )
        self.prepared.move_to_end(key)
        while len(self.prepared) > 64:
            self.prepared.popitem(last=False)
        return result

    def park_optimizer(self):
        before = self.memory_snapshot()
        moments_bytes = move_adamw_moments(self.optimizer, to_cpu=True)
        self.torch.cuda.empty_cache()
        self.emit(
            "optimizer_memory",
            policy=OPTIMIZER_MEMORY_POLICY,
            moments_bytes=moments_bytes,
            before=before,
            after=self.memory_snapshot(),
        )

    def optimizer_moments_on_cpu(self):
        moments = [
            value
            for state in self.optimizer.state.values()
            for name, value in state.items()
            if name != "step"
        ]
        return bool(moments) and all(v.device.type == "cpu" for v in moments)

    def restore(self, state, *, optimizer=True):
        super().restore(state, optimizer=optimizer)
        if optimizer:
            self.park_optimizer()

    def train_units(self, units, real, regions, state):
        tick = time.monotonic()
        self.phase = f"training_{STAGE}_epoch_{state['epoch']}"
        self.optimizer.zero_grad(set_to_none=True)
        self.park_optimizer()
        counts = {"page": 0, "berrutti_lines": 0, "luisa": 0, "synthetic": 0}
        tokens, loss_sum = 0, 0.0
        weights = (
            self.manifest["family_weights_mixed"]
            if state["epoch"] <= 3
            else self.manifest["family_weights_consolidation"]
        )
        for position, unit in enumerate(units, state["group"] * 8 + 1):
            page = real[unit["page_id"]]
            families = {
                "page": [self.prepare(page, training=True)],
                "berrutti_lines": [
                    self.prepare(page, line, training=True)
                    for line in regions[unit["page_id"]]
                ],
            }
            if state["epoch"] <= 3:
                families.update(
                    luisa=[self.prepare(row, training=True) for row in unit["luisa"]],
                    synthetic=[
                        self.prepare(row, training=True) for row in unit["synthetic"]
                    ],
                )
            streams = []
            for name, examples in families.items():
                if not examples:
                    raise ValueError(
                        "Every mixed training unit needs all declared families"
                    )
                total = sum(example["tokens"] for example in examples)
                streams.append(
                    [
                        (example, weights[name] * example["tokens"] / total)
                        for example in examples
                    ]
                )
                counts[name] += len(examples)
                tokens += total
            for row in zip_longest(*streams):
                for pair in row:
                    if pair is not None:
                        example, weight = pair
                        loss_sum += weight * self.backward(example, weight / len(units))
            self.emit(
                "sample_completed",
                epoch=state["epoch"],
                stage=STAGE,
                source_page=position,
                samples_completed=sum(counts.values()),
                representation="interleaved",
            )
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if any(
            p.grad is None or not self.torch.isfinite(p.grad).all() for p in parameters
        ):
            raise RuntimeError("Missing or non-finite mixed adapter gradient")
        norm = self.torch.nn.utils.clip_grad_norm_(
            parameters, 1.0, error_if_nonfinite=True, foreach=False
        )
        lr = learning_rate(
            state["global_step"],
            self.config["total_optimizer_steps"],
            self.config["learning_rate"],
            self.config["minimum_learning_rate"],
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        move_adamw_moments(self.optimizer, to_cpu=False)
        self.optimizer.step()
        if any(not self.torch.isfinite(p).all() for p in parameters):
            raise RuntimeError("Non-finite mixed adapter after update")
        self.optimizer.zero_grad(set_to_none=True)
        self.park_optimizer()
        return {
            "samples": sum(counts.values()),
            "source_pages": len(units),
            "supervised_tokens": tokens,
            "unit_loss_sum": loss_sum,
            "seconds": time.monotonic() - tick,
            **{f"count_{k}": v for k, v in counts.items()},
        }, {"loss": loss_sum / len(units), "lr": lr, "gradient_norm": float(norm)}


def run_training(runtime, cohort, pages, manifest, should_yield):
    root = runtime.root
    baseline = json.loads(
        (
            Path(runtime.config["operation_root"]) / "baseline" / "baseline.json"
        ).read_text()
    )
    if (
        baseline["runtime_identity"] != runtime.identity
        or baseline["optimizer_steps"] != 0
    ):
        raise ValueError("Experiment four requires its own identical fresh baseline")
    if not (root / "baseline.json").exists():
        closed_json(root / "baseline.json", baseline)
    state_path = root / "state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {
            "epoch": 1,
            "group": 0,
            "global_step": 0,
            "phase": "training",
            "checkpoint": None,
            "epoch_checkpoints": {},
            "stats": {},
            "runtime_identity": runtime.identity,
        }
    )
    runtime.restore(state)
    real = {p["page_id"]: p for p in pages["train"]}
    regions = {p["page_id"]: p["regions"] for p in cohort["pages"]}
    while state["epoch"] <= 4:
        epoch = state["epoch"]
        entry = manifest["epochs"][epoch - 1]
        groups = unit_groups(entry["units"], runtime.config["seed"], epoch)
        if state["phase"] == "training":
            while state["group"] < 16:
                if should_yield():
                    runtime.progress(state)
                    return 85
                stats, metrics = runtime.train_units(
                    groups[state["group"]], real, regions, state
                )
                for key, value in stats.items():
                    state["stats"][key] = state["stats"].get(key, 0) + value
                state["global_step"] += 1
                state["group"] += 1
                if state["group"] == 16:
                    state["phase"] = "validation"
                elif state["group"] % 4 == 0:
                    state["phase"] = "progress_validation"
                checkpoint = runtime.checkpoint(state)
                runtime.emit(
                    "optimizer_step",
                    epoch=epoch,
                    stage=STAGE,
                    page=min(state["group"] * 8, 125),
                    total=125,
                    optimizer_steps=state["global_step"],
                    **metrics,
                )
                if state["phase"] == "validation":
                    state["epoch_checkpoints"][str(epoch)] = checkpoint
                    runtime.progress(state)
                if state["phase"] != "training":
                    break
        if state["phase"] == "progress_validation":
            marker = 100 * epoch + state["group"]
            validation = runtime.evaluate(
                pages["validation"],
                marker,
                "validation",
                STAGE + "_progress",
                runtime.adapter_digest(),
                should_yield,
            )
            if validation is None:
                runtime.progress(state)
                return 85
            path = root / "epochs" / f"epoch-{marker:02d}.json"
            if not path.exists():
                closed_json(
                    path,
                    {
                        "epoch": marker,
                        "training_epoch": epoch,
                        "optimizer_steps": state["global_step"],
                        "validation": validation,
                    },
                )
            state["phase"] = "training"
            runtime.progress(state)
            continue
        if state["phase"] != "validation":
            raise ValueError("Unexpected mixed continuation phase")
        if str(epoch) not in state["epoch_checkpoints"]:
            state["epoch_checkpoints"][str(epoch)] = state["checkpoint"]
        stats = state["stats"]
        expected_luisa = manifest["audit"]["luisa_selected"] if epoch <= 3 else 0
        if (
            stats["samples"] != entry["samples"]
            or stats["source_pages"] != 125
            or state["global_step"] != epoch * 16
            or any(
                stats["count_" + name] != count
                for name, count in {
                    "page": 125,
                    "berrutti_lines": 5212,
                    "luisa": expected_luisa,
                    "synthetic": entry["synthetic_pages"],
                }.items()
            )
        ):
            raise RuntimeError(
                "Mixed epoch did not cover every required example exactly once"
            )
        training_path = root / "epochs" / f"epoch-{epoch:02d}-training.json"
        if not training_path.exists():
            training = {
                "epoch": epoch,
                "stage": STAGE,
                "source_pages": 125,
                "samples": stats["samples"],
                "optimizer_steps": 16,
                "mean_loss": stats["unit_loss_sum"] / 125,
                "supervised_tokens": stats["supervised_tokens"],
                "seconds": stats["seconds"],
                "family_counts": {
                    k: v for k, v in stats.items() if k.startswith("count_")
                },
            }
            closed_json(training_path, training)
            runtime.emit("training_epoch", **training)
        validation = runtime.evaluate(
            pages["validation"],
            epoch,
            "validation",
            STAGE,
            runtime.adapter_digest(),
            should_yield,
        )
        if validation is None:
            runtime.progress(state)
            return 85
        path = root / "epochs" / f"epoch-{epoch:02d}.json"
        if not path.exists():
            closed_json(path, {"epoch": epoch, "validation": validation})
        state.update(epoch=epoch + 1, group=0, phase="training", stats={})
        runtime.progress(state)
    validations = [
        json.loads((root / "epochs" / f"epoch-{e:02d}.json").read_text())["validation"]
        for e in (1, 2, 3, 4)
    ]
    best = select_epoch(validations)
    if best is None:
        raise RuntimeError("No complete mixed validation is eligible")
    selected_checkpoint = state["epoch_checkpoints"][str(best["epoch"])]
    runtime.restore({**state, "checkpoint": selected_checkpoint}, optimizer=False)
    test = runtime.evaluate(
        pages["test"],
        best["epoch"],
        "test",
        STAGE,
        runtime.adapter_digest(),
        should_yield,
    )
    if test is None:
        return 85
    path = root / "epochs" / f"epoch-{best['epoch']:02d}-selected-test.json"
    if not path.exists():
        closed_json(path, {"epoch": best["epoch"], "test": test})
    summary = {
        "success": True,
        "result_complete": test["all_pages_completed"],
        "best_epoch": best["epoch"],
        "best_validation_cer": best["cer"],
        "test_cer_at_selected_epoch": test["cer"],
        "selected_test_failed_pages": test["failed_pages"],
        "optimizer_steps": state["global_step"],
        "selected_checkpoint": selected_checkpoint,
        "source_document_independence_certified": False,
        "validation_interpretation": manifest["audit"]["validation_interpretation"],
    }
    closed_json(root / "completed.json", summary)
    runtime.emit("completed", **summary)
    return 0


def assay_examples(config, pages, manifest):
    """Keep the original difficult probes, independently of their generation outcome."""
    entry = manifest["epochs"][0]
    chosen = [
        unit for unit in entry["units"] if unit["page_id"] in config["smoke_page_ids"]
    ]
    if len(chosen) != 2:
        raise ValueError("Mixed assay needs both representative Berrutti pages")
    all_external = [row for unit in entry["units"] for row in unit["luisa"]]
    longest = sorted(all_external, key=lambda r: len(r["reference_text"]), reverse=True)
    all_synthetic = [row for unit in entry["units"] for row in unit["synthetic"]]
    by_columns = {
        n: max(
            (row for row in all_synthetic if row["columns"] == n),
            key=lambda r: len(r["reference_text"]),
        )
        for n in (1, 2)
    }
    units = [
        {
            **unit,
            "luisa": longest[:2] if i == 0 else longest[2:4],
            "synthetic": [by_columns[i + 1]],
        }
        for i, unit in enumerate(chosen)
    ]
    samples = [
        pages["validation"][0],
        pages["test"][0],
        by_columns[1],
        by_columns[2],
        longest[0],
    ]
    return units, samples


def assay_contract(config, pages, manifest):
    units, samples = assay_examples(config, pages, manifest)
    return {
        "protocol": ASSAY_PROTOCOL,
        "generation_policy": InstructSamplingPolicy().metadata(),
        "training_epochs": [1, 1, 4],
        "optimizer_steps": 3,
        "optimizer_memory_policy": OPTIMIZER_MEMORY_POLICY,
        "decode_precision_policy": DECODE_PRECISION_POLICY,
        "decode_attention_policy": DECODE_ATTENTION_POLICY,
        "decode_convolution_policy": DECODE_CONVOLUTION_POLICY,
        "decode_linear_policy": DECODE_LINEAR_POLICY,
        "source_pages": [unit["page_id"] for unit in units],
        "generation_probes": [
            {
                "page_id": sample["page_id"],
                "image_sha256": sample["image_sha256"],
                "kind": sample.get("source", "berrutti_page"),
                "columns": sample.get("columns"),
            }
            for sample in samples
        ],
    }


def exact_state_digest(value, torch):
    """Fingerprint every AdamW moment and RNG, including tensor dtype and shape."""
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(
                json.dumps(["tensor", str(tensor.dtype), list(tensor.shape)]).encode()
            )
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            digest.update(str(len(item)).encode())
            for child in item:
                visit(child)
        else:
            digest.update(
                json.dumps([type(item).__name__, item], allow_nan=False).encode()
            )

    visit(value)
    return digest.hexdigest()


def validate_assay_receipt(receipt, contract):
    """A successful runtime or partial generation never substitutes for this gate."""
    flags = (
        "success",
        "diagnostic_only",
        "checkpoint_reload",
        "generation_checked",
        "training_finite",
        "adamw_exact",
        "scheduler_exact",
        "rng_exact",
        "optimizer_parking_exact",
        "decode_precision_checked",
        "decode_attention_checked",
        "decode_convolution_checked",
        "decode_linear_checked",
    )
    if (
        contract["protocol"] != ASSAY_PROTOCOL
        or contract.get("optimizer_memory_policy") != OPTIMIZER_MEMORY_POLICY
        or contract.get("decode_precision_policy") != DECODE_PRECISION_POLICY
        or contract.get("decode_attention_policy") != DECODE_ATTENTION_POLICY
        or contract.get("decode_convolution_policy") != DECODE_CONVOLUTION_POLICY
        or contract.get("decode_linear_policy") != DECODE_LINEAR_POLICY
        or receipt.get("assay_contract") != contract
        or receipt.get("optimizer_steps") != 3
        or receipt.get("cuda_device_count") != 2
        or not all(receipt.get(flag) is True for flag in flags)
    ):
        raise ValueError("The complete mixed/consolidation GPU assay is required")
    if contract.get("generation_policy") != InstructSamplingPolicy().metadata():
        raise ValueError("Approved generation policy differs")
    sampling = receipt.get("decode_sampling_controls", [])
    if len(sampling) != 6 or any(
        row.get("policy") != InstructSamplingPolicy().metadata()["version"]
        or row.get("calls") not in (1, 2)
        or len(row.get("seeds", [])) != row["calls"]
        or row.get("attempt_offset") != 0
        or row.get("attempts") != list(range(1, row["calls"] + 1))
        or row.get("presence_calls", 0) < 1
        or any(
            row.get(flag) is not True
            for flag in ("inputs_unchanged", "rng_restored", "interfaces_restored")
        )
        for row in sampling
    ):
        raise ValueError("All six seeded sampling controls must pass")
    validate_linear_controls(receipt)
    for row, probe in zip(
        sampling,
        [contract["generation_probes"][3], *contract["generation_probes"]],
        strict=True,
    ):
        if (
            row["page_id"] != probe["page_id"]
            or row["image_sha256"] != probe["image_sha256"]
            or row["source_seed"] != 20261007
            or row["parameters"] != InstructSamplingPolicy().metadata()["sampling"]
            or row["seeds"]
            != [
                page_seed(20261007, probe, attempt)
                for attempt in range(1, row["calls"] + 1)
            ]
        ):
            raise ValueError("Sampling seeds or fixed page/parameters differ")
    control = receipt.get("optimizer_parking_control", {})
    if (
        control.get("policy") != OPTIMIZER_MEMORY_POLICY
        or control.get("cuda_device_count") != 2
        or control.get("updates_per_device") != 4
        or not all(
            control.get(flag) is True
            for flag in ("parameters_exact", "moments_exact", "finite")
        )
    ):
        raise ValueError("AdamW relocation has not passed its GPU identity control")
    updates = receipt.get("training_updates", [])
    if len(updates) != 3:
        raise ValueError("Missing diagnostic updates")
    for step, (row, phase) in enumerate(
        zip(updates, ("mixed", "mixed", "consolidation"), strict=True), 1
    ):
        counts, metrics = row["training_counts"], row["metrics"]
        if (
            row["optimizer_step"] != step
            or row["phase"] != phase
            or row.get("optimizer_moments_on_cpu") is not True
            or not all(
                row.get(flag) is True
                for flag in (
                    "checkpoint_reload",
                    "adamw_exact",
                    "scheduler_exact",
                    "rng_exact",
                )
            )
            or counts["source_pages"] != 2
            or counts["count_page"] != 2
            or counts["count_berrutti_lines"] <= 0
            or any(
                not math.isfinite(metrics[k]) for k in ("loss", "lr", "gradient_norm")
            )
            or metrics["lr"] <= 0
            or metrics["gradient_norm"] < 0
            or (
                phase == "mixed"
                and (counts["count_luisa"] != 4 or counts["count_synthetic"] != 2)
            )
            or (
                phase == "consolidation"
                and (counts["count_luisa"] != 0 or counts["count_synthetic"] != 0)
            )
        ):
            raise ValueError("Diagnostic update or exact restoration is incomplete")
    generations = receipt.get("generations", [])
    precision = receipt.get("decode_precision_controls", [])
    if len(precision) != 6 or any(
        row.get("policy") != DECODE_PRECISION_POLICY
        or row.get("calls", 0) < 1
        or row.get("finite") is not True
        or row.get("reference_exact") is not True
        or row.get("stored_weights_exact") is not True
        for row in precision
    ):
        raise ValueError(
            "All six decode controls must verify the FP32 output projection"
        )
    if len(generations) != 5:
        raise ValueError("All five original generation probes are required")
    convolution_identity = receipt.get("decode_convolution_control", {})
    if (
        convolution_identity.get("policy") != DECODE_CONVOLUTION_POLICY
        or convolution_identity.get("cuda_device_count") != 2
        or convolution_identity.get("cases_per_device") != 4
        or not all(
            convolution_identity.get(k) is True
            for k in (
                "finite",
                "reference_close",
                "cache_exact",
                "parameters_unchanged",
            )
        )
    ):
        raise ValueError("Missing GPU convolution numerical/cache control")
    convolution = receipt.get("decode_convolution_controls", [])
    if len(convolution) != 6 or any(
        row.get("policy") != DECODE_CONVOLUTION_POLICY
        or any(
            row.get(k, 0) < 1
            for k in (
                "calls",
                "prefill_calls",
                "cached_calls",
                "reference_checks",
                "cache_checks",
            )
        )
        or any(
            row.get(k) is not True
            for k in (
                "finite",
                "reference_close",
                "cache_exact",
                "parameters_unchanged",
            )
        )
        or not math.isfinite(row.get("maximum_reference_relative_l2", float("inf")))
        or not 0 <= row["maximum_reference_relative_l2"] <= 1e-5
        for row in convolution
    ):
        raise ValueError("All six causal convolution controls must pass")
    attention = receipt.get("decode_attention_controls", [])
    if len(attention) != 6 or any(
        row.get("policy") != DECODE_ATTENTION_POLICY
        or any(
            row.get(key, 0) < 1
            for key in ("calls", "prefill_calls", "cached_calls", "reference_checks")
        )
        or any(
            row.get(key) is not True
            for key in ("finite", "reference_close", "parameters_unchanged")
        )
        or not math.isfinite(row.get("maximum_reference_relative_l2", float("inf")))
        or not 0 <= row["maximum_reference_relative_l2"] <= 1e-5
        for row in attention
    ):
        raise ValueError("All six decoder attention controls must pass")
    for actual, expected in zip(
        generations, contract["generation_probes"], strict=True
    ):
        if (
            any(actual.get(key) != value for key, value in expected.items())
            or actual.get("all_pages_completed") is not True
            or actual.get("failed_pages") != 0
            or actual.get("retried_pages") not in (0, 1)
        ):
            raise ValueError("Generation probe changed or did not complete")


def smoke(runtime, cohort, pages, manifest):
    from peft import set_peft_model_state_dict
    from transformers.models.qwen3_5 import modeling_qwen3_5

    units, samples = assay_examples(runtime.config, pages, manifest)
    contract = assay_contract(runtime.config, pages, manifest)
    initial = runtime.adapter()
    parking_control = verify_optimizer_parking(runtime.torch)
    runtime.torch.cuda.empty_cache()
    runtime.emit("optimizer_parking_identity", diagnostic_only=True, **parking_control)
    convolution_control = verify_decode_convolution(runtime.torch, modeling_qwen3_5)
    runtime.emit(
        "decode_convolution_identity", diagnostic_only=True, **convolution_control
    )
    linear_control = verify_decode_linear(runtime.torch, modeling_qwen3_5)
    runtime.emit("decode_linear_identity", diagnostic_only=True, **linear_control)
    # Distinguish a pre-existing decode failure from an update/restore regression.
    # This diagnostic does not relax the five mandatory post-training probes.
    rng_before = runtime.rng()
    initial_generation = runtime.evaluate(
        [samples[3]],
        0,
        "diagnostic",
        "experiment4_smoke_initial",
        runtime.initial_adapter_sha256,
        lambda: False,
    )
    runtime.restore_rng(rng_before)
    runtime.emit(
        "assay_initial_generation",
        all_pages_completed=initial_generation["all_pages_completed"],
        failed_pages=initial_generation["failed_pages"],
        retried_pages=initial_generation["retried_pages"],
        diagnostic_only=True,
    )
    state = {
        "epoch": 1,
        "group": 0,
        "global_step": 0,
        "runtime_identity": runtime.identity,
    }
    training = []
    for epoch in contract["training_epochs"]:
        state["epoch"] = epoch
        stats, metrics = runtime.train_units(
            units,
            {p["page_id"]: p for p in pages["train"]},
            {p["page_id"]: p["regions"] for p in cohort["pages"]},
            state,
        )
        state["global_step"] += 1
        saved = runtime.adapter_digest()
        rng_digest = exact_state_digest(runtime.rng(), runtime.torch)
        optimizer_digest = exact_state_digest(
            runtime.optimizer.state_dict(), runtime.torch
        )
        runtime.checkpoint(state)
        set_peft_model_state_dict(runtime.model, initial)
        runtime.optimizer = runtime.new_optimizer()
        random.random()
        runtime.torch.rand(3)
        for device in range(2):
            runtime.torch.rand(3, device=f"cuda:{device}")
        runtime.restore(state)
        if (
            runtime.adapter_digest() != saved
            or exact_state_digest(runtime.rng(), runtime.torch) != rng_digest
            or exact_state_digest(runtime.optimizer.state_dict(), runtime.torch)
            != optimizer_digest
            or runtime.optimizer.param_groups[0]["lr"]
            != learning_rate(
                state["global_step"] - 1,
                runtime.config["total_optimizer_steps"],
                runtime.config["learning_rate"],
                runtime.config["minimum_learning_rate"],
            )
        ):
            raise RuntimeError("Mixed checkpoint restoration was not exact")
        row = {
            "optimizer_step": state["global_step"],
            "phase": "mixed" if epoch <= 3 else "consolidation",
            "training_counts": stats,
            "metrics": metrics,
            "checkpoint_reload": True,
            "adamw_exact": True,
            "scheduler_exact": True,
            "rng_exact": True,
            "optimizer_moments_on_cpu": runtime.optimizer_moments_on_cpu(),
        }
        training.append(row)
        runtime.emit("assay_update_restored", diagnostic_only=True, **row)
    generations = []
    for epoch, sample in enumerate(samples, 1):
        result = runtime.evaluate(
            [sample], epoch, "diagnostic", "experiment4_smoke", saved, lambda: False
        )
        generations.append(
            {
                **contract["generation_probes"][epoch - 1],
                "all_pages_completed": result["all_pages_completed"],
                "failed_pages": result["failed_pages"],
                "retried_pages": result["retried_pages"],
            }
        )
    closed_json(
        runtime.root / "diagnosis.json",
        {
            "diagnostic_only": True,
            "assay_contract": contract,
            "initial_generation": initial_generation,
            "optimizer_parking_control": parking_control,
            "decode_convolution_control": convolution_control,
            "decode_linear_control": linear_control,
            "training_updates": training,
            "generations": generations,
            "decode_precision_controls": runtime.decode_precision_controls,
            "decode_attention_controls": runtime.decode_attention_controls,
            "decode_convolution_controls": runtime.decode_convolution_controls,
            "decode_linear_controls": runtime.decode_linear_controls,
            "decode_sampling_controls": runtime.decode_sampling_controls,
        },
    )
    if not all(row["all_pages_completed"] for row in generations):
        raise RuntimeError("Mixed assay generation incomplete")
    counts = {
        key: sum(row["training_counts"][key] for row in training)
        for key in training[0]["training_counts"]
    }
    receipt = {
        "success": True,
        "diagnostic_only": True,
        "cuda_device_count": 2,
        "checkpoint_reload": True,
        "generation_checked": True,
        "training_finite": True,
        "optimizer_steps": contract["optimizer_steps"],
        "runtime_identity": runtime.identity,
        "initial_adapter_sha256": runtime.initial_adapter_sha256,
        "assay_contract": contract,
        "training_counts": counts,
        "training_updates": training,
        "adamw_exact": True,
        "scheduler_exact": True,
        "rng_exact": True,
        "optimizer_parking_exact": True,
        "optimizer_parking_control": parking_control,
        "decode_precision_checked": True,
        "decode_precision_controls": runtime.decode_precision_controls,
        "decode_attention_checked": True,
        "decode_attention_controls": runtime.decode_attention_controls,
        "decode_convolution_checked": True,
        "decode_convolution_control": convolution_control,
        "decode_convolution_controls": runtime.decode_convolution_controls,
        "decode_linear_checked": True,
        "decode_linear_control": linear_control,
        "decode_linear_controls": runtime.decode_linear_controls,
        "decode_sampling_controls": runtime.decode_sampling_controls,
        "generations": generations,
    }
    validate_assay_receipt(receipt, contract)
    closed_json(runtime.root / "completed.json", receipt)
    runtime.emit("runner_smoke_completed", **receipt)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("preflight", "smoke", "baseline", "train"), required=True
    )
    args = parser.parse_args()
    os.umask(0o077)
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "jobs").mkdir(exist_ok=True)
    started, runtime = time.monotonic(), None
    with (args.root / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.root / "completed.json").exists():
            return 0

        def emit(event, **values):
            row = {
                "event": event,
                "elapsed_segment": time.monotonic() - started,
                **values,
            }
            with (args.root / "events.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                )
            print(
                json.dumps(
                    {k: v for k, v in row.items() if k != "selected_checkpoint"},
                    allow_nan=False,
                ),
                flush=True,
            )

        try:
            config = json.loads(args.config.read_text())
            if (
                sum(
                    path.is_dir() and path.name.isdigit()
                    for path in (args.root / "jobs").iterdir()
                )
                > config["maximum_segments"]
            ):
                raise RuntimeError("Maximum mixed continuation segments exhausted")
            cohort, pages, manifest = verify_manifest(config, emit)
            if args.mode == "preflight":
                receipt = {
                    "success": True,
                    "gpu_check": False,
                    "optimizer_steps": 0,
                    "source_pages": 125,
                    "line_images": 5212,
                    "luisa_lines": manifest["audit"]["luisa_selected"],
                    "synthetic_coverage": manifest["synthetic_coverage"],
                    "assay_contract": assay_contract(config, pages, manifest),
                }
                closed_json(args.root / "completed.json", receipt)
                emit("preflight_completed", **receipt)
                return 0
            runtime = MixedRuntime(config, args.root, emit, manifest)
            if args.mode == "smoke":
                return smoke(runtime, cohort, pages, manifest)
            approved = json.loads(
                (
                    Path(config["operation_root"]) / "smoke" / "completed.json"
                ).read_text()
            )
            validate_assay_receipt(approved, assay_contract(config, pages, manifest))
            if (
                approved.get("success") is not True
                or approved["runtime_identity"] != runtime.identity
                or not approved["checkpoint_reload"]
                or not approved["generation_checked"]
            ):
                raise ValueError(
                    "Experiment four's exact runtime has not passed its GPU assay"
                )
            if args.mode == "baseline":
                summaries = {
                    split: runtime.evaluate(
                        pages[split],
                        0,
                        split,
                        "baseline",
                        runtime.initial_adapter_sha256,
                        lambda: False,
                    )
                    for split in ("validation", "test")
                }
                closed_json(
                    args.root / "baseline.json",
                    {
                        "epoch": 0,
                        "optimizer_steps": 0,
                        "runtime_identity": runtime.identity,
                        **summaries,
                    },
                )
                closed_json(
                    args.root / "completed.json",
                    {"success": True, "optimizer_steps": 0},
                )
                emit("completed", optimizer_steps=0)
                return 0
            return run_training(
                runtime,
                cohort,
                pages,
                manifest,
                lambda: time.monotonic() - started >= config["segment_seconds"],
            )
        except Exception as error:
            emit(
                "failed",
                diagnosis={
                    "phase": runtime.phase if runtime else "setup",
                    "error_type": type(error).__name__,
                    "success": False,
                    **({"gpu_memory": runtime.memory_snapshot()} if runtime else {}),
                },
            )
            raise


if __name__ == "__main__":
    raise SystemExit(main())

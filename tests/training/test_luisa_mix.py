"""Full line coverage, exact layout targets and four-epoch selection contracts."""

import hashlib
import copy
import io
import json
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from finetune.recipes.berrutti_qwen38.luisa_mix import (
    ASSAY_PROTOCOL,
    DECODE_PRECISION_POLICY,
    DECODE_ATTENTION_POLICY,
    DECODE_CONVOLUTION_POLICY,
    OPTIMIZER_MEMORY_POLICY,
    assay_examples,
    move_adamw_moments,
    precise_decode_head,
    precise_decode_attention,
    precise_decode_convolution,
    run_training,
    select_epoch,
    unit_groups,
    validate_assay_receipt,
    verify_mixed_image_hashes,
)
from data.preparation.luisa import (
    choose_lines,
    compose_epoch,
    distribute,
    overlap_reasons,
)
from finetune.recipes.berrutti_qwen38.linear_decode import (
    DECODE_LINEAR_POLICY,
    NATIVE_QWEN_SHA256,
    precise_decode_linear,
    tensor_identity,
)
from data.preparation.synthetic_pages import CropStore, layout, render
from finetune.recipes.berrutti_qwen38.qwen_sampling import (
    InstructSamplingPolicy,
    page_seed,
)
from reporting.berrutti.monitor_status import (
    experiment_focus,
    read_additional_experiments,
    read_run,
)


class AssayGateTests(unittest.TestCase):
    def test_inference_tensor_identity_requires_value_checks_without_versions(self):
        class Tensor:
            dtype, device = "bf16", "cuda:0"

            def data_ptr(self):
                return 123

            @property
            def _version(self):
                raise RuntimeError("Inference tensors do not track version counter.")

        tensor = Tensor()
        self.assertEqual(
            tensor_identity(tensor), (id(tensor), 123, None, "bf16", "cuda:0")
        )

    def test_linear_scope_restores_dispatch_and_precision_after_exception(self):
        from types import SimpleNamespace

        originals = (lambda *a, **k: "chunk", lambda *a, **k: "recurrent")
        modeling = SimpleNamespace(
            torch_chunk_gated_delta_rule=originals[0],
            torch_recurrent_gated_delta_rule=originals[1],
        )
        parameter = SimpleNamespace(
            data_ptr=lambda: 123, _version=0, dtype="bf16", device="cuda:0"
        )
        model = SimpleNamespace(parameters=lambda: [parameter])
        matmul = SimpleNamespace(allow_tf32=True)
        torch = SimpleNamespace(
            is_grad_enabled=lambda: True,
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=matmul)),
        )
        events = []
        with patch(
            "finetune.recipes.berrutti_qwen38.linear_decode.native_chunk_body",
            return_value=originals[0],
        ):
            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                with precise_decode_linear(
                    model, torch, modeling, lambda *a, **k: events.append(k)
                ):
                    self.assertFalse(matmul.allow_tf32)
                    self.assertIs(
                        modeling.torch_recurrent_gated_delta_rule, originals[1]
                    )
                    with self.assertRaisesRegex(ValueError, "inference-only"):
                        modeling.torch_chunk_gated_delta_rule(*([object()] * 5))
                    raise RuntimeError("generation failed")
            self.assertIs(modeling.torch_chunk_gated_delta_rule, originals[0])
            self.assertTrue(matmul.allow_tf32)
            self.assertTrue(events[-1]["interfaces_restored"])
            with self.assertRaisesRegex(
                RuntimeError, "parameters or native interfaces"
            ):
                with precise_decode_linear(
                    model, torch, modeling, lambda *a, **k: None
                ):
                    parameter._version += 1
            self.assertIs(modeling.torch_chunk_gated_delta_rule, originals[0])
            self.assertTrue(matmul.allow_tf32)

    def test_linear_gate_rejects_partial_coverage_and_unverified_state(self):
        contract, receipt = self.receipt()
        validate_assay_receipt(receipt, contract)
        for mutation in (
            lambda r: r.update(decode_linear_checked=False),
            lambda r: r["decode_linear_control"].update(success=False),
            lambda r: r["decode_linear_control"]["rows"].pop(),
            lambda r: r["decode_linear_control"]["rows"][0].update(device=1),
            lambda r: r["decode_linear_control"]["rows"][0].update(state_exact=False),
            lambda r: r["decode_linear_control"]["rows"][0].update(
                state_reference_relative_l2=1e-4
            ),
            lambda r: r["decode_linear_control"]["rows"][0].update(
                output_reference_relative_l2=float("nan")
            ),
            lambda r: r["decode_linear_controls"].pop(),
            lambda r: r["decode_linear_controls"][0].update(reference_checks=0),
            lambda r: r["decode_linear_controls"][0].update(
                native_source_sha256="other"
            ),
            lambda r: r["decode_linear_controls"][0].update(
                cached_kernel_unchanged=False
            ),
            lambda r: r["decode_linear_controls"][0].update(inputs_unchanged=False),
            lambda r: r["decode_linear_controls"][0].update(interfaces_restored=False),
            lambda r: r["decode_linear_controls"][0].update(
                maximum_reference_relative_l2=1e-4
            ),
        ):
            invalid = copy.deepcopy(receipt)
            mutation(invalid)
            with self.assertRaises(ValueError):
                validate_assay_receipt(invalid, contract)

    def test_full_mixed_hashes_cover_reused_files_and_preserve_bytes(self):
        from collections import Counter
        from threading import Lock

        from data.integrity import file_sha256

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = [b"a" * (2 * 1024 * 1024 + 13), b"synthetic-pixels"]
            examples = []
            for index, payload in enumerate(payloads):
                path = root / f"image-{index}.png"
                path.write_bytes(payload)
                examples.append(
                    {
                        "image": str(path),
                        "image_sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
            manifest = {
                "epochs": [
                    {"units": [{"luisa": examples[:1], "synthetic": examples[1:]}]}
                    for _ in range(3)
                ]
            }
            reads, lock = Counter(), Lock()

            def record(path):
                with lock:
                    reads[str(path)] += 1
                return file_sha256(path)

            with patch(
                "finetune.recipes.berrutti_qwen38.luisa_mix.file_sha256",
                side_effect=record,
            ):
                self.assertEqual(verify_mixed_image_hashes(manifest), 2)
            self.assertEqual(reads, {example["image"]: 1 for example in examples})
            self.assertEqual(
                [Path(example["image"]).read_bytes() for example in examples], payloads
            )

    def test_mixed_hashes_reject_changed_missing_and_conflicting_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            path.write_bytes(b"original")
            example = {
                "image": str(path),
                "image_sha256": hashlib.sha256(b"original").hexdigest(),
            }
            unit = {"luisa": [example], "synthetic": []}
            manifest = {"epochs": [{"units": [unit]}]}
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Mixed input image changed"):
                verify_mixed_image_hashes(manifest)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                verify_mixed_image_hashes(manifest)
            unit["synthetic"] = [{**example, "image_sha256": "0" * 64}]
            with self.assertRaisesRegex(ValueError, "Conflicting hashes"):
                verify_mixed_image_hashes(manifest)

    def test_convolution_restores_native_interfaces_and_refuses_training(self):
        from types import SimpleNamespace

        originals = (lambda *a, **k: "prefill", lambda *a, **k: "cached")
        modeling = SimpleNamespace(
            causal_conv1d_fn=originals[0], causal_conv1d_update=originals[1]
        )
        parameter = SimpleNamespace(
            data_ptr=lambda: 123, _version=0, dtype="bf16", device="cuda:0"
        )
        model = SimpleNamespace(parameters=lambda: [parameter])
        torch = SimpleNamespace(is_grad_enabled=lambda: True)
        events = []
        hidden, weight = SimpleNamespace(shape=(1, 2, 3)), SimpleNamespace(shape=(2, 4))
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            with precise_decode_convolution(
                model, torch, modeling, lambda *a, **k: events.append(k)
            ):
                self.assertIsNot(modeling.causal_conv1d_fn, originals[0])
                self.assertIsNot(modeling.causal_conv1d_update, originals[1])
                with self.assertRaisesRegex(ValueError, "inference causal convolution"):
                    modeling.causal_conv1d_fn(hidden, weight, activation="silu")
                with self.assertRaisesRegex(ValueError, "inference-only"):
                    modeling.causal_conv1d_update(
                        hidden, object(), weight, activation="silu"
                    )
                raise RuntimeError("generation failed")
        self.assertIs(modeling.causal_conv1d_fn, originals[0])
        self.assertIs(modeling.causal_conv1d_update, originals[1])
        self.assertTrue(events[-1]["parameters_unchanged"])
        self.assertEqual(events[-1]["calls"], 0)
        with self.assertRaisesRegex(RuntimeError, "stored parameters"):
            with precise_decode_convolution(
                model, torch, modeling, lambda *a, **k: None
            ):
                parameter._version += 1
        self.assertIs(modeling.causal_conv1d_fn, originals[0])
        self.assertIs(modeling.causal_conv1d_update, originals[1])

    def test_attention_accepts_native_vision_and_decoder_keyword_masks(self):
        from types import SimpleNamespace

        calls = []

        def native(module, query, key, value, attention_mask, **kwargs):
            calls.append((module, query, key, value, attention_mask, kwargs))
            return "native"

        registry = SimpleNamespace(
            get_interface=lambda *a: native,
            register=lambda name, interface: setattr(registry, "interface", interface),
        )
        model = SimpleNamespace(parameters=lambda: [])
        torch = SimpleNamespace(bfloat16="bf16")
        vision = type("Qwen3_5VisionAttention", (), {})()
        decoder = type("Qwen3_5Attention", (), {"training": True})()
        query, key, value, mask = (object() for _ in range(4))
        with precise_decode_attention(model, torch, registry, lambda *a, **k: None):
            # Native Qwen vision supplies the mask by keyword, including None.
            for actual_mask in (None, mask):
                self.assertEqual(
                    registry.interface(
                        vision,
                        query,
                        key,
                        value,
                        attention_mask=actual_mask,
                        scaling=0.125,
                        dropout=0.0,
                        is_causal=False,
                    ),
                    "native",
                )
                self.assertEqual(
                    calls[-1],
                    (
                        vision,
                        query,
                        key,
                        value,
                        actual_mask,
                        {"scaling": 0.125, "dropout": 0.0, "is_causal": False},
                    ),
                )
            # Decoder uses the same name; reach its fail-closed training guard.
            with self.assertRaisesRegex(ValueError, "Unexpected decoder SDPA"):
                registry.interface(
                    decoder,
                    SimpleNamespace(dtype="bf16"),
                    key,
                    value,
                    attention_mask=mask,
                )
        self.assertIs(registry.interface, native)

    def test_attention_scope_restores_registry_and_rejects_training(self):
        from types import SimpleNamespace

        class Registry:
            def __init__(self):
                self.interface = lambda *a, **k: "native"

            def get_interface(self, name, fallback):
                return self.interface

            def register(self, name, interface):
                self.interface = interface

        parameter = SimpleNamespace(
            data_ptr=lambda: 123, _version=0, dtype="bf16", device="cuda:0"
        )
        model = SimpleNamespace(parameters=lambda: [parameter])
        torch = SimpleNamespace(bfloat16="bf16")
        registry = Registry()
        original = registry.interface
        decoder = type("Qwen3_5Attention", (), {"training": True})()
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            with precise_decode_attention(model, torch, registry, lambda *a, **k: None):
                self.assertEqual(
                    registry.interface(object(), None, None, None, None), "native"
                )
                with self.assertRaisesRegex(ValueError, "Unexpected decoder"):
                    registry.interface(
                        decoder, SimpleNamespace(dtype="bf16"), None, None, None
                    )
                raise RuntimeError("generation failed")
        self.assertIs(registry.interface, original)
        with self.assertRaisesRegex(RuntimeError, "stored parameters"):
            with precise_decode_attention(model, torch, registry, lambda *a, **k: None):
                parameter._version += 1
        self.assertIs(registry.interface, original)

    def test_decode_head_restores_native_routing_even_when_generation_fails(self):
        from types import SimpleNamespace

        class Weight:
            requires_grad = False
            dtype = "bf16"
            device = SimpleNamespace(type="cuda")
            shape = (0,)

            def detach(self):
                return self

            def float(self):
                return self

        class Linear:
            bias = None
            weight = Weight()

            def forward(self, hidden):
                return hidden

        torch = SimpleNamespace(
            nn=SimpleNamespace(Linear=Linear),
            bfloat16="bf16",
            cuda=SimpleNamespace(empty_cache=lambda: None),
        )
        for routed in (False, True):
            head = Linear()
            if routed:
                head._hf_hook = object()
                head._old_forward = head.forward
            slot = "_old_forward" if routed else "forward"
            original = getattr(head, slot)
            model = SimpleNamespace(get_output_embeddings=lambda: head)
            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                with precise_decode_head(model, torch, lambda *a, **k: None):
                    self.assertNotEqual(getattr(head, slot), original)
                    raise RuntimeError("generation failed")
            self.assertEqual(getattr(head, slot), original)
            self.assertIs(head.weight, Linear.weight)
        head.weight = SimpleNamespace(requires_grad=True)
        with self.assertRaises(ValueError):
            with precise_decode_head(model, torch, lambda *a, **k: None):
                self.fail("Trainable output weights must be rejected")

    def receipt(self):
        probes = [
            {"page_id": str(i), "image_sha256": f"hash-{i}", "columns": c}
            for i, c in enumerate((None, None, 1, 2, None))
        ]
        contract = {
            "protocol": ASSAY_PROTOCOL,
            "generation_policy": InstructSamplingPolicy().metadata(),
            "training_epochs": [1, 1, 4],
            "optimizer_steps": 3,
            "optimizer_memory_policy": OPTIMIZER_MEMORY_POLICY,
            "decode_precision_policy": DECODE_PRECISION_POLICY,
            "decode_attention_policy": DECODE_ATTENTION_POLICY,
            "decode_convolution_policy": DECODE_CONVOLUTION_POLICY,
            "decode_linear_policy": DECODE_LINEAR_POLICY,
            "generation_probes": probes,
        }
        flags = {
            key: True
            for key in (
                "checkpoint_reload",
                "adamw_exact",
                "scheduler_exact",
                "rng_exact",
                "optimizer_parking_exact",
                "decode_precision_checked",
                "decode_attention_checked",
                "decode_convolution_checked",
                "decode_linear_checked",
            )
        }
        updates = [
            {
                **flags,
                "optimizer_step": i,
                "phase": "mixed" if i < 3 else "consolidation",
                "optimizer_moments_on_cpu": True,
                "training_counts": {
                    "source_pages": 2,
                    "count_page": 2,
                    "count_berrutti_lines": 90,
                    "count_luisa": 4 if i < 3 else 0,
                    "count_synthetic": 2 if i < 3 else 0,
                },
                "metrics": {"loss": 0.5, "lr": 0.00005, "gradient_norm": 0.7},
            }
            for i in (1, 2, 3)
        ]
        receipt = {
            **flags,
            "success": True,
            "diagnostic_only": True,
            "generation_checked": True,
            "training_finite": True,
            "assay_contract": contract,
            "optimizer_steps": 3,
            "cuda_device_count": 2,
            "training_updates": updates,
            "decode_sampling_controls": [
                {
                    "policy": InstructSamplingPolicy().metadata()["version"],
                    "parameters": InstructSamplingPolicy().metadata()["sampling"],
                    "page_id": probe["page_id"],
                    "image_sha256": probe["image_sha256"],
                    "source_seed": 20261007,
                    "calls": 1,
                    "attempt_offset": 0,
                    "attempts": [1],
                    "seeds": [page_seed(20261007, probe, 1)],
                    "presence_calls": 100,
                    "inputs_unchanged": True,
                    "rng_restored": True,
                    "interfaces_restored": True,
                }
                for probe in [probes[3], *probes]
            ],
            "optimizer_parking_control": {
                "policy": OPTIMIZER_MEMORY_POLICY,
                "cuda_device_count": 2,
                "updates_per_device": 4,
                "parameters_exact": True,
                "moments_exact": True,
                "finite": True,
            },
            "decode_precision_controls": [
                {
                    "policy": DECODE_PRECISION_POLICY,
                    "calls": 100,
                    "finite": True,
                    "reference_exact": True,
                    "stored_weights_exact": True,
                }
                for _ in range(6)
            ],
            "decode_attention_controls": [
                {
                    "policy": DECODE_ATTENTION_POLICY,
                    "calls": 100,
                    "prefill_calls": 8,
                    "cached_calls": 92,
                    "reference_checks": 2,
                    "maximum_reference_relative_l2": 1e-7,
                    "finite": True,
                    "reference_close": True,
                    "parameters_unchanged": True,
                }
                for _ in range(6)
            ],
            "decode_convolution_control": {
                "policy": DECODE_CONVOLUTION_POLICY,
                "cuda_device_count": 2,
                "cases_per_device": 4,
                "finite": True,
                "reference_close": True,
                "cache_exact": True,
                "parameters_unchanged": True,
            },
            "decode_convolution_controls": [
                {
                    "policy": DECODE_CONVOLUTION_POLICY,
                    "calls": 100,
                    "prefill_calls": 8,
                    "cached_calls": 92,
                    "cache_checks": 92,
                    "reference_checks": 2,
                    "maximum_reference_relative_l2": 1e-7,
                    "finite": True,
                    "reference_close": True,
                    "cache_exact": True,
                    "parameters_unchanged": True,
                }
                for _ in range(6)
            ],
            "generations": [
                {
                    **p,
                    "all_pages_completed": True,
                    "failed_pages": 0,
                    "retried_pages": 0,
                }
                for p in probes
            ],
        }
        linear = {
            "policy": DECODE_LINEAR_POLICY,
            "native_source_sha256": NATIVE_QWEN_SHA256,
            "calls": 48,
            "prefill_calls": 48,
            "reference_checks": 1,
            "maximum_reference_relative_l2": 1e-7,
            **{
                k: True
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
            },
        }
        receipt["decode_linear_controls"] = [dict(linear) for _ in range(6)]
        receipt["decode_linear_control"] = {
            "policy": DECODE_LINEAR_POLICY,
            "success": True,
            "cuda_device_count": 2,
            "cases_per_device": 4,
            "rows": [
                dict(
                    linear,
                    device=d,
                    case=c,
                    output_reference_relative_l2=1e-7,
                    state_reference_relative_l2=1e-7,
                    state_exact=True,
                )
                for d in range(2)
                for c in range(4)
            ],
        }
        return contract, receipt

    def test_sampling_gate_rejects_wrong_policy_page_seed_or_missing_restoration(self):
        contract, receipt = self.receipt()
        validate_assay_receipt(receipt, contract)
        for mutation in (
            lambda r: r["decode_sampling_controls"].pop(),
            lambda r: r["decode_sampling_controls"][0].update(calls=0),
            lambda r: r["decode_sampling_controls"][0].update(calls=3),
            lambda r: r["decode_sampling_controls"][0].update(presence_calls=0),
            lambda r: r["decode_sampling_controls"][0].update(attempt_offset=1),
            lambda r: r["decode_sampling_controls"][0].update(attempts=[2]),
            lambda r: r["decode_sampling_controls"][0].update(seeds=[0]),
            lambda r: r["decode_sampling_controls"][0].update(page_id="substituted"),
            lambda r: r["decode_sampling_controls"][0].update(image_sha256="changed"),
            lambda r: r["decode_sampling_controls"][0].update(source_seed=99),
            lambda r: r["decode_sampling_controls"][0].update(parameters={}),
            lambda r: r["decode_sampling_controls"][0].update(inputs_unchanged=False),
            lambda r: r["decode_sampling_controls"][0].update(rng_restored=False),
            lambda r: r["decode_sampling_controls"][0].update(
                interfaces_restored=False
            ),
        ):
            invalid = copy.deepcopy(receipt)
            mutation(invalid)
            with self.assertRaises(ValueError):
                validate_assay_receipt(invalid, contract)
        legacy = copy.deepcopy(receipt)
        legacy["assay_contract"].pop("generation_policy")
        with self.assertRaises(ValueError):
            validate_assay_receipt(legacy, legacy["assay_contract"])

    def test_convolution_gate_rejects_unverified_cache_math_and_scope(self):
        contract, receipt = self.receipt()
        validate_assay_receipt(receipt, contract)
        for mutation in (
            lambda r: r.update(decode_convolution_checked=False),
            lambda r: r["decode_convolution_control"].update(cuda_device_count=1),
            lambda r: r["decode_convolution_control"].update(cache_exact=False),
            lambda r: r["decode_convolution_controls"].pop(),
            lambda r: r["decode_convolution_controls"][0].update(prefill_calls=0),
            lambda r: r["decode_convolution_controls"][1].update(cached_calls=0),
            lambda r: r["decode_convolution_controls"][2].update(cache_checks=0),
            lambda r: r["decode_convolution_controls"][3].update(cache_exact=False),
            lambda r: r["decode_convolution_controls"][4].update(
                parameters_unchanged=False
            ),
            lambda r: r["decode_convolution_controls"][5].update(
                maximum_reference_relative_l2=float("nan")
            ),
            lambda r: r["decode_convolution_controls"][5].update(
                maximum_reference_relative_l2=1e-4
            ),
            lambda r: r["decode_convolution_controls"][5].update(policy="unverified"),
        ):
            invalid = copy.deepcopy(receipt)
            mutation(invalid)
            with self.assertRaises(ValueError):
                validate_assay_receipt(invalid, contract)

    def test_gpu_success_cannot_mask_incomplete_generation_or_restore(self):
        contract, receipt = self.receipt()
        validate_assay_receipt(receipt, contract)
        for mutation in (
            lambda r: r["generations"][3].update(
                all_pages_completed=False, failed_pages=1
            ),
            lambda r: r["generations"][3].update(image_sha256="easier-replacement"),
            lambda r: r["generations"].pop(),
            lambda r: r["training_updates"][1].update(adamw_exact=False),
            lambda r: r["training_updates"][1]["metrics"].update(loss=float("nan")),
            lambda r: r["training_updates"][2]["training_counts"].update(count_luisa=1),
            lambda r: r.update(optimizer_steps=1),
            lambda r: r["optimizer_parking_control"].update(moments_exact=False),
            lambda r: r["training_updates"][1].update(optimizer_moments_on_cpu=False),
            lambda r: r["decode_precision_controls"].pop(),
            lambda r: r["decode_precision_controls"][0].update(finite=False),
            lambda r: r["decode_precision_controls"][4].update(
                stored_weights_exact=False
            ),
            lambda r: r["decode_attention_controls"].pop(),
            lambda r: r["decode_attention_controls"][0].update(cached_calls=0),
            lambda r: r["decode_attention_controls"][3].update(
                parameters_unchanged=False
            ),
            lambda r: r["decode_attention_controls"][4].update(
                maximum_reference_relative_l2=float("nan")
            ),
        ):
            invalid = copy.deepcopy(receipt)
            mutation(invalid)
            with self.assertRaises(ValueError):
                validate_assay_receipt(invalid, contract)

    def test_parking_preserves_cpu_step_counters_and_parameter_device_placement(self):
        from types import SimpleNamespace

        class Tensor:
            def __init__(self, device):
                self.device = device

            def to(self, *, device):
                return Tensor(device)

            def numel(self):
                return 8

            def element_size(self):
                return 4

        parameters = [Tensor("cuda:0"), Tensor("cuda:1")]
        counters = [Tensor("cpu"), Tensor("cpu")]
        optimizer = SimpleNamespace(
            param_groups=[{"params": parameters}],
            state={
                p: {
                    "step": counter,
                    "exp_avg": Tensor(p.device),
                    "exp_avg_sq": Tensor(p.device),
                }
                for p, counter in zip(parameters, counters, strict=True)
            },
        )
        self.assertEqual(move_adamw_moments(optimizer, to_cpu=True), 128)
        for p, counter in zip(parameters, counters, strict=True):
            self.assertEqual(optimizer.state[p]["exp_avg"].device, "cpu")
            self.assertIs(optimizer.state[p]["step"], counter)
        move_adamw_moments(optimizer, to_cpu=False)
        for p, counter in zip(parameters, counters, strict=True):
            self.assertEqual(optimizer.state[p]["exp_avg"].device, p.device)
            self.assertEqual(optimizer.state[p]["exp_avg_sq"].device, p.device)
            self.assertIs(optimizer.state[p]["step"], counter)
        optimizer.state[parameters[0]]["unexpected_state"] = Tensor("cuda:0")
        with self.assertRaises(ValueError):
            move_adamw_moments(optimizer, to_cpu=True)

    def test_assay_retains_densest_original_layouts_and_longest_line(self):
        external = [
            {"reference_text": "x" * n, "page_id": f"l{n}"} for n in range(1, 6)
        ]
        synthetic = [
            {"reference_text": "x" * n, "page_id": f"s{c}-{n}", "columns": c}
            for c in (1, 2)
            for n in (2, 9)
        ]
        units = [
            {"page_id": p, "luisa": external, "synthetic": synthetic}
            for p in ("a", "b")
        ]
        selected, probes = assay_examples(
            {"smoke_page_ids": ["a", "b"]},
            {"validation": [{"page_id": "v"}], "test": [{"page_id": "t"}]},
            {"epochs": [{"units": units}]},
        )
        self.assertEqual(
            [p["page_id"] for p in probes], ["v", "t", "s1-9", "s2-9", "l5"]
        )
        self.assertEqual(
            [u["synthetic"][0]["page_id"] for u in selected], ["s1-9", "s2-9"]
        )


class SyntheticPagesTests(unittest.TestCase):
    def lines(self, root, count):
        path = root / "line.png"
        Image.new("RGB", (80, 12), "black").save(path)
        return [
            {
                "line_id": str(i),
                "image": str(path),
                "image_sha256": "source-digest",
                "source": "luisa",
                "size": [80, 12],
                "reference_text": f"human line {i}",
            }
            for i in range(count)
        ]

    def test_all_lines_retained_once_with_bounded_nonoverlapping_boxes(self):
        with tempfile.TemporaryDirectory() as directory:
            lines = self.lines(Path(directory), 119)
            seen, index = [], 0
            while len(seen) < len(lines):
                placements = layout(
                    lines[len(seen) :], (400, 600), 10 + index, 1 + index % 2
                )
                self.assertTrue(placements)
                for row in placements:
                    x, y, right, bottom = row["box"]
                    self.assertTrue(0 <= x < right <= 400 and 0 <= y < bottom <= 600)
                for a, b in zip(placements, placements[1:]):
                    self.assertTrue(
                        a["column"] < b["column"] or a["box"][3] <= b["box"][1]
                    )
                seen.extend(row["line_id"] for row in placements)
                index += 1
            self.assertEqual(seen, [row["line_id"] for row in lines])

    def test_two_columns_have_explicit_order_and_reproducible_pixels_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lines = self.lines(root, 48)
            a = render(lines, (400, 600), 1234, 2, root / "a.png")
            b = render(lines, (400, 600), 1234, 2, root / "b.png")
            self.assertEqual({p["column"] for p in a["sources"]}, {0, 1})
            self.assertEqual(
                (root / "a.png").read_bytes(), (root / "b.png").read_bytes()
            )
            self.assertEqual(
                a["reference_text"],
                "\n".join(row["reference_text"] for row in lines[: len(a["sources"])]),
            )
            self.assertEqual(a["sources"], b["sources"])
            self.assertFalse(a["artificial_text"])
            self.assertTrue(a["no_source_line_clipped"])
            with self.assertRaises(FileExistsError):
                render(lines, (400, 600), 1234, 2, root / "a.png")
            c = render(lines, (400, 600), 1235, 2, root / "c.png")
            self.assertNotEqual(a["sources"], c["sources"])

    def test_render_uses_entire_annotated_crop_not_background_outside_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lines = self.lines(root, 1)
            source = Image.new("RGB", (100, 40), "red")
            source.paste(Image.new("RGB", (80, 12), "black"), (10, 10))
            source.save(root / "line.png")
            lines[0]["crop_xyxy"] = [10, 10, 90, 22]
            item = render(lines, (400, 600), 1, 1, root / "page.png")
            with Image.open(root / "page.png") as page:
                crop = page.crop(item["sources"][0]["box"])
                self.assertEqual(set(crop.tobytes()), {0})

    def test_parallel_cached_compositions_match_original_pixels_targets_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lines = self.lines(root, 119)
            original = Image.new("RGB", (100, 40), "red")
            crop = Image.new("RGB", (80, 12), "white")
            for x in range(80):
                for y in range(12):
                    crop.putpixel((x, y), (x * 3, y * 20, (x + y) * 2))
            original.paste(crop, (10, 10))
            original.save(root / "line.png")
            encoded = io.BytesIO()
            crop.save(encoded, format="PNG")
            for line in lines:
                line.update(source="berrutti_line", crop_xyxy=[10, 10, 90, 22])
            seed, sizes = 20261007, [(400, 600), (420, 630)]
            ordered = lines.copy()
            rng = random.Random(seed + 10000)
            rng.shuffle(ordered)
            expected, cursor = [], 0
            while cursor < len(ordered):
                index = len(expected)
                item = render(
                    ordered[cursor:],
                    sizes[rng.randrange(len(sizes))],
                    seed + 100000 + index,
                    1 + index % 2,
                    root / "original" / f"{index}.png",
                )
                expected.append(item)
                cursor += len(item["sources"])
            # A one-byte budget forces disk fallback; four workers share it safely.
            with CropStore(root / "crops.sqlite", cache_bytes=1) as store:
                for line in lines:
                    store.add(
                        line,
                        encoded.getvalue(),
                        expected_rgb=hashlib.sha256(crop.tobytes()).hexdigest(),
                    )
                store.seal()
                observed = compose_epoch(
                    lines,
                    sizes,
                    seed,
                    1,
                    root / "parallel",
                    store,
                    4,
                    lambda *a, **k: None,
                    check_identity=True,
                )
                self.assertLessEqual(store.bytes, 1)
            self.assertEqual(len(observed), len(expected))
            for actual, reference in zip(observed, expected, strict=True):
                self.assertEqual(
                    Path(actual["image"]).read_bytes(),
                    Path(reference["image"]).read_bytes(),
                )
                for key in ("sources", "reference_text", "size", "seed", "columns"):
                    self.assertEqual(actual[key], reference[key])
                receipt = root / "parallel" / "pages" / (actual["page_id"] + ".json")
                self.assertEqual(json.loads(receipt.read_text()), actual)
            identities = list((root / "parallel" / "identity").glob("*.json"))
            self.assertEqual(len(identities), 2)
            self.assertEqual(
                {json.loads(p.read_text())["columns"] for p in identities}, {1, 2}
            )

    def test_source_cache_rejects_changed_pixels_references_and_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            line = self.lines(root, 1)[0]
            encoded = Path(line["image"]).read_bytes()
            with CropStore(root / "crops.sqlite", cache_bytes=1) as store:
                with self.assertRaisesRegex(ValueError, "pixels"):
                    store.add(line, encoded, expected_rgb="changed")
                store.add(line, encoded)
                with self.assertRaisesRegex(ValueError, "reference"):
                    store.load({**line, "reference_text": "changed"})
                with self.assertRaisesRegex(ValueError, "dimensions"):
                    store.load({**line, "size": [81, 12]})
                store.connection.execute("UPDATE crops SET rgb_sha='changed'")
                store.seal()
                with self.assertRaisesRegex(ValueError, "pixels"):
                    store.load(line)


class MixedCorpusTests(unittest.TestCase):
    def test_extension_monitor_recovers_four_epochs_without_changing_primary_arms(self):
        module = "reporting.berrutti.monitor_status"
        spec = {
            "name": "experiment4",
            "job_id": 7,
            "run_root": "/sealed/mixed",
            "epochs": 4,
            "progress_markers": [104, 108, 112],
            "definition": {"exploratory": True},
        }
        metric = {
            "epoch": 4,
            "validation": {"pages": 25, "cer": 0.03, "failed_pages": 0},
        }
        with patch(module + ".remote", side_effect=["", json.dumps(metric)]) as remote:
            run = read_run(spec, {7: "RUNNING"})
        self.assertEqual(run["evaluations"][0]["epoch"], 4)
        self.assertIn("epoch-04.json", remote.call_args.args[0])
        self.assertIn("epoch-104.json", remote.call_args.args[0])
        with (
            patch(module + ".configure_cluster"),
            patch(module + ".resolve_continuation", return_value=spec),
            patch(module + ".job_states", return_value={7: "RUNNING"}),
            patch(module + ".pending_queue", return_value={}),
            patch(module + ".read_run", return_value=run),
        ):
            additional = read_additional_experiments([spec], "Ruche")
        self.assertEqual(additional["experiment4"]["state"], "RUNNING")
        coord = {
            "experiment_plan": {
                "protocol": "three-arms-125-v1",
                "arms": {arm: {} for arm in ("pages", "lines", "combined")},
            },
            "additional_experiments": [spec],
        }
        self.assertEqual(
            set(experiment_focus(coord, {})["primary_experiments"]),
            {"pages", "lines", "combined"},
        )

    def test_interrupted_progress_validation_resumes_without_repeating_updates(self):
        class Runtime:
            identity = "mixed-runtime"

            def __init__(self, root):
                self.root = root
                self.config = {"operation_root": str(root.parent), "seed": 20261007}
                self.updates = []
                self.restored = []
                self.interrupt = True

            def restore(self, state, *, optimizer=True):
                self.restored.append((state["global_step"], optimizer))

            def progress(self, state):
                (self.root / "state.json").write_text(json.dumps(state))

            def checkpoint(self, state):
                state["checkpoint"] = f"checkpoint-{state['global_step']}"
                self.progress(state)
                return state["checkpoint"]

            def adapter_digest(self):
                return "adapter"

            def emit(self, *args, **kwargs):
                pass

            def train_units(self, units, real, regions, state):
                self.updates.append((state["epoch"], state["group"]))
                n = len(units)
                return {
                    "samples": n * 4 if state["epoch"] <= 3 else n * 2,
                    "source_pages": n,
                    "unit_loss_sum": n,
                    "supervised_tokens": n * 20,
                    "seconds": 1,
                    "count_page": n,
                    "count_berrutti_lines": 5212 if state["group"] == 15 else 0,
                    "count_luisa": n if state["epoch"] <= 3 else 0,
                    "count_synthetic": n if state["epoch"] <= 3 else 0,
                }, {"loss": 1, "lr": 0.00001}

            def evaluate(self, pages, epoch, split, stage, identity, should_yield):
                if epoch == 104 and self.interrupt:
                    self.interrupt = False
                    return None
                return {
                    "epoch": epoch,
                    "split": split,
                    "pages": 25,
                    "all_pages_completed": True,
                    "failed_pages": 0,
                    "retried_pages": 0,
                    "cer": 0.001 if epoch > 100 else 0.1 / epoch,
                }

        with tempfile.TemporaryDirectory() as directory:
            operation = Path(directory)
            root = operation / "train"
            root.mkdir()
            (operation / "baseline").mkdir()
            (operation / "baseline" / "baseline.json").write_text(
                json.dumps({"runtime_identity": "mixed-runtime", "optimizer_steps": 0})
            )
            units = [{"page_id": str(i)} for i in range(125)]
            manifest = {
                "audit": {
                    "luisa_selected": 125,
                    "validation_interpretation": "exploratory",
                },
                "epochs": [
                    {
                        "units": units,
                        "samples": 500 if epoch <= 3 else 250,
                        "synthetic_pages": 125 if epoch <= 3 else 0,
                    }
                    for epoch in (1, 2, 3, 4)
                ],
            }
            cohort = {"pages": [{**unit, "regions": []} for unit in units]}
            pages = {"train": units, "validation": [{}] * 25, "test": [{}] * 25}
            runtime = Runtime(root)
            self.assertEqual(
                run_training(runtime, cohort, pages, manifest, lambda: False), 85
            )
            cursor = json.loads((root / "state.json").read_text())
            self.assertEqual(
                (cursor["global_step"], cursor["phase"]), (4, "progress_validation")
            )
            self.assertEqual(
                run_training(runtime, cohort, pages, manifest, lambda: False), 0
            )
            self.assertEqual(
                runtime.updates, [(e, g) for e in (1, 2, 3, 4) for g in range(16)]
            )
            summary = json.loads((root / "completed.json").read_text())
            self.assertEqual(
                (summary["optimizer_steps"], summary["best_epoch"]), (64, 4)
            )
            self.assertEqual(summary["selected_checkpoint"], "checkpoint-64")

    def test_explicit_inclusion_policy_and_heldout_screening(self):
        overlap = {
            "line_id": "overlap",
            "reference_text": "A sufficiently long held out line",
            "visual_key": "different",
            "source_split": "train",
        }
        overlap["overlap_reasons"] = overlap_reasons(
            overlap, set(), ["asufficientlylongheldoutlineonapage"], set()
        )
        self.assertEqual(overlap["overlap_reasons"], ["heldout_page_text"])
        reserved = {
            **overlap,
            "line_id": "test",
            "source_split": "test",
            "overlap_reasons": [],
        }
        safe = {**overlap, "line_id": "safe", "overlap_reasons": []}
        lines = [overlap, reserved, safe]
        self.assertEqual(len(choose_lines(lines, "include-all")[0]), 3)
        selected, excluded = choose_lines(lines, "exclude-known-overlap")
        self.assertEqual([line["line_id"] for line in selected], ["safe"])
        self.assertEqual(len(excluded), 2)
        with self.assertRaises(ValueError):
            choose_lines(lines, "implicit")

    def test_distribution_and_grouping_cover_all_auxiliary_and_source_examples(self):
        rows = [{"line_id": str(i)} for i in range(23774)]
        buckets = distribute(rows, 125, 9)
        self.assertEqual(
            Counter(r["line_id"] for group in buckets for r in group),
            Counter(r["line_id"] for r in rows),
        )
        self.assertLessEqual(max(map(len, buckets)) - min(map(len, buckets)), 1)
        units = [
            {"page_id": str(i), "luisa": bucket} for i, bucket in enumerate(buckets)
        ]
        for epoch in (1, 2, 3, 4):
            groups = unit_groups(units, 20261007, epoch)
            self.assertEqual([len(g) for g in groups], [8] * 15 + [5])
            self.assertEqual(len({u["page_id"] for g in groups for u in g}), 125)
        self.assertNotEqual(unit_groups(units, 9, 1), unit_groups(units, 9, 2))

    def test_selection_requires_four_epochs_and_complete_validation_not_progress(self):
        rows = [
            {
                "epoch": i,
                "split": "validation",
                "pages": 25,
                "all_pages_completed": True,
                "failed_pages": 0,
                "cer": 0.01 * i,
            }
            for i in (1, 2, 3, 4)
        ]
        self.assertIsNone(select_epoch(rows[:3]))
        rows[0].update(all_pages_completed=False, failed_pages=1)
        self.assertEqual(select_epoch(rows)["epoch"], 2)
        self.assertIsNone(select_epoch(rows + [{**rows[1], "epoch": 104, "cer": 0.0}]))
        rows[1]["cer"] = float("nan")
        self.assertEqual(select_epoch(rows)["epoch"], 3)


if __name__ == "__main__":
    unittest.main()

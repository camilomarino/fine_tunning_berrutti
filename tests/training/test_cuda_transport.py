"""Transport evidence supplements the unchanged full-model assay gate."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

from finetune.recipes.berrutti_qwen38.cuda_transport import (
    POLICY,
    host_staged_transfers,
    validate_transport_receipt,
)


def receipt():
    return {
        "policy": POLICY,
        "success": True,
        "full_qwen_gate": False,
        "transfer_cases": [
            dict(source=a, destination=b, dtype=dtype, count=n, exact=True, overloads=5)
            for a, b in [(0, 1), (1, 0)]
            for dtype in ("torch.int64", "torch.float32", "torch.bfloat16")
            for n in (1, 127)
        ],
        "routing_control": dict(
            success=True, cuda_device_count=2, samples_completed=128
        ),
        "runtime_control": dict(
            policy=POLICY,
            calls=400,
            exact=True,
            source_unchanged=True,
            dtype_shape_preserved=True,
            interfaces_restored=True,
        ),
    }


class TransportGuards(unittest.TestCase):
    def test_complete_transport_evidence_remains_separate_from_qwen_gate(self):
        closed = receipt()
        validate_transport_receipt(closed)
        self.assertFalse(closed["full_qwen_gate"])

    def test_rejects_partial_duplicate_or_inexact_transfer_coverage(self):
        for change in ("partial", "duplicate", "inexact", "overload"):
            closed = receipt()
            if change == "partial":
                closed["transfer_cases"].pop()
            elif change == "duplicate":
                closed["transfer_cases"][0] = deepcopy(closed["transfer_cases"][1])
            elif change == "inexact":
                closed["transfer_cases"][0]["exact"] = False
            else:
                closed["transfer_cases"][0]["overloads"] = 4
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_transport_receipt(closed)

    def test_rejects_weakened_original_gradient_control(self):
        for field, value in (
            ("success", False),
            ("samples_completed", 127),
            ("cuda_device_count", 1),
        ):
            closed = receipt()
            closed["routing_control"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_transport_receipt(closed)

    def test_rejects_failed_runtime_identity_and_missing_restoration(self):
        for field in (
            "exact",
            "source_unchanged",
            "dtype_shape_preserved",
            "interfaces_restored",
        ):
            closed = receipt()
            closed["runtime_control"][field] = False
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_transport_receipt(closed)

    def test_restores_original_interface_after_exception_and_rejects_nesting(self):
        class Tensor:
            def to(self, *args, **kwargs):
                raise AssertionError("No tensor operation belongs in this control")

        torch = SimpleNamespace(Tensor=Tensor)
        original = Tensor.to
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with host_staged_transfers(torch) as control:
                self.assertIsNot(Tensor.to, original)
                with self.assertRaisesRegex(RuntimeError, "already installed"):
                    with host_staged_transfers(torch):
                        pass
                raise RuntimeError("deliberate")
        self.assertIs(Tensor.to, original)
        self.assertTrue(control["interfaces_restored"])


if __name__ == "__main__":
    unittest.main()

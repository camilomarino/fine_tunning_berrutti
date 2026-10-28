"""Annotation agreement and projection wrapper checks shared by the three arms."""

import unittest
from functools import partial

from finetune.recipes.berrutti_qwen38.comparison import (
    aligned_regions,
    check_native_forward,
)


def region(text, y, index=0):
    return {
        "shape_attributes": {
            "name": "rect",
            "x": index * 5,
            "y": y,
            "width": 50,
            "height": 10,
        },
        "region_attributes": {"text": text},
    }


class SharedContracts(unittest.TestCase):
    def test_native_check_preserves_accelerate_routing_and_rejects_a_patch(self):
        class Native:
            def forward(self):
                return "native"

        def new_forward(module):
            return module._old_forward()

        new_forward.__module__ = "accelerate.hooks"
        module = Native()
        check_native_forward(module, Native.forward)
        module._old_forward = module.forward
        module._hf_hook = object()
        module.forward = partial(new_forward, module)
        wrapped = module.forward
        check_native_forward(module, Native.forward)
        self.assertIs(module.forward, wrapped)
        self.assertEqual(module.forward(), "native")
        module._old_forward = lambda: "patched"
        with self.assertRaises(RuntimeError):
            check_native_forward(module, Native.forward)

    def test_complete_content_includes_short_numeric_lines(self):
        result = aligned_regions(
            "123\n456", [region("123", 10), region("456", 30)], (100, 100)
        )
        self.assertEqual([row["target"] for row in result["regions"]], ["123", "456"])

    def test_missing_or_changed_transcription_is_flagged_as_disagreement(self):
        self.assertIsNone(
            aligned_regions(
                "first second missing",
                [region("first", 10), region("second", 30)],
                (100, 100),
            )
        )
        self.assertIsNone(
            aligned_regions(
                "first second", [region("first", 10), region("wrong", 30)], (100, 100)
            )
        )

    def test_reference_multiset_requires_each_repeated_line_exactly_once(self):
        result = aligned_regions(
            "a b a", [region("a", 10), region("a b", 30)], (100, 100)
        )
        self.assertEqual(result["order"], "reference_verified_multiset")
        self.assertEqual([row["target"] for row in result["regions"]], ["a b", "a"])
        self.assertIsNone(
            aligned_regions("a b a a", [region("a", 10), region("a b", 30)], (100, 100))
        )

    def test_invalid_coordinates_are_rejected(self):
        with self.assertRaises(ValueError):
            aligned_regions("first", [region("first", 95)], (100, 100))


if __name__ == "__main__":
    unittest.main()

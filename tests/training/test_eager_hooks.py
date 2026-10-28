"""A compiler wrapper may be bypassed only when its original routing is exact."""

import functools
import unittest

from finetune.recipes.berrutti_qwen38.eager_hooks import original_hook


def fake_wrapper(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not hasattr(wrapper, "_compiled_fn"):
            wrapper._compiled_fn = fn
        return wrapper._compiled_fn(*args, **kwargs)

    return wrapper


class HookGuards(unittest.TestCase):
    def test_preserves_actual_routing_body_and_rejects_false_original(self):
        def routing(value):
            return value + 1

        routing.__module__ = "accelerate.hooks"
        wrapper = fake_wrapper(routing)
        self.assertIs(original_hook(wrapper), routing)
        self.assertEqual(original_hook(wrapper)(3), wrapper(3))
        wrapper.__wrapped__ = lambda value: value
        with self.assertRaises(ValueError):
            original_hook(wrapper)

    def test_rejects_unknown_decorators(self):
        with self.assertRaises(ValueError):
            original_hook(lambda value: value)


if __name__ == "__main__":
    unittest.main()

"""Approved generation policy, consumed attempts and isolated random streams."""

import importlib.util
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from finetune.recipes.berrutti_qwen38.generation_guard import GenerationPolicy
from finetune.recipes.berrutti_qwen38.qwen_sampling import (
    InstructSamplingPolicy,
    PresencePenalty,
    page_seed,
    seeded_sampling,
)


class SamplingPolicyTests(unittest.TestCase):
    def test_sampling_preserves_guard_budgets_and_original_policy(self):
        original, sampled = GenerationPolicy(), InstructSamplingPolicy()
        for reason in (None, "repetition", "time_limit", "interrupted"):
            before = original.attempt_parameters(reason)
            after = sampled.attempt_parameters(reason)
            self.assertEqual(after["max_new_tokens"], before["max_new_tokens"])
            self.assertEqual(after["max_seconds"], before["max_seconds"])
            self.assertTrue(after["do_sample"])
            self.assertEqual(after["repetition_penalty"], 1.0)
        for key in (
            "max_attempts",
            "min_repeat_tokens",
            "max_repeat_period",
            "min_repeats",
            "check_interval",
        ):
            self.assertEqual(sampled.metadata()[key], original.metadata()[key])
        self.assertEqual(sampled.tail_tokens, original.tail_tokens)
        self.assertFalse(original.attempt_parameters(None)["do_sample"])

    def test_seed_depends_only_on_source_seed_image_identity_and_attempt(self):
        page = {"page_id": "fixture", "image_sha256": "a" * 64}
        first = page_seed(20261007, page, 1)
        self.assertEqual(
            first,
            page_seed(
                20261007,
                {
                    **page,
                    "reference_text": "unused",
                    "checkpoint": "other",
                    "root": "other",
                },
                1,
            ),
        )
        self.assertNotEqual(first, page_seed(20261007, page, 2))
        self.assertNotEqual(
            first, page_seed(20261007, {**page, "image_sha256": "b" * 64}, 1)
        )
        with self.assertRaises(ValueError):
            page_seed(20261007, page, 3)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Requires real CPU torch")
class SamplingTensorTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        # These controls never allocate a workstation GPU.
        cuda = SimpleNamespace(
            device_count=lambda: 0,
            _is_in_bad_fork=lambda: False,
            manual_seed_all=lambda _: None,
        )
        self.addCleanup(patch.stopall)
        patch.object(torch, "cuda", cuda).start()
        patch.dict(
            sys.modules, {"transformers": SimpleNamespace(LogitsProcessorList=list)}
        ).start()

    def test_presence_penalty_excludes_prompt_and_counts_repeats_once(self):
        torch = self.torch
        ids = torch.tensor([[1, 2, 1, 3, 3]])
        scores = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])
        saved = scores.clone()
        self.assertTrue(
            torch.equal(
                PresencePenalty(torch, 3)(ids, scores),
                torch.tensor([[0.0, 1.0, 2.0, 1.5, 4.0]]),
            )
        )
        self.assertTrue(torch.equal(scores, saved))
        self.assertTrue(
            torch.equal(PresencePenalty(torch, 3)(ids[:, :3], scores), scores)
        )

    def test_resume_uses_second_seed_without_changing_global_rng_or_interface(self):
        torch = self.torch

        class Model:
            def generate(self, **kwargs):
                scores = kwargs["logits_processor"][0](
                    kwargs["input_ids"], torch.zeros(1, 5)
                )
                return torch.multinomial(torch.softmax(scores, dim=-1), 1)

        model, policy = Model(), InstructSamplingPolicy()
        page = {"page_id": "fixture", "image_sha256": "a" * 64}
        kwargs = {
            **policy.attempt_parameters(None),
            "input_ids": torch.tensor([[1, 2, 3]]),
        }
        saved_rng, original = torch.get_rng_state().clone(), model.generate
        with seeded_sampling(
            model, torch, policy, page, 20261007, lambda *a, **k: None
        ) as fresh:
            model.generate(**kwargs)
            second = model.generate(**kwargs)
            with self.assertRaises(ValueError):
                model.generate(**kwargs)
        with seeded_sampling(
            model,
            torch,
            policy,
            page,
            20261007,
            lambda *a, **k: None,
            completed_attempts=1,
        ) as resumed:
            resumed_second = model.generate(**kwargs)
        self.assertEqual(resumed["seeds"], fresh["seeds"][1:])
        self.assertEqual(resumed["attempts"], [2])
        self.assertTrue(torch.equal(second, resumed_second))
        self.assertTrue(torch.equal(saved_rng, torch.get_rng_state()))
        self.assertEqual(model.generate, original)
        self.assertNotIn("generate", model.__dict__)
        self.assertTrue(resumed["rng_restored"] and resumed["interfaces_restored"])

    def test_exception_restores_rng_and_existing_instance_override(self):
        torch = self.torch

        def fail(**kwargs):
            torch.rand(2)
            raise RuntimeError("fixture")

        model, policy = SimpleNamespace(generate=fail), InstructSamplingPolicy()
        page = {"page_id": "fixture", "image_sha256": "a" * 64}
        saved_rng = torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with seeded_sampling(
                model, torch, policy, page, 20261007, lambda *a, **k: None
            ) as control:
                model.generate(
                    **policy.attempt_parameters(None), input_ids=torch.tensor([[1, 2]])
                )
        self.assertIs(model.generate, fail)
        self.assertTrue(torch.equal(saved_rng, torch.get_rng_state()))
        self.assertTrue(control["rng_restored"] and control["interfaces_restored"])

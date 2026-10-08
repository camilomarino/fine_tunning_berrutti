"""Offline runtime tests with tiny random weights; run in the training environment."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    LogitsProcessor,
    LogitsProcessorList,
)

from experiments.berrutti_qwen38 import train
from experiments.berrutti_qwen38.generation_guard import GenerationPolicy


class ForcedTokens(LogitsProcessor):
    def __init__(self, sequence):
        self.sequence = sequence

    def __call__(self, input_ids, scores):
        index = input_ids.shape[-1] - 1
        scores[:] = -torch.inf
        scores[:, self.sequence[index % len(self.sequence)]] = 0
        return scores


class GenerationRuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tokenizer = SimpleNamespace(
            eos_token_id=7,
            pad_token_id=0,
            decode=lambda ids, **kwargs: " ".join(
                str(i) for i in ids if i not in (7, 9)
            ),
        )
        self.batch = {
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.ones(1, 1, dtype=torch.long),
        }

    def run_sequence(self, sequence, policy):
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=256,
                n_positions=256,
                n_embd=8,
                n_layer=1,
                n_head=1,
                bos_token_id=1,
                eos_token_id=[7, 9],
                pad_token_id=0,
            )
        ).eval()
        model.generation_config.forced_eos_token_id = 7
        original = model.generate
        model.generate = lambda **kwargs: original(
            **kwargs, logits_processor=LogitsProcessorList([ForcedTokens(sequence)])
        )
        return train.generate_attempt(
            model, self.tokenizer, self.batch, policy, policy.attempt_parameters(None)
        )

    def test_real_generate_accepts_secondary_eos_at_cap(self):
        result = self.run_sequence([3, 4, 9], GenerationPolicy(max_new_tokens=3))
        self.assertEqual(result["stop_reason"], "eos")
        self.assertEqual(result["last_token_id"], 9)

    def test_real_generate_stops_repetition_before_cap(self):
        result = self.run_sequence([3, 4], GenerationPolicy(max_new_tokens=200))
        self.assertEqual(result["stop_reason"], "repetition")
        self.assertEqual(result["generated_tokens"], 128)

    def test_real_generate_does_not_force_fake_eos_at_cap(self):
        result = self.run_sequence(
            list(range(20, 40)), GenerationPolicy(max_new_tokens=4)
        )
        self.assertEqual(result["stop_reason"], "token_limit")
        self.assertEqual(result["last_token_id"], 23)

    def test_real_generate_cooperative_timeout(self):
        result = self.run_sequence([3, 4], GenerationPolicy(max_seconds=0.000001))
        self.assertEqual(result["stop_reason"], "time_limit")

    def test_empty_eos_not_success(self):
        result = self.run_sequence([9], GenerationPolicy())
        self.assertEqual(result["stop_reason"], "empty_output")

    def test_missing_training_eos_rejected(self):
        with self.assertRaisesRegex(ValueError, "no EOS"):
            train.validate_target_eos(torch.tensor([[-100, 1, 2]]), 7)
        train.validate_target_eos(torch.tensor([[-100, 1, 7, 2]]), 7)

    def test_incomplete_page_does_not_abort_or_disappear_and_resume_reuses_it(self):
        rows = [
            {"page_id": "a", "image_sha256": "aaa", "reference_text": "abc"},
            {"page_id": "b", "image_sha256": "bbb", "reference_text": "def"},
        ]
        base = {
            "generated_tokens": 10,
            "seconds": 1.0,
            "last_token_id": 3,
            "eos_token_ids": [7, 9],
            "repeat_period_tokens": None,
        }
        outputs = [
            {**base, "prediction": "abc" * 100, "stop_reason": "repetition"},
            {**base, "prediction": "abc" * 100, "stop_reason": "repetition"},
            {**base, "prediction": "def", "stop_reason": "eos"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = SimpleNamespace(eval=lambda: None)
            processor = SimpleNamespace(tokenizer=self.tokenizer)
            with (
                patch.object(train, "encoded", return_value=(self.batch, None)),
                patch.object(
                    train, "generate_attempt", side_effect=outputs
                ) as generate,
            ):
                metrics = train.evaluate(
                    model, processor, rows, root, "validation", 0, 4096
                )
                self.assertEqual(generate.call_count, 3)
            self.assertEqual(metrics["pages"], 2)
            self.assertEqual(metrics["char_count"], 6)
            self.assertEqual(metrics["failed_pages"], 1)
            self.assertGreater(metrics["cer"], 1)
            saved = json.loads(
                (root / "predictions/epoch-00/validation/a.json").read_text()
            )
            self.assertEqual(saved["status"], "incomplete")
            with patch.object(
                train,
                "encoded",
                side_effect=AssertionError("Should reuse saved outputs"),
            ):
                resumed = train.evaluate(
                    model, processor, rows, root, "validation", 0, 4096
                )
            self.assertEqual(resumed, metrics)

    def test_saved_epoch_evaluation_can_complete_after_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train.write_new_json(
                root / "checkpoints/epoch-01/epoch.json",
                {
                    "epoch": 1,
                    "training": {"mean_loss": 0.1},
                    "evaluation_pending": True,
                },
            )
            with patch.object(
                train, "evaluate", side_effect=RuntimeError("hardware failure")
            ):
                with self.assertRaises(RuntimeError):
                    train.evaluate_saved_epoch(
                        None, None, {"validation": [], "test": []}, root, 1, 4096
                    )
            self.assertTrue((root / "checkpoints/epoch-01/epoch.json").is_file())
            self.assertFalse((root / "epochs/epoch-01.json").exists())
            with patch.object(
                train, "evaluate", return_value={"cer": 0.02, "failed_pages": 0}
            ):
                result = train.evaluate_saved_epoch(
                    None, None, {"validation": [], "test": []}, root, 1, 4096
                )
            self.assertNotIn("evaluation_pending", result)
            self.assertEqual(result["training"]["mean_loss"], 0.1)


if __name__ == "__main__":
    unittest.main()

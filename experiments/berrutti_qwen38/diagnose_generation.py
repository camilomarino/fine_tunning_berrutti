"""Reproduce the failed baseline page and then apply guards, without training.

Run only inside a two-A100 allocation. Evidence goes to a new private output.
The reference is used for diagnostics/scoring, never to select a decoding retry.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from experiments.berrutti_qwen38.common import file_sha256, load_pages
from experiments.berrutti_qwen38.generation_guard import (
    GenerationPolicy,
    eos_ids,
    repeated_suffix,
)
from experiments.berrutti_qwen38.train import (
    MODEL,
    REVISION,
    TARGETS,
    emit,
    encoded,
    evaluate,
    patch_visual_projection,
    write_new_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--page-id", required=True)
    parser.add_argument(
        "--guarded-only",
        action="store_true",
        help="Skip the already-recorded legacy reproduction",
    )
    args = parser.parse_args()
    os.umask(0o077)
    if torch.cuda.device_count() != 2:
        raise RuntimeError("Diagnostic requires two allocated A100 GPUs")
    for index in range(2):
        if "A100" not in torch.cuda.get_device_name(index):
            raise RuntimeError("Diagnostic requires A100 GPUs")
        if float(torch.ones(8, device=f"cuda:{index}").sum()) != 8:
            raise RuntimeError("GPU allocation unavailable")
    import causal_conv1d  # noqa: F401

    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    policy = GenerationPolicy()
    write_new_json(
        root / "config.json",
        {
            "model": MODEL,
            "revision": REVISION,
            "page_id": args.page_id,
            "guarded_only": args.guarded_only,
            "split_sha256": file_sha256(args.split),
            "generation_policy": policy.metadata(),
            "purpose": "failure diagnosis only; no training or test-based quality tuning",
            "code_sha256": {
                name: file_sha256(Path(__file__).with_name(name))
                for name in (
                    "train.py",
                    "generation_guard.py",
                    "diagnose_generation.py",
                )
            },
        },
    )
    pages = load_pages(args.dataset, args.split)
    rows = [
        row for rows in pages.values() for row in rows if row["page_id"] == args.page_id
    ]
    if len(rows) != 1:
        raise ValueError("Diagnostic page missing or duplicated")
    random.seed(20261007)
    torch.manual_seed(20261007)
    torch.cuda.manual_seed_all(20261007)
    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True
    )
    processor.image_processor.size = {"shortest_edge": 65536, "longest_edge": 2097152}
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL,
        revision=REVISION,
        dtype=torch.bfloat16,
        device_map="balanced",
        max_memory={0: "34GiB", 1: "34GiB"},
        attn_implementation="sdpa",
        local_files_only=True,
    )
    if any(
        not str(parameter.device).startswith("cuda:")
        for parameter in model.parameters()
    ):
        raise RuntimeError("Diagnostic disallows CPU/disk offload")
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            target_modules=TARGETS,
            task_type="CAUSAL_LM",
        ),
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.eval()
    example, labels = encoded(processor, pages["train"][0], target=True)
    patch_visual_projection(model, example, labels, root)
    del example, labels
    summary = None
    if not args.guarded_only:
        batch, _ = encoded(processor, rows[0], target=False)
        prompt_length = batch["input_ids"].shape[-1]
        emit(root, event="legacy_probe_started", max_new_tokens=4096)
        began = time.monotonic()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.generate(
                **batch,
                do_sample=False,
                max_new_tokens=4096,
                use_cache=True,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        tokens = output[0, prompt_length:].tolist()
        eos = eos_ids(
            model.generation_config.eos_token_id, processor.tokenizer.eos_token_id
        )
        ended = bool(tokens) and tokens[-1] in eos
        summary = {
            "generated_tokens": len(tokens),
            "last_token_id": tokens[-1] if tokens else None,
            "eos_token_ids": list(eos),
            "tokenizer_eos_token_id": processor.tokenizer.eos_token_id,
            "recognized_model_eos": ended,
            "legacy_would_raise": len(tokens) >= 4096
            and tokens[-1] != processor.tokenizer.eos_token_id,
            "repeat_period_tokens": repeated_suffix(
                tokens[:-1] if ended else tokens, policy
            ),
            "reference_tokens": len(
                processor.tokenizer.encode(
                    rows[0]["reference_text"], add_special_tokens=False
                )
            ),
            "seconds": time.monotonic() - began,
        }
        write_new_json(
            root / "legacy-output.json",
            {
                **summary,
                "prediction": processor.tokenizer.decode(
                    tokens, skip_special_tokens=True
                ),
            },
        )
        emit(root, event="legacy_probe_result", **summary)
        del batch, output
    metrics = evaluate(model, processor, rows, root, "diagnostic", 0, 4096)
    write_new_json(root / "complete.json", {"legacy": summary, "guarded": metrics})
    emit(root, event="diagnostic_completed", legacy=summary, guarded=metrics)


if __name__ == "__main__":
    main()

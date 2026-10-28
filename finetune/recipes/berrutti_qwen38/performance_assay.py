"""Bounded two-A100 performance/identity assay; no optimizer updates or epochs."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

from data.integrity import file_sha256
from finetune.recipes.berrutti_qwen38.performance import FrozenVisionCache, cache_key


def identity_metrics(reference, candidate):
    """Quantify differences, including the uncached repeat control, on CPU."""
    import torch

    reference_loss, reference_gradients, _ = reference
    loss, gradients, _ = candidate
    if gradients.keys() != reference_gradients.keys():
        raise RuntimeError("Gradient coverage changed between identical inputs")
    different = 0
    maximum_absolute = maximum_relative = 0.0
    for name, gradient in gradients.items():
        original = reference_gradients[name]
        if not torch.equal(original, gradient):
            different += 1
            delta = gradient.float() - original.float()
            maximum_absolute = max(maximum_absolute, float(delta.abs().max()))
            maximum_relative = max(
                maximum_relative,
                float(delta.norm() / original.float().norm().clamp_min(1e-12)),
            )
    return {
        "identical_loss_and_gradients": reference_loss == loss and different == 0,
        "loss_identical": reference_loss == loss,
        "loss_absolute_difference": abs(reference_loss - loss),
        "different_gradients": different,
        "maximum_gradient_absolute_difference": maximum_absolute,
        "maximum_gradient_relative_l2": maximum_relative,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    events = args.output / "events.jsonl"

    def emit(event, **data):
        row = {"event": event, "elapsed": time.monotonic() - started, **data}
        with events.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)

    started = time.monotonic()
    emit("starting", optimizer_steps=0, purpose="performance assay, not training")
    import importlib.metadata

    import torch
    from PIL import Image
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from transformers import StoppingCriteria, StoppingCriteriaList

    from finetune.recipes.berrutti_qwen38 import eager_hooks, projection_runtime
    from finetune.recipes.berrutti_qwen38.generation_guard import (
        DecodeGuard,
        GenerationPolicy,
        eos_ids,
        generate_with_retries,
    )

    config = json.loads(args.config.read_text())
    if torch.cuda.device_count() != 2:
        raise RuntimeError("This assay requires exactly two visible A100 GPUs")
    for device in range(2):
        if "A100" not in torch.cuda.get_device_name(device):
            raise RuntimeError("Unexpected GPU class")
        with torch.cuda.device(device):
            assert torch.ones(8, device="cuda").sum().item() == 8
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "transformers",
            "accelerate",
            "peft",
            "flash-linear-attention",
        )
    }
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    emit("gpu_runtime", cuda_device_count=2, versions=versions, success=True)
    emit("routing", **eager_hooks.install())
    emit("routing_smoke", **eager_hooks.verify_routing())
    emit("model_loading")
    loaded_at = time.monotonic()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        config["model_root"],
        dtype=torch.bfloat16,
        device_map="balanced",
        max_memory={0: "34GiB", 1: "34GiB"},
        attn_implementation="sdpa",
        local_files_only=True,
    )
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise RuntimeError(
            "CPU/disk model offload is not an accepted performance profile"
        )
    targets = r".*language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)"
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            target_modules=targets,
            task_type="CAUSAL_LM",
        ),
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    visual = model.base_model.model.model.visual
    visual.eval()
    projection = projection_runtime.install(model)
    emit(
        "loaded",
        seconds=time.monotonic() - loaded_at,
        trainable_parameters=sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    )
    processor = AutoProcessor.from_pretrained(
        config["model_root"], local_files_only=True
    )
    processor.tokenizer.padding_side = "left"
    instruction = "Transcribe fielmente todo el texto de esta imagen en orden de lectura. Conserva los saltos de línea. Devuelve solo la transcripción, sin explicaciones ni correcciones."
    dataset = Path(config["dataset_root"])
    prepared = []
    preparation_started = time.monotonic()
    for page in config["samples"]:
        for key in ("image", "reference"):
            if file_sha256(dataset / page[key]) != page[f"{key}_sha256"]:
                raise ValueError("Assay input differs from sealed source data")
        with Image.open(dataset / page["image"]) as source:
            image = source.convert("RGB")
        examples = [
            (
                page["sample"],
                image,
                (dataset / page["reference"]).read_text(),
                config["page_max_pixels"],
            )
        ]
        examples += [
            (
                f"{page['sample']}-line-{index}",
                image.crop(line["crop_xyxy"]),
                line["target"],
                config["line_max_pixels"],
            )
            for index, line in enumerate(page["lines"])
        ]
        for name, example_image, reference, max_pixels in examples:
            processor.image_processor.size = {
                "shortest_edge": config["min_pixels"],
                "longest_edge": max_pixels,
            }
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example_image},
                        {"type": "text", "text": instruction},
                    ],
                }
            ]
            kwargs = dict(
                tokenize=True,
                enable_thinking=False,
                return_dict=True,
                return_tensors="pt",
            )
            prefix = processor.apply_chat_template(
                messages, add_generation_prompt=True, **kwargs
            )
            full = processor.apply_chat_template(
                messages
                + [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": reference}],
                    }
                ],
                add_generation_prompt=False,
                **kwargs,
            )
            prompt_tokens = prefix["input_ids"].shape[-1]
            if not torch.equal(
                prefix["input_ids"], full["input_ids"][:, :prompt_tokens]
            ):
                raise RuntimeError("Prompt prefix mismatch")
            keep = full["input_ids"].shape[-1] - prompt_tokens + 1
            labels = full["input_ids"][:, -keep:].clone()
            labels[:, 0] = -100
            identity = {
                "model_revision": config["revision"],
                "image": page["image_sha256"],
                "example": name,
                "pixel_sha256": __import__("hashlib")
                .sha256(full["pixel_values"].numpy().tobytes())
                .hexdigest(),
                "grid": full["image_grid_thw"].tolist(),
                "projection": projection_runtime.POLICY,
                "versions": versions,
                "dtype": "bf16",
                "attention": "sdpa",
            }
            prepared.append(
                {
                    "name": name,
                    "prefix": prefix,
                    "full": full,
                    "labels": labels,
                    "keep": keep,
                    "key": cache_key(identity),
                }
            )
    emit(
        "inputs_prepared",
        seconds=time.monotonic() - preparation_started,
        samples=len(prepared),
    )

    def sync():
        for device in range(2):
            torch.cuda.synchronize(device)

    def training(example, cache=None):
        model.train()
        visual.eval()
        model.config.use_cache = False
        batch = {key: value.to("cuda:0") for key, value in example["full"].items()}
        model.zero_grad(set_to_none=True)
        sync()
        tick = time.monotonic()
        with cache.activate(example["key"]) if cache else nullcontext():
            output = model(
                **batch,
                labels=example["labels"].to("cuda:0"),
                logits_to_keep=example["keep"],
                use_cache=False,
            )
            loss = output.loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
        sync()
        seconds = time.monotonic() - tick
        gradients = {}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise RuntimeError("Missing or non-finite LoRA gradient")
                gradients[name] = parameter.grad.detach().cpu().clone()
        result = (float(loss.detach()), gradients, seconds)
        del output, loss, batch
        model.zero_grad(set_to_none=True)
        return result

    model.eval()
    emit(
        "projection",
        **projection_runtime.verify(projection, prepared[0]["full"]["pixel_values"]),
    )
    cache = FrozenVisionCache(visual)
    if config.get("identity_diagnostic", False):
        example = prepared[0]
        emit("uncached_control_reference_start")
        reference = training(example)
        results = []
        for phase, active_cache in (
            ("uncached_repeat", None),
            ("cache_fill", cache),
            ("cache_hit", cache),
            ("cache_hit_repeat", cache),
        ):
            emit("identity_call_start", representation=phase)
            candidate = training(example, active_cache)
            row = {
                "representation": phase,
                "seconds": candidate[2],
                **identity_metrics(reference, candidate),
            }
            emit("identity_control", **row)
            results.append(row)
            del candidate
        cache.close()
        summary = {
            "success": True,
            "diagnostic_only": True,
            "optimization_accepted": False,
            "training_identity_passed": all(
                row["identical_loss_and_gradients"] for row in results
            ),
            "cuda_device_count": 2,
            "optimizer_steps": 0,
            "control_results": results,
            "generation_checked": False,
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        emit("performance_identity_diagnostic_completed", **summary)
        return
    measurements = []
    selected = [
        prepared[0],
        next(row for row in prepared if row["name"] == "page-2"),
        prepared[1],
    ]
    for example in selected:
        emit("training_reference_start", sample=example["name"])
        reference_loss, reference_gradients, reference_seconds = training(example)
        # One cold fill is separately timed; subsequent epochs can reuse its features.
        _, _, fill_seconds = training(example, cache)
        loss, gradients, hit_seconds = training(example, cache)
        identical = reference_loss == loss and all(
            torch.equal(reference_gradients[name], gradients[name])
            for name in gradients
        )
        row = {
            "sample": example["name"],
            "reference_seconds": reference_seconds,
            "cache_fill_seconds": fill_seconds,
            "cache_hit_seconds": hit_seconds,
            "loss": loss,
            "identical_loss_and_gradients": identical,
        }
        measurements.append(row)
        emit("training_identity", **row)
        if not identical:
            raise RuntimeError("Frozen visual cache changed loss or gradients")
        del reference_gradients, gradients

    policy = GenerationPolicy()
    eos = eos_ids(
        model.generation_config.eos_token_id, processor.tokenizer.eos_token_id
    )

    def decode(example, cache_enabled):
        model.eval()
        model.gradient_checkpointing_disable()
        prefix = {key: value.to("cuda:0") for key, value in example["prefix"].items()}
        prompt_tokens = prefix["input_ids"].shape[-1]

        def attempt(parameters):
            guard = DecodeGuard(policy, parameters, eos)
            tick = time.monotonic()

            class Stop(StoppingCriteria):
                def __call__(self, ids, scores, **kwargs):
                    generated = ids.shape[-1] - prompt_tokens
                    reason = guard.observe(
                        ids[0, prompt_tokens:][-policy.tail_tokens :].tolist(),
                        generated=generated,
                        elapsed=time.monotonic() - tick,
                    )
                    return torch.full(
                        (1,), reason is not None, dtype=torch.bool, device=ids.device
                    )

            arguments = {
                key: value for key, value in parameters.items() if key != "max_seconds"
            }
            with (
                torch.inference_mode(),
                cache.activate(example["key"]) if cache_enabled else nullcontext(),
            ):
                ids = model.generate(
                    **prefix,
                    **arguments,
                    use_cache=True,
                    stopping_criteria=StoppingCriteriaList([Stop()]),
                )
            sync()
            tokens = ids[0, prompt_tokens:].tolist()
            reason = guard.observe(
                tokens[-policy.tail_tokens :],
                generated=len(tokens),
                elapsed=time.monotonic() - tick,
                final=True,
            )
            return {
                "prediction": processor.tokenizer.decode(
                    tokens, skip_special_tokens=True
                ),
                "tokens": tokens,
                "stop_reason": reason,
                "seconds": time.monotonic() - tick,
            }

        result = generate_with_retries(attempt, policy, lambda row: None)
        return result

    emit("generation_reference_start")
    reference = decode(prepared[0], False)
    candidate = decode(prepared[0], True)
    (args.output / "generation-private.json").write_text(
        json.dumps({"reference": reference, "cached": candidate}, ensure_ascii=False)
    )
    identical = (
        reference["status"] == candidate["status"] == "complete"
        and reference["attempts"][-1]["tokens"] == candidate["attempts"][-1]["tokens"]
    )
    emit(
        "generation_identity",
        identical_tokens=identical,
        reference_status=reference["status"],
        candidate_status=candidate["status"],
        reference_seconds=sum(a["seconds"] for a in reference["attempts"]),
        cache_seconds=sum(a["seconds"] for a in candidate["attempts"]),
        generated_tokens=len(candidate["attempts"][-1]["tokens"]),
        failed_pages=int(candidate["status"] != "complete"),
        retried_pages=int(len(candidate["attempts"]) == 2),
    )
    if not identical:
        raise RuntimeError(
            "Cached generation did not reproduce complete reference output"
        )
    cache.close()
    summary = {
        "success": True,
        "optimizer_steps": 0,
        "cuda_device_count": 2,
        "training_identity": measurements,
        "generation_tokens_identical": True,
        "full_generated_pages": 1,
        "not_complete_validation": True,
        "median_cache_hit_ratio": statistics.median(
            row["cache_hit_seconds"] / row["reference_seconds"] for row in measurements
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    emit("performance_assay_completed", **summary)


if __name__ == "__main__":
    main()

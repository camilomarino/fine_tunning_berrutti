"""Bounded 4B FP16 V100 fallback: prove the OCR loop while 27B waits for A100s."""

from __future__ import annotations

import json
import os
import random
import time
import types
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from experiments.berrutti_qwen38.common import file_sha256, load_pages, score_pairs
from experiments.berrutti_qwen38.train import TARGETS, encoded

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def emit(root: Path, **value: object) -> None:
    line = json.dumps(
        {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **value}
    )
    with (root / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def use_equivalent_visual_projection(model, root: Path) -> None:
    projection = model.base_model.model.model.visual.patch_embed.proj
    if not (
        projection.kernel_size == projection.stride
        and projection.padding == (0, 0, 0)
        and projection.dilation == (1, 1, 1)
        and projection.groups == 1
    ):
        raise RuntimeError("Visual patch projection cannot be replaced safely")
    sample = torch.randn(
        32,
        projection.in_channels,
        *projection.kernel_size,
        device=projection.weight.device,
        dtype=projection.weight.dtype,
    )
    with torch.inference_mode():
        convolution = projection(sample)
        linear = torch.nn.functional.linear(
            sample.flatten(1),
            projection.weight.flatten(1),
            projection.bias,
        ).view_as(convolution)
    relative_l2 = float(
        (convolution.float() - linear.float()).norm() / convolution.float().norm()
    )
    if relative_l2 >= 0.01:
        raise RuntimeError(f"Visual projection mismatch: {relative_l2}")

    def forward(self, value):
        if tuple(value.shape[2:]) != self.kernel_size:
            raise ValueError("Unexpected patch shape")
        return torch.nn.functional.linear(
            value.flatten(1),
            self.weight.flatten(1),
            self.bias,
        )[:, :, None, None, None]

    projection.forward = types.MethodType(forward, projection)
    emit(root, event="visual_projection_parity", relative_l2=relative_l2)


def generate(model, processor, row: dict, root: Path, stage: str) -> float:
    batch, _ = encoded(processor, row, target=False)
    prefix = batch["input_ids"].shape[-1]
    model.eval()
    with torch.inference_mode():
        output = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=2048,
            use_cache=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    new_ids = output[0, prefix:]
    if len(new_ids) >= 2048 and int(new_ids[-1]) != processor.tokenizer.eos_token_id:
        raise RuntimeError(f"Generation was capped: {row['page_id']}")
    prediction = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
    metrics = score_pairs([(row["reference_text"], prediction)])
    path = root / "predictions" / stage
    path.mkdir(parents=True, exist_ok=True)
    with (path / f"{row['page_id']}.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {"reference": row["reference_text"], "prediction": prediction},
            stream,
            ensure_ascii=False,
        )
    emit(root, event="generation", stage=stage, page_id=row["page_id"], **metrics)
    return metrics["cer"]


def main() -> None:
    os.umask(0o077)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one V100")
    device_name = torch.cuda.get_device_name(0)
    if "V100" not in device_name:
        raise RuntimeError(f"Expected V100, got {device_name}")
    if float(torch.ones(8, device="cuda:0").sum()) != 8.0:
        raise RuntimeError("GPU allocation smoke failed")
    import causal_conv1d  # noqa: F401

    root = Path(os.environ["EXPERIMENT_ROOT"])
    output = root / "runs/v100-4b-fp16-sanity-r1"
    output.mkdir(parents=True, exist_ok=False)
    random.seed(20261007)
    torch.manual_seed(20261007)
    torch.set_num_threads(4)
    split = root / "code/experiments/berrutti_qwen38/split-v1.json"
    pages = load_pages(root / "dataset", split)
    train_ids = {"low_r0290_0317", "medium_r1060-050"}
    train = [row for row in pages["train"] if row["page_id"] in train_ids]
    validation = min(pages["validation"], key=lambda row: len(row["reference_text"]))
    test = min(pages["test"], key=lambda row: len(row["reference_text"]))
    emit(
        output,
        event="start",
        model=MODEL,
        revision=REVISION,
        dtype="float16",
        device=device_name,
        train_pages=len(train),
        validation_pages=1,
        test_pages=1,
        split_sha256=file_sha256(split),
        code_sha256=file_sha256(Path(__file__)),
    )
    processor = AutoProcessor.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True
    )
    processor.image_processor.size = {"shortest_edge": 65536, "longest_edge": 1048576}
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL,
        revision=REVISION,
        dtype=torch.float16,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
        local_files_only=True,
    )
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
    model.base_model.model.model.visual.eval()
    use_equivalent_visual_projection(model, output)
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(parameters, lr=5e-5, weight_decay=0.01)
    emit(
        output, event="loaded", trainable_parameters=sum(p.numel() for p in parameters)
    )
    before = {
        "validation": generate(
            model, processor, validation, output, "baseline-validation"
        ),
        "test": generate(model, processor, test, output, "baseline-test"),
    }
    losses = []
    for epoch in range(1, 5):
        model.train()
        model.base_model.model.model.visual.eval()
        for row in train:
            batch, labels = encoded(processor, row, target=True)
            optimizer.zero_grad(set_to_none=True)
            loss = model(
                **batch, labels=labels, logits_to_keep=labels.shape[-1], use_cache=False
            ).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss: {row['page_id']}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError(f"Non-finite gradient: {row['page_id']}")
            optimizer.step()
            losses.append(float(loss.detach()))
            emit(
                output,
                event="train_step",
                epoch=epoch,
                page_id=row["page_id"],
                loss=losses[-1],
                gradient_norm=float(norm),
            )
            del batch, labels, loss
    after = {
        "validation": generate(
            model, processor, validation, output, "trained-validation"
        ),
        "test": generate(model, processor, test, output, "trained-test"),
    }
    if not sum(losses[-2:]) < sum(losses[:2]):
        raise RuntimeError("Tiny-fit loss did not decrease")
    with (output / "complete.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "initial_mean_loss": sum(losses[:2]) / 2,
                "final_mean_loss": sum(losses[-2:]) / 2,
                "baseline_cer": before,
                "trained_cer": after,
            },
            stream,
            indent=2,
        )
    emit(
        output,
        event="completed",
        initial_mean_loss=sum(losses[:2]) / 2,
        final_mean_loss=sum(losses[-2:]) / 2,
        baseline_cer=before,
        trained_cer=after,
    )


if __name__ == "__main__":
    main()

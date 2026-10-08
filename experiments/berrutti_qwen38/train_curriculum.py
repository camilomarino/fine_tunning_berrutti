"""Independent Qwen3.8-27B crop curriculum plus conservative page augmentation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import time
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from experiments.berrutti_qwen38.augmentation import (
    conservative_augment,
    padded_crop,
    presentation_seed,
)
from experiments.berrutti_qwen38.common import NORMALIZERS, file_sha256, load_pages
from experiments.berrutti_qwen38.generation_guard import (
    GenerationPolicy,
    select_complete_validation_epoch,
)
from experiments.berrutti_qwen38.train import (
    KERNEL_REVISION,
    MODEL,
    PROMPT,
    REVISION,
    TARGETS,
    emit,
    evaluate_saved_epoch,
    patch_visual_projection,
    save_checkpoint,
    validate_target_eos,
    write_new_json,
)
from experiments.berrutti_qwen38.scoring import normalize_text


def encode_training(
    processor, row: dict, *, epoch: int, position: int, stage: str
) -> tuple[dict, torch.Tensor, dict]:
    with Image.open(row["image"]) as source:
        image = source.convert("RGB")
    is_page = stage == "page"
    if not is_page:
        image = padded_crop(image, row["crop_xyxy"])
    seed = presentation_seed(
        20261007,
        stage=stage,
        epoch=epoch,
        position=position,
        page_id=row["page_id"],
        region=row.get("region_index"),
    )
    image, operations = conservative_augment(image, seed, page=is_page)
    processor.image_processor.size = {
        "shortest_edge": 65536,
        "longest_edge": 2097152 if is_page else 262144,
    }
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    prefix = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )
    full = processor.apply_chat_template(
        messages
        + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": row["reference_text"]}],
            }
        ],
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )
    length = prefix["input_ids"].shape[-1]
    if not torch.equal(prefix["input_ids"], full["input_ids"][:, :length]):
        raise ValueError(f"Prompt/target prefix mismatch: {row['page_id']}")
    keep = full["input_ids"].shape[-1] - length + 1
    if keep < 2 or full["input_ids"].shape[-1] > 8192:
        raise ValueError(f"Empty or oversized supervised target: {row['page_id']}")
    labels = full["input_ids"][:, -keep:].clone()
    labels[:, 0] = -100
    validate_target_eos(labels, processor.tokenizer.eos_token_id)
    inputs = {key: value.to("cuda:0") for key, value in full.items()}
    return inputs, labels.to("cuda:0"), operations


def load_crops(path: Path, dataset: Path, train_pages: list[dict]) -> list[dict]:
    pages = {row["page_id"]: row for row in train_pages}
    crops = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        crop = json.loads(line)
        page_id = crop["page_id"]
        if page_id not in pages:
            raise ValueError(f"Crop outside training split: {page_id}")
        page = pages[page_id]
        image = (dataset / crop["image"]).resolve()
        if image != page["image"] or crop["image_sha256"] != page["image_sha256"]:
            raise ValueError(f"Crop references a different image: {page_id}")
        target = crop["target"]
        if hashlib.sha256(target.encode()).hexdigest() != crop["target_sha256"]:
            raise ValueError(f"Crop target changed: {page_id}")
        if normalize_text(target, NORMALIZERS) not in normalize_text(
            page["reference_text"], NORMALIZERS
        ):
            raise ValueError(f"Crop target unaligned with page: {page_id}")
        key = (page_id, crop["region_index"])
        if key in seen:
            raise ValueError(f"Duplicate crop: {key}")
        seen.add(key)
        crops.append({**crop, "image": image, "reference_text": target})
    if len(crops) != 989:
        raise ValueError(f"Expected the sealed 989-crop curriculum, got {len(crops)}")
    return crops


def make_scheduler(optimizer, *, steps: int, warmup_ratio: float):
    warmup = max(1, math.ceil(steps * warmup_ratio))

    def factor(step: int) -> float:
        if step < warmup:
            return max(0.01, (step + 1) / warmup)
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_stage(
    model,
    processor,
    rows: list[dict],
    root: Path,
    *,
    stage: str,
    epoch: int,
    optimizer,
    scheduler,
    parameters,
    rng: random.Random,
) -> dict:
    model.train()
    model.base_model.model.model.visual.eval()
    ordered = rows.copy()
    rng.shuffle(ordered)
    optimizer.zero_grad(set_to_none=True)
    losses = []
    began = time.perf_counter()
    manifest = root / "presentations" / f"{stage}-epoch-{epoch:02d}.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest.exists():
        raise FileExistsError(manifest)
    with manifest.open("x", encoding="utf-8") as presentations:
        for index, row in enumerate(ordered, 1):
            batch, labels, operations = encode_training(
                processor, row, epoch=epoch, position=index, stage=stage
            )
            presentations.write(
                json.dumps(
                    {
                        "page_id": row["page_id"],
                        "region_index": row.get("region_index"),
                        "source_image_sha256": row["image_sha256"],
                        "target_sha256": row.get("target_sha256"),
                        "augmentations": operations,
                    }
                )
                + "\n"
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(
                    **batch,
                    labels=labels,
                    logits_to_keep=labels.shape[-1],
                    use_cache=False,
                ).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite {stage} loss: {row['page_id']}")
            losses.append(float(loss.detach()))
            group_start = ((index - 1) // 8) * 8
            group_size = min(8, len(ordered) - group_start)
            (loss / group_size).backward()
            if index % 8 == 0 or index == len(ordered):
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                if not torch.isfinite(norm):
                    raise RuntimeError(
                        f"Non-finite {stage} gradients: {row['page_id']}"
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                emit(
                    root,
                    event="optimizer_step",
                    stage=stage,
                    epoch=epoch,
                    page=index,
                    total=len(ordered),
                    loss=losses[-1],
                    gradient_norm=float(norm),
                    lr=scheduler.get_last_lr()[0],
                    peak_gpu_bytes=[
                        torch.cuda.max_memory_reserved(i) for i in range(2)
                    ],
                )
            del batch, labels, loss
    result = {
        "mean_loss": sum(losses) / len(losses),
        "seconds": time.perf_counter() - began,
        "presentations": len(ordered),
    }
    emit(root, event="training_epoch", stage=stage, epoch=epoch, **result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("sanity", "full"), required=True)
    args = parser.parse_args()
    generation_policy = GenerationPolicy()
    os.umask(0o077)
    if torch.cuda.device_count() != 2:
        raise RuntimeError("Curriculum experiment requires two A100s")
    for index in range(2):
        if "A100" not in torch.cuda.get_device_name(index):
            raise RuntimeError("Non-A100 GPU allocation")
        if float(torch.ones(8, device=f"cuda:{index}").sum()) != 8:
            raise RuntimeError(f"GPU {index} unavailable")
    import causal_conv1d  # noqa: F401

    dataset = args.dataset.resolve()
    pages = load_pages(dataset, args.split)
    crops = load_crops(args.crops, dataset, pages["train"])
    if args.mode == "sanity":
        chosen = sorted({crop["page_id"] for crop in crops})[:2]
        crops = [
            crop
            for page_id in chosen
            for crop in [row for row in crops if row["page_id"] == page_id][:2]
        ]
        pages["train"] = [row for row in pages["train"] if row["page_id"] in chosen]
        pages["validation"] = sorted(
            pages["validation"], key=lambda row: len(row["reference_text"])
        )[:1]
        pages["test"] = sorted(
            pages["test"], key=lambda row: len(row["reference_text"])
        )[:1]
        if len(crops) != 4 or len(pages["train"]) != 2:
            raise ValueError("Sanity curriculum requires four crops from two pages")
    else:
        sanity = args.output.parent / "curriculum-sanity-v1" / "complete.json"
        if not sanity.is_file():
            raise RuntimeError("The curriculum sanity run has not completed")
        sanity_result = json.loads(sanity.read_text())
        sanity_config = json.loads((sanity.parent / "config.json").read_text())
        if sanity_config.get("generation_policy") != generation_policy.metadata():
            raise RuntimeError("Curriculum sanity must use the same generation guards")
        if sanity_result.get("best_validation_cer") is None:
            raise RuntimeError("Curriculum sanity has no complete validation epoch")
        if sanity_result["epochs"] != 2:
            raise RuntimeError("Curriculum sanity run has incomplete evaluations")
    baseline = json.loads((args.baseline / "complete.json").read_text())
    baseline_config = json.loads((args.baseline / "config.json").read_text())
    if baseline_config.get("generation_policy") != generation_policy.metadata():
        raise ValueError("Baseline decoding policy differs; comparison withheld")
    if baseline.get("best_validation_cer") is None:
        raise ValueError("Baseline has no complete validation epoch")
    if baseline_config["model"] != MODEL or baseline_config[
        "split_sha256"
    ] != file_sha256(args.split):
        raise ValueError("Baseline model or split differs")
    if (
        baseline_config["max_pixels"] != 2097152
        or baseline_config["revision"] != REVISION
    ):
        raise ValueError("Baseline image budget or checkpoint revision differs")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    config = {
        "model": MODEL,
        "revision": REVISION,
        "kernel_revision": KERNEL_REVISION,
        "split_sha256": file_sha256(args.split),
        "dataset_manifest_sha256": file_sha256(dataset / "manifest.json"),
        "crop_manifest_sha256": file_sha256(args.crops),
        "baseline_complete_sha256": file_sha256(args.baseline / "complete.json"),
        "code_sha256": file_sha256(Path(__file__)),
        "evaluation_code_sha256": file_sha256(Path(__file__).with_name("train.py")),
        "generation_guard_sha256": file_sha256(
            Path(__file__).with_name("generation_guard.py")
        ),
        "generation_policy": generation_policy.metadata(),
        "augmentation_code_sha256": file_sha256(
            Path(__file__).with_name("augmentation.py")
        ),
        "seed": 20261007,
        "mode": args.mode,
        "crop_presentations": len(crops),
        "page_presentations": (3 if args.mode == "full" else 1) * len(pages["train"]),
        "page_max_pixels": 2097152,
        "crop_max_pixels": 262144,
        "adaptation": "LoRA r16 alpha32 dropout0.1 language only; frozen vision",
        "selection": "lowest generated validation CER across region and page stages",
        "baseline_best_validation_cer": baseline["best_validation_cer"],
        "normalizers": list(NORMALIZERS),
    }
    write_new_json(root / "config.json", config)
    random.seed(20261007)
    torch.manual_seed(20261007)
    torch.cuda.manual_seed_all(20261007)
    torch.set_num_threads(4)
    emit(
        root,
        event="start",
        **config,
        versions={
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "flash-linear-attention")
        },
    )
    processor = AutoProcessor.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True
    )
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
        raise RuntimeError("CPU/disk offload is outside this experiment")
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            target_modules=TARGETS,
            task_type="CAUSAL_LM",
        ),
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.base_model.model.model.visual.eval()
    if any(
        parameter.requires_grad
        for parameter in model.base_model.model.model.visual.parameters()
    ):
        raise RuntimeError("Vision encoder was not frozen")
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    first_batch, first_labels, _ = encode_training(
        processor, crops[0], epoch=0, position=0, stage="region"
    )
    model.eval()  # disable LoRA dropout for the before/after projection parity gate
    patch_visual_projection(model, first_batch, first_labels, root)
    del first_batch, first_labels
    emit(root, event="loaded", trainable_parameters=sum(p.numel() for p in parameters))
    rng = random.Random(20261007)
    region_optimizer = torch.optim.AdamW(parameters, lr=5e-5, weight_decay=0.01)
    region_scheduler = make_scheduler(
        region_optimizer, steps=math.ceil(len(crops) / 8), warmup_ratio=0.05
    )
    region_train = train_stage(
        model,
        processor,
        crops,
        root,
        stage="region",
        epoch=0,
        optimizer=region_optimizer,
        scheduler=region_scheduler,
        parameters=parameters,
        rng=rng,
    )
    processor.image_processor.size = {"shortest_edge": 65536, "longest_edge": 2097152}
    region_meta = {
        "epoch": 0,
        "stage": "region",
        "training": region_train,
        "evaluation_pending": True,
    }
    save_checkpoint(model, region_optimizer, region_scheduler, root, 0, region_meta)
    evaluate_saved_epoch(model, processor, pages, root, 0, 4096)
    page_optimizer = torch.optim.AdamW(parameters, lr=2e-5, weight_decay=0.01)
    page_epochs = 3 if args.mode == "full" else 1
    page_scheduler = make_scheduler(
        page_optimizer,
        steps=math.ceil(len(pages["train"]) / 8) * page_epochs,
        warmup_ratio=0.05,
    )
    for epoch in range(1, page_epochs + 1):
        train = train_stage(
            model,
            processor,
            pages["train"],
            root,
            stage="page",
            epoch=epoch,
            optimizer=page_optimizer,
            scheduler=page_scheduler,
            parameters=parameters,
            rng=rng,
        )
        processor.image_processor.size = {
            "shortest_edge": 65536,
            "longest_edge": 2097152,
        }
        metadata = {
            "epoch": epoch,
            "stage": "page",
            "training": train,
            "evaluation_pending": True,
        }
        save_checkpoint(model, page_optimizer, page_scheduler, root, epoch, metadata)
        evaluate_saved_epoch(model, processor, pages, root, epoch, 4096)
    summaries = [
        json.loads(path.read_text())
        for path in sorted((root / "epochs").glob("epoch-*.json"))
    ]
    best = select_complete_validation_epoch(summaries)
    variant_wins = (
        best is not None and best["validation"]["cer"] < baseline["best_validation_cer"]
    )
    write_new_json(
        root / "complete.json",
        {
            "best_epoch_by_validation_cer": best["epoch"] if best else None,
            "best_validation_cer": best["validation"]["cer"] if best else None,
            "test_cer_at_selected_epoch": best["test"]["cer"] if best else None,
            "selected_test_failed_pages": best["test"]["failed_pages"]
            if best
            else None,
            "selection_status": "selected" if best else "no_complete_validation_epoch",
            "baseline_best_validation_cer": baseline["best_validation_cer"],
            "recommended_run_by_validation_cer": (
                ("curriculum-v1" if variant_wins else "full")
                if args.mode == "full"
                else None
            ),
            "selection_uses_test": False,
            "epochs": len(summaries),
        },
    )
    emit(
        root,
        event="completed",
        best_epoch=best["epoch"] if best else None,
        best_validation_cer=best["validation"]["cer"] if best else None,
    )


if __name__ == "__main__":
    main()

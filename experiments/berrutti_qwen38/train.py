"""Single-process, two-A100 BF16 LoRA OCR experiment for Berrutti v3.

Run only inside a Slurm GPU allocation. The model is partitioned by layer;
ordinary two-process DDP would replicate its weights and exceed A100 memory.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import random
import time
import types
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoProcessor,
    Qwen3_5ForConditionalGeneration,
    StoppingCriteria,
    StoppingCriteriaList,
)

from experiments.berrutti_qwen38.common import file_sha256, load_pages, score_pairs
from experiments.berrutti_qwen38.generation_guard import (
    DecodeGuard,
    GenerationPolicy,
    eos_ids,
    generate_with_retries,
    select_complete_validation_epoch,
)

MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
KERNEL_REVISION = "f2651e776f66069cdcf842840db637583def1223"
PROMPT = (
    "Transcribe fielmente todo el texto de esta imagen en orden de lectura. "
    "Conserva los saltos de línea. Devuelve solo la transcripción, "
    "sin explicaciones ni correcciones."
)
TARGETS = (
    r".*language_model\..*\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|"
    r"in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)"
)
_REPORT_STARTED = time.monotonic()
_LAST_REPORT = _REPORT_STARTED
_LATEST: dict[str, object] = {}


def emit(root: Path, **payload: object) -> None:
    global _LAST_REPORT
    line = json.dumps(
        {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **payload}
    )
    with (root / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(line, flush=True)
    event = payload.get("event")
    if event == "optimizer_step":
        _LATEST["training"] = {
            key: payload.get(key) for key in ("epoch", "page", "loss", "lr")
        }
    elif event == "evaluation":
        _LATEST[str(payload["split"])] = {
            key: payload.get(key) for key in ("epoch", "cer", "wer", "pages")
        }
    now = time.monotonic()
    if now - _LAST_REPORT >= 7200:
        report = {
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_hours": (now - _REPORT_STARTED) / 3600,
            "latest": _LATEST.copy(),
            "last_event": event,
        }
        with (root / "progress-2h.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(report) + "\n")
        print(json.dumps({"event": "two_hour_report", **report}), flush=True)
        _LAST_REPORT = now


def write_new_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def encoded(processor, row: dict, *, target: bool) -> tuple[dict, torch.Tensor | None]:
    with Image.open(row["image"]) as source:
        image = source.convert("RGB")
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
    if not target:
        return {key: value.to("cuda:0") for key, value in prefix.items()}, None
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
        raise ValueError(f"Prompt/target token prefix differs: {row['page_id']}")
    keep = full["input_ids"].shape[-1] - length + 1
    if keep < 2 or full["input_ids"].shape[-1] > 8192:
        raise ValueError(f"Empty or oversized supervised sequence: {row['page_id']}")
    labels = full["input_ids"][:, -keep:].clone()
    labels[:, 0] = -100  # first label is the unsupervised prompt boundary
    validate_target_eos(labels, processor.tokenizer.eos_token_id)
    return {key: value.to("cuda:0") for key, value in full.items()}, labels.to("cuda:0")


def validate_target_eos(labels, eos: int | None) -> None:
    if eos is None or not bool((labels[:, 1:] == eos).any()):
        raise ValueError(
            "Supervised assistant target has no EOS; refusing to train a non-terminating target"
        )


def patch_visual_projection(
    model, batch: dict, labels: torch.Tensor, root: Path
) -> None:
    """Check the exact-patch Conv3d/linear equivalence before enabling it."""
    projection = model.base_model.model.model.visual.patch_embed.proj
    if not (
        projection.kernel_size == projection.stride
        and projection.padding == (0, 0, 0)
        and projection.dilation == (1, 1, 1)
        and projection.groups == 1
    ):
        raise ValueError(
            "Visual patch convolution no longer satisfies linear equivalence"
        )
    value = batch["pixel_values"].to(
        projection.weight.device, dtype=projection.weight.dtype
    )
    value = value.view(-1, projection.in_channels, *projection.kernel_size)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        convolution = projection(value)
        linear = torch.nn.functional.linear(
            value.flatten(1), projection.weight.flatten(1), projection.bias
        ).view_as(convolution)
        relative_l2 = float(
            (convolution.float() - linear.float()).norm() / convolution.float().norm()
        )
        if relative_l2 >= 0.01:
            raise ValueError(f"Visual projection changed output: {relative_l2}")
        keep = labels.shape[-1]
        before = model(
            **batch, labels=labels, logits_to_keep=keep, use_cache=False
        ).loss.detach()

    def linear_forward(self, x):
        if tuple(x.shape[2:]) != self.kernel_size:
            raise ValueError("Unexpected visual patch shape")
        return torch.nn.functional.linear(
            x.flatten(1), self.weight.flatten(1), self.bias
        )[:, :, None, None, None]

    projection.forward = types.MethodType(linear_forward, projection)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        after = model(
            **batch, labels=labels, logits_to_keep=keep, use_cache=False
        ).loss.detach()
    if not torch.allclose(before, after, rtol=0.01, atol=1e-4):
        raise ValueError(
            f"Visual projection changed supervised loss: {before} -> {after}"
        )
    emit(
        root,
        event="visual_projection_parity",
        relative_l2=relative_l2,
        loss_before=float(before),
        loss_after=float(after),
    )


class GenerationStopping(StoppingCriteria):
    """Single-page stop criterion; inspect generated tokens, never the prompt."""

    def __init__(self, guard: DecodeGuard, prompt_length: int, started: float):
        self.guard = guard
        self.prompt_length = prompt_length
        self.started = started

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[0] != 1:
            raise ValueError("OCR guard supports one page/beam at a time")
        generated = input_ids.shape[-1] - self.prompt_length
        last = int(input_ids[0, -1])
        tail = [last]
        if last in self.guard.eos or generated % self.guard.policy.check_interval == 0:
            start = max(
                self.prompt_length,
                input_ids.shape[-1] - self.guard.policy.tail_tokens - 1,
            )
            tail = input_ids[0, start:].tolist()
        reason = self.guard.observe(
            tail, generated=generated, elapsed=time.monotonic() - self.started
        )
        return torch.tensor(
            [reason is not None], dtype=torch.bool, device=input_ids.device
        )


def generate_attempt(
    model, tokenizer, batch: dict, policy: GenerationPolicy, parameters: dict
) -> dict:
    """Return a bounded native output and its actual termination reason.

    Time limits are cooperative: they cannot interrupt a stuck CUDA kernel.
    CUDA/OOM/invalid-input errors remain fatal rather than being mislabeled as
    bad OCR; the caller keeps the last pre-evaluation checkpoint in that case.
    """
    eos = eos_ids(model.generation_config.eos_token_id, tokenizer.eos_token_id)
    prompt_length = batch["input_ids"].shape[-1]
    guard = DecodeGuard(policy, parameters, eos)
    started = time.monotonic()
    stopping = GenerationStopping(guard, prompt_length, started)
    kwargs = {key: value for key, value in parameters.items() if key != "max_seconds"}
    kwargs.update(
        eos_token_id=list(eos),
        pad_token_id=tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else eos[0],
        use_cache=True,
        stopping_criteria=StoppingCriteriaList([stopping]),
        max_time=None,
        min_new_tokens=0,
        min_length=0,
        return_dict_in_generate=False,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
    )
    with (
        torch.inference_mode(),
        torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=batch["input_ids"].device.type == "cuda",
        ),
    ):
        output = model.generate(**batch, **kwargs)
    new_ids = output[0, prompt_length:].tolist()
    elapsed = time.monotonic() - started
    reason = guard.observe(new_ids, generated=len(new_ids), elapsed=elapsed, final=True)
    prediction = tokenizer.decode(new_ids, skip_special_tokens=True)
    if reason == "eos" and not prediction.strip():
        reason = "empty_output"
    return {
        "prediction": prediction,
        "stop_reason": reason,
        "generated_tokens": len(new_ids),
        "last_token_id": new_ids[-1] if new_ids else None,
        "eos_token_ids": list(eos),
        "seconds": elapsed,
        "repeat_period_tokens": guard.repeat_period,
    }


def evaluate(
    model,
    processor,
    rows: list[dict],
    root: Path,
    name: str,
    epoch: int,
    max_new: int,
    *,
    policy: GenerationPolicy | None = None,
    stage: str | None = None,
) -> dict:
    policy = policy or GenerationPolicy(
        max_new_tokens=max_new, retry_max_new_tokens=max_new * 2
    )
    model.eval()
    predictions = []
    failed_pages = retried_pages = 0
    attempt_seconds = 0.0
    stop_counts: Counter = Counter()
    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        batch = None
        attempt_root = root / "attempts" / f"epoch-{epoch:02d}" / name / row["page_id"]
        previous_attempts = [
            json.loads(path.read_text())
            for path in sorted(attempt_root.glob("attempt-*.json"))
        ]
        for saved in previous_attempts:
            if (
                saved["page_id"] != row["page_id"]
                or saved["image_sha256"] != row["image_sha256"]
            ):
                raise ValueError("Saved generation is bound to a different page")

        def generate(parameters):
            nonlocal batch
            if batch is None:
                batch, _ = encoded(processor, row, target=False)
            return generate_attempt(
                model, processor.tokenizer, batch, policy, parameters
            )

        def record(attempt):
            path = attempt_root / f"attempt-{attempt['attempt']:02d}.json"
            payload = {
                "page_id": row["page_id"],
                "image_sha256": row["image_sha256"],
                **attempt,
            }
            if path.exists():
                if json.loads(path.read_text()) != payload:
                    raise ValueError("Saved attempt changed")
            else:
                write_new_json(path, payload)
                emit(
                    root,
                    event="generation_attempt",
                    epoch=epoch,
                    split=name,
                    page=index,
                    attempt=attempt["attempt"],
                    stop_reason=attempt["stop_reason"],
                    generated_tokens=attempt["generated_tokens"],
                    seconds=attempt["seconds"],
                )
            stop_counts[attempt["stop_reason"]] += 1

        result = generate_with_retries(
            generate,
            policy,
            record,
            previous_attempts=previous_attempts,
        )
        prediction = result["prediction"]
        failed_pages += result["status"] != "complete"
        retried_pages += len(result["attempts"]) > 1
        attempt_seconds += sum(attempt["seconds"] for attempt in result["attempts"])
        predictions.append((row["reference_text"], prediction))
        prediction_path = root / "predictions"
        prediction_path = (
            prediction_path / f"epoch-{epoch:02d}" / name / f"{row['page_id']}.json"
        )
        prediction_record = {
            "page_id": row["page_id"],
            "reference": row["reference_text"],
            "prediction": prediction,
            "image_sha256": row["image_sha256"],
            "status": result["status"],
            "stop_reason": result["stop_reason"],
            "selected_attempt": len(result["attempts"]),
        }
        if prediction_path.exists():
            if json.loads(prediction_path.read_text()) != prediction_record:
                raise ValueError("Saved prediction changed")
        else:
            write_new_json(prediction_path, prediction_record)
        if index % 5 == 0 or index == len(rows):
            emit(
                root,
                event="eval_progress",
                epoch=epoch,
                split=name,
                completed=index,
                total=len(rows),
            )
        del batch
    metrics = score_pairs(predictions)
    metrics["seconds"] = time.perf_counter() - started
    metrics.update(
        stage=stage or ("baseline" if epoch == 0 else "page"),
        failed_pages=failed_pages,
        retried_pages=retried_pages,
        generation_stop_counts=dict(stop_counts),
        cer_includes_incomplete_outputs=True,
        all_pages_completed=failed_pages == 0,
        generation_seconds_including_prior_attempts=attempt_seconds,
    )
    summary_path = root / "evaluations" / f"epoch-{epoch:02d}" / f"{name}.json"
    if summary_path.exists():
        prior = json.loads(summary_path.read_text())
        if {k: v for k, v in prior.items() if k != "seconds"} != {
            k: v for k, v in metrics.items() if k != "seconds"
        }:
            raise ValueError("Saved evaluation changed")
        return prior
    write_new_json(summary_path, metrics)
    emit(root, event="evaluation", epoch=epoch, split=name, **metrics)
    return metrics


def evaluate_saved_epoch(
    model, processor, pages, root: Path, epoch: int, max_new: int
) -> dict:
    """Finish an epoch's evaluation after its training state is durably saved."""
    metadata = json.loads(
        (root / "checkpoints" / f"epoch-{epoch:02d}" / "epoch.json").read_text()
    )
    metadata.pop("evaluation_pending", None)
    for name in ("validation", "test"):
        metadata[name] = evaluate(
            model,
            processor,
            pages[name],
            root,
            name,
            epoch,
            max_new,
            stage=metadata.get("stage", "page"),
        )
    metadata["checkpoint_selection"] = "complete validation CER only; test descriptive"
    write_new_json(root / "epochs" / f"epoch-{epoch:02d}.json", metadata)
    return metadata


def save_checkpoint(
    model, optimizer, scheduler, root: Path, epoch: int, metadata: dict
) -> None:
    destination = root / "checkpoints" / f"epoch-{epoch:02d}"
    temporary = root / "checkpoints" / f".epoch-{epoch:02d}.incomplete"
    if destination.exists() or temporary.exists():
        raise FileExistsError(destination)
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary / "adapter", safe_serialization=True)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "python_rng": random.getstate(),
        },
        temporary / "training_state.pt",
    )
    write_new_json(temporary / "epoch.json", metadata)
    temporary.rename(destination)
    emit(root, event="checkpoint", epoch=epoch, path=str(destination))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("sanity", "full"), required=True)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--max-pixels", type=int, default=2097152)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    generation_policy = GenerationPolicy(
        max_new_tokens=args.max_new_tokens, retry_max_new_tokens=2 * args.max_new_tokens
    )
    if torch.cuda.device_count() != 2:
        raise RuntimeError("Qwen3.8-27B BF16 LoRA requires exactly two visible A100s")
    for index in range(2):
        with torch.cuda.device(index):
            allocated = torch.ones(8, device=f"cuda:{index}")
            if float(allocated.sum()) != 8.0:
                raise RuntimeError(f"GPU {index} failed its allocation smoke test")
            del allocated
    peer = torch.arange(1024, device="cuda:0", dtype=torch.float32)
    if not torch.equal(peer.to("cuda:1").to("cuda:0"), peer):
        raise RuntimeError("GPU peer transfer failed")
    del peer
    if args.epochs < 0 or args.max_pixels <= 0 or args.max_new_tokens <= 0:
        raise ValueError("Invalid training/evaluation budget")
    import causal_conv1d  # noqa: F401

    os.umask(0o077)
    root = args.output.resolve()
    sanity_run = None
    if args.mode == "full":
        candidates = [
            path
            for path in root.parent.glob("sanity*")
            if path.is_dir() and (path / "complete.json").is_file()
        ]
        if len(candidates) != 1:
            raise RuntimeError("Expected exactly one complete tiny-training run")
        sanity_root = candidates[0]
        sanity_run = sanity_root.name
        sanity_summary = json.loads((sanity_root / "complete.json").read_text())
        sanity_config = json.loads((sanity_root / "config.json").read_text())
        if sanity_config.get("generation_policy") != generation_policy.metadata():
            raise RuntimeError("Tiny training must validate the same generation policy")
        if sanity_summary.get("best_validation_cer") is None:
            raise RuntimeError("Tiny training has no complete validation epoch")
        sanity_epochs = [
            json.loads(path.read_text())
            for path in sorted((sanity_root / "epochs").glob("epoch-*.json"))
        ]
        if len(sanity_epochs) < 4 or sanity_summary["epochs"] != len(sanity_epochs):
            raise RuntimeError("Tiny training did not finish its evaluation epochs")
        if not (
            sanity_epochs[-1]["training"]["mean_loss"]
            < sanity_epochs[0]["training"]["mean_loss"]
        ):
            raise RuntimeError("Tiny training did not reduce loss; full run withheld")
    pages = load_pages(args.dataset, args.split)
    if args.mode == "sanity":
        train_ids = {"low_r0290_0317", "medium_r1060-050"}
        pages["train"] = [row for row in pages["train"] if row["page_id"] in train_ids]
        pages["validation"] = sorted(
            pages["validation"], key=lambda row: len(row["reference_text"])
        )[:1]
        pages["test"] = sorted(
            pages["test"], key=lambda row: len(row["reference_text"])
        )[:1]
    epochs = args.epochs or (4 if args.mode == "sanity" else 3)
    accumulation = 1 if args.mode == "sanity" else 8
    learning_rate = 5e-5 if args.mode == "sanity" else 2e-5
    config = {
        "model": MODEL,
        "revision": REVISION,
        "kernel_revision": KERNEL_REVISION,
        "mode": args.mode,
        "epochs": epochs,
        "accumulation": accumulation,
        "learning_rate": learning_rate,
        "max_pixels": args.max_pixels,
        "max_new_tokens": args.max_new_tokens,
        "generation_policy": generation_policy.metadata(),
        "generation_guard_sha256": file_sha256(
            Path(__file__).with_name("generation_guard.py")
        ),
        "split_sha256": file_sha256(args.split),
        "dataset_manifest_sha256": file_sha256(args.dataset / "manifest.json"),
        "code_sha256": file_sha256(Path(__file__)),
        "seed": 20261007,
        "sanity_gate_run": sanity_run,
        "page_counts": {name: len(rows) for name, rows in pages.items()},
        "prompt": PROMPT,
        "normalizers": ["anyascii@1", "all-whitespace@1"],
    }
    if args.resume:
        prior = json.loads((root / "config.json").read_text())
        if prior != config:
            raise ValueError("Resume config differs from original run")
    else:
        root.mkdir(parents=True, exist_ok=False)
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
        devices=[torch.cuda.get_device_name(i) for i in range(2)],
    )
    processor = AutoProcessor.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True
    )
    processor.image_processor.size = {
        "shortest_edge": 65536,
        "longest_edge": args.max_pixels,
    }
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
        raise RuntimeError("Model loaded with CPU/disk offload")
    model.config.use_cache = False
    prior_checkpoint = None
    if args.resume:
        checkpoints = sorted((root / "checkpoints").glob("epoch-*"))
        prior_checkpoint = checkpoints[-1] if checkpoints else None
    if prior_checkpoint is not None:
        model = PeftModel.from_pretrained(
            model, prior_checkpoint / "adapter", is_trainable=True
        )
    else:
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
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if any(
        parameter.requires_grad
        for parameter in model.base_model.model.model.visual.parameters()
    ):
        raise RuntimeError("Vision encoder was not frozen")
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(pages["train"]) / accumulation)
    total_steps = steps_per_epoch * epochs
    warmup_steps = 0 if args.mode == "sanity" else max(1, math.ceil(total_steps * 0.05))

    def lr_factor(step: int) -> float:
        if args.mode == "sanity":
            return 1.0
        if step < warmup_steps:
            return max(0.01, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    first_epoch = 1
    if prior_checkpoint is not None:
        state = torch.load(prior_checkpoint / "training_state.pt", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        first_epoch = int(prior_checkpoint.name.split("-")[-1]) + 1
    example, labels = encoded(processor, pages["train"][0], target=True)
    model.eval()
    patch_visual_projection(model, example, labels, root)
    del example, labels
    emit(
        root,
        event="loaded",
        resumed=args.resume,
        trainable_parameters=sum(p.numel() for p in parameters),
        device_map=getattr(model.base_model.model, "hf_device_map", None),
    )
    if first_epoch == 1:
        for name in ("validation", "test"):
            evaluate(model, processor, pages[name], root, name, 0, args.max_new_tokens)
    elif not (root / "epochs" / f"epoch-{first_epoch - 1:02d}.json").exists():
        evaluate_saved_epoch(
            model, processor, pages, root, first_epoch - 1, args.max_new_tokens
        )
    for epoch in range(first_epoch, epochs + 1):
        model.train()
        model.base_model.model.model.visual.eval()
        train_rows = pages["train"].copy()
        random.shuffle(train_rows)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        start = time.perf_counter()
        for index, row in enumerate(train_rows, 1):
            batch, labels = encoded(processor, row, target=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(
                    **batch,
                    labels=labels,
                    logits_to_keep=labels.shape[-1],
                    use_cache=False,
                ).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss on {row['page_id']}")
            losses.append(float(loss.detach()))
            group_start = ((index - 1) // accumulation) * accumulation
            group_size = min(accumulation, len(train_rows) - group_start)
            (loss / group_size).backward()
            if index % accumulation == 0 or index == len(train_rows):
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                if not torch.isfinite(norm):
                    raise RuntimeError(f"Non-finite gradient norm on {row['page_id']}")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                emit(
                    root,
                    event="optimizer_step",
                    epoch=epoch,
                    page=index,
                    loss=losses[-1],
                    gradient_norm=float(norm),
                    lr=scheduler.get_last_lr()[0],
                    peak_gpu_bytes=[
                        torch.cuda.max_memory_reserved(i) for i in range(2)
                    ],
                )
            del batch, labels, loss
        train_metrics = {
            "mean_loss": sum(losses) / len(losses),
            "seconds": time.perf_counter() - start,
        }
        emit(root, event="training_epoch", epoch=epoch, **train_metrics)
        metadata = {
            "epoch": epoch,
            "training": train_metrics,
            "evaluation_pending": True,
        }
        # Preserve the learned weights even if evaluation hits a hardware failure.
        save_checkpoint(model, optimizer, scheduler, root, epoch, metadata)
        evaluate_saved_epoch(model, processor, pages, root, epoch, args.max_new_tokens)
    summaries = [
        json.loads(path.read_text())
        for path in sorted((root / "epochs").glob("epoch-*.json"))
    ]
    best = select_complete_validation_epoch(summaries)
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
            "epochs": len(summaries),
        },
    )
    emit(root, event="completed", best_epoch=best["epoch"] if best else None)


if __name__ == "__main__":
    main()

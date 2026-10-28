"""One checkpointable GPU runner for the three sealed Berrutti strategies."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import math
import os
import random
import sqlite3
import time
from collections import Counter
from pathlib import Path
from uuid import uuid4

from data.berrutti import load_pages
from data.integrity import closed_json, file_sha256
from data.preparation.contracts import RUNTIME_PROFILE as RUNTIME_PROFILE
from data.preparation.contracts import verify_inputs as verify_dataset_inputs
from evaluation.scoring import score_pairs
from finetune.recipes.berrutti_qwen38.generation_guard import (
    DecodeGuard,
    GenerationPolicy,
    eos_ids,
    generate_with_retries,
)
from finetune.recipes.berrutti_qwen38.performance import cache_key, merge_page_metrics
from finetune.recipes.berrutti_qwen38.three_arm_plan import ARMS, select_test_epoch
from finetune.recipes.berrutti_qwen38.three_arm_plan import PROTOCOL as PROTOCOL

INSTRUCTION = "Transcribe fielmente todo el texto de esta imagen en orden de lectura. Conserva los saltos de línea. Devuelve solo la transcripción, sin explicaciones ni correcciones."
TARGETS = r".*language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)"


def verify_inputs(config):
    """Keep injectable hash/page readers while delegating CPU input validation."""
    return verify_dataset_inputs(config, hash_file=file_sha256, page_loader=load_pages)


def training_groups(pages: list[dict], seed: int, epoch: int, size: int = 8):
    if epoch not in (1, 2, 3) or not pages or size < 1:
        raise ValueError("Invalid epoch or source-page group")
    if len({page["page_id"] for page in pages}) != len(pages):
        raise ValueError("Duplicate training source page")
    ordered = sorted(pages, key=lambda page: page["page_id"])
    random.Random(seed + epoch).shuffle(ordered)
    return [ordered[start : start + size] for start in range(0, len(ordered), size)]


def example_weights(
    arm: str, supervised_tokens: list[int]
) -> tuple[float, list[float]]:
    """One page is one unit; padding and line count never change its weight."""
    if arm not in ARMS or not supervised_tokens or min(supervised_tokens) < 1:
        raise ValueError("Invalid strategy or empty supervised line")
    page_weight = {"pages": 1.0, "lines": 0.0, "combined": 0.5}[arm]
    total = sum(supervised_tokens)
    return page_weight, [
        (1 - page_weight) * count / total for count in supervised_tokens
    ]


def learning_rate(completed: int, total: int, maximum: float, minimum: float) -> float:
    if total < 2 or not 0 <= completed < total or not 0 < minimum <= maximum:
        raise ValueError("Invalid scheduler state")
    return (
        minimum
        + (maximum - minimum) * (1 + math.cos(math.pi * completed / (total - 1))) / 2
    )


def resumed_attempts(root: Path, policy: GenerationPolicy) -> list[dict]:
    """A killed generation consumes its reserved attempt; never permit a third."""
    attempts = []
    reason = None
    for number in (1, 2):
        path = root / f"attempt-{number}.json"
        started = root / f"attempt-{number}.started.json"
        if not path.exists() and started.exists():
            reservation = json.loads(started.read_text())
            if reservation["parameters"] != policy.attempt_parameters(reason):
                raise ValueError("Interrupted attempt has a different policy")
            closed_json(
                path,
                {
                    **reservation,
                    "prediction": "",
                    "stop_reason": "interrupted",
                    "seconds": 0.0,
                },
            )
        if not path.exists():
            break
        row = json.loads(path.read_text())
        if row["attempt"] != number or row["parameters"] != policy.attempt_parameters(
            reason
        ):
            raise ValueError("Saved attempt policy mismatch")
        attempts.append(row)
        reason = row["stop_reason"]
        if reason == "eos":
            if (root / "attempt-2.started.json").exists() and number == 1:
                raise ValueError("Unexpected attempt after successful generation")
            break
    return attempts


def check_gpu_names(names: list[str], expected: str = "A100") -> None:
    """Allow only explicitly selected, two-device BF16 hardware profiles."""
    if (
        expected not in ("A100", "A40")
        or len(names) != 2
        or any(expected not in name.replace("-", " ").split() for name in names)
    ):
        raise RuntimeError(f"Exactly two visible {expected} GPUs are required")


class Runtime:
    def __init__(self, config: dict, root: Path, emit):
        import importlib.metadata

        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        from finetune.recipes.berrutti_qwen38 import eager_hooks, projection_runtime

        self.torch, self.config, self.root, self.emit = torch, config, root, emit
        self.phase = "gpu_runtime"
        gpu_names = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        check_gpu_names(gpu_names, config.get("gpu_type", "A100"))
        for device in range(2):
            assert torch.ones(8, device=f"cuda:{device}").sum().item() == 8
        torch.set_num_threads(4)
        emit("gpu_probe", runtime=True, success=True, torch={"device_count": 2})
        eager_hooks.install()
        emit("runtime_smoke", **eager_hooks.verify_routing())
        self.phase = "model_loading"
        emit("phase", phase=self.phase)
        tick = time.monotonic()
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            config["model_root"],
            dtype=torch.bfloat16,
            device_map="balanced",
            max_memory={0: "34GiB", 1: "34GiB"},
            attn_implementation="sdpa",
            local_files_only=True,
        )
        if any(p.device.type != "cuda" for p in self.model.parameters()):
            raise RuntimeError("CPU/disk weight offload is forbidden in this profile")
        torch.manual_seed(config["seed"])
        random.seed(config["seed"])
        self.model = get_peft_model(
            self.model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.0,
                bias="none",
                target_modules=TARGETS,
                task_type="CAUSAL_LM",
            ),
        )
        self.visual = self.model.base_model.model.model.visual
        self.visual.eval()
        self.projection = projection_runtime.install(self.model)
        self.processor = AutoProcessor.from_pretrained(
            config["model_root"], local_files_only=True
        )
        self.processor.tokenizer.padding_side = "left"
        self.policy = GenerationPolicy()
        self.eos = eos_ids(
            self.model.generation_config.eos_token_id,
            self.processor.tokenizer.eos_token_id,
        )
        self.prepared = {}
        self.initial_adapter_sha256 = self.adapter_digest()
        self.identity = cache_key(
            {
                "profile": config["runtime_profile"],
                "gpu_names": gpu_names,
                "model_revision": config["revision"],
                "initial_adapter": self.initial_adapter_sha256,
                "cohort": config["cohort_sha256"],
                "split": config["split_sha256"],
                "sources": config["source_hashes"],
                "prompt": INSTRUCTION,
                "policy": self.policy.metadata(),
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in ("torch", "transformers", "peft", "accelerate")
                },
                "pixels": {
                    k: config[k]
                    for k in ("page_max_pixels", "line_max_pixels", "min_pixels")
                },
            }
        )
        self.cache = sqlite3.connect(f"file:{config['line_cache']}?mode=ro", uri=True)
        receipt = json.loads(
            self.cache.execute(
                "SELECT value FROM metadata WHERE key='closed_receipt'"
            ).fetchone()[0]
        )
        if (
            receipt.get("complete") is not True
            or receipt["cohort_sha256"] != config["cohort_sha256"]
            or receipt["line_images"] != 5212
        ):
            raise ValueError("Crop cache does not match the complete sealed cohort")
        self.optimizer = self.new_optimizer()
        emit(
            "runtime_loaded",
            seconds=time.monotonic() - tick,
            initial_adapter_sha256=self.initial_adapter_sha256,
            runtime_identity=self.identity,
            optimizer_steps=0,
        )

    def new_optimizer(self):
        return self.torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config["learning_rate"],
            weight_decay=0.0,
        )

    def adapter(self):
        from peft import get_peft_model_state_dict

        return {
            k: v.detach().cpu().clone()
            for k, v in get_peft_model_state_dict(self.model).items()
        }

    def adapter_digest(self):
        digest = hashlib.sha256()
        for name, tensor in sorted(self.adapter().items()):
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(tensor.contiguous().view(self.torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def rng(self):
        return {
            "python": random.getstate(),
            "torch": self.torch.get_rng_state(),
            "cuda": self.torch.cuda.get_rng_state_all(),
        }

    def restore_rng(self, state):
        random.setstate(state["python"])
        self.torch.set_rng_state(state["torch"])
        self.torch.cuda.set_rng_state_all(state["cuda"])

    def progress(self, state):
        path = self.root / "cursors" / f"{uuid4().hex}.pt"
        path.parent.mkdir(exist_ok=True)
        self.torch.save({"rng": self.rng(), "global_step": state["global_step"]}, path)
        value = {**state, "rng_cursor": str(path), "runtime_identity": self.identity}
        temporary = self.root / f".state-{uuid4().hex}.tmp"
        temporary.write_text(json.dumps(value, allow_nan=False) + "\n")
        temporary.replace(self.root / "state.json")
        state.update(value)

    def checkpoint(self, state):
        path = (
            self.root / "checkpoints" / f"step-{state['global_step']:03d}-{uuid4().hex}"
        )
        path.mkdir(parents=True, exist_ok=False)
        self.torch.save(
            {
                "adapter": self.adapter(),
                "optimizer": self.optimizer.state_dict(),
                "rng": self.rng(),
                "scheduler": {
                    "completed_steps": state["global_step"],
                    "total_steps": self.config.get("total_optimizer_steps", 48),
                    "version": "cosine-v1",
                },
                "runtime_identity": self.identity,
            },
            path / "state.pt",
        )
        closed_json(
            path / "closed.json",
            {
                "global_step": state["global_step"],
                "runtime_identity": self.identity,
                "state_sha256": file_sha256(path / "state.pt"),
            },
        )
        state["checkpoint"] = str(path)
        self.progress(state)
        return str(path)

    def restore(self, state, *, optimizer=True):
        from peft import set_peft_model_state_dict

        if state["runtime_identity"] != self.identity:
            raise ValueError("Resume runtime differs from the saved experiment")
        if state.get("checkpoint"):
            path = Path(state["checkpoint"])
            receipt = json.loads((path / "closed.json").read_text())
            if receipt["state_sha256"] != file_sha256(path / "state.pt"):
                raise ValueError("Checkpoint bytes differ from their closed receipt")
            payload = self.torch.load(
                path / "state.pt", map_location="cpu", weights_only=False
            )
            if (
                payload["runtime_identity"] != self.identity
                or receipt["global_step"] != payload["scheduler"]["completed_steps"]
                or payload["scheduler"]["total_steps"]
                != self.config.get("total_optimizer_steps", 48)
            ):
                raise ValueError("Invalid checkpoint identity or scheduler")
            if (
                optimizer
                and payload["scheduler"]["completed_steps"] != state["global_step"]
            ):
                raise ValueError("Optimizer cursor differs from checkpoint")
            set_peft_model_state_dict(self.model, payload["adapter"])
            if optimizer:
                self.optimizer.load_state_dict(payload["optimizer"])
                self.restore_rng(payload["rng"])
        if optimizer and state.get("rng_cursor"):
            cursor = self.torch.load(
                state["rng_cursor"], map_location="cpu", weights_only=False
            )
            if cursor["global_step"] != state["global_step"]:
                raise ValueError("RNG cursor differs from optimizer step")
            self.restore_rng(cursor["rng"])

    def prepare(self, page, line=None, *, training=False):
        from PIL import Image

        key = (
            page["page_id"],
            None if line is None else line["region_index"],
            training,
        )
        if key in self.prepared:
            return self.prepared[key]
        if line is None:
            with Image.open(page["image"]) as source:
                image = source.convert("RGB")
            reference, max_pixels = (
                page["reference_text"],
                page.get("max_pixels", self.config["page_max_pixels"]),
            )
        else:
            row = self.cache.execute(
                "SELECT target,png,rgb_sha256 FROM crops WHERE page_id=? AND region_index=?",
                key[:2],
            ).fetchone()
            if row is None or row[0] != line["target"]:
                raise ValueError("Missing crop or altered original human label")
            with Image.open(io.BytesIO(row[1])) as source:
                image = source.convert("RGB")
            if hashlib.sha256(image.tobytes()).hexdigest() != row[2]:
                raise ValueError("Cached crop pixels changed")
            reference, max_pixels = line["target"], self.config["line_max_pixels"]
        self.processor.image_processor.size = {
            "shortest_edge": self.config["min_pixels"],
            "longest_edge": max_pixels,
        }
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": INSTRUCTION},
                ],
            }
        ]
        kwargs = dict(
            tokenize=True, enable_thinking=False, return_dict=True, return_tensors="pt"
        )
        prefix = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, **kwargs
        )
        if not training:
            result = {"inputs": prefix}
        else:
            full = self.processor.apply_chat_template(
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
            prompt = prefix["input_ids"].shape[-1]
            if not self.torch.equal(prefix["input_ids"], full["input_ids"][:, :prompt]):
                raise RuntimeError("Supervised prompt prefix mismatch")
            keep = full["input_ids"].shape[-1] - prompt + 1
            labels = full["input_ids"][:, -keep:].clone()
            labels[:, 0] = -100
            result = {
                "inputs": full,
                "labels": labels,
                "keep": keep,
                "tokens": int((labels != -100).sum()),
            }
            if result["tokens"] < 1:
                raise ValueError("Empty supervised example")
        self.prepared[key] = result
        return result

    def backward(self, example, weight):
        self.model.train()
        self.visual.eval()
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.model.config.use_cache = False
        batch = {k: v.to("cuda:0") for k, v in example["inputs"].items()}
        output = self.model(
            **batch,
            labels=example["labels"].to("cuda:0"),
            logits_to_keep=example["keep"],
            use_cache=False,
        )
        loss = output.loss
        if not self.torch.isfinite(loss):
            raise RuntimeError("Non-finite supervised loss")
        # Routed logits are large and unused here. Retain only the loss graph;
        # release allocator cache so Triton's backward kernels have headroom.
        del output
        self.torch.cuda.empty_cache()
        if not getattr(self, "backward_diagnostic_emitted", False):
            self.emit("training_memory", devices=self.memory_snapshot())
            self.backward_diagnostic_emitted = True
        (loss * weight).backward()
        value = float(loss.detach())
        del loss, batch
        return value

    def memory_snapshot(self):
        cuda = self.torch.cuda
        return [
            {
                "device": device,
                "allocated_bytes": cuda.memory_allocated(device),
                "reserved_bytes": cuda.memory_reserved(device),
                "peak_allocated_bytes": cuda.max_memory_allocated(device),
                "free_bytes": cuda.mem_get_info(device)[0],
            }
            for device in range(cuda.device_count())
        ]

    def train_group(self, arm, pages, regions, global_step, epoch, first_page):
        tick = time.monotonic()
        self.phase = f"training_{arm}_epoch_{epoch}"
        self.optimizer.zero_grad(set_to_none=True)
        loss = 0.0
        samples = tokens = 0
        for position, page in enumerate(pages, first_page + 1):
            lines = (
                [
                    self.prepare(page, line, training=True)
                    for line in regions[page["page_id"]]
                ]
                if arm != "pages"
                else []
            )
            page_weight, weights = example_weights(
                arm, [row["tokens"] for row in lines] or [1]
            )
            if page_weight:
                example = self.prepare(page, training=True)
                loss += page_weight * self.backward(example, page_weight / len(pages))
                samples += 1
                tokens += example["tokens"]
            for example, weight in zip(lines, weights, strict=False):
                loss += weight * self.backward(example, weight / len(pages))
                samples += 1
                tokens += example["tokens"]
            self.emit(
                "sample_completed",
                epoch=epoch,
                stage=arm,
                source_page=position,
                samples_completed=samples,
                representation="page" if arm == "pages" else "line",
            )
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if any(
            p.grad is None or not self.torch.isfinite(p.grad).all() for p in parameters
        ):
            raise RuntimeError("Missing or non-finite adapter gradient")
        norm = self.torch.nn.utils.clip_grad_norm_(
            parameters, 1.0, error_if_nonfinite=True, foreach=False
        )
        lr = learning_rate(
            global_step,
            self.config.get("total_optimizer_steps", 48),
            self.config["learning_rate"],
            self.config["minimum_learning_rate"],
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        if any(not self.torch.isfinite(p).all() for p in parameters):
            raise RuntimeError("Non-finite updated adapter weight")
        self.optimizer.zero_grad(set_to_none=True)
        self.emit(
            "optimizer_step",
            epoch=epoch,
            stage=arm,
            page=first_page + len(pages),
            total=125,
            loss=loss / len(pages),
            lr=lr,
            optimizer_steps=global_step + 1,
            gradient_norm=float(norm),
        )
        return {
            "samples": samples,
            "supervised_tokens": tokens,
            "source_pages": len(pages),
            "unit_loss_sum": loss,
            "seconds": time.monotonic() - tick,
        }

    def decode(self, page, root):
        from transformers import StoppingCriteria, StoppingCriteriaList

        self.model.eval()
        self.model.gradient_checkpointing_disable()
        prefix = {k: v.to("cuda:0") for k, v in self.prepare(page)["inputs"].items()}
        prompt = prefix["input_ids"].shape[-1]
        previous = resumed_attempts(root, self.policy)
        attempts = previous.copy()

        def generate(parameters):
            number = len(attempts) + 1
            closed_json(
                root / f"attempt-{number}.started.json",
                {"attempt": number, "parameters": parameters},
            )
            guard, tick = (
                DecodeGuard(self.policy, parameters, self.eos),
                time.monotonic(),
            )

            class Stop(StoppingCriteria):
                def __call__(_, ids, scores, **kwargs):
                    reason = guard.observe(
                        ids[0, prompt:][-self.policy.tail_tokens :].tolist(),
                        generated=ids.shape[-1] - prompt,
                        elapsed=time.monotonic() - tick,
                    )
                    return self.torch.full(
                        (1,),
                        reason is not None,
                        dtype=self.torch.bool,
                        device=ids.device,
                    )

            with self.torch.inference_mode():
                ids = self.model.generate(
                    **prefix,
                    **{k: v for k, v in parameters.items() if k != "max_seconds"},
                    use_cache=True,
                    stopping_criteria=StoppingCriteriaList([Stop()]),
                )
            tokens = ids[0, prompt:].tolist()
            reason = guard.observe(
                tokens[-self.policy.tail_tokens :],
                generated=len(tokens),
                elapsed=time.monotonic() - tick,
                final=True,
            )
            return {
                "prediction": self.processor.tokenizer.decode(
                    tokens, skip_special_tokens=True
                ),
                "stop_reason": reason,
                "seconds": time.monotonic() - tick,
                "generated_tokens": len(tokens),
            }

        def record(attempt):
            path = root / f"attempt-{attempt['attempt']}.json"
            if not path.exists():
                closed_json(path, attempt)
                attempts.append(attempt)

        return generate_with_retries(
            generate, self.policy, record, previous_attempts=previous
        )

    def evaluate(self, pages, epoch, split, stage, checkpoint_identity, should_yield):
        self.phase = f"evaluation_{stage}_epoch_{epoch}_{split}"
        self.emit("phase", phase=self.phase)
        root = self.root / "evaluations" / f"epoch-{epoch:02d}-{split}"
        identity = cache_key(
            {"runtime": self.identity, "adapter": checkpoint_identity, "split": split}
        )
        rows, stops = [], Counter()
        for index, page in enumerate(pages, 1):
            page_root = root / "pages" / page["page_id"]
            path = page_root / "closed.json"
            if path.exists():
                row = json.loads(path.read_text())
            else:
                if should_yield():
                    return None
                result = self.decode(page, page_root)
                row = {
                    "page_id": page["page_id"],
                    "evaluation_key": identity,
                    "status": result["status"],
                    "attempts": len(result["attempts"]),
                    "seconds": sum(a["seconds"] for a in result["attempts"]),
                    "generation_stop_counts": dict(
                        Counter(a["stop_reason"] for a in result["attempts"])
                    ),
                    **{
                        k: v
                        for k, v in score_pairs(
                            [(page["reference_text"], result["prediction"])]
                        ).items()
                        if k not in ("pages", "cer", "wer")
                    },
                }
                closed_json(path, row)
            if row["evaluation_key"] != identity:
                raise ValueError(
                    "Saved prediction belongs to another checkpoint or policy"
                )
            rows.append(row)
            stops.update(row["generation_stop_counts"])
            self.emit(
                "eval_progress",
                epoch=epoch,
                split=split,
                completed=index,
                total=len(pages),
                stage=stage,
            )
        summary = {
            **merge_page_metrics([rows], [p["page_id"] for p in pages], identity),
            "seconds": sum(r["seconds"] for r in rows),
            "generation_stop_counts": dict(stops),
            "epoch": epoch,
            "split": split,
            "stage": stage,
        }
        receipt = root / "closed.json"
        if not receipt.exists():
            closed_json(receipt, summary)
        self.emit("evaluation", **summary)
        return summary


def run_arm(runtime, pages, cohort, mode, should_yield):
    root, emit = runtime.root, runtime.emit
    baseline = json.loads(
        (
            Path(runtime.config["operation_root"]) / "baseline" / "baseline.json"
        ).read_text()
    )
    if baseline["runtime_identity"] != runtime.identity:
        raise ValueError("Shared baseline does not match fresh initialization/runtime")
    if not (root / "baseline.json").exists():
        closed_json(root / "baseline.json", baseline)
        for split in ("validation", "test"):
            emit("evaluation", **baseline[split])
    state_path = root / "state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {
            "epoch": 1,
            "group": 0,
            "global_step": 0,
            "phase": "training",
            "checkpoint": None,
            "epoch_checkpoints": {},
            "stats": {},
            "runtime_identity": runtime.identity,
        }
    )
    runtime.restore(state)
    regions = {p["page_id"]: p["regions"] for p in cohort["pages"]}
    while state["epoch"] <= 3:
        epoch = state["epoch"]
        groups = training_groups(pages["train"], runtime.config["seed"], epoch)
        if state["phase"] == "training":
            for index in range(state["group"], len(groups)):
                if should_yield():
                    runtime.progress(state)
                    return 85
                result = runtime.train_group(
                    mode, groups[index], regions, state["global_step"], epoch, index * 8
                )
                for key, value in result.items():
                    state["stats"][key] = state["stats"].get(key, 0) + value
                state["global_step"] += 1
                state["group"] = index + 1
                if state["group"] == len(groups):
                    state["phase"] = "validation"
                checkpoint = runtime.checkpoint(state)
                if state["phase"] == "validation":
                    state["epoch_checkpoints"][str(epoch)] = checkpoint
                    runtime.progress(state)
            stats = state["stats"]
            expected = {"pages": 125, "lines": 5212, "combined": 5337}[mode]
            if (
                stats["source_pages"] != 125
                or stats["samples"] != expected
                or state["global_step"] != epoch * 16
            ):
                raise RuntimeError(
                    "An epoch did not cover every required example exactly once"
                )
            emit(
                "training_epoch",
                epoch=epoch,
                stage=mode,
                source_pages=125,
                samples=expected,
                optimizer_steps=16,
                mean_loss=stats["unit_loss_sum"] / 125,
                seconds=stats["seconds"],
                allocated_gpu_seconds=2 * stats["seconds"],
                supervised_tokens=stats["supervised_tokens"],
            )
        if str(epoch) not in state["epoch_checkpoints"]:
            state["epoch_checkpoints"][str(epoch)] = state["checkpoint"]
        validation = runtime.evaluate(
            pages["validation"],
            epoch,
            "validation",
            mode,
            runtime.adapter_digest(),
            should_yield,
        )
        if validation is None:
            runtime.progress(state)
            return 85
        path = root / "epochs" / f"epoch-{epoch:02d}.json"
        if not path.exists():
            closed_json(path, {"epoch": epoch, "validation": validation})
        state.update(epoch=epoch + 1, group=0, phase="training", stats={})
        runtime.progress(state)
    validations = [
        json.loads((root / "epochs" / f"epoch-{epoch:02d}.json").read_text())[
            "validation"
        ]
        for epoch in (1, 2, 3)
    ]
    selected = select_test_epoch(validations)
    if selected is None:
        raise RuntimeError("No complete validation epoch is eligible for selection")
    best_epoch = selected["epoch"]
    runtime.restore(
        {**state, "checkpoint": state["epoch_checkpoints"][str(best_epoch)]},
        optimizer=False,
    )
    test = runtime.evaluate(
        pages["test"], best_epoch, "test", mode, runtime.adapter_digest(), should_yield
    )
    if test is None:
        return 85
    epoch_path = root / "epochs" / f"epoch-{best_epoch:02d}-selected-test.json"
    if not epoch_path.exists():
        closed_json(epoch_path, {"epoch": best_epoch, "test": test})
    summary = {
        "best_epoch": best_epoch,
        "best_validation_cer": selected["cer"],
        "test_cer_at_selected_epoch": test["cer"],
        "selected_test_failed_pages": test["failed_pages"],
        "optimizer_steps": state["global_step"],
        "selected_checkpoint": state["epoch_checkpoints"][str(best_epoch)],
    }
    closed_json(root / "completed.json", summary)
    emit("completed", **summary)
    return 0


def run_smoke(runtime, pages, cohort):
    from peft import set_peft_model_state_dict

    root, emit = runtime.root, runtime.emit
    by_id = {p["page_id"]: p for p in cohort["pages"]}
    samples = [
        p for p in pages["train"] if p["page_id"] in runtime.config["smoke_page_ids"]
    ]
    if len(samples) != 2 or all(
        by_id[p["page_id"]]["page_and_lines_concordant"] for p in samples
    ):
        raise ValueError("Smoke needs two pages including an annotation disagreement")
    initial, initial_rng = runtime.adapter(), runtime.rng()
    identities = []
    for arm in ARMS:
        set_peft_model_state_dict(runtime.model, initial)
        runtime.restore_rng(initial_rng)
        runtime.optimizer = runtime.new_optimizer()
        identities.append(runtime.adapter_digest())
        runtime.train_group(
            arm, samples, {k: v["regions"] for k, v in by_id.items()}, 0, 1, 0
        )
        state = {"global_step": 1, "runtime_identity": runtime.identity}
        saved = runtime.adapter_digest()
        expected_rng = runtime.rng()
        parameter = next(iter(runtime.optimizer.state))
        expected_moment = runtime.optimizer.state[parameter]["exp_avg"].cpu().clone()
        runtime.checkpoint(state)
        set_peft_model_state_dict(runtime.model, initial)
        runtime.optimizer = runtime.new_optimizer()
        random.random()
        runtime.torch.rand(3)
        for device in range(2):
            runtime.torch.rand(3, device=f"cuda:{device}")
        runtime.restore(state)
        restored_rng = runtime.rng()
        if (
            runtime.adapter_digest() != saved
            or not runtime.optimizer.state
            or restored_rng["python"] != expected_rng["python"]
            or not runtime.torch.equal(restored_rng["torch"], expected_rng["torch"])
            or any(
                not runtime.torch.equal(actual, expected)
                for actual, expected in zip(
                    restored_rng["cuda"], expected_rng["cuda"], strict=True
                )
            )
            or not runtime.torch.equal(
                runtime.optimizer.state[parameter]["exp_avg"].cpu(), expected_moment
            )
            or runtime.optimizer.param_groups[0]["lr"]
            != runtime.config["learning_rate"]
        ):
            raise RuntimeError(
                "Checkpoint did not restore adapter, optimizer and scheduler"
            )
        for split in ("validation", "test"):
            # Different arm epochs keep independent immutable diagnostic output.
            result = runtime.evaluate(
                pages[split][:1],
                ARMS.index(arm) + 1,
                split,
                f"smoke_{arm}",
                saved,
                lambda: False,
            )
            if result["all_pages_completed"] is not True:
                raise RuntimeError("Small assay did not complete full-page generation")
    if len(set(identities)) != 1:
        raise RuntimeError(
            "The three strategies did not share fresh adapter initialization"
        )
    summary = {
        "success": True,
        "cuda_device_count": 2,
        "source_pages": 2,
        "strategies_checked": 3,
        "checkpoint_reload": True,
        "diagnostic_only": True,
        "runtime_identity": runtime.identity,
    }
    closed_json(root / "completed.json", summary)
    emit("runner_smoke_completed", **summary)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=(*ARMS, "preflight", "smoke", "baseline"), required=True
    )
    args = parser.parse_args()
    os.umask(0o077)
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "jobs").mkdir(exist_ok=True)
    lock = (args.root / "execution.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.root / "completed.json").exists():
        return 0
    started = time.monotonic()

    def emit(event, **value):
        row = {"event": event, "elapsed_segment": time.monotonic() - started, **value}
        with (args.root / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        print(
            json.dumps(
                {k: v for k, v in row.items() if k not in ("selected_checkpoint",)},
                allow_nan=False,
            ),
            flush=True,
        )

    runtime = None
    try:
        config = json.loads(args.config.read_text())
        if (
            sum(
                path.is_dir() and path.name.isdigit()
                for path in (args.root / "jobs").iterdir()
            )
            > config["maximum_segments"]
        ):
            raise RuntimeError("Maximum checkpointed Slurm segments exhausted")
        cohort, pages = verify_inputs(config)
        if args.mode == "preflight":
            receipt = {
                "success": True,
                "optimizer_steps": 0,
                "source_pages": len(pages["train"]),
                "line_images": cohort["summary"]["train_lines"],
                "gpu_check": False,
                "runtime_profile": config["runtime_profile"],
            }
            closed_json(args.root / "completed.json", receipt)
            emit("preflight_completed", **receipt)
            return 0
        emit("phase", phase="runtime_imports")
        runtime = Runtime(config, args.root, emit)
        from finetune.recipes.berrutti_qwen38 import projection_runtime

        pixels = runtime.prepare(pages["train"][0], training=True)["inputs"][
            "pixel_values"
        ]
        emit("projection", **projection_runtime.verify(runtime.projection, pixels))
        if args.mode == "smoke":
            return run_smoke(runtime, pages, cohort)
        smoke = json.loads(
            (Path(config["operation_root"]) / "smoke" / "completed.json").read_text()
        )
        if (
            smoke.get("success") is not True
            or smoke["runtime_identity"] != runtime.identity
        ):
            raise ValueError(
                "Representative GPU assay did not approve this exact runtime"
            )

        def should_yield():
            return time.monotonic() - started >= config["segment_seconds"]

        if args.mode == "baseline":
            summaries = {}
            for split in ("validation", "test"):
                summaries[split] = runtime.evaluate(
                    pages[split],
                    0,
                    split,
                    "baseline",
                    runtime.initial_adapter_sha256,
                    lambda: False,
                )
            summary = {
                "epoch": 0,
                "runtime_identity": runtime.identity,
                "optimizer_steps": 0,
                **summaries,
            }
            closed_json(args.root / "baseline.json", summary)
            closed_json(
                args.root / "completed.json", {"success": True, "optimizer_steps": 0}
            )
            emit("completed", optimizer_steps=0)
            return 0
        return run_arm(runtime, pages, cohort, args.mode, should_yield)
    except Exception as error:
        diagnosis = {
            "phase": runtime.phase if runtime else "setup",
            "error_type": type(error).__name__,
            "success": False,
        }
        if runtime:
            try:
                diagnosis["gpu_memory"] = runtime.memory_snapshot()
            except Exception as memory_error:
                diagnosis["memory_diagnostic_error_type"] = type(memory_error).__name__
        emit(
            "failed",
            diagnosis=diagnosis,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())

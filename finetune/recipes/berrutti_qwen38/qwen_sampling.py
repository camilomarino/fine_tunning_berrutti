"""User-approved instruct sampling for experiment four only."""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json

from finetune.recipes.berrutti_qwen38.generation_guard import GenerationPolicy

SAMPLING_POLICY = "qwen38-instruct-seeded-sampling-v1"
PARAMETERS = dict(
    do_sample=True,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    min_p=0.0,
    repetition_penalty=1.0,
)


@dataclass(frozen=True)
class InstructSamplingPolicy(GenerationPolicy):
    retry_repetition_penalty: float = 1.0

    def metadata(self):
        result = super().metadata()
        result.update(
            version=SAMPLING_POLICY,
            sampling={**PARAMETERS, "presence_penalty": 1.5},
            seed_scope="source seed, page/image identity and attempt; no checkpoint/root/reference dependence",
            presence_scope="generated tokens only, once per distinct token",
            seed_rng_restored=True,
        )
        # Fixed vendor penalty replaces the old conditional retry penalty explicitly.
        result["retry_repetition_penalty"] = 1.0
        return result

    def attempt_parameters(self, previous_reason):
        return {**super().attempt_parameters(previous_reason), **PARAMETERS}


def page_seed(seed, page, attempt):
    if type(seed) is not int or type(attempt) is not int or attempt not in (1, 2):
        raise ValueError("Invalid fixed generation seed/attempt")
    key = json.dumps(
        [SAMPLING_POLICY, seed, page["page_id"], page["image_sha256"], attempt],
        separators=(",", ":"),
    )
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % (2**63)


class PresencePenalty:
    def __init__(self, torch, prompt_length, penalty=1.5):
        if type(prompt_length) is not int or prompt_length < 1 or penalty != 1.5:
            raise ValueError("Unexpected vendor presence policy")
        self.torch = torch
        self.prompt_length = prompt_length
        self.penalty = penalty
        self.calls = 0
        self.penalized_tokens = 0

    def __call__(self, input_ids, scores):
        if (
            input_ids.ndim != 2
            or scores.ndim != 2
            or input_ids.shape[0] != 1
            or scores.shape[0] != 1
        ):
            raise ValueError("Single-page generation is required")
        if (
            input_ids.shape[1] < self.prompt_length
            or not self.torch.isfinite(scores).all()
        ):
            raise ValueError("Invalid presence-penalty inputs/logits")
        seen = self.torch.unique(input_ids[0, self.prompt_length :]).to(scores.device)
        result = scores.clone()
        if seen.numel():
            if bool((seen < 0).any()) or bool((seen >= scores.shape[-1]).any()):
                raise ValueError("Generated token outside vocabulary")
            result[0, seen] -= self.penalty
        self.calls += 1
        self.penalized_tokens += seen.numel()
        return result


@contextmanager
def seeded_sampling(model, torch, policy, page, seed, emit, *, completed_attempts=0):
    if policy.metadata()["version"] != SAMPLING_POLICY:
        raise ValueError("Unexpected sampling contract")
    if type(completed_attempts) is not int or completed_attempts not in (0, 1, 2):
        raise ValueError("Invalid consumed generation attempts")
    original = model.generate
    had_slot = "generate" in model.__dict__
    slot = model.__dict__.get("generate")
    control = dict(
        policy=SAMPLING_POLICY,
        parameters={**PARAMETERS, "presence_penalty": 1.5},
        page_id=page["page_id"],
        image_sha256=page["image_sha256"],
        source_seed=seed,
        calls=0,
        attempt_offset=completed_attempts,
        attempts=[],
        seeds=[],
        presence_calls=0,
        penalized_tokens=0,
        inputs_unchanged=True,
        rng_restored=False,
        interfaces_restored=False,
    )
    cpu_rng = torch.get_rng_state().clone()
    devices = list(range(torch.cuda.device_count()))
    gpu_rng = [torch.cuda.get_rng_state(d).clone() for d in devices]

    def generate(*args, **kwargs):
        if args or any(kwargs.get(name) != value for name, value in PARAMETERS.items()):
            raise ValueError(
                "Native generation differs from the approved sampling policy"
            )
        if "logits_processor" in kwargs or "generator" in kwargs:
            raise ValueError("Unexpected additional generation processor/RNG")
        attempt = completed_attempts + control["calls"] + 1
        actual_seed = page_seed(seed, page, attempt)
        ids = kwargs["input_ids"]
        before = ids.clone()
        processor = PresencePenalty(torch, int(ids.shape[1]))
        from transformers import LogitsProcessorList

        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(actual_seed)
            output = original(
                **kwargs, logits_processor=LogitsProcessorList([processor])
            )
        if not torch.equal(ids, before):
            control["inputs_unchanged"] = False
            raise RuntimeError("Native sampled generation mutated prefix ids")
        control["calls"] += 1
        control["attempts"].append(attempt)
        control["seeds"].append(actual_seed)
        control["presence_calls"] += processor.calls
        control["penalized_tokens"] += processor.penalized_tokens
        return output

    model.generate = generate
    try:
        yield control
    finally:
        if had_slot:
            model.generate = slot
        else:
            delattr(model, "generate")
        control["interfaces_restored"] = model.generate == original
        control["rng_restored"] = torch.equal(torch.get_rng_state(), cpu_rng) and all(
            torch.equal(torch.cuda.get_rng_state(d), saved)
            for d, saved in zip(devices, gpu_rng, strict=True)
        )
        emit("seeded_sampling_control", **control)
        if not control["rng_restored"] or not control["interfaces_restored"]:
            raise RuntimeError("Sampled generation did not restore RNG/interfaces")

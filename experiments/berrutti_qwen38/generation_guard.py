"""Bounded OCR decoding policy; pure Python, independent of model runtimes.

This experiment-layer module imports no benchmark orchestration or ML runtime.
The caller persists every attempt before retrying; references never enter the
retry/selection policy. All thresholds are heuristics, not OCR quality claims.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class GenerationPolicy:
    max_new_tokens: int = 4096
    retry_max_new_tokens: int = 8192
    max_seconds: float = 600.0
    retry_max_seconds: float = 900.0
    min_repeat_tokens: int = 128
    min_repeats: int = 4
    max_repeat_period: int = 128
    check_interval: int = 16
    retry_repetition_penalty: float = 1.05

    def __post_init__(self) -> None:
        for name in (
            "max_new_tokens",
            "retry_max_new_tokens",
            "min_repeat_tokens",
            "min_repeats",
            "max_repeat_period",
            "check_interval",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("max_seconds", "retry_max_seconds", "retry_repetition_penalty"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            self.retry_max_new_tokens < self.max_new_tokens
            or self.retry_max_seconds < self.max_seconds
            or self.min_repeats < 2
            or self.retry_repetition_penalty < 1
        ):
            raise ValueError("Invalid retry or repetition budget")

    @property
    def tail_tokens(self) -> int:
        return max(
            self.min_repeat_tokens + self.max_repeat_period - 1,
            self.min_repeats * self.max_repeat_period,
        )

    def metadata(self) -> dict:
        return {
            "version": "bounded-ocr-v2",
            **asdict(self),
            "max_attempts": 2,
            "time_limit": "cooperative, checked between decoding steps",
            "selection": "first successful attempt, otherwise last raw output",
            "failed_pages": "included in CER; validation checkpoint ineligible",
        }

    def attempt_parameters(self, previous_reason: str | None) -> dict:
        retry = previous_reason is not None
        return {
            "max_new_tokens": (
                self.retry_max_new_tokens if retry else self.max_new_tokens
            ),
            "max_seconds": self.retry_max_seconds if retry else self.max_seconds,
            "repetition_penalty": (
                self.retry_repetition_penalty
                if previous_reason == "repetition"
                else 1.0
            ),
            "do_sample": False,
            "num_beams": 1,
            "no_repeat_ngram_size": 0,
            # Never manufacture a successful EOS at the token limit.
            "forced_eos_token_id": None,
        }


def eos_ids(
    configured: int | Sequence[int] | None, fallback: int | None
) -> tuple[int, ...]:
    value = configured if configured is not None else fallback
    values = (value,) if isinstance(value, int) else tuple(value or ())
    if not values or any(type(item) is not int or item < 0 for item in values):
        raise ValueError("Missing or invalid EOS token configuration")
    return tuple(dict.fromkeys(values))


def repeated_suffix(tokens: Sequence[int], policy: GenerationPolicy) -> int | None:
    """Return the smallest sustained, exact period in a bounded generated tail."""
    tail = tuple(tokens[-policy.tail_tokens :])
    for period in range(
        1, min(policy.max_repeat_period, len(tail) // policy.min_repeats) + 1
    ):
        repeats = max(policy.min_repeats, math.ceil(policy.min_repeat_tokens / period))
        span = repeats * period
        if span <= len(tail) and tail[-span:] == tail[-period:] * repeats:
            return period
    return None


class DecodeGuard:
    def __init__(
        self, policy: GenerationPolicy, parameters: dict, eos: tuple[int, ...]
    ):
        self.policy = policy
        self.parameters = parameters
        self.eos = eos
        self.reason: str | None = None
        self.repeat_period: int | None = None

    def observe(
        self,
        tail: Sequence[int],
        *,
        generated: int,
        elapsed: float,
        final: bool = False,
    ) -> str | None:
        if self.reason is not None:
            return self.reason
        ended = bool(tail) and tail[-1] in self.eos
        if final or ended or generated % self.policy.check_interval == 0:
            self.repeat_period = repeated_suffix(
                tail[:-1] if ended else tail, self.policy
            )
            if self.repeat_period is not None:
                self.reason = "repetition"
        if self.reason is None:
            if ended:
                self.reason = "eos"
            elif elapsed >= self.parameters["max_seconds"]:
                self.reason = "time_limit"
            elif generated >= self.parameters["max_new_tokens"]:
                self.reason = "token_limit"
            elif final:
                self.reason = "unexpected_stop"
        return self.reason


def generate_with_retries(
    generate: Callable[[dict], dict],
    policy: GenerationPolicy,
    record: Callable[[dict], None],
    *,
    previous_attempts: Sequence[dict] = (),
) -> dict:
    """Persist bounded failures and retry once; never inspect the reference/CER."""
    reason = None
    attempts = []
    if len(previous_attempts) > 2:
        raise ValueError("Too many saved attempts")
    for number in (1, 2):
        parameters = policy.attempt_parameters(reason)
        if number <= len(previous_attempts):
            attempt = previous_attempts[number - 1]
            if attempt["attempt"] != number or attempt["parameters"] != parameters:
                raise ValueError("Saved attempt differs from generation policy")
        else:
            result = generate(parameters)
            attempt = {**result, "attempt": number, "parameters": parameters}
        reason = attempt["stop_reason"]
        record(attempt)
        attempts.append(attempt)
        if reason == "eos":
            if len(previous_attempts) > number:
                raise ValueError("Unexpected attempts after successful generation")
            break
    return {
        "prediction": attempts[-1]["prediction"],
        "status": "complete" if reason == "eos" else "incomplete",
        "stop_reason": reason,
        "attempts": attempts,
    }


def select_complete_validation_epoch(summaries: list[dict]) -> dict | None:
    eligible = [
        item
        for item in summaries
        if item["validation"].get("failed_pages", 0) == 0
        and math.isfinite(item["validation"]["cer"])
    ]
    return min(
        eligible,
        key=lambda item: (item["validation"]["cer"], item["epoch"]),
        default=None,
    )

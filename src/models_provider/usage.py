"""Provider-neutral usage and account-limit records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any

__all__ = ["ModelUsage", "UsageLedger", "UsageSnapshot", "UsageWindow"]


def _integer(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _decimal(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True, slots=True)
class ModelUsage:
    """Normalized usage for one model response, regardless of provider wire format."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    input_audio_tokens: int = 0
    output_audio_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens

    @property
    def completion_tokens(self) -> int:
        return self.output_tokens

    @property
    def cached_tokens(self) -> int:
        return self.cache_read_tokens

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "ModelUsage":
        value = value or {}
        input_tokens = _integer(value.get("input_tokens", value.get("prompt_tokens")))
        output_tokens = _integer(value.get("output_tokens", value.get("completion_tokens")))
        total_tokens = _integer(value.get("total_tokens")) or input_tokens + output_tokens
        output_details = value.get("output_token_details") or value.get("completion_tokens_details") or {}
        input_details = value.get("input_token_details") or value.get("prompt_tokens_details") or {}
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=_integer(value.get("reasoning_tokens", output_details.get("reasoning_tokens"))),
            cache_read_tokens=_integer(value.get("cache_read_tokens", value.get("cached_tokens", input_details.get("cache_read", input_details.get("cached_tokens"))))),
            cache_write_tokens=_integer(value.get("cache_write_tokens", input_details.get("cache_creation", input_details.get("cache_write_tokens")))),
            input_audio_tokens=_integer(value.get("input_audio_tokens")),
            output_audio_tokens=_integer(value.get("output_audio_tokens")),
            cost_usd=_decimal(value.get("cost_usd", value.get("cost"))),
        )

    def combined_with(self, other: "ModelUsage") -> "ModelUsage":
        return ModelUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            input_audio_tokens=self.input_audio_tokens + other.input_audio_tokens,
            output_audio_tokens=self.output_audio_tokens + other.output_audio_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )


@dataclass(frozen=True, slots=True)
class UsageWindow:
    """One provider-reported rate-limit window."""

    name: str
    used_percent: float = 0.0
    duration_minutes: int = 0
    resets_at: int | None = None


@dataclass(frozen=True, slots=True)
class UsageSnapshot:
    """A provider account's latest quota snapshot, safe to serialize and publish."""

    provider: str
    captured_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    plan: str = ""
    active_limit: str = ""
    credit_balance: float | None = None
    has_credits: bool = False
    unlimited_credits: bool = False
    windows: tuple[UsageWindow, ...] = ()


class UsageLedger:
    """Thread-safe in-memory totals keyed by provider-qualified model identifier."""

    def __init__(self) -> None:
        self._totals: dict[str, ModelUsage] = {}
        self._lock = Lock()

    def record(self, model_identifier: str, usage: ModelUsage) -> ModelUsage:
        with self._lock:
            total = self._totals.get(model_identifier, ModelUsage()).combined_with(usage)
            self._totals[model_identifier] = total
            return total

    def get(self, model_identifier: str) -> ModelUsage:
        with self._lock:
            return self._totals.get(model_identifier, ModelUsage())

    def snapshot(self) -> dict[str, ModelUsage]:
        with self._lock:
            return dict(self._totals)

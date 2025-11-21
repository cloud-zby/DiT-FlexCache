from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class _MetricsState:
    lookups: int = 0
    hits: int = 0
    skip_step_total: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    saves: int = 0
    saved_steps_total: int = 0

    def reset(self) -> None:
        self.lookups = 0
        self.hits = 0
        self.skip_step_total = 0
        self.by_type.clear()
        self.saves = 0
        self.saved_steps_total = 0


class CacheMetrics:
    _default: "CacheMetrics" | None = None

    def __init__(self) -> None:
        self._state = _MetricsState()

    @classmethod
    def get(cls) -> "CacheMetrics":
        if cls._default is None:
            cls._default = CacheMetrics()
        return cls._default

    def reset(self) -> None:
        self._state.reset()

    def record_lookup(self, hit: bool, skip_step: int = 0, cache_type: str = "none") -> None:
        self._state.lookups += 1
        if hit:
            self._state.hits += 1
            self._state.skip_step_total += max(0, int(skip_step))
            self._state.by_type[cache_type] = self._state.by_type.get(cache_type, 0) + 1

    def record_save(self, saved_steps_count: int) -> None:
        self._state.saves += 1
        self._state.saved_steps_total += max(0, int(saved_steps_count))

    def snapshot(self) -> Dict[str, object]:
        lookups = self._state.lookups
        hits = self._state.hits
        hit_ratio = (hits / lookups) if lookups else 0.0
        avg_skip = (self._state.skip_step_total / hits) if hits else 0.0
        avg_saved_steps = (self._state.saved_steps_total / self._state.saves) if self._state.saves else 0.0
        return {
            "lookups": lookups,
            "hits": hits,
            "hit_ratio": round(hit_ratio, 4),
            "avg_skip_step": round(avg_skip, 2),
            "by_type": dict(sorted(self._state.by_type.items())),
            "saves": self._state.saves,
            "avg_saved_steps": round(avg_saved_steps, 2),
        }


"""Reward-driven sampling allocation, independent of environments and PPO loss."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from uni_rl.ipc.mix_schedule import allocate_counts, normalized_ratios

_DEFAULTS = {
    "metric": "mean_step_reward",
    "strategy": "learning_progress",
    "interval": 50,
    "warmup": 100,
    "ema_alpha": 0.2,
    "temperature": 1.0,
    "min_ratio": 0.05,
    "max_ratio": 0.5,
    "smoothing": 0.3,
    "freeze_after": 2000,
}
_METRIC_WEIGHTS = {
    "mean_step_reward": "samples",
    "mean_episode_reward": "completed_episodes",
    "mean_episode_length": "completed_episodes",
}


def validate_adaptive_config(
    config: Mapping[str, Any] | None, names: tuple[str, ...], total: int
) -> dict[str, Any] | None:
    if config is None:
        return None
    if type(total) is not int or total < 1:
        raise ValueError("Adaptive total must be a positive integer")
    if not isinstance(config, Mapping) or set(config) - set(_DEFAULTS):
        raise ValueError("adaptive contains unknown settings or is not a mapping")
    result = {**_DEFAULTS, **config}
    if len(names) < 2:
        raise ValueError("Adaptive mixing requires at least two training backends")
    if not isinstance(result["metric"], str) or result["metric"] not in _METRIC_WEIGHTS:
        raise ValueError(f"adaptive.metric must be one of {tuple(_METRIC_WEIGHTS)}")
    if not isinstance(result["strategy"], str) or result["strategy"] not in {
        "learning_progress",
        "low_reward",
    }:
        raise ValueError("adaptive.strategy must be learning_progress or low_reward")
    for key in ("interval", "warmup", "freeze_after"):
        value = result[key]
        if type(value) is not int or value < (1 if key != "warmup" else 0):
            raise ValueError(f"adaptive.{key} must be a valid nonnegative iteration count")
    if result["freeze_after"] <= max(result["warmup"], result["interval"]):
        raise ValueError("adaptive.freeze_after must leave time after warmup for adaptation")
    for key in ("ema_alpha", "temperature", "min_ratio", "max_ratio", "smoothing"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"adaptive.{key} must be numeric")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"adaptive.{key} must be positive and finite")
    for key in ("ema_alpha", "smoothing", "max_ratio"):
        if result[key] > 1:
            raise ValueError(f"adaptive.{key} must be at most one")
    floor = max(result["min_ratio"], 1 / total)
    if floor > result["max_ratio"] or len(names) * floor > 1 + 1e-12:
        raise ValueError("Adaptive minimum allocation cannot fit all backends in num_envs")
    if len(names) * result["max_ratio"] < 1 - 1e-12:
        raise ValueError("Adaptive maximum allocation cannot sum to one")
    return result


def _bounded_distribution(values: list[float], floor: float, ceiling: float) -> list[float]:
    """Euclidean simplex projection with both lower and upper bounds."""
    lo, hi = min(values) - ceiling, max(values) - floor
    for _ in range(80):
        shift = (lo + hi) / 2
        if math.fsum(min(ceiling, max(floor, v - shift)) for v in values) > 1:
            lo = shift
        else:
            hi = shift
    return [min(ceiling, max(floor, value - (lo + hi) / 2)) for value in values]


class AdaptiveMixController:
    """Update between rollouts using sample-weighted windows and bounded smoothing.

    Learning progress is the magnitude of change in each backend's smoothed
    reward, normalized by its recent reward scale. Low reward allocates more
    samples to the weakest backend under the common task reward. Neither is
    an optimizer of held-out performance; a separate experiment must select
    and test a frozen ratio. State contains JSON-compatible primitives only.
    """

    def __init__(self, config: Mapping[str, Any], ratios: Mapping[str, float], total: int):
        self.names = tuple(ratios)
        self.total = total
        validated = validate_adaptive_config(config, self.names, total)
        assert validated is not None
        self.config = validated
        self.ratios = normalized_ratios(ratios)
        self.floor = max(float(self.config["min_ratio"]), 1 / total)
        self.ceiling = float(self.config["max_ratio"])
        if any(v < self.floor - 1e-12 or v > self.ceiling + 1e-12 for v in self.ratios.values()):
            raise ValueError("Initial adaptive ratios must satisfy min_ratio and max_ratio")
        allocate_counts(self.ratios, total)
        self.ema: dict[str, float] = {}
        self.scale: dict[str, float] = {}
        self.sums = dict.fromkeys(self.names, 0.0)
        self.weights = dict.fromkeys(self.names, 0.0)
        self.window_iterations = 0
        self.last_iteration = -1
        self.updates = 0
        self.frozen = False

    def _clear_window(self) -> None:
        self.sums = dict.fromkeys(self.names, 0.0)
        self.weights = dict.fromkeys(self.names, 0.0)
        self.window_iterations = 0

    def observe(
        self, iteration: int, metrics: Mapping[str, Mapping[str, float]], *, paused: bool = False
    ) -> dict[str, Any] | None:
        if type(iteration) is not int or iteration <= self.last_iteration:
            raise ValueError("Adaptive observations must have increasing iteration numbers")
        self.last_iteration = iteration
        completed = iteration + 1
        if self.frozen:
            return None
        if completed >= self.config["freeze_after"]:
            self.frozen = True
            self._clear_window()
            return {"event": "adaptive_frozen", "iteration": iteration, "ratios": dict(self.ratios)}
        if paused:
            self._clear_window()
            return None
        metric = self.config["metric"]
        weight_key = _METRIC_WEIGHTS[metric]
        rows = []
        for name in self.names:
            row = metrics.get(name, {})
            weight = row.get(weight_key, 0.0)
            value = row.get(metric)
            if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
                self._clear_window()
                return {
                    "event": "adaptive_skipped",
                    "iteration": iteration,
                    "reason": f"Invalid weight: {name}",
                }
            if weight > 0 and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                self._clear_window()
                return {
                    "event": "adaptive_skipped",
                    "iteration": iteration,
                    "reason": f"Missing/nonfinite {metric}: {name}",
                }
            if weight > 0:
                assert value is not None
                rows.append((name, float(value), float(weight)))
            else:
                rows.append((name, 0.0, 0.0))
        for name, value, weight in rows:
            self.sums[name] += value * weight
            self.weights[name] += weight
        self.window_iterations += 1
        if self.window_iterations < self.config["interval"] or completed < self.config["warmup"]:
            return None
        if any(weight <= 0 for weight in self.weights.values()):
            self._clear_window()
            return {
                "event": "adaptive_skipped",
                "iteration": iteration,
                "reason": "No metric observations for every backend",
            }
        means = {name: self.sums[name] / self.weights[name] for name in self.names}
        self._clear_window()
        alpha = self.config["ema_alpha"]
        progress = {}
        for name, value in means.items():
            previous = self.ema.get(name, value)
            scale = self.scale.get(name, abs(value))
            smoothed = previous + alpha * (value - previous)
            # Normalize using the previous scale, preserving invariance to
            # per-backend reward magnitude in the progress strategy.
            progress[name] = abs(smoothed - previous) / max(scale, abs(value), 1e-8)
            self.ema[name] = smoothed
            self.scale[name] = (1 - alpha) * scale + alpha * abs(value)
        scores = list(progress.values())
        if self.config["strategy"] == "low_reward":
            scores = [-self.ema[name] for name in self.names]
        center = math.fsum(scores) / len(scores)
        spread = math.sqrt(math.fsum((v - center) ** 2 for v in scores) / len(scores))
        # Tiny numerical differences must not create a strong allocation signal.
        standardized = [(v - center) / max(spread, 1e-6) for v in scores]
        logits = [v / self.config["temperature"] for v in standardized]
        peak = max(logits)
        exp = [math.exp(v - peak) for v in logits]
        target = _bounded_distribution([v / math.fsum(exp) for v in exp], self.floor, self.ceiling)
        smoothing = self.config["smoothing"]
        previous_ratios = dict(self.ratios)
        self.ratios = {
            name: (1 - smoothing) * self.ratios[name] + smoothing * target[index]
            for index, name in enumerate(self.names)
        }
        allocate_counts(self.ratios, self.total)
        self.updates += 1
        return {
            "event": "adaptive_updated",
            "iteration": iteration,
            "applies_at_iteration": completed,
            "metric": metric,
            "strategy": self.config["strategy"],
            "window_means": means,
            "ema": dict(self.ema),
            "scores": dict(zip(self.names, standardized, strict=True)),
            "previous_ratios": previous_ratios,
            "ratios": dict(self.ratios),
            "counts": allocate_counts(self.ratios, self.total),
        }

    def state_dict(self) -> dict[str, Any]:
        return deepcopy(
            {
                "version": 1,
                "config": self.config,
                "names": self.names,
                "total": self.total,
                "ratios": self.ratios,
                "ema": self.ema,
                "scale": self.scale,
                "sums": self.sums,
                "weights": self.weights,
                "window_iterations": self.window_iterations,
                "last_iteration": self.last_iteration,
                "updates": self.updates,
                "frozen": self.frozen,
            }
        )

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("Invalid adaptive checkpoint")
        for key, expected in (("version", 1), ("config", self.config), ("total", self.total)):
            if state.get(key) != expected:
                raise ValueError(f"Incompatible adaptive checkpoint: {key}")
        if tuple(state.get("names", ())) != self.names:
            raise ValueError("Incompatible adaptive checkpoint: names")
        for key in ("window_iterations", "last_iteration", "updates"):
            if type(state.get(key)) is not int or state[key] < (
                -1 if key == "last_iteration" else 0
            ):
                raise ValueError(f"Invalid adaptive checkpoint: {key}")
        if type(state.get("frozen")) is not bool:
            raise ValueError("Invalid adaptive checkpoint: frozen")
        for key in ("ratios", "ema", "scale", "sums", "weights"):
            values = state.get(key)
            if not isinstance(values, Mapping) or (
                set(values) != set(self.names) and not (key in {"ema", "scale"} and not values)
            ):
                raise ValueError(f"Invalid adaptive checkpoint: {key}")
            if any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in values.values()
            ):
                raise ValueError(f"Nonfinite adaptive checkpoint: {key}")
            if key in {"weights", "scale"} and any(v < 0 for v in values.values()):
                raise ValueError(f"Negative adaptive checkpoint: {key}")
        ratios = normalized_ratios(state["ratios"])
        if not math.isclose(math.fsum(state["ratios"].values()), 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Invalid adaptive checkpoint: ratios must sum to one")
        if any(v < self.floor - 1e-12 or v > self.ceiling + 1e-12 for v in ratios.values()):
            raise ValueError("Invalid adaptive checkpoint: ratio bounds")
        allocate_counts(ratios, self.total)
        for key in (
            "ratios",
            "ema",
            "scale",
            "sums",
            "weights",
            "window_iterations",
            "last_iteration",
            "updates",
            "frozen",
        ):
            setattr(self, key, deepcopy(state[key]))

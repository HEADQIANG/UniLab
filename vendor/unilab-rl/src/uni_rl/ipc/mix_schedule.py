"""Deterministic mixed-backend allocation and iteration-boundary control."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any


def _weights(ratios: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(ratios, Mapping) or not ratios:
        raise ValueError("Ratios must be a nonempty mapping")
    result = {}
    for name, value in ratios.items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError(f"Invalid backend name: {name!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Weight for {name} must be numeric")
        try:
            weight = float(value)
        except OverflowError as exc:
            raise ValueError(f"Weight for {name} must be finite") from exc
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Weight for {name} must be finite and nonnegative")
        result[name] = weight
    if not any(result.values()):
        raise ValueError("At least one backend weight must be positive")
    return result


def parse_mix(text: str) -> dict[str, float]:
    """Parse the ordered CLI form ``mujoco=0.7,motrix=0.3``."""
    ratios: dict[str, float] = {}
    for item in text.split(","):
        name, separator, value = item.strip().partition("=")
        if not separator or name in ratios:
            raise ValueError("Expected unique comma-separated backend=weight pairs")
        try:
            ratios[name] = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid weight for {name!r}: {value!r}") from exc
    return _weights(ratios)


def normalized_ratios(ratios: Mapping[str, float]) -> dict[str, float]:
    """Normalize without overflowing when users provide large finite weights."""
    weights = _weights(ratios)
    scale = max(weights.values())
    denominator = math.fsum(value / scale for value in weights.values())
    return {name: (value / scale) / denominator for name, value in weights.items()}


def allocate_counts(ratios: Mapping[str, float], total: int) -> dict[str, int]:
    """Largest-remainder allocation with declaration-order ties and no starvation."""
    if type(total) is not int or total < 1:
        raise ValueError("Total environment count must be a positive integer")
    fractions = normalized_ratios(ratios)
    quotas = [value * total for value in fractions.values()]
    counts = [math.floor(value) for value in quotas]
    remaining = total - sum(counts)
    order = sorted(range(len(counts)), key=lambda index: -(quotas[index] - counts[index]))
    for index in order[:remaining]:
        counts[index] += 1
    result = dict(zip(fractions, counts, strict=True))
    for name, count in result.items():
        if ratios[name] > 0 and count == 0:
            raise ValueError(
                f"Positive weight for {name} received zero environments; increase total"
            )
    return result


def _declared_ratios(ratios: Mapping[str, float], backend_names: Sequence[str]) -> dict[str, float]:
    weights = _weights(ratios)
    if set(weights) != set(backend_names):
        raise ValueError("Ratios must specify exactly the backends declared at startup")
    return {name: weights[name] for name in backend_names}


def validate_stages(
    stages: Sequence[Mapping[str, Any]],
    backend_names: Sequence[str],
    total: int | None = None,
) -> list[dict[str, Any]]:
    """Validate an ascending table of absolute (zero-based) iteration boundaries."""
    if not isinstance(stages, (list, tuple)):
        raise ValueError("Stages must be a list")
    result = []
    previous = -1
    for stage in stages:
        if not isinstance(stage, Mapping) or set(stage) != {"iteration", "ratios"}:
            raise ValueError("Each stage requires only iteration and ratios")
        iteration = stage["iteration"]
        if type(iteration) is not int or iteration < 0 or iteration <= previous:
            raise ValueError("Stage iterations must be unique, ascending nonnegative integers")
        ratios = _declared_ratios(stage["ratios"], backend_names)
        if total is not None:
            allocate_counts(ratios, total)
        result.append({"iteration": iteration, "ratios": ratios})
        previous = iteration
    return result


def atomic_write_control(
    run_dir: str | Path, ratios: Mapping[str, float] | None = None, *, clear: bool = False
) -> Path:
    """Atomically replace a run's control file; monotonic nanoseconds version requests."""
    if clear == (ratios is not None):
        raise ValueError("Specify either ratios or clear=True")
    directory = Path(run_dir)
    if not directory.is_dir():
        raise ValueError(f"Run directory does not exist: {directory}")
    path = directory / "control.json"
    weights = _weights(ratios) if ratios is not None else None
    manifest_path = directory / "mix_manifest.json"
    seen_version = 0
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        path = Path(manifest["control_file"])
        seen_version = int(manifest.get("seen_version", 0))
        if weights is not None:
            weights = _declared_ratios(weights, manifest["backend_names"])
            allocate_counts(weights, manifest["total"])
    try:
        current = json.loads(path.read_text())
        if type(current.get("version")) is int:
            seen_version = max(seen_version, current["version"])
    except (OSError, ValueError, AttributeError, RecursionError):
        pass
    payload: dict[str, Any] = {"version": max(time.time_ns(), seen_version + 1)}
    if clear:
        payload["clear"] = True
    else:
        payload["ratios"] = weights
    fd, temporary = tempfile.mkstemp(prefix=".mix-control-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


class MixSchedule:
    """Persistent manual overrides over a deterministic staged sampling schedule."""

    def __init__(
        self,
        initial_ratios: Mapping[str, float],
        total: int,
        stages: Sequence[Mapping[str, Any]] = (),
        control_file: str | Path | None = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.initial_ratios = _weights(initial_ratios)
        allocate_counts(self.initial_ratios, total)
        self.total = total
        self.backend_names = tuple(self.initial_ratios)
        self.stages = validate_stages(stages, self.backend_names, total)
        self.control_file = Path(control_file) if control_file is not None else None
        self.event_callback = event_callback
        self.seen_version = 0
        self.manual_ratios: dict[str, float] | None = None
        self.effective_ratios = dict(self.initial_ratios)
        self._last_invalid_text: str | None = None

    def _event(self, **event: Any) -> None:
        if self.event_callback is not None:
            self.event_callback(event)

    def _read_control(self, iteration: int) -> None:
        if self.control_file is None:
            return
        try:
            text = self.control_file.read_text()
        except FileNotFoundError:
            return
        except (OSError, UnicodeError) as exc:
            invalid = f"read-error: {exc}"
            if invalid != self._last_invalid_text:
                self._event(event="control_rejected", iteration=iteration, reason=str(exc))
                self._last_invalid_text = invalid
            return
        try:
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError("Control request must be an object")
            version = payload.get("version")
            if type(version) is not int or version < 1:
                raise ValueError("Control version must be a positive integer")
            if version <= self.seen_version:
                return
            self.seen_version = version
            if set(payload) == {"version", "clear"} and payload["clear"] is True:
                self.manual_ratios = None
            elif set(payload) == {"version", "ratios"}:
                ratios = _declared_ratios(payload["ratios"], self.backend_names)
                allocate_counts(ratios, self.total)
                self.manual_ratios = ratios
            else:
                raise ValueError("Control request requires ratios or clear=true")
            self._event(
                event="control_accepted",
                iteration=iteration,
                version=version,
                manual_ratios=self.manual_ratios,
            )
        except (ValueError, TypeError, RecursionError) as exc:
            if text != self._last_invalid_text:
                self._event(event="control_rejected", iteration=iteration, reason=str(exc))
                self._last_invalid_text = text

    def resolve(
        self, iteration: int, *, fallback_ratios: Mapping[str, float] | None = None
    ) -> dict[str, int]:
        if type(iteration) is not int or iteration < 0:
            raise ValueError("Iteration must be a nonnegative integer")
        self._read_control(iteration)
        ratios = (
            self.initial_ratios
            if fallback_ratios is None
            else _declared_ratios(fallback_ratios, self.backend_names)
        )
        for stage in self.stages:
            if stage["iteration"] > iteration:
                break
            ratios = stage["ratios"]
        self.effective_ratios = dict(self.manual_ratios or ratios)
        return allocate_counts(self.effective_ratios, self.total)

    def state_dict(self) -> dict[str, Any]:
        return deepcopy(
            {
                "initial_ratios": self.initial_ratios,
                "total": self.total,
                "backend_names": self.backend_names,
                "stages": self.stages,
                "seen_version": self.seen_version,
                "manual_ratios": self.manual_ratios,
                "effective_ratios": self.effective_ratios,
            }
        )

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for key in ("initial_ratios", "total", "backend_names", "stages"):
            if state.get(key) != getattr(self, key):
                raise ValueError(f"Incompatible mixed schedule checkpoint: {key}")
        version = state.get("seen_version")
        if type(version) is not int or version < 0:
            raise ValueError("Invalid checkpoint control version")
        manual = state.get("manual_ratios")
        if manual is not None:
            manual = _declared_ratios(manual, self.backend_names)
            allocate_counts(manual, self.total)
        effective = _declared_ratios(state["effective_ratios"], self.backend_names)
        allocate_counts(effective, self.total)
        self.seen_version = version
        self.manual_ratios = manual
        self.effective_ratios = effective

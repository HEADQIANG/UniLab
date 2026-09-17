"""Synchronous, shared-memory composition of injected vector environments.

Only small control messages and diagnostic metadata cross the pipes. Each
worker owns one backend, while the caller owns policy inference and rollout
storage. Factories must keep backend imports inside their call bodies.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
import signal
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.connection import wait as wait_connections
from multiprocessing.shared_memory import SharedMemory
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from uni_rl.env_contract import EnvAlgoCapabilities, EnvFactory, EnvProtocol, get_algo_capabilities

_CLOSE_GRACE_SECONDS = 15.0
_CLOSE_TERMINATE_SECONDS = 1.0
_CLOSE_KILL_SECONDS = 1.0


@dataclass(frozen=True)
class WorkerSpec:
    """One declared backend; zero-count entries are not launched."""

    name: str
    factory: EnvFactory
    env_cfg_override: Mapping[str, Any] | None = None
    process_env: dict[str, str] | None = None
    cpu_ids: tuple[int, ...] | None = None
    timeout_s: float = 120.0
    seed: int = 1
    metadata_factory: Callable[[EnvProtocol], dict[str, Any]] | None = None


@dataclass
class MixedState:
    obs: dict[str, np.ndarray]
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: dict[str, Any]
    final_observation: dict[str, np.ndarray] | None


def _plain(value: Any) -> Any:
    """Normalize cold-path metadata without discarding sequence order."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported semantic metadata type: {type(value).__name__}")


def _metadata(env: EnvProtocol, spec: WorkerSpec) -> dict[str, Any]:
    caps = get_algo_capabilities(env)
    action_shape = tuple(env.action_space.shape)
    if len(action_shape) != 1:
        raise ValueError("Mixed environments require flat action spaces")
    metadata: dict[str, Any] = {
        "obs_groups": list(env.obs_groups_spec.items()),
        "observation_shape": tuple(env.observation_space.shape),
        "action_shape": action_shape,
        "action_low": (
            caps.action_low
            if caps.action_low is not None
            else getattr(env.action_space, "low", None)
        ),
        "action_high": (
            caps.action_high
            if caps.action_high is not None
            else getattr(env.action_space, "high", None)
        ),
        "joint_names": caps.joint_names,
        "ctrl_dt": float(env.cfg.ctrl_dt),
        "max_episode_seconds": float(env.cfg.max_episode_seconds),
        "semantic": spec.metadata_factory(env) if spec.metadata_factory else {},
    }
    for key in ("ctrl_dt", "max_episode_seconds"):
        if not np.isfinite(metadata[key]) or metadata[key] <= 0:
            raise ValueError("ctrl_dt and max_episode_seconds must be finite and positive")
    if "obs" not in env.obs_groups_spec or any(dim <= 0 for dim in env.obs_groups_spec.values()):
        raise ValueError("Observation groups must have positive dimensions and include 'obs'")
    if action_shape[0] <= 0:
        raise ValueError("Action dimension must be positive")
    for name in ("action_low", "action_high"):
        if metadata[name] is not None:
            bound = np.asarray(metadata[name])
            if bound.shape != action_shape:
                raise ValueError(f"{name}: expected shape {action_shape}, got {bound.shape}")
            if bound.dtype.kind not in "bifu" or np.isnan(bound).any():
                raise ValueError(f"{name}: bounds must be numeric and contain no NaN")
    if metadata["action_low"] is not None and metadata["action_high"] is not None:
        if np.any(np.asarray(metadata["action_low"]) > np.asarray(metadata["action_high"])):
            raise ValueError("action_low must not exceed action_high")
    return {key: _plain(value) for key, value in metadata.items()}


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {result.shape}")
    if result.dtype.kind not in "bifu":
        raise ValueError(f"{name}: expected numeric array, got {result.dtype}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name}: non-finite values")
    if result.dtype.kind == "f" and np.any(np.abs(result) > np.finfo(np.float32).max):
        raise ValueError(f"{name}: values exceed float32 transport range")
    return result


def _logs(info: Mapping[str, Any]) -> dict[str, float]:
    result = {}
    for name, value in info.get("log", {}).items():
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if array.size and array.dtype.kind in "bifu":
            result[str(name)] = float(array.mean())
    return result


def _publish_state(
    state: Any, buffers: dict[str, np.ndarray], groups: dict[str, int], count: int
) -> dict[str, float]:
    if list(state.obs) != list(groups):
        raise ValueError("Observation group order changed at runtime")
    for name, dim in groups.items():
        buffers[f"obs/{name}"][:] = _array(state.obs[name], (count, dim), f"obs/{name}")
    for name in ("reward", "terminated", "truncated"):
        buffers[name][:] = _array(getattr(state, name), (count,), name)
    info = state.info
    done = buffers["terminated"] | buffers["truncated"]
    final = state.final_observation
    if final is None:
        final = info.get("final_observation")
    valid = done.copy()
    if "_final_observation" in info:
        valid &= _array(info["_final_observation"], (count,), "_final_observation").astype(bool)
    if final is None:
        valid[:] = False
    if np.any(buffers["truncated"] & ~valid):
        raise ValueError("Truncated environments require valid pre-reset final_observation")
    buffers["final_valid"][:] = valid
    for name, dim in groups.items():
        target = buffers[f"final/{name}"]
        target.fill(0)
        if np.any(valid):
            if not isinstance(final, dict) or list(final) != list(groups):
                raise ValueError("final_observation must contain every ordered observation group")
            value = np.asarray(final[name])
            if value.shape == (count, dim):
                target[valid] = _array(value[valid], (int(valid.sum()), dim), f"final/{name}")
            else:
                target[valid] = _array(value, (int(valid.sum()), dim), f"final/{name}")
    if "steps" in info:
        buffers["steps"][:] = _array(info["steps"], (count,), "steps")
    else:
        buffers["steps"] += 1
        buffers["steps"][done] = 0
    return _logs(info)


def _worker_main(spec: WorkerSpec, count: int, generation: int, connection: Connection) -> None:
    env: EnvProtocol | None = None
    memory: list[SharedMemory] = []
    try:
        if hasattr(os, "setsid"):
            os.setsid()
        if spec.process_env:
            os.environ.update(spec.process_env)
        if spec.cpu_ids is not None:
            if not hasattr(os, "sched_setaffinity"):
                raise RuntimeError("CPU affinity is unavailable on this platform")
            os.sched_setaffinity(0, spec.cpu_ids)
        seed = (spec.seed + generation * 1_000_003) % (2**32)
        random.seed(seed)
        np.random.seed(seed)
        env = spec.factory(count, spec.env_cfg_override)
        if env.num_envs != count:
            raise ValueError(f"Factory returned {env.num_envs} environments instead of {count}")
        metadata = _metadata(env, spec)
        initial_state = env.init_state() if env.state is None else env.state
        connection.send(("ready", metadata))
        command, payload = connection.recv()
        if command != "attach":
            raise RuntimeError("Expected shared-memory attachment")
        descriptors, start, stop = payload
        buffers = {}
        for name, shm_name, shape, dtype in descriptors:
            shm = SharedMemory(name=shm_name)
            memory.append(shm)
            buffers[name] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)[start:stop]
        groups = dict(env.obs_groups_spec)
        log = _publish_state(initial_state, buffers, groups, count)
        if "steps" not in initial_state.info:
            buffers["steps"].fill(0)
        connection.send(("ok", {"log": log}))
        while True:
            command, payload = connection.recv()
            started = time.monotonic()
            if command == "close":
                break
            if command == "step":
                state = env.step(buffers["actions"].copy())
                log = _publish_state(state, buffers, groups, count)
                response = {"log": log, "step_seconds": time.monotonic() - started}
            elif command == "reset":
                indices = buffers["reset_indices"][:payload].copy()
                obs, info = env.reset(indices)
                if list(obs) != list(groups):
                    raise ValueError("Reset observation group order differs")
                for name, dim in groups.items():
                    buffers[f"obs/{name}"][indices] = _array(
                        obs[name], (len(indices), dim), f"reset/{name}"
                    )
                for name in ("reward", "terminated", "truncated", "final_valid", "steps"):
                    buffers[name][indices] = 0
                response = {"log": _logs(info)}
            elif command == "episode_lengths":
                setter = getattr(env, "set_episode_length_buf", None)
                if not callable(setter):
                    raise TypeError("Backend cannot set its episode length buffer")
                setter(buffers["episode_lengths"].copy())
                buffers["steps"][:] = buffers["episode_lengths"]
                response = {}
            elif command == "nan_guard":
                env.set_nan_guard(payload)
                response = {}
            else:
                raise ValueError(f"Unknown mixed environment command: {command}")
            connection.send(("ok", response))
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        for shm in memory:
            shm.close()
        connection.close()


class MixedProcessEnv:
    """A fixed-size batch assembled from synchronous backend subprocesses.

    State arrays are snapshots, not mutable shared-memory views: advancing a
    backend never overwrites an observation retained by the caller.
    """

    def __init__(
        self,
        specs: Sequence[WorkerSpec],
        counts: Mapping[str, int],
        *,
        generation: int = 0,
    ) -> None:
        if not specs or len({spec.name for spec in specs}) != len(specs):
            raise ValueError("Worker names must be nonempty and unique")
        for spec in specs:
            if not spec.name or not np.isfinite(spec.timeout_s) or spec.timeout_s <= 0:
                raise ValueError("Workers require a name and a finite positive timeout")
            if spec.cpu_ids is not None and not spec.cpu_ids:
                raise ValueError("cpu_ids must be nonempty when specified")
        self._specs = tuple(specs)
        self.backend_names = tuple(spec.name for spec in specs)
        self.backend_counts = self._validate_counts(counts)
        self.num_envs = sum(self.backend_counts.values())
        self.generation = generation
        self.state: MixedState | None = None
        self._metadata: dict[str, Any] | None = None
        self._processes: dict[str, Any] = {}
        self._connections: dict[str, Connection] = {}
        self._slices: dict[str, slice] = {}
        self._memory: list[SharedMemory] = []
        self._buffers: dict[str, np.ndarray] = {}
        self._closed = True
        self._stats = {
            name: {
                "samples": 0.0,
                "reward_sum": 0.0,
                "episodes": 0.0,
                "censored_episodes": 0.0,
                "episode_reward_sum": 0.0,
                "episode_length_sum": 0.0,
                "step_seconds": 0.0,
                "interrupted_episodes": 0.0,
            }
            for name in self.backend_names
        }
        self._start()

    def _validate_counts(self, counts: Mapping[str, int]) -> dict[str, int]:
        if set(counts) != set(self.backend_names):
            raise ValueError("counts must include exactly the declared backend names")
        result = {}
        for name in self.backend_names:
            count = counts[name]
            if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
                raise ValueError(f"Invalid environment count for {name}: {count}")
            result[name] = int(count)
        if sum(result.values()) == 0:
            raise ValueError("At least one backend must have environments")
        return result

    def _receive(self, name: str, deadline: float) -> tuple[str, Any]:
        connection = self._connections[name]
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not connection.poll(remaining):
            raise TimeoutError(f"Mixed backend {name!r} exceeded its operation timeout")
        try:
            status, response = connection.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError(f"Mixed backend {name!r} exited unexpectedly") from exc
        if status == "error":
            raise RuntimeError(f"Mixed backend {name!r} failed:\n{response}")
        return status, response

    def _receive_all(self, deadlines: Mapping[str, float]) -> dict[str, tuple[str, Any]]:
        pending = {self._connections[name]: name for name in deadlines}
        responses = {}
        while pending:
            earliest = min(pending.values(), key=lambda name: deadlines[name])
            remaining = deadlines[earliest] - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Mixed backend {earliest!r} exceeded its operation timeout")
            # Waiting by completion order prevents another worker's long deadline
            # from hiding a short-deadline worker's late response.
            ready = cast(list[Connection], wait_connections(list(pending), timeout=remaining))
            for connection in ready:
                name = pending.pop(connection)
                responses[name] = self._receive(name, deadlines[name])
        return responses

    def _start(self) -> None:
        self._closed = False
        self.state = None
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int64)
        self._episode_censored = np.zeros(self.num_envs, dtype=np.bool_)
        context = mp.get_context("spawn")
        deadlines = {}
        try:
            offset = 0
            for spec in self._specs:
                count = self.backend_counts[spec.name]
                if not count:
                    continue
                self._slices[spec.name] = slice(offset, offset + count)
                offset += count
                parent, child = context.Pipe()
                process = context.Process(
                    target=_worker_main,
                    args=(spec, count, self.generation, child),
                    name=f"mixed-env-{spec.name}",
                )
                self._connections[spec.name] = parent
                self._processes[spec.name] = process
                previous = {key: os.environ.get(key) for key in spec.process_env or {}}
                try:
                    os.environ.update(spec.process_env or {})
                    process.start()
                finally:
                    child.close()
                    for key, value in previous.items():
                        if value is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = value
                deadlines[spec.name] = time.monotonic() + spec.timeout_s
            reference = self._metadata
            responses = self._receive_all(deadlines)
            for name in self._connections:
                status, metadata = responses[name]
                if status != "ready":
                    raise RuntimeError(f"Invalid startup response from {name}")
                if reference is None:
                    reference = metadata
                elif reference != metadata:
                    differing = [key for key in reference if reference[key] != metadata.get(key)]
                    raise ValueError(f"Incompatible backend {name!r}: {', '.join(differing)}")
            assert reference is not None
            self._metadata = reference
            self.obs_groups_spec = dict(reference["obs_groups"])
            self.observation_space = SimpleNamespace(shape=tuple(reference["observation_shape"]))
            low = reference["action_low"]
            high = reference["action_high"]
            self.action_space = SimpleNamespace(
                shape=tuple(reference["action_shape"]),
                low=None if low is None else np.asarray(low, dtype=np.float32),
                high=None if high is None else np.asarray(high, dtype=np.float32),
            )
            self.algo_capabilities = EnvAlgoCapabilities(
                action_low=self.action_space.low,
                action_high=self.action_space.high,
                joint_names=(
                    None if reference["joint_names"] is None else tuple(reference["joint_names"])
                ),
            )
            self.cfg = SimpleNamespace(
                ctrl_dt=reference["ctrl_dt"],
                max_episode_seconds=reference["max_episode_seconds"],
            )
            self.play_capabilities = SimpleNamespace(supports_physics_state_playback=False)
            shapes = {
                "actions": ((self.num_envs, *self.action_space.shape), np.float32),
                "reward": ((self.num_envs,), np.float32),
                "terminated": ((self.num_envs,), np.bool_),
                "truncated": ((self.num_envs,), np.bool_),
                "final_valid": ((self.num_envs,), np.bool_),
                "steps": ((self.num_envs,), np.int64),
                "reset_indices": ((self.num_envs,), np.int64),
                "episode_lengths": ((self.num_envs,), np.int64),
            }
            for name, dim in self.obs_groups_spec.items():
                shapes[f"obs/{name}"] = ((self.num_envs, dim), np.float32)
                shapes[f"final/{name}"] = ((self.num_envs, dim), np.float32)
            descriptors = []
            for name, (shape, dtype) in shapes.items():
                size = int(np.prod(shape)) * np.dtype(dtype).itemsize
                shm = SharedMemory(create=True, size=max(size, 1))
                self._memory.append(shm)
                array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                array.fill(0)
                self._buffers[name] = array
                descriptors.append((name, shm.name, shape, np.dtype(dtype).str))
            payloads = {
                name: (descriptors, section.start, section.stop)
                for name, section in self._slices.items()
            }
            self._exchange("attach", payloads)
        except BaseException:
            self.close()
            raise

    def _exchange(self, command: str, payloads: Mapping[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Mixed environment is closed")
        deadlines = {}
        timeouts = {spec.name: spec.timeout_s for spec in self._specs}
        try:
            for name, payload in payloads.items():
                self._connections[name].send((command, payload))
                deadlines[name] = time.monotonic() + timeouts[name]
            result = {}
            for name, (status, response) in self._receive_all(deadlines).items():
                if status != "ok":
                    raise RuntimeError(f"Invalid {command} response from backend {name}")
                result[name] = response
            return result
        except BaseException:
            self.close()
            raise

    def _snapshot(self, responses: Mapping[str, Any] | None = None) -> MixedState:
        logs = {}
        for name, response in (responses or {}).items():
            for key, value in response.get("log", {}).items():
                logs[f"Mixed/{name}/{key}"] = value
        valid = self._buffers["final_valid"].copy()
        return MixedState(
            obs={name: self._buffers[f"obs/{name}"].copy() for name in self.obs_groups_spec},
            reward=self._buffers["reward"].copy(),
            terminated=self._buffers["terminated"].copy(),
            truncated=self._buffers["truncated"].copy(),
            info={"log": logs, "_final_observation": valid, "steps": self._buffers["steps"].copy()},
            final_observation=(
                {name: self._buffers[f"final/{name}"].copy() for name in self.obs_groups_spec}
                if np.any(valid)
                else None
            ),
        )

    def init_state(self) -> MixedState:
        if self._closed:
            raise RuntimeError("Mixed environment is closed")
        if self.state is None:
            self.state = self._snapshot()
        return self.state

    def step(self, actions: np.ndarray) -> MixedState:
        if self.state is None:
            raise RuntimeError("Call init_state() before stepping")
        if self._closed:
            raise RuntimeError("Mixed environment is closed")
        self._buffers["actions"][:] = _array(
            actions, (self.num_envs, *self.action_space.shape), "actions"
        )
        responses = self._exchange("step", dict.fromkeys(self._connections))
        self.state = self._snapshot(responses)
        self._episode_returns += self.state.reward
        self._episode_lengths += 1
        done = self.state.terminated | self.state.truncated
        for name, section in self._slices.items():
            stats = self._stats[name]
            finished = done[section] & ~self._episode_censored[section]
            stats["samples"] += self.backend_counts[name]
            stats["reward_sum"] += float(self.state.reward[section].sum())
            stats["episodes"] += int(finished.sum())
            stats["censored_episodes"] += int(
                (done[section] & self._episode_censored[section]).sum()
            )
            stats["episode_reward_sum"] += float(self._episode_returns[section][finished].sum())
            stats["episode_length_sum"] += int(self._episode_lengths[section][finished].sum())
            stats["step_seconds"] += responses[name]["step_seconds"]
        self._episode_returns[done] = 0
        self._episode_lengths[done] = 0
        self._episode_censored[done] = False
        return self.state

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Mixed environment is closed")
        indices = np.asarray(env_indices)
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise ValueError("reset indices must be a one-dimensional integer array")
        if (
            np.any(indices < 0)
            or np.any(indices >= self.num_envs)
            or len(set(indices)) != len(indices)
        ):
            raise ValueError("reset indices must be unique and within the environment batch")
        if self.state is None:
            self.init_state()
        payloads = {}
        for name, section in self._slices.items():
            local = indices[(indices >= section.start) & (indices < section.stop)] - section.start
            if local.size:
                self._buffers["reset_indices"][section][: len(local)] = local
                payloads[name] = len(local)
        responses = self._exchange("reset", payloads)
        self._episode_returns[indices] = 0
        self._episode_lengths[indices] = 0
        self._episode_censored[indices] = False
        self.state = self._snapshot(responses)
        return {name: value[indices].copy() for name, value in self.state.obs.items()}, {
            "log": self.state.info["log"],
            "steps": self.state.info["steps"][indices].copy(),
        }

    def set_episode_length_buf(self, value: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("Mixed environment is closed")
        lengths = _array(value, (self.num_envs,), "episode_lengths")
        if lengths.dtype.kind not in "iu" or np.any(lengths < 0):
            raise ValueError("Episode lengths must be nonnegative integers")
        self._buffers["episode_lengths"][:] = lengths
        self._exchange("episode_lengths", dict.fromkeys(self._connections))
        # Random timeout offsets are not observed transitions. Their first
        # partial episodes must never enter complete-episode reward statistics.
        self._episode_censored |= lengths != self._episode_lengths
        self._episode_lengths[:] = lengths
        if self.state is not None:
            self.state.info["steps"] = lengths.copy()

    def set_nan_guard(self, guard: Any) -> None:
        self._exchange("nan_guard", dict.fromkeys(self._connections, guard))

    def get_stats(self) -> dict[str, dict[str, float]]:
        """Return cumulative counters, retained across backend rebuilds."""
        return {name: dict(values) for name, values in self._stats.items()}

    def reconfigure(self, counts: Mapping[str, int]) -> bool:
        """Rebuild all workers only when the integer allocation changes."""
        updated = self._validate_counts(counts)
        if sum(updated.values()) != self.num_envs:
            raise ValueError("Reconfiguration cannot change the total environment count")
        if self._closed:
            raise RuntimeError("Cannot reconfigure a closed mixed environment")
        if updated == self.backend_counts:
            return False
        for name, section in self._slices.items():
            self._stats[name]["interrupted_episodes"] += int(
                np.count_nonzero(self._episode_lengths[section])
            )
        self.close()
        self.backend_counts = updated
        self.generation += 1
        self._start()
        return True

    def close(self) -> None:
        """Release shared memory and the workers' own process groups."""
        if self._closed:
            return
        self._closed = True
        for connection in self._connections.values():
            try:
                connection.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        processes = [process for process in self._processes.values() if process.pid is not None]

        def join_until(deadline: float) -> None:
            for process in processes:
                process.join(timeout=max(0.0, deadline - time.monotonic()))

        # Engine shutdown can release nested workers' shared memory only after
        # their application has closed. All backends get the same grace period;
        # adding a backend must not multiply the shutdown timeout.
        join_until(time.monotonic() + _CLOSE_GRACE_SECONDS)
        for sig, timeout in (
            (signal.SIGTERM, _CLOSE_TERMINATE_SECONDS),
            (signal.SIGKILL, _CLOSE_KILL_SECONDS),
        ):
            deadline = time.monotonic() + timeout
            for process in processes:
                group_signaled = False
                if hasattr(os, "killpg"):
                    try:
                        # The outer worker may have exited while an external
                        # engine descendant still belongs to its process group.
                        os.killpg(process.pid, sig)
                        group_signaled = True
                    except ProcessLookupError:
                        pass
                if not group_signaled and process.is_alive():
                    if sig == signal.SIGTERM:
                        process.terminate()
                    else:
                        process.kill()
            join_until(deadline)
        for process in processes:
            process.close()
        for connection in self._connections.values():
            connection.close()
        self._connections.clear()
        self._processes.clear()
        self._slices.clear()
        self._buffers.clear()
        for shm in self._memory:
            shm.close()
            shm.unlink()
        self._memory.clear()

    def __enter__(self) -> MixedProcessEnv:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

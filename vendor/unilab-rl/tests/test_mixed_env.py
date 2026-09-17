"""Spawn-based mixed vector environment protocol and lifecycle tests."""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import numpy as np
import pytest

from uni_rl.env_contract import EnvAlgoCapabilities, EnvProtocol
from uni_rl.ipc import mixed_env
from uni_rl.ipc.mixed_env import MixedProcessEnv, MixedState, WorkerSpec


class FakeEnv:
    def __init__(self, count: int, cfg: Any) -> None:
        self.options = dict(cfg or {})
        time.sleep(self.options.get("init_delay", 0))
        if self.options.get("init_error"):
            raise RuntimeError("test startup failed")
        self.num_envs = count
        self.obs_groups_spec = {"obs": 2, "critic": 3}
        if self.options.get("reverse_groups"):
            self.obs_groups_spec = {"critic": 3, "obs": 2}
        self.cfg = SimpleNamespace(
            ctrl_dt=self.options.get("ctrl_dt", 0.02), max_episode_seconds=1.0
        )
        self.observation_space = SimpleNamespace(shape=(2,))
        self.action_space = SimpleNamespace(
            shape=(2,),
            low=np.asarray(self.options.get("action_low", [-1, -1])),
            high=np.asarray(self.options.get("action_high", [1, 1])),
        )
        self.algo_capabilities = EnvAlgoCapabilities(
            action_low=None if self.options.get("space_bounds") else self.action_space.low,
            action_high=None if self.options.get("space_bounds") else self.action_space.high,
            joint_names=tuple(self.options.get("joints", ("a", "b"))),
        )
        self.play_capabilities = SimpleNamespace(supports_physics_state_playback=False)
        self.state: MixedState | None = None
        self.steps = np.zeros(count, dtype=np.int64)
        self.offset = self.options.get("offset", 0.0)
        if self.options.get("read_env"):
            self.offset = float(os.environ["MIXED_TEST_OFFSET"])
        self.child = None
        if self.options.get("descendant"):
            self.child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.close_memory = (
            SharedMemory(name=self.options["close_memory_name"], create=True, size=1)
            if "close_memory_name" in self.options
            else None
        )

    def _obs(self, value: np.ndarray) -> dict[str, np.ndarray]:
        return {
            name: np.repeat(value[:, None], dim, axis=1)
            for name, dim in self.obs_groups_spec.items()
        }

    def init_state(self) -> MixedState:
        self.state = MixedState(
            obs=self._obs(np.full(self.num_envs, self.offset)),
            reward=np.zeros(self.num_envs),
            terminated=np.zeros(self.num_envs, dtype=bool),
            truncated=np.zeros(self.num_envs, dtype=bool),
            info={"steps": self.steps.copy()},
            final_observation=None,
        )
        return self.state

    def step(self, actions: np.ndarray) -> MixedState:
        if "ready_event" in self.options:
            self.options["ready_event"].set()
            if not self.options["peer_event"].wait(2):
                raise RuntimeError("step actions were not dispatched to every backend")
        if self.options.get("crash"):
            os._exit(19)
        if self.options.get("error"):
            raise RuntimeError("test backend failed")
        time.sleep(self.options.get("delay", 0))
        self.steps += 1
        done = self.steps >= self.options.get("horizon", 2)
        obs = self._obs(actions.sum(axis=1) + self.offset)
        final = {name: array.copy() + 100 for name, array in obs.items()}
        for array in obs.values():
            array[done] = self.offset
        self.steps[done] = 0
        self.state = MixedState(
            obs=obs,
            reward=actions.sum(axis=1).astype(np.float64) + self.offset,
            terminated=np.zeros(self.num_envs, dtype=bool),
            truncated=done,
            info={
                "steps": self.steps.copy(),
                "log": {"offset": np.full(self.num_envs, self.offset)},
            },
            final_observation=final
            if done.any() and not self.options.get("missing_final")
            else None,
        )
        if self.options.get("compact_final") and done.any():
            self.state.final_observation = {name: value[done] for name, value in final.items()}
        if self.options.get("invalid_mask"):
            self.state.info["_final_observation"] = np.zeros(self.num_envs, dtype=bool)
        return self.state

    def reset(self, indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.steps[indices] = 0
        return self._obs(np.full(len(indices), self.offset)), {}

    def set_episode_length_buf(self, value: np.ndarray) -> None:
        self.steps[:] = value

    def set_nan_guard(self, guard: Any) -> None:
        pass

    def close(self) -> None:
        # Leave the optional external engine process alive to test group cleanup.
        if self.options.get("ignore_termination"):
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(self.options.get("close_delay", 0))
        if self.close_memory is not None:
            self.close_memory.close()
            self.close_memory.unlink()
            Path(self.options["close_marker"]).write_text(self.close_memory.name)


def make_fake(count: int, cfg: Any) -> FakeEnv:
    return FakeEnv(count, cfg)


def fake_metadata(env: Any) -> dict[str, Any]:
    return {"task": env.options.get("task", "shared"), "order": ["a", "b"]}


def child_metadata(env: Any) -> dict[str, Any]:
    return {"child_pid": env.child.pid}


def spec(name: str, **cfg: Any) -> WorkerSpec:
    return WorkerSpec(name, make_fake, cfg, timeout_s=10, metadata_factory=fake_metadata)


def test_shared_slices_float_conversion_final_observation_and_stats() -> None:
    with MixedProcessEnv(
        [spec("slow", delay=0.08), spec("fast", offset=10)], {"slow": 7, "fast": 3}
    ) as env:
        assert isinstance(env, EnvProtocol)
        assert env.state is None
        initial = env.init_state()
        actions = np.arange(20, dtype=np.float64).reshape(10, 2)
        expected = actions.sum(axis=1) + np.r_[np.zeros(7), np.full(3, 10)]
        first = env.step(actions)
        np.testing.assert_allclose(first.reward, expected)
        np.testing.assert_allclose(first.obs["obs"][:, 0], expected)
        assert first.obs["obs"].dtype == np.float32
        assert first.final_observation is None
        second = env.step(actions)
        np.testing.assert_allclose(second.final_observation["obs"][:, 0], expected + 100)
        assert second.truncated.all()
        assert second.info["_final_observation"].all()
        np.testing.assert_allclose(initial.obs["obs"][:, 0], np.r_[np.zeros(7), np.full(3, 10)])
        np.testing.assert_allclose(first.obs["obs"][:, 0], expected)
        stats = env.get_stats()
        assert stats["slow"]["samples"] == 14
        assert stats["fast"]["samples"] == 6
        assert stats["slow"]["episodes"] == 7
        assert stats["slow"]["episode_length_sum"] == 14
        assert second.info["log"]["Mixed/fast/offset"] == 10


def test_all_actions_are_sent_before_waiting_for_any_backend() -> None:
    context = mp.get_context("spawn")
    a_ready, b_ready = context.Event(), context.Event()
    workers = [
        spec("a", ready_event=a_ready, peer_event=b_ready),
        spec("b", ready_event=b_ready, peer_event=a_ready),
    ]
    with MixedProcessEnv(workers, {"a": 1, "b": 1}) as env:
        env.init_state()
        env.step(np.zeros((2, 2)))


def test_random_initial_timeouts_exclude_partial_episode_statistics() -> None:
    with MixedProcessEnv([spec("a", horizon=3, offset=1)], {"a": 2}) as env:
        env.init_state()
        env.set_episode_length_buf(np.array([2, 0], dtype=np.int64))
        env.step(np.zeros((2, 2)))
        assert env.get_stats()["a"]["episodes"] == 0
        assert env.get_stats()["a"]["censored_episodes"] == 1
        env.step(np.zeros((2, 2)))
        env.step(np.zeros((2, 2)))
        stats = env.get_stats()["a"]
        assert stats["samples"] == 6
        assert stats["reward_sum"] == 6
        assert stats["episodes"] == 1
        assert stats["episode_length_sum"] == 3
        assert stats["episode_reward_sum"] == 3
        env.step(np.zeros((2, 2)))
        assert env.get_stats()["a"]["episodes"] == 2


@pytest.mark.parametrize("space_bounds", [False, True])
def test_unbounded_action_metadata_allows_finite_samples_only(space_bounds: bool) -> None:
    cfg = {
        "space_bounds": space_bounds,
        "action_low": [-np.inf, -np.inf],
        "action_high": [np.inf, np.inf],
    }
    with MixedProcessEnv([spec("a", **cfg), spec("b", **cfg)], {"a": 1, "b": 1}) as env:
        assert np.isneginf(env.action_space.low).all()
        assert np.isposinf(env.action_space.high).all()
        env.init_state()
        assert np.isfinite(env.step(np.ones((2, 2))).reward).all()
        for invalid in (np.inf, -np.inf, np.nan):
            with pytest.raises(ValueError, match="non-finite"):
                env.step(np.full((2, 2), invalid))


@pytest.mark.parametrize(
    "cfg, message",
    [
        ({"action_low": [np.nan, -1]}, "no NaN"),
        ({"action_high": [np.nan, 1]}, "no NaN"),
        ({"action_low": [-1]}, "expected shape"),
        ({"action_high": [0, -2]}, "must not exceed"),
    ],
)
def test_invalid_action_bounds_fail_startup(cfg: dict[str, Any], message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        MixedProcessEnv([spec("a", **cfg)], {"a": 1})


def test_startup_failure_leaves_no_workers() -> None:
    before = {child.pid for child in mp.active_children()}
    with pytest.raises(RuntimeError, match="test startup failed"):
        MixedProcessEnv([spec("a"), spec("b", init_error=True)], {"a": 1, "b": 1})
    assert {child.pid for child in mp.active_children()} == before


def test_newly_activated_incompatible_backend_fails_closed() -> None:
    with MixedProcessEnv([spec("a"), spec("b", task="other")], {"a": 2, "b": 0}) as env:
        env.init_state()
        with pytest.raises(ValueError, match="semantic"):
            env.reconfigure({"a": 1, "b": 1})
        assert env._closed


def test_partial_reset_keeps_requested_row_order() -> None:
    with MixedProcessEnv([spec("a", offset=2), spec("b", offset=9)], {"a": 2, "b": 2}) as env:
        env.init_state()
        env.step(np.ones((4, 2)))
        obs, info = env.reset(np.array([3, 0], dtype=np.int64))
        np.testing.assert_array_equal(obs["obs"][:, 0], [9, 2])
        np.testing.assert_array_equal(env.state.obs["obs"][:, 0], [2, 4, 11, 9])
        np.testing.assert_array_equal(info["steps"], [0, 0])
        empty, _ = env.reset(np.array([], dtype=np.int64))
        assert empty["critic"].shape == (0, 3)
        with pytest.raises(ValueError, match="unique"):
            env.reset(np.array([1, 1]))


def test_episode_lengths_and_compact_terminal_rows() -> None:
    with MixedProcessEnv([spec("a", compact_final=True, horizon=3)], {"a": 3}) as env:
        env.init_state()
        env.set_episode_length_buf(np.array([2, 0, 1], dtype=np.int64))
        state = env.step(np.ones((3, 2)))
        np.testing.assert_array_equal(state.truncated, [True, False, False])
        np.testing.assert_allclose(state.final_observation["critic"][0], 102)
        np.testing.assert_array_equal(state.info["steps"], [0, 1, 2])


def test_reconfigure_restarts_only_changed_counts_and_keeps_metadata() -> None:
    with MixedProcessEnv([spec("a"), spec("b", offset=5)], {"a": 3, "b": 0}) as env:
        env.init_state()
        env.step(np.ones((3, 2)))
        first_pid = env._processes["a"].pid
        assert env.reconfigure({"a": 3, "b": 0}) is False
        assert env._processes["a"].pid == first_pid
        assert env.reconfigure({"a": 1, "b": 2}) is True
        assert env.generation == 1
        assert env.state is None
        assert env._processes["a"].pid != first_pid
        np.testing.assert_array_equal(env.init_state().obs["obs"][:, 0], [0, 5, 5])
        assert env.get_stats()["a"]["interrupted_episodes"] == 3
        assert env.get_stats()["a"]["samples"] == 3
        with pytest.raises(ValueError, match="total"):
            env.reconfigure({"a": 1, "b": 1})


@pytest.mark.parametrize(
    "cfg, reason",
    [
        ({"task": "other"}, "semantic"),
        ({"ctrl_dt": 0.01}, "ctrl_dt"),
        ({"reverse_groups": True}, "obs_groups"),
        ({"joints": ("b", "a")}, "joint_names"),
    ],
)
def test_semantic_mismatch_fails_before_step(cfg: dict[str, Any], reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        MixedProcessEnv([spec("a"), spec("b", **cfg)], {"a": 1, "b": 1})


@pytest.mark.parametrize("cfg", [{"missing_final": True}, {"invalid_mask": True}])
def test_missing_timeout_final_observation_fails_closed(cfg: dict[str, Any]) -> None:
    with MixedProcessEnv([spec("a", horizon=1, **cfg)], {"a": 2}) as env:
        env.init_state()
        with pytest.raises(RuntimeError, match="valid pre-reset final_observation"):
            env.step(np.zeros((2, 2)))
        assert env._closed


@pytest.mark.parametrize(
    "cfg, message",
    [({"crash": True}, "exited unexpectedly"), ({"error": True}, "test backend failed")],
)
def test_backend_failure_closes_all_workers_and_memory(cfg: dict[str, Any], message: str) -> None:
    env = MixedProcessEnv([spec("a", **cfg), spec("b")], {"a": 1, "b": 1})
    names = [shm.name for shm in env._memory]
    env.init_state()
    with pytest.raises(RuntimeError, match=message):
        env.step(np.zeros((2, 2)))
    assert not env._processes
    for name in names:
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=name)
    env.close()


def test_timeout_closes_workers() -> None:
    with MixedProcessEnv([spec("a", delay=1)], {"a": 1}) as env:
        env.init_state()
        env._specs = (replace(env._specs[0], timeout_s=0.05),)
        with pytest.raises(TimeoutError, match="a.*timeout"):
            env.step(np.zeros((1, 2)))
        assert env._closed


def test_short_deadline_is_not_hidden_by_an_earlier_slow_worker() -> None:
    workers = [spec("a", delay=0.3), spec("b", delay=0.2)]
    with MixedProcessEnv(workers, {"a": 1, "b": 1}) as env:
        env.init_state()
        env._specs = (
            replace(workers[0], timeout_s=1),
            replace(workers[1], timeout_s=0.05),
        )
        with pytest.raises(TimeoutError, match="b.*timeout"):
            env.step(np.zeros((2, 2)))
        assert env._closed
        assert not env._processes


def test_startup_deadlines_are_checked_concurrently() -> None:
    workers = [
        spec("a", init_delay=2),
        replace(spec("b", init_delay=1), timeout_s=0.75),
    ]
    before = {child.pid for child in mp.active_children()}
    with pytest.raises(TimeoutError, match="b.*timeout"):
        MixedProcessEnv(workers, {"a": 1, "b": 1})
    assert {child.pid for child in mp.active_children()} == before


def test_process_environment_is_isolated() -> None:
    a = replace(spec("a", read_env=True), process_env={"MIXED_TEST_OFFSET": "12"})
    b = replace(spec("b", read_env=True), process_env={"MIXED_TEST_OFFSET": "34"})
    previous = os.environ.get("MIXED_TEST_OFFSET")
    with MixedProcessEnv([a, b], {"a": 1, "b": 1}) as env:
        np.testing.assert_array_equal(env.init_state().obs["obs"][:, 0], [12, 34])
    assert os.environ.get("MIXED_TEST_OFFSET") == previous


def test_delayed_close_releases_worker_owned_shared_memory(tmp_path: Path) -> None:
    names = [f"mixed-close-{uuid4().hex}" for _ in range(2)]
    markers = [tmp_path / f"closed-{index}" for index in range(2)]
    workers = [
        spec(
            str(index),
            close_delay=1.0,
            close_memory_name=name,
            close_marker=str(marker),
        )
        for index, (name, marker) in enumerate(zip(names, markers, strict=True))
    ]
    env = MixedProcessEnv(workers, {worker.name: 1 for worker in workers})
    try:
        env.close()
        for name, marker in zip(names, markers, strict=True):
            assert marker.read_text() == name, "Worker cleanup was interrupted"
            with pytest.raises(FileNotFoundError):
                SharedMemory(name=name)
    finally:
        env.close()
        # A failing regression must not leave its deliberately owned memory behind.
        for name in names:
            try:
                memory = SharedMemory(name=name)
            except FileNotFoundError:
                continue
            memory.close()
            memory.unlink()


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="POSIX termination signals")
def test_stuck_workers_share_shutdown_deadlines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mixed_env, "_CLOSE_GRACE_SECONDS", 0.25)
    monkeypatch.setattr(mixed_env, "_CLOSE_TERMINATE_SECONDS", 0.25)
    monkeypatch.setattr(mixed_env, "_CLOSE_KILL_SECONDS", 0.25)
    workers = [spec(str(index), close_delay=60, ignore_termination=True) for index in range(4)]
    env = MixedProcessEnv(workers, {worker.name: 1 for worker in workers})
    worker_pids = {process.pid for process in env._processes.values()}
    started = time.monotonic()
    env.close()
    elapsed = time.monotonic() - started
    assert elapsed < 1.5, f"Shutdown waited per worker instead of per phase: {elapsed:.2f}s"
    assert not worker_pids.intersection(child.pid for child in mp.active_children())
    assert not env._processes
    env.close()


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="POSIX worker process groups")
def test_external_engine_descendant_is_stopped_on_close() -> None:
    worker = replace(spec("a", descendant=True), metadata_factory=child_metadata)
    env = MixedProcessEnv([worker], {"a": 1})
    child_pid = env._metadata["semantic"]["child_pid"]
    env.close()
    # A terminated adopted child may remain a zombie briefly on minimal CI init systems.
    deadline = time.monotonic() + 2
    while True:
        try:
            with open(f"/proc/{child_pid}/stat") as handle:
                status = handle.read().split()[2]
        except FileNotFoundError:
            break
        if status == "Z":
            break
        assert time.monotonic() < deadline, "External engine survived process-group cleanup"
        time.sleep(0.01)


@pytest.mark.parametrize(
    "counts", [{"a": 0}, {"a": -1}, {"a": 1.5}, {"a": True}, {"a": 1, "other": 2}]
)
def test_invalid_counts_rejected(counts: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        MixedProcessEnv([spec("a")], counts)

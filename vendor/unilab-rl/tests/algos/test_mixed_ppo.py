from __future__ import annotations

import json
import random
from copy import deepcopy

import numpy as np
import pytest
import torch
from conftest import FakeEnvFactory, FakeVecEnv
from rsl_rl.runners import OnPolicyRunner

from uni_rl.algos.mixed_ppo import MixedOnPolicyRunner, resolve_mixed_ppo_runtime
from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper
from uni_rl.ipc.mix_schedule import atomic_write_control
from uni_rl.ipc.mixed_env import MixedProcessEnv, WorkerSpec


class _MixedEnv(FakeVecEnv):
    def __init__(self, counts=None):
        self.backend_counts = dict(counts or {"a": 2, "b": 2})
        self.backend_names = tuple(self.backend_counts)
        self.generation = 0
        self.transitions = []
        self.stats = {
            name: dict.fromkeys(
                [
                    "samples",
                    "reward_sum",
                    "episodes",
                    "episode_reward_sum",
                    "episode_length_sum",
                    "step_seconds",
                    "interrupted_episodes",
                ],
                0.0,
            )
            for name in self.backend_names
        }
        super().__init__(sum(self.backend_counts.values()), {"obs": 3, "critic": 4}, 2)
        self.steps = np.zeros(self.num_envs, dtype=int)
        self.returns = np.zeros(self.num_envs, dtype=np.float32)

    def init_state(self):
        super().init_state()
        self.reset(np.arange(self.num_envs))
        return self.state

    def reset(self, indices):
        if self.state is None:
            super().init_state()
        self.steps[indices] = 0
        self.returns[indices] = 0
        labels = np.concatenate(
            [
                np.full(count, index + 1, dtype=np.float32)
                for index, count in enumerate(self.backend_counts.values())
            ]
        )
        for observations in self.state.obs.values():
            observations[indices] = 0
            observations[indices, 0] = labels[indices]
        self.state.terminated[indices] = False
        self.state.truncated[indices] = False
        return {key: value[indices] for key, value in self.state.obs.items()}, {}

    def step(self, actions):
        self._state = self.state.replace(
            obs={key: value.copy() for key, value in self.state.obs.items()}
        )
        self.steps += 1
        rewards = self.state.obs["obs"][:, 0].copy() + actions.sum(1) * 0.01
        self.returns += rewards
        for observations in self.state.obs.values():
            observations[:, 1] = self.steps
        final = {key: value.copy() for key, value in self.state.obs.items()}
        done = self.steps == 3
        offset = 0
        for name, count in self.backend_counts.items():
            indices = slice(offset, offset + count)
            completed = np.arange(offset, offset + count)[done[indices]]
            stats = self.stats[name]
            stats["samples"] += count
            stats["reward_sum"] += float(rewards[indices].sum())
            stats["episodes"] += len(completed)
            stats["episode_reward_sum"] += float(self.returns[completed].sum())
            stats["episode_length_sum"] += float(self.steps[completed].sum())
            stats["step_seconds"] += 0.01 if count else 0
            offset += count
        self.reset(np.flatnonzero(done))
        self._state = self.state.replace(
            reward=rewards,
            terminated=np.zeros(self.num_envs, dtype=bool),
            truncated=done,
            final_observation=final,
        )
        return self.state

    def reconfigure(self, counts):
        if counts == self.backend_counts:
            return False
        self.transitions.append(dict(counts))
        self.backend_counts = dict(counts)
        self.generation += 1
        self.init_state()
        return True

    def get_stats(self):
        return deepcopy(self.stats)


def _config(ratios=None, stages=None):
    model = {"class_name": "MLPModel", "hidden_dims": [4], "obs_normalization": True}
    return {
        "actor": {
            **model,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.5},
        },
        "critic": dict(model),
        "algorithm": {
            "class_name": "uni_rl.algos.rsl_rl_ppo:FinalObservationAwarePPO",
            "num_learning_epochs": 1,
            "num_mini_batches": 2,
        },
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "num_steps_per_env": 2,
        "save_interval": 10,
        "mixed": {
            "initial_ratios": ratios or {"a": 1, "b": 1},
            "stages": stages or [],
            "fingerprint": "synthetic-common-task-v1",
        },
    }


def _runner(config=None, counts=None, log_dir=None):
    return MixedOnPolicyRunner(RslRlVecEnvWrapper(_MixedEnv(counts)), config or _config(), log_dir)


def test_runtime_selects_existing_wrapper_and_mixed_runner():
    runtime = resolve_mixed_ppo_runtime(_config())
    assert runtime.wrapper_cls is RslRlVecEnvWrapper
    assert runtime.runner_cls is MixedOnPolicyRunner


def test_one_learner_mixed_rollout_and_safe_boundary(monkeypatch):
    config = _config(stages=[{"iteration": 1, "ratios": {"a": 3, "b": 1}}])
    runner = _runner(config)
    ids = [id(runner.alg), id(runner.alg.optimizer), id(runner.alg.actor.obs_normalizer)]
    initial = deepcopy(runner.alg.actor.state_dict())
    events = []
    original_update = runner.alg.update
    original_reconfigure = runner.mixed_env.reconfigure
    batches = []

    def update():
        batches.append(runner.alg.storage.observations["policy"].clone())
        events.append("update")
        return original_update()

    def reconfigure(counts):
        events.append("reconfigure")
        assert runner.alg.storage.step == 0
        assert runner.next_iteration == 1
        return original_reconfigure(counts)

    monkeypatch.setattr(runner.alg, "update", update)
    monkeypatch.setattr(runner.mixed_env, "reconfigure", reconfigure)
    monkeypatch.setattr(runner.logger, "log", lambda **kwargs: events.append("log"))
    runner.learn(2)
    assert events == ["update", "log", "reconfigure", "update", "log"]
    assert [id(runner.alg), id(runner.alg.optimizer), id(runner.alg.actor.obs_normalizer)] == ids
    assert torch.equal(batches[0][0, :, 0], torch.tensor([1.0, 1.0, 2.0, 2.0]))
    assert torch.equal(batches[1][0, :, 0], torch.tensor([1.0, 1.0, 1.0, 2.0]))
    assert torch.equal(batches[1][0, :, 1], torch.zeros(4))
    assert runner.next_iteration == 2
    assert any(
        not torch.equal(value, initial[key]) for key, value in runner.alg.actor.state_dict().items()
    )


def test_single_backend_is_numerically_identical_to_upstream():
    config = _config({"a": 1})
    torch.manual_seed(19)
    plain = OnPolicyRunner(RslRlVecEnvWrapper(_MixedEnv({"a": 4})), deepcopy(config))
    plain.learn(3)
    torch.manual_seed(19)
    mixed = _runner(config, {"a": 4})
    mixed.learn(3)
    for original_model, mixed_model in [
        (plain.alg.actor, mixed.alg.actor),
        (plain.alg.critic, mixed.alg.critic),
    ]:
        for key, value in original_model.state_dict().items():
            assert torch.equal(value, mixed_model.state_dict()[key]), key
    assert (
        plain.alg.optimizer.state_dict()["param_groups"]
        == mixed.alg.optimizer.state_dict()["param_groups"]
    )


def test_adaptive_changes_next_rollout_only_and_preserves_learner(tmp_path, monkeypatch):
    config = _config()
    config["mixed"]["adaptive"] = {
        "strategy": "low_reward",
        "interval": 1,
        "warmup": 0,
        "freeze_after": 4,
        "max_ratio": 0.75,
        "smoothing": 1.0,
    }
    runner = _runner(config)
    identity = id(runner.alg.optimizer)
    calls = []
    reconfigure = runner.mixed_env.reconfigure

    def check_boundary(counts):
        assert runner.alg.storage.step == 0
        calls.append((runner.next_iteration, dict(counts)))
        return reconfigure(counts)

    monkeypatch.setattr(runner.mixed_env, "reconfigure", check_boundary)
    runner.learn(2)
    assert calls == [(1, {"a": 3, "b": 1})]
    assert id(runner.alg.optimizer) == identity
    path = str(tmp_path / "adaptive.pt")
    runner.save(path)
    resumed = _runner(config)
    resumed.load(path, map_location="cpu")
    assert resumed.adaptive.state_dict() == runner.adaptive.state_dict()
    resumed.learn(3)
    assert resumed.adaptive.frozen
    assert resumed.next_iteration == 5


def test_adaptive_resume_mismatch_rejected_before_policy_mutation(tmp_path):
    cfg = _config()
    cfg["mixed"]["adaptive"] = {"max_ratio": 0.75}
    runner = _runner(cfg)
    runner.learn(1)
    path = str(tmp_path / "adaptive.pt")
    runner.save(path)
    cfg["mixed"]["adaptive"]["metric"] = "mean_episode_length"
    other = _runner(cfg)
    original = deepcopy(other.alg.actor.state_dict())
    with pytest.raises(ValueError, match="adaptive checkpoint"):
        other.load(path)
    assert all(
        torch.equal(value, other.alg.actor.state_dict()[key]) for key, value in original.items()
    )


def test_checkpoint_restores_next_iteration_override_optimizer_and_rng(tmp_path):
    config = _config(stages=[{"iteration": 1, "ratios": {"a": 3, "b": 1}}])
    config["mixed"]["control_file"] = str(tmp_path / "control.json")
    atomic_write_control(tmp_path, {"a": 1, "b": 3})
    runner = _runner(config)
    runner.learn(2)
    path = str(tmp_path / "model.pt")
    runner.save(path, {"user": "preserved"})
    policy_checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    assert policy_checkpoint["infos"]["user"] == "preserved"
    assert isinstance(policy_checkpoint["infos"]["uni_rl_mixed_ppo"]["rng"]["numpy"][1], list)
    expected_rng = (random.random(), np.random.random(), torch.rand(3))
    resumed = _runner(config)
    resumed.logger.init_logging_writer()
    infos = resumed.load(path, map_location="cpu")
    assert infos["user"] == "preserved"
    assert resumed.next_iteration == 2
    assert resumed.current_learning_iteration == 2
    assert resumed.schedule.manual_ratios == {"a": 1, "b": 3}
    assert resumed.mixed_env.backend_counts == {"a": 1, "b": 3}
    assert random.random() == expected_rng[0]
    assert np.random.random() == expected_rng[1]
    assert torch.equal(torch.rand(3), expected_rng[2])
    assert (
        resumed.alg.optimizer.state_dict()["param_groups"]
        == runner.alg.optimizer.state_dict()["param_groups"]
    )
    for index, values in runner.alg.optimizer.state_dict()["state"].items():
        for key, value in values.items():
            assert torch.equal(value, resumed.alg.optimizer.state_dict()["state"][index][key])
    (tmp_path / "control.json").write_text(json.dumps({"version": 1, "clear": True}))
    resumed.learn(1)
    assert resumed.next_iteration == 3
    assert resumed.mixed_env.backend_counts == {"a": 1, "b": 3}


def test_legacy_raw_numpy_rng_checkpoint_still_resumes_when_trusted(tmp_path):
    runner = _runner()
    runner.learn(1)
    path = str(tmp_path / "legacy_model.pt")
    runner.save(path)
    checkpoint = torch.load(path, weights_only=True)
    numpy_rng = np.random.get_state()
    checkpoint["infos"]["uni_rl_mixed_ppo"]["rng"]["numpy"] = numpy_rng
    torch.save(checkpoint, path)
    expected = np.random.random()
    resumed = _runner()
    resumed.load(path, map_location="cpu")
    assert np.random.random() == expected


def test_invalid_checkpoint_rejected_before_policy_mutation(tmp_path):
    runner = _runner()
    runner.learn(1)
    path = str(tmp_path / "model.pt")
    runner.save(path)
    config = _config()
    config["mixed"]["fingerprint"] = "different-task"
    resumed = _runner(config)
    original = deepcopy(resumed.alg.actor.state_dict())
    with pytest.raises(ValueError, match="fingerprint"):
        resumed.load(path)
    for key, value in original.items():
        assert torch.equal(value, resumed.alg.actor.state_dict()[key])


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("num_mini_batches", 3, "divisible"),
        ("rnd_cfg", {"weight": 1}, "RND"),
        ("symmetry_cfg", {}, "symmetry"),
        ("class_name", "PPO", "FinalObservationAwarePPO"),
    ],
)
def test_rejects_unsupported_algorithms_and_dropped_samples(key, value, match):
    config = _config()
    config["algorithm"][key] = value
    with pytest.raises(ValueError, match=match):
        _runner(config)


def test_rejects_multi_learner_before_distributed_initialization(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="one learner"):
        _runner()


def test_worker_failure_aborts_and_closes_everything(monkeypatch):
    runner = _runner()

    def fail(actions):
        raise RuntimeError("worker timed out")

    monkeypatch.setattr(runner.env, "step", fail)
    with pytest.raises(RuntimeError, match="worker timed out"):
        runner.learn(1)
    assert runner.env.env.closed


def test_spawned_workers_joint_learner_logs_and_rebuild(tmp_path):
    config = _config(stages=[{"iteration": 1, "ratios": {"a": 3, "b": 1}}])
    factory = FakeEnvFactory({"obs": 3, "critic": 4}, 2)
    env = MixedProcessEnv([WorkerSpec(name, factory) for name in ("a", "b")], {"a": 2, "b": 2})
    try:
        runner = MixedOnPolicyRunner(RslRlVecEnvWrapper(env), config, str(tmp_path))
        runner.learn(2)
        events = [
            json.loads(line) for line in (tmp_path / "mix_events.jsonl").read_text().splitlines()
        ]
        iterations = [event for event in events if event["event"] == "iteration"]
        assert [event["backends"]["a"]["samples"] for event in iterations] == [4, 6]
        assert [event["backends"]["b"]["samples"] for event in iterations] == [4, 2]
        assert [event["iteration"] for event in events if event["event"] == "reconfigured"] == [1]
        assert env.generation == 1
        manifest = json.loads((tmp_path / "mix_manifest.json").read_text())
        assert manifest["backend_names"] == ["a", "b"]
        assert manifest["total"] == 4
        assert (tmp_path / "model_1.pt").exists()
    finally:
        env.close()

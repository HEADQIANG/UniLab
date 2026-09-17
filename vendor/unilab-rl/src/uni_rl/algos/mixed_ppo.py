# The learn loop is adapted from RSL-RL 5.0.1 OnPolicyRunner.
# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# SPDX-License-Identifier: BSD-3-Clause
"""Synchronous mixed-backend sampling with one unmodified PPO learner."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
import time
from collections.abc import Mapping
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import check_nan, resolve_callable

from uni_rl.algos.adaptive_mix import AdaptiveMixController
from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper
from uni_rl.algos.rsl_rl_ppo import FinalObservationAwarePPO
from uni_rl.algos.rsl_rl_runtime import RslRlPPORuntime
from uni_rl.ipc.mix_schedule import MixSchedule, allocate_counts, normalized_ratios

_STATE_KEY = "uni_rl_mixed_ppo"
_STATE_VERSION = 1


@runtime_checkable
class MixedSamplingEnv(Protocol):
    """Explicit public scheduling extension supplied by the composite environment."""

    backend_names: tuple[str, ...]
    backend_counts: dict[str, int]
    generation: int

    def reconfigure(self, counts: Mapping[str, int]) -> bool: ...

    def get_stats(self) -> dict[str, dict[str, float]]: ...


def resolve_mixed_ppo_runtime(rl_cfg: dict[str, Any]) -> RslRlPPORuntime:
    """Consumer runtime-resolver entrypoint, compatible with normal PPO assembly."""
    if not isinstance(rl_cfg.get("mixed"), Mapping):
        raise ValueError("Mixed PPO requires algo.mixed configuration")
    return RslRlPPORuntime(wrapper_cls=RslRlVecEnvWrapper, runner_cls=MixedOnPolicyRunner)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".mixed-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class MixedOnPolicyRunner(OnPolicyRunner):
    """Keep upstream PPO math and add scheduling only between complete iterations."""

    def __init__(
        self,
        env: RslRlVecEnvWrapper,
        train_cfg: dict[str, Any],
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        if version("rsl-rl-lib") != "5.0.1":
            raise RuntimeError("Mixed PPO requires the pinned rsl-rl-lib==5.0.1 runtime")
        if int(os.environ.get("WORLD_SIZE", "1")) != 1 or train_cfg.get("multi_gpu"):
            raise ValueError(
                "Mixed PPO supports exactly one learner; distributed training is disabled"
            )
        if not isinstance(env, RslRlVecEnvWrapper) or not isinstance(env.env, MixedSamplingEnv):
            raise TypeError("Mixed PPO requires RslRlVecEnvWrapper over a MixedSamplingEnv")
        self.wrapped_env = env
        self.mixed_env: MixedSamplingEnv = env.env
        config = deepcopy(train_cfg)
        algorithm = config.get("algorithm", {})
        if resolve_callable(algorithm.get("class_name", "PPO")) is not FinalObservationAwarePPO:
            raise ValueError(
                "Mixed PPO requires FinalObservationAwarePPO, without HORA or custom losses"
            )
        if algorithm.get("rnd_cfg") or algorithm.get("symmetry_cfg") is not None:
            raise ValueError("Mixed PPO does not support RND or symmetry")
        steps = config.get("num_steps_per_env")
        minibatches = algorithm.get("num_mini_batches", 4)
        if type(steps) is not int or steps < 1 or type(minibatches) is not int or minibatches < 1:
            raise ValueError("Rollout steps and minibatch count must be positive integers")
        if (env.num_envs * steps) % minibatches:
            raise ValueError("num_envs * num_steps_per_env must be divisible by num_mini_batches")
        mixed = config.get("mixed")
        if (
            not isinstance(mixed, Mapping)
            or not isinstance(mixed.get("fingerprint"), str)
            or not mixed["fingerprint"]
        ):
            raise ValueError("Mixed PPO requires a nonempty common-task fingerprint")
        self.fingerprint = mixed["fingerprint"]
        self.mix_log_dir = Path(log_dir).resolve() if log_dir is not None else None
        if self.mix_log_dir is not None:
            self.mix_log_dir.mkdir(parents=True, exist_ok=True)
        control_file = mixed.get("control_file")
        if control_file is None and self.mix_log_dir is not None:
            control_file = self.mix_log_dir / "control.json"
        self.schedule = MixSchedule(
            mixed["initial_ratios"],
            env.num_envs,
            mixed.get("stages", []),
            control_file,
            self._record_event,
        )
        if self.schedule.backend_names != self.mixed_env.backend_names:
            raise ValueError("Mixed runner and environment must declare identical ordered backends")
        initial_counts = allocate_counts(self.schedule.initial_ratios, env.num_envs)
        if initial_counts != self.mixed_env.backend_counts:
            raise ValueError("Initial environment counts do not match mixed runner configuration")
        adaptive_config = mixed.get("adaptive")
        if adaptive_config is not None and self.schedule.stages:
            raise ValueError("Adaptive mixing cannot be combined with fixed ratio stages")
        self.adaptive = (
            AdaptiveMixController(adaptive_config, self.schedule.initial_ratios, env.num_envs)
            if adaptive_config is not None
            else None
        )
        self.next_iteration = 0
        super().__init__(cast(VecEnv, env), config, log_dir, device)
        if self.alg.actor.is_recurrent or self.alg.critic.is_recurrent:
            raise ValueError("Mixed PPO supports only feedforward actor and critic models")
        if self.alg.rnd or self.alg.symmetry:
            raise ValueError("Mixed PPO does not support RND or symmetry")
        self._write_manifest()

    def _event(self, **event: Any) -> None:
        self._record_event(event)

    def _record_event(self, event: dict[str, Any]) -> None:
        if self.mix_log_dir is not None:
            with (self.mix_log_dir / "mix_events.jsonl").open("a") as stream:
                stream.write(json.dumps({"time": time.time(), **event}, allow_nan=False) + "\n")

    def _write_manifest(self) -> None:
        if self.mix_log_dir is not None:
            assert self.schedule.control_file is not None
            _write_json(
                self.mix_log_dir / "mix_manifest.json",
                {
                    "backend_names": self.schedule.backend_names,
                    "total": self.schedule.total,
                    "fingerprint": self.fingerprint,
                    "control_file": str(self.schedule.control_file.resolve()),
                    "seen_version": self.schedule.seen_version,
                    "adaptive": self.adaptive.config if self.adaptive is not None else None,
                },
            )

    def _clear_episode_bookkeeping(self) -> None:
        self.wrapped_env.episode_returns.zero_()
        self.wrapped_env.episode_lengths.zero_()
        self.logger.cur_reward_sum.zero_()
        self.logger.cur_episode_length.zero_()
        self.logger.ep_extras.clear()

    def _apply_boundary(self, iteration: int) -> bool:
        previous_ratios = dict(self.schedule.effective_ratios)
        counts = self.schedule.resolve(
            iteration, fallback_ratios=self.adaptive.ratios if self.adaptive is not None else None
        )
        previous_counts = dict(self.mixed_env.backend_counts)
        changed = previous_counts != counts
        if changed:
            interrupted = int(torch.count_nonzero(self.wrapped_env.episode_lengths).item())
            start = time.perf_counter()
            self.mixed_env.reconfigure(counts)
            self.wrapped_env.reset()
            self._clear_episode_bookkeeping()
            self._event(
                event="reconfigured",
                iteration=iteration,
                counts=counts,
                previous_counts=previous_counts,
                interrupted_episodes=interrupted,
                rebuild_seconds=time.perf_counter() - start,
                generation=self.mixed_env.generation,
            )
        if changed or previous_ratios != self.schedule.effective_ratios or iteration == 0:
            self._event(
                event="allocation",
                iteration=iteration,
                requested_ratios=self.schedule.effective_ratios,
                normalized_ratios=normalized_ratios(self.schedule.effective_ratios),
                counts=counts,
                actual_ratios={name: count / self.env.num_envs for name, count in counts.items()},
            )
        self._write_manifest()
        return changed

    def _log_mix(self, iteration: int, baseline: dict[str, dict[str, float]]) -> None:
        current = self.mixed_env.get_stats()
        metrics: dict[str, dict[str, float]] = {}
        fractions = normalized_ratios(self.schedule.effective_ratios)
        for name, count in self.mixed_env.backend_counts.items():
            before, after = baseline.get(name, {}), current.get(name, {})
            delta = {key: value - before.get(key, 0.0) for key, value in after.items()}
            samples = count * self.cfg["num_steps_per_env"]
            if "samples" in delta and delta["samples"] != samples:
                raise RuntimeError(
                    f"Backend {name} collected {delta['samples']} samples, expected {samples}"
                )
            values = {
                "count": count,
                "samples": samples,
                "requested_ratio": fractions[name],
                "actual_ratio": count / self.env.num_envs,
                "sampling_seconds": delta.get("step_seconds", 0.0),
            }
            if samples:
                values["mean_step_reward"] = delta.get("reward_sum", 0.0) / samples
                if values["sampling_seconds"] > 0:
                    values["samples_per_second"] = samples / values["sampling_seconds"]
            episodes = delta.get("episodes", 0.0)
            values["completed_episodes"] = episodes
            if episodes:
                values["mean_episode_reward"] = delta["episode_reward_sum"] / episodes
                values["mean_episode_length"] = delta["episode_length_sum"] / episodes
            metrics[name] = values
            if self.logger.writer is not None:
                for metric, value in values.items():
                    self.logger.writer.add_scalar(f"Mixed/{name}/{metric}", value, iteration)
        self._event(event="iteration", iteration=iteration, backends=metrics)
        if self.adaptive is not None:
            event = self.adaptive.observe(
                iteration, metrics, paused=self.schedule.manual_ratios is not None
            )
            if event is not None:
                self._record_event(event)
            if self.mix_log_dir is not None:
                _write_json(self.mix_log_dir / "adaptive_mix.json", self.adaptive.state_dict())

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """RSL-RL 5.0.1 loop, with scheduling before rollout and checkpoint state after log."""
        if type(num_learning_iterations) is not int or num_learning_iterations < 0:
            raise ValueError("num_learning_iterations must be a nonnegative integer")
        try:
            self._learn(num_learning_iterations, init_at_random_ep_len)
        except BaseException:
            self.wrapped_env.close()
            raise

    def _learn(self, num_learning_iterations: int, init_at_random_ep_len: bool) -> None:
        self._apply_boundary(self.next_iteration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        self.logger.init_logging_writer()
        start_it = self.next_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            if it != start_it and self._apply_boundary(it):
                obs = self.env.get_observations().to(self.device)
            baseline = self.mixed_env.get_stats()
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    self.logger.process_env_step(rewards, dones, extras, None)
                stop = time.time()
                collect_time = stop - start
                start = stop
                self.alg.compute_returns(obs)
            loss_dict = self.alg.update()
            if not all(math.isfinite(float(value)) for value in loss_dict.values()):
                raise FloatingPointError(f"Nonfinite mixed PPO loss at iteration {it}: {loss_dict}")
            learn_time = time.time() - start
            self.current_learning_iteration = it
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=None,
            )
            self._log_mix(it, baseline)
            self.next_iteration = it + 1
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                assert self.logger.log_dir is not None
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
        if self.logger.writer is not None:
            assert self.logger.log_dir is not None
            self.save(
                os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt")
            )
            self.logger.stop_logging_writer()

    def save(self, path: str, infos: dict | None = None) -> None:
        numpy_rng = np.random.get_state(legacy=True)
        assert isinstance(numpy_rng, tuple)
        checkpoint_infos = dict(infos or {})
        checkpoint_infos[_STATE_KEY] = {
            "version": _STATE_VERSION,
            "next_iteration": self.next_iteration,
            "fingerprint": self.fingerprint,
            "num_steps_per_env": self.cfg["num_steps_per_env"],
            "schedule": self.schedule.state_dict(),
            "adaptive": self.adaptive.state_dict() if self.adaptive is not None else None,
            "generation": self.mixed_env.generation,
            "rng": {
                "python": random.getstate(),
                # Policy inspection uses torch.load(weights_only=True). Keep
                # NumPy's RNG array out of the checkpoint's pickle globals.
                "numpy": (
                    numpy_rng[0],
                    numpy_rng[1].tolist(),
                    numpy_rng[2],
                    numpy_rng[3],
                    numpy_rng[4],
                ),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
        }
        super().save(path, checkpoint_infos)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        if load_cfg is not None:
            raise ValueError(
                "Mixed resume requires a complete learner checkpoint; use the standard runner for inference"
            )
        checkpoint = torch.load(path, weights_only=False, map_location=map_location)
        infos = checkpoint.get("infos") if isinstance(checkpoint, Mapping) else None
        state = infos.get(_STATE_KEY) if isinstance(infos, Mapping) else None
        if (
            not isinstance(state, Mapping)
            or type(state.get("version")) is not int
            or state["version"] != _STATE_VERSION
        ):
            raise ValueError("Missing or unsupported mixed PPO checkpoint state")
        if (
            state.get("fingerprint") != self.fingerprint
            or state.get("num_steps_per_env") != self.cfg["num_steps_per_env"]
        ):
            raise ValueError("Incompatible mixed PPO task fingerprint or rollout length")
        next_iteration = state.get("next_iteration")
        if type(next_iteration) is not int or next_iteration < 0:
            raise ValueError("Invalid checkpoint next_iteration")
        schedule = MixSchedule(
            self.schedule.initial_ratios, self.schedule.total, self.schedule.stages
        )
        schedule.load_state_dict(state["schedule"])
        adaptive_state = state.get("adaptive")
        if (adaptive_state is None) != (self.adaptive is None):
            raise ValueError("Incompatible adaptive configuration on resume")
        if self.adaptive is not None:
            # Validate on a copy before loading any learner/optimizer tensors.
            if not isinstance(adaptive_state, Mapping):
                raise ValueError("Missing adaptive checkpoint state")
            adaptive = deepcopy(self.adaptive)
            adaptive.load_state_dict(adaptive_state)
            if adaptive.last_iteration != next_iteration - 1:
                raise ValueError("Adaptive checkpoint iteration does not match learner")
        rng = state["rng"]
        numpy_rng = (
            rng["numpy"][0],
            np.asarray(rng["numpy"][1], dtype=np.uint32),
            rng["numpy"][2],
            rng["numpy"][3],
            rng["numpy"][4],
        )
        random.Random().setstate(rng["python"])
        np.random.RandomState().set_state(numpy_rng)
        torch.Generator().set_state(rng["torch"].cpu())
        if rng["cuda"] and len(rng["cuda"]) != torch.cuda.device_count():
            raise ValueError("Checkpoint CUDA RNG topology does not match the current learner")
        try:
            result: dict = super().load(path, load_cfg, strict, map_location)
            self.schedule.load_state_dict(state["schedule"])
            if self.adaptive is not None:
                assert isinstance(adaptive_state, Mapping)
                self.adaptive.load_state_dict(adaptive_state)
            self.next_iteration = next_iteration
            self.current_learning_iteration = next_iteration
            self.mixed_env.reconfigure(
                allocate_counts(self.schedule.effective_ratios, self.env.num_envs)
            )
            self.wrapped_env.reset()
            self._clear_episode_bookkeeping()
            random.setstate(rng["python"])
            np.random.set_state(numpy_rng)
            torch.set_rng_state(rng["torch"].cpu())
            if rng["cuda"]:
                torch.cuda.set_rng_state_all([value.cpu() for value in rng["cuda"]])
            self._write_manifest()
            self._event(
                event="resumed", iteration=next_iteration, previous_generation=state["generation"]
            )
            return result
        except BaseException:
            self.wrapped_env.close()
            raise

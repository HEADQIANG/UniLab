"""Task-owned configuration and process factories for mixed-backend PPO."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any, cast

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

MIXED_TASK_BACKENDS = {
    "go2_joystick_flat": ("mujoco", "motrix", "drake"),
    "g1_walk_flat": (
        "mujoco",
        "motrix",
        "drake",
        "mjwarp",
        "isaacgym",
        "isaacsim",
        "genesis",
        "newton",
    ),
}
CPU_BACKENDS = {"mujoco", "motrix", "drake"}
BACKEND_OPTIONS = {
    "mujoco": {"chunk_size", "adaptive_chunk_size", "post_step_forward_sensor"},
    "motrix": {"motrix_max_iterations"},
    "drake": {"drake_backend_mode", "drake_nthread"},
    "mjwarp": {"mjwarp_nconmax", "mjwarp_njmax"},
    "genesis": {
        "genesis_integrator",
        "genesis_constraint_solver",
        "genesis_friction_cone",
        "genesis_solver_iterations",
        "genesis_device_id",
    },
    "newton": {"newton_device", "newton_nconmax", "newton_njmax", "newton_capacity_check_steps"},
    "isaacgym": {"isaacgym_device_id", "isaacgym_worker_timeout_s"},
    "isaacsim": {
        "isaacsim_device_id",
        "isaacsim_worker_timeout_s",
        "isaacsim_render_mode",
        "isaacsim_render_width",
        "isaacsim_render_height",
    },
}
RUNTIME_FIELDS = set().union(*BACKEND_OPTIONS.values(), {"cpu_ids", "seed"})


def check_mix_task(task: str, ratios: Mapping[str, float]) -> None:
    supported = MIXED_TASK_BACKENDS.get(task)
    if supported is None:
        raise ValueError(
            f"Mixed PPO needs a common task profile; supported tasks: {tuple(MIXED_TASK_BACKENDS)}"
        )
    unsupported = set(ratios) - set(supported)
    if unsupported:
        raise ValueError(
            f"Task {task} has no mixed profile for {sorted(unsupported)}; supported: {supported}"
        )


def load_mix_config(path: str | None, names: tuple[str, ...], total: int) -> dict[str, Any]:
    from uni_rl.algos.adaptive_mix import validate_adaptive_config
    from uni_rl.ipc.mix_schedule import validate_stages

    data: Any = {} if not path else OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(data, dict) or set(data) - {
        "backends",
        "stages",
        "adaptive",
        "held_out_backends",
    }:
        raise ValueError("--mix-config accepts backends, stages, adaptive and held_out_backends")
    held_out = data.get("held_out_backends", [])
    if (
        not isinstance(held_out, list)
        or any(not isinstance(n, str) or n not in BACKEND_OPTIONS for n in held_out)
        or len(set(held_out)) != len(held_out)
    ):
        raise ValueError("held_out_backends must be distinct known backend names")
    if set(held_out) & set(names):
        raise ValueError("Held-out evaluation backends cannot contribute training samples")
    backends = data.get("backends", {})
    if not isinstance(backends, dict) or set(backends) - set(names):
        raise ValueError("mix-config backends must be declared by --sim-mix")
    for name, options in backends.items():
        if not isinstance(options, dict) or set(options) - {
            "device",
            "cpu_ids",
            "timeout_s",
            "options",
        }:
            raise ValueError(f"Invalid process settings for {name}")
        timeout = options.get("timeout_s", 120.0)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError(f"{name}.timeout_s must be positive and finite")
        ids = options.get("cpu_ids")
        if ids is not None and (
            not isinstance(ids, list)
            or not ids
            or any(type(i) is not int or i < 0 for i in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError(f"{name}.cpu_ids must be distinct nonnegative integers")
        runtime = options.get("options", {})
        if not isinstance(runtime, dict) or set(runtime) - BACKEND_OPTIONS[name]:
            raise ValueError(
                f"{name}.options may only set backend runtime fields, not task semantics"
            )
    stages = validate_stages(data.get("stages", []), names, total=total)
    adaptive = validate_adaptive_config(data.get("adaptive"), names, total)
    if adaptive is not None and stages:
        raise ValueError("Adaptive mixing cannot be combined with fixed ratio stages")
    return {
        "backends": backends,
        "stages": stages,
        "adaptive": adaptive,
        "held_out_backends": held_out,
    }


def compose_mixed_profile(task: str, backend: str) -> DictConfig:
    overrides = [f"task={task}/{backend}_mixed"]
    if GlobalHydra.instance().is_initialized():
        return compose(config_name="config", overrides=overrides)
    root = Path(__file__).resolve().parents[1] / "conf" / "ppo"
    with initialize_config_dir(version_base="1.3", config_dir=str(root)):
        return compose(config_name="config", overrides=overrides)


def ordered_contract(cfg: DictConfig) -> dict[str, Any]:
    """Keep mapping insertion order, including observation/action term order."""
    env = OmegaConf.to_container(cfg.env, resolve=True)
    assert isinstance(env, dict)
    env = {k: v for k, v in env.items() if k not in RUNTIME_FIELDS}
    algo_fields = (
        "obs_groups",
        "empirical_normalization",
        "obs_normalization",
        "policy",
        "actor",
        "critic",
    )
    return {
        "env": env,
        "reward": OmegaConf.to_container(cfg.reward, resolve=True),
        "algo": {
            key: OmegaConf.to_container(value, resolve=True)
            if OmegaConf.is_config(value)
            else value
            for key in algo_fields
            if (value := OmegaConf.select(cfg, f"algo.{key}")) is not None
        },
    }


def contract_fingerprint(cfg: DictConfig) -> str:
    payload = json.dumps(
        ordered_contract(cfg), ensure_ascii=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def make_mixed_task_env(
    task_name: str,
    backend: str,
    num_envs: int,
    env_cfg_override: Mapping[str, Any] | None = None,
    *,
    nan_guard: dict[str, Any] | None = None,
):
    """Spawn-safe owner factory; simulator imports stay inside the worker."""
    import numpy as np

    from unilab.base.env_factory import make_registry_env

    overrides = dict(env_cfg_override or {})
    overrides["seed"] = int(np.random.randint(0, 2**31 - 1))
    if backend == "mjwarp":
        from unilab.base.process_device import bind_backend_process_device

        bind_backend_process_device("cuda:0")
    env = make_registry_env(task_name, backend, num_envs, overrides)
    try:
        if nan_guard is not None:
            from unilab.training.run import apply_env_nan_guard

            apply_env_nan_guard(env, OmegaConf.create({"nan_guard": nan_guard}))
        return env
    except BaseException:
        env.close()
        raise


def mixed_env_metadata(env: Any) -> dict[str, Any]:
    """Compare resolved public manager/entity metadata, not backend internals."""
    import numpy as np

    from unilab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from unilab.envs.mdp.actions.actions import BaseAction

    if not isinstance(env, ManagerBasedRlEnv):
        raise TypeError("Mixed task profiles require ManagerBasedRlEnv")

    def row(value: Any) -> Any:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim > 1:
            if not np.allclose(array, array[0], rtol=0, atol=1e-6):
                raise ValueError("Mixed action defaults must be common to all environments")
            array = array[0]
        return np.round(array, 6).tolist()

    actions = []
    for name in env.action_manager.active_terms:
        term = env.action_manager.get_term(name)
        if not isinstance(term, BaseAction):
            raise TypeError(f"Mixed action {name} must expose BaseAction mapping metadata")
        actions.append((name, term.target_names, row(term.scale), row(term.offset)))

    return {
        "observations": [
            (group, list(terms), env.observation_manager.group_obs_term_dim[group])
            for group, terms in env.observation_manager.active_terms.items()
        ],
        "actions": actions,
        "entities": [
            (
                name,
                list(entity.joint_names),
                list(entity.actuator_names),
                row(entity.data.default_joint_pos),
            )
            for name, entity in env.scene.entities.items()
        ],
        "ctrl_dt": env.cfg.ctrl_dt,
        "max_episode_seconds": env.cfg.max_episode_seconds,
    }


def worker_device(
    backend: str, device: str, visible: str | None = None
) -> tuple[dict[str, str], dict[str, Any]]:
    if backend in CPU_BACKENDS:
        if device != "cpu":
            raise ValueError(
                f"{backend} physics requires device=cpu (learner device is independent)"
            )
        return {"CUDA_VISIBLE_DEVICES": ""}, {}
    if not isinstance(device, str) or not device.startswith("cuda:") or not device[5:].isdigit():
        raise ValueError(f"{backend} physics needs explicit device=cuda:N")
    index = int(device[5:])
    token = str(index)
    if visible is not None:
        tokens = [part.strip() for part in visible.split(",") if part.strip()]
        if index >= len(tokens):
            raise ValueError(f"{device} is outside CUDA_VISIBLE_DEVICES={visible!r}")
        token = tokens[index]
    field = {
        "genesis": "genesis_device_id",
        "isaacgym": "isaacgym_device_id",
        "isaacsim": "isaacsim_device_id",
        "newton": "newton_device",
    }.get(backend)
    overrides = {} if field is None else {field: "cuda:0" if backend == "newton" else 0}
    return {"CUDA_VISIBLE_DEVICES": token}, overrides


def _backend_worker_seed(training_seed: int, backend: str) -> int:
    """Name-stable uint32 seed; see docs/mixed_ppo.md for the versioned recipe."""
    namespace = f"unilab-mixed-worker-v1:{backend}".encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(namespace).digest()[:4], "big")
    return (training_seed + offset) % (2**32)


def build_worker_specs(
    cfg: DictConfig, ratios: Mapping[str, float], settings: dict[str, Any], learner_device: str
):
    from uni_rl.ipc.mixed_env import WorkerSpec

    from unilab.base.config_adapter import BackendAdapter

    task = str(cfg.training.mix_task)
    check_mix_task(task, ratios)
    common = BackendAdapter(cfg, root_dir=Path.cwd(), algo_name="ppo").build_task_env_cfg_override()
    if common.get("curriculum"):
        raise ValueError(
            "Mixed PPO v1 does not support stateful curricula across environment rebuilds"
        )
    enabled_events = {key for key, value in common.get("events", {}).items() if value is not None}
    if enabled_events != {"reset_scene_to_default"}:
        raise ValueError(
            "Mixed v1 requires only reset_scene_to_default; physical DR and random root resets are disabled"
        )
    nan_guard = OmegaConf.to_container(cfg.training.nan_guard, resolve=True)
    if not isinstance(nan_guard, dict) or any(not isinstance(key, str) for key in nan_guard):
        raise ValueError("training.nan_guard must be a mapping with string keys")
    # No backend-specific changes to policy I/O, commands, rewards, or reset semantics.
    specs = []
    for name in ratios:
        profile = compose_mixed_profile(task, name)
        runtime = {
            key: OmegaConf.to_container(value, resolve=True)
            if OmegaConf.is_config(value)
            else value
            for key in BACKEND_OPTIONS[name]
            if (value := OmegaConf.select(profile, f"env.{key}")) is not None
        }
        options = settings["backends"].get(name, {})
        runtime.update(options.get("options", {}))
        device = options.get(
            "device",
            "cpu"
            if name in CPU_BACKENDS
            else (learner_device if learner_device.startswith("cuda:") else "cuda:0"),
        )
        process_env, binding = worker_device(name, device, os.environ.get("CUDA_VISIBLE_DEVICES"))
        runtime.update(binding)
        timeout = float(options.get("timeout_s", 120.0))
        if name in {"isaacgym", "isaacsim"}:
            runtime[f"{name}_worker_timeout_s"] = timeout
        cpu_ids = options.get("cpu_ids")
        if cpu_ids:
            runtime["cpu_ids"] = cpu_ids
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMBA_NUM_THREADS",
            ):
                process_env[key] = str(len(cpu_ids))
        specs.append(
            WorkerSpec(
                name=name,
                factory=partial(
                    make_mixed_task_env,
                    str(cfg.training.task_name),
                    name,
                    nan_guard=cast(dict[str, Any], nan_guard),
                ),
                env_cfg_override={**common, **runtime},
                process_env=process_env,
                cpu_ids=tuple(cpu_ids) if cpu_ids else None,
                timeout_s=timeout,
                seed=_backend_worker_seed(int(cfg.algo.seed), name),
                metadata_factory=mixed_env_metadata,
            )
        )
    return specs

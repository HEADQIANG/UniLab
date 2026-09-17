"""Mixed workers and evaluation must share an ordered, backend-neutral task."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

CONF_DIR = Path(__file__).parents[2] / "src" / "unilab" / "conf" / "ppo"
TASK_BACKENDS = {
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
RUNTIME_FIELDS = {
    "mujoco": (),
    "motrix": (),
    "drake": ("drake_backend_mode", "drake_nthread"),
    "mjwarp": ("mjwarp_nconmax", "mjwarp_njmax"),
    "isaacgym": ("isaacgym_device_id",),
    "isaacsim": (
        "isaacsim_device_id",
        "isaacsim_worker_timeout_s",
        "isaacsim_render_width",
        "isaacsim_render_height",
    ),
    "genesis": ("genesis_device_id", "genesis_integrator"),
    "newton": (
        "newton_device",
        "newton_nconmax",
        "newton_njmax",
        "newton_capacity_check_steps",
    ),
}
PROFILE_CASES = [
    (task, backend) for task, backends in TASK_BACKENDS.items() for backend in backends
]


@lru_cache(maxsize=None)
def _compose(task: str, profile: str) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose("config", overrides=[f"task={task}/{profile}"])


def _ordered(value):
    """Dict equality alone misses reordered observation or actuator declarations."""
    if isinstance(value, Mapping):
        return tuple((key, _ordered(item)) for key, item in value.items())
    if isinstance(value, list):
        return tuple(_ordered(item) for item in value)
    return value


def _value(cfg: DictConfig, path: str):
    value = OmegaConf.select(cfg, path)
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    return _ordered(value)


@pytest.mark.parametrize("task,backend", PROFILE_CASES)
def test_mixed_profile_preserves_complete_ordered_task(task, backend):
    baseline = _compose(task, "mixed")
    profile = _compose(task, f"{backend}_mixed")
    runtime_fields = set(RUNTIME_FIELDS[backend])
    baseline_env = OmegaConf.to_container(baseline.env, resolve=True)
    profile_env = OmegaConf.to_container(profile.env, resolve=True)
    assert isinstance(baseline_env, dict) and isinstance(profile_env, dict)
    for key in runtime_fields:
        profile_env.pop(key, None)

    assert _ordered(profile_env) == _ordered(baseline_env)
    assert _value(profile, "reward") == _value(baseline, "reward")
    for path in (
        "algo.policy",
        "algo.algorithm",
        "algo.obs_groups",
        "algo.empirical_normalization",
        "algo.num_steps_per_env",
        "training.task_name",
    ):
        assert _value(profile, path) == _value(baseline, path), path
    assert profile.training.sim_backend == backend
    assert profile.training.no_play is False
    assert profile.training.play_render_mode == ("record" if backend == "mjwarp" else "auto")
    assert profile.algo.runtime_resolver is None
    assert OmegaConf.select(profile, "algo.runtime_impl") is None


@pytest.mark.parametrize("task,backend", PROFILE_CASES)
def test_mixed_profile_keeps_only_original_backend_runtime_settings(task, backend):
    original = _compose(task, backend)
    profile = _compose(task, f"{backend}_mixed")
    for field in RUNTIME_FIELDS[backend]:
        assert _value(profile, f"env.{field}") == _value(original, f"env.{field}")
    if backend == "drake":
        assert profile.training.device == "cpu"
    if backend in {"isaacsim", "newton"}:
        assert profile.play_profile.enabled is False


@pytest.mark.parametrize("task", TASK_BACKENDS)
def test_mixed_baseline_only_disables_declared_physical_terms(task):
    original = _compose(task, "mujoco")
    mixed = _compose(task, "mixed")
    original_env = OmegaConf.to_container(original.env, resolve=True)
    assert isinstance(original_env, dict)
    original_env["events"]["reset_root_state_uniform"] = None
    original_env["events"]["pd_gains"] = None
    assert _ordered(original_env) == _value(mixed, "env")
    assert [key for key, term in mixed.env.events.items() if term is not None] == [
        "reset_scene_to_default"
    ]
    assert OmegaConf.select(mixed, "env.domain_rand") is None

    original_reward = OmegaConf.to_container(original.reward, resolve=True)
    assert isinstance(original_reward, dict)
    disabled_reward = "contact" if task == "go2_joystick_flat" else "feet_air_time"
    original_reward[disabled_reward] = None
    assert _ordered(original_reward) == _value(mixed, "reward")
    assert mixed.training.sim_backend == "mixed"
    assert mixed.training.no_play is True
    assert mixed.training.play_render_mode == "none"
    assert mixed.training.mix_task == task
    assert mixed.training.mix_ratios is None
    assert mixed.training.mix_config is None
    assert mixed.algo.runtime_resolver == "uni_rl.algos.mixed_ppo:resolve_mixed_ppo_runtime"


def test_mixed_profile_backend_union_and_no_unsupported_task_pairings():
    assert {backend for backends in TASK_BACKENDS.values() for backend in backends} == {
        "mujoco",
        "motrix",
        "drake",
        "mjwarp",
        "isaacgym",
        "isaacsim",
        "genesis",
        "newton",
    }
    for task, backends in TASK_BACKENDS.items():
        profiles = {path.stem for path in (CONF_DIR / "task" / task).glob("*_mixed.yaml")}
        assert profiles == {f"{backend}_mixed" for backend in backends}


def test_original_single_backend_tuning_is_not_replaced():
    go2 = _compose("go2_joystick_flat", "motrix")
    assert go2.env.commands.twist.ranges.lin_vel_x == [0.5, 0.5]
    assert go2.reward.contact is not None
    g1 = _compose("g1_walk_flat", "motrix")
    assert g1.env.actions.joint_pos.scale == 0.5
    assert g1.algo.empirical_normalization is True
    assert g1.env.terminations.tilt.params.max_tilt_deg == 35.0
    assert g1.reward.feet_air_time is not None
    for task in TASK_BACKENDS:
        mujoco = _compose(task, "mujoco")
        assert mujoco.env.events.pd_gains is not None
        assert mujoco.env.events.reset_root_state_uniform is not None
        assert OmegaConf.select(mujoco, "algo.runtime_resolver") is None

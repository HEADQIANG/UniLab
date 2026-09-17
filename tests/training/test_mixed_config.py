import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from unilab.training.mixed_config import (
    MIXED_TASK_BACKENDS,
    build_worker_specs,
    compose_mixed_profile,
    contract_fingerprint,
    load_mix_config,
    mixed_env_metadata,
    worker_device,
)


def test_cpu_and_remapped_gpu_devices():
    assert worker_device("mujoco", "cpu")[0] == {"CUDA_VISIBLE_DEVICES": ""}
    env, override = worker_device("genesis", "cuda:1", "4,5")
    assert env == {"CUDA_VISIBLE_DEVICES": "5"}
    assert override == {"genesis_device_id": 0}
    assert worker_device("newton", "cuda:0", "GPU-uuid")[1] == {"newton_device": "cuda:0"}


@pytest.mark.parametrize(
    "backend,device,visible",
    [
        ("motrix", "cuda:0", None),
        ("newton", "cpu", None),
        ("genesis", "cuda:1", "0"),
        ("genesis", "cuda:0", ""),
    ],
)
def test_bad_device_rejected(backend, device, visible):
    with pytest.raises(ValueError):
        worker_device(backend, device, visible)


@pytest.mark.parametrize(
    "payload",
    [
        {"ratios": {}},
        {"backends": {"newton": {}}},
        {"backends": {"mujoco": {"timeout_s": 0}}},
        {"backends": {"mujoco": {"cpu_ids": [1, 1]}}},
        {"backends": {"mujoco": {"options": {"ctrl_dt": 0.1}}}},
        {"stages": [{"iteration": 1, "ratios": {"mujoco": 1}}]},
    ],
)
def test_invalid_file_rejected(tmp_path, payload):
    path = tmp_path / "mix.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_mix_config(str(path), ("mujoco", "motrix"), 10)


def test_compatible_fingerprints():
    first = compose_mixed_profile("g1_walk_flat", "mujoco")
    second = compose_mixed_profile("g1_walk_flat", "motrix")
    assert contract_fingerprint(first) == contract_fingerprint(second)
    second.env.actions.joint_pos.scale = 0.5
    assert contract_fingerprint(first) != contract_fingerprint(second)


def test_fingerprint_detects_observation_order():
    cfg = compose_mixed_profile("go2_joystick_flat", "mujoco")
    original = contract_fingerprint(cfg)
    terms = OmegaConf.to_container(cfg.env.observations.policy.terms, resolve=True)
    cfg.env.observations.policy.terms = dict(reversed(list(terms.items())))
    assert contract_fingerprint(cfg) != original


def test_factory_specs_do_not_inherit_original_motrix_tuning():
    cfg = compose_mixed_profile("g1_walk_flat", "mujoco")
    specs = build_worker_specs(
        cfg, {"mujoco": 1, "motrix": 1}, {"backends": {}, "stages": []}, "cpu"
    )
    assert len(specs) == 2
    assert specs[0].env_cfg_override["actions"] == specs[1].env_cfg_override["actions"]
    assert specs[1].env_cfg_override["actions"]["joint_pos"]["scale"] == 0.25
    assert specs[0].env_cfg_override["rewards"] == specs[1].env_cfg_override["rewards"]


def test_worker_seeds_match_across_single_mixed_and_reordered_backends(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    cfg = compose_mixed_profile("g1_walk_flat", "mujoco")
    cfg.algo.seed = 11
    names = MIXED_TASK_BACKENDS["g1_walk_flat"]
    settings = {"backends": {}, "stages": []}
    mixed = {
        spec.name: spec.seed
        for spec in build_worker_specs(cfg, dict.fromkeys(names, 1.0), settings, "cpu")
    }
    reordered = {
        spec.name: spec.seed
        for spec in build_worker_specs(cfg, dict.fromkeys(reversed(names), 1.0), settings, "cpu")
    }
    singles = {
        name: build_worker_specs(cfg, {name: 1.0}, settings, "cpu")[0].seed for name in names
    }
    assert mixed == reordered == singles
    assert len(set(mixed.values())) == len(names)
    assert all(type(seed) is int and 0 <= seed < 2**32 for seed in mixed.values())


def test_worker_seeds_change_with_training_seed_and_remain_uint32():
    cfg = compose_mixed_profile("g1_walk_flat", "mujoco")
    settings = {"backends": {}, "stages": []}
    cfg.algo.seed = 1
    first = build_worker_specs(cfg, {"motrix": 1.0, "drake": 1.0}, settings, "cpu")
    cfg.algo.seed = 2**32 - 1
    last = build_worker_specs(cfg, {"motrix": 1.0, "drake": 1.0}, settings, "cpu")
    for before, after in zip(first, last, strict=True):
        assert before.seed != after.seed
        assert 0 <= before.seed < 2**32
        assert 0 <= after.seed < 2**32


def test_factory_rejects_reenabled_domain_randomization():
    cfg = compose_mixed_profile("go2_joystick_flat", "mujoco")
    cfg.env.events.pd_gains = {"func": "unilab.envs.mdp.pd_gains", "mode": "reset"}
    with pytest.raises(ValueError, match="physical DR"):
        build_worker_specs(cfg, {"mujoco": 1}, {"backends": {}, "stages": []}, "cpu")


def test_explicit_resume_does_not_resolve_latest(monkeypatch, tmp_path):
    from unilab.training.mixed_ppo import resolve_mixed_resume

    checkpoint = tmp_path / "model_10.pt"
    checkpoint.write_bytes(b"test")
    cfg = compose_mixed_profile("go2_joystick_flat", "mujoco")
    cfg.algo.resume_path = str(checkpoint)
    monkeypatch.setattr(
        "unilab.training.parse_checkpoint_path",
        lambda *args, **kwargs: pytest.fail("Must honor explicit path"),
    )
    assert resolve_mixed_resume(cfg) == checkpoint


def test_missing_explicit_resume_fails_without_fallback(tmp_path):
    from unilab.training.mixed_ppo import resolve_mixed_resume

    cfg = compose_mixed_profile("go2_joystick_flat", "mujoco")
    cfg.algo.resume_path = str(tmp_path / "missing.pt")
    with pytest.raises(ValueError, match="Explicit mixed checkpoint not found"):
        resolve_mixed_resume(cfg)


def test_metadata_rejects_actions_without_mapping_contract():
    from unilab.envs.manager_based_rl_env import ManagerBasedRlEnv

    env = object.__new__(ManagerBasedRlEnv)
    env.action_manager = SimpleNamespace(active_terms=["custom"], get_term=lambda name: object())
    with pytest.raises(TypeError, match="BaseAction mapping metadata"):
        mixed_env_metadata(env)


def test_holdout_prevents_mujoco_training_even_at_zero_weight(tmp_path):
    path = tmp_path / "holdout.yaml"
    path.write_text("held_out_backends: [mujoco]\n")
    with pytest.raises(ValueError, match="Held-out"):
        load_mix_config(str(path), ("mujoco", "motrix"), 32)
    assert load_mix_config(str(path), ("motrix", "newton"), 32)["held_out_backends"] == ["mujoco"]


def test_adaptive_config_composes_and_rejects_conflicting_stages(tmp_path):
    path = tmp_path / "adaptive.json"
    data = {"held_out_backends": ["mujoco"], "adaptive": {"max_ratio": 0.8}}
    path.write_text(json.dumps(data))
    assert (
        load_mix_config(str(path), ("motrix", "newton"), 32)["adaptive"]["metric"]
        == "mean_step_reward"
    )
    data["stages"] = [{"iteration": 2, "ratios": {"motrix": 1, "newton": 1}}]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="combined"):
        load_mix_config(str(path), ("motrix", "newton"), 32)

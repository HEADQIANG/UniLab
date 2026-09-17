"""Mixed routing is explicit and cannot bypass task/algorithm contracts."""

import json

import pytest

from unilab import cli


def command(monkeypatch, **kwargs):
    checked = []
    monkeypatch.setattr(cli, "_check_runtime_requirements", lambda algo, sim: checked.append(sim))
    result = cli.build_command(
        mode="train",
        algo="ppo",
        task="go2_joystick_flat",
        sim=None,
        overrides=[],
        sim_mix="mujoco=0.7,motrix=0.3",
        **kwargs,
    )
    return result, checked


def test_mix_routes_to_common_owner(monkeypatch):
    result, checked = command(monkeypatch)
    assert "task=go2_joystick_flat/mixed" in result
    assert "training.mix_task=go2_joystick_flat" in result
    assert checked == ["mujoco", "motrix"]
    raw = next(v.split("=", 1)[1] for v in result if v.startswith("training.mix_ratios="))
    assert json.loads(raw) == "mujoco=0.7,motrix=0.3"


@pytest.mark.parametrize(
    "extra",
    [
        {"sim": "mujoco"},
        {"algo": "sac"},
        {"mode": "eval"},
        {"profile": "hora"},
        {"task": "go1_joystick_flat"},
        {"sim_mix": "mujoco=1,genesis=1"},
        {"sim_mix": "mujoco=1,mujoco=2"},
        {"sim_mix": "mujoco=nan"},
        {"overrides": ["algo.runtime_resolver=custom"]},
    ],
)
def test_reject_invalid_mix(monkeypatch, extra):
    monkeypatch.setattr(cli, "_check_runtime_requirements", lambda *args: None)
    args = dict(
        mode="train",
        algo="ppo",
        task="go2_joystick_flat",
        sim=None,
        overrides=[],
        sim_mix="mujoco=1,motrix=1",
    )
    args.update(extra)
    with pytest.raises(SystemExit):
        cli.build_command(**args)


def test_parser_requires_one_simulation_mode():
    parser = cli._train_eval_parser(mode="train")
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--algo",
                "ppo",
                "--task",
                "go2_joystick_flat",
                "--sim",
                "mujoco",
                "--sim-mix",
                "mujoco=1",
            ]
        )


def test_single_backend_routing_unchanged(monkeypatch):
    monkeypatch.setattr(cli, "_check_runtime_requirements", lambda *args: None)
    result = cli.build_command(
        mode="train", algo="ppo", task="go2_joystick_flat", sim="mujoco", overrides=[]
    )
    assert "task=go2_joystick_flat/mujoco" in result
    assert not any("mix_" in v for v in result)


def test_config_path_quoted(monkeypatch, tmp_path):
    path = tmp_path / "two words.yaml"
    path.write_text("stages: []\n")
    result, _ = command(monkeypatch, mix_config=str(path))
    assert f"training.mix_config={json.dumps(str(path))}" in result


@pytest.mark.parametrize(
    "filename",
    [
        "\u6587\u6863 \u6bd4\u4f8b.yaml",
        "quote\"double and 'single'.yaml",
        "back\\slash.yaml",
        '\u6bd4\u4f8b,= [stages] "quoted" \\ end.yaml',
        r"before\"quote.yaml",
        "ending\\",
    ],
)
def test_mix_config_path_roundtrips_hydra_argv(monkeypatch, tmp_path, filename):
    from hydra import compose, initialize_config_dir

    from unilab.training.mixed_config import load_mix_config

    directory = tmp_path / "\u5b9e\u9a8c\u76ee\u5f55"
    directory.mkdir()
    path = directory / filename
    path.write_text("stages: []\n", encoding="utf-8")
    result, _ = command(monkeypatch, mix_config=str(path))
    with initialize_config_dir(
        config_dir=str(cli.package_root() / "conf" / "ppo"), version_base="1.3"
    ):
        cfg = compose("config", overrides=result[2:])
    assert cfg.training.mix_config == str(path.resolve())
    assert cfg.training.mix_ratios == "mujoco=0.7,motrix=0.3"
    assert load_mix_config(cfg.training.mix_config, ("mujoco", "motrix"), 10) == {
        "backends": {},
        "stages": [],
        "adaptive": None,
        "held_out_backends": [],
    }


def test_eval_common_profile(monkeypatch):
    monkeypatch.setattr(cli, "_check_runtime_requirements", lambda *args: None)
    result = cli.build_command(
        mode="eval", algo="ppo", task="g1_walk_flat", sim="motrix", profile="mixed", overrides=[]
    )
    assert "task=g1_walk_flat/motrix_mixed" in result
    assert "training.play_only=true" in result


@pytest.mark.parametrize(
    "task,backend",
    [
        ("go2_joystick_flat", "mujoco"),
        ("go2_joystick_flat", "motrix"),
        ("go2_joystick_flat", "drake"),
        ("g1_walk_flat", "mujoco"),
        ("g1_walk_flat", "motrix"),
        ("g1_walk_flat", "drake"),
        ("g1_walk_flat", "mjwarp"),
        ("g1_walk_flat", "isaacgym"),
        ("g1_walk_flat", "isaacsim"),
        ("g1_walk_flat", "genesis"),
        ("g1_walk_flat", "newton"),
    ],
)
def test_eval_mixed_legal_profiles_preserve_explicit_checkpoint(monkeypatch, task, backend):
    monkeypatch.setattr(cli, "_check_runtime_requirements", lambda *args: None)
    checkpoint = "algo.load_run=/tmp/mixed-checkpoint/model_10.pt"
    result = cli.build_command(
        mode="eval",
        algo="ppo",
        task=task,
        sim=backend,
        profile="mixed",
        render_mode="record",
        overrides=[checkpoint],
    )
    assert f"task={task}/{backend}_mixed" in result
    assert checkpoint in result
    assert "training.play_only=true" in result
    assert "training.play_render_mode=record" in result
    assert not any(value.startswith("training.sim_backend=") for value in result)


@pytest.mark.parametrize(
    "algo,task,backend",
    [
        ("ppo", "go2_joystick_flat", "genesis"),
        ("ppo", "go1_joystick_flat", "mujoco"),
        ("sac", "g1_walk_flat", "mujoco"),
    ],
)
def test_eval_mixed_rejects_unsupported_pair_before_dependency_checks(
    monkeypatch, algo, task, backend
):
    def unexpected_dependency_check(*args):
        pytest.fail("Unsupported mixed profiles must not probe backend dependencies")

    monkeypatch.setattr(cli, "_check_runtime_requirements", unexpected_dependency_check)
    with pytest.raises(SystemExit, match="mixed|Mixed"):
        cli.build_command(
            mode="eval", algo=algo, task=task, sim=backend, profile="mixed", overrides=[]
        )


def test_eval_mixed_missing_owner_does_not_fallback(monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(cli, "_check_runtime_requirements", lambda *args: None)
    real_is_file = Path.is_file
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda path: False if path.name == "motrix_mixed.yaml" else real_is_file(path),
    )
    with pytest.raises(SystemExit, match="No owner config exists"):
        cli.build_command(
            mode="eval",
            algo="ppo",
            task="g1_walk_flat",
            sim="motrix",
            profile="mixed",
            overrides=[],
        )


def test_eval_mixed_interactive_selects_common_owner(monkeypatch):
    monkeypatch.setattr(cli, "_check_runtime_requirements", lambda *args: None)
    monkeypatch.setattr(cli, "find_spec", lambda name: object())
    result = cli.build_command(
        mode="eval",
        algo="ppo",
        task="go2_joystick_flat",
        sim="mujoco",
        profile="mixed",
        render_mode="interactive",
        overrides=[],
    )
    assert result[1].endswith("play_interactive.py")
    assert result[result.index("--sim") + 1] == "mujoco_mixed"
    assert "interactive.action_mode=policy" in result

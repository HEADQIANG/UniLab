from __future__ import annotations

import json

import pytest

from uni_rl.ipc.mix_schedule import (
    MixSchedule,
    allocate_counts,
    atomic_write_control,
    parse_mix,
    validate_stages,
)


def test_allocation_weights_ties_zero_and_large_numbers():
    assert allocate_counts(parse_mix("mujoco=0.7,motrix=0.3"), 1000) == {
        "mujoco": 700,
        "motrix": 300,
    }
    assert allocate_counts({"a": 1, "b": 1, "c": 1}, 5) == {"a": 2, "b": 2, "c": 1}
    assert allocate_counts({"a": 0, "b": 10}, 5) == {"a": 0, "b": 5}
    assert allocate_counts({"a": 1e308, "b": 1e308}, 4) == {"a": 2, "b": 2}


@pytest.mark.parametrize("text", ["", "a=1,a=2", "a=nan", "a=inf", "a=-1", "a=0", "a"])
def test_invalid_cli(text):
    with pytest.raises(ValueError):
        parse_mix(text)


@pytest.mark.parametrize("ratios,total", [({"a": 1, "b": 1}, 1), ({"a": True}, 2), ({"a": 1}, 0)])
def test_allocation_rejects_empty_or_starved(ratios, total):
    with pytest.raises(ValueError):
        allocate_counts(ratios, total)


@pytest.mark.parametrize("iterations", [[1, 1], [2, 1], [-1], [True], [1.0]])
def test_stage_validation(iterations):
    with pytest.raises(ValueError):
        validate_stages([{"iteration": it, "ratios": {"a": 1}} for it in iterations], ["a"])


def test_stage_manual_persistence_clear_and_resume(tmp_path):
    stages = [{"iteration": 2, "ratios": {"a": 1, "b": 3}}]
    schedule = MixSchedule({"a": 3, "b": 1}, 8, stages, tmp_path / "control.json")
    assert schedule.resolve(0) == {"a": 6, "b": 2}
    atomic_write_control(tmp_path, {"a": 1, "b": 1})
    assert schedule.resolve(1) == {"a": 4, "b": 4}
    assert schedule.resolve(2) == {"a": 4, "b": 4}
    state = schedule.state_dict()
    resumed = MixSchedule({"a": 3, "b": 1}, 8, stages, tmp_path / "control.json")
    resumed.load_state_dict(state)
    assert resumed.resolve(3) == {"a": 4, "b": 4}
    atomic_write_control(tmp_path, clear=True)
    assert resumed.resolve(4) == {"a": 2, "b": 6}
    (tmp_path / "control.json").write_text(json.dumps({"version": 1, "ratios": {"a": 1, "b": 1}}))
    assert resumed.resolve(5) == {"a": 2, "b": 6}


def test_invalid_hot_update_preserves_ratio_and_logs_once(tmp_path):
    events = []
    schedule = MixSchedule(
        {"a": 1, "b": 1}, 4, control_file=tmp_path / "control.json", event_callback=events.append
    )
    atomic_write_control(tmp_path, {"a": 1, "other": 1})
    assert schedule.resolve(1) == {"a": 2, "b": 2}
    assert schedule.resolve(2) == {"a": 2, "b": 2}
    assert [event["event"] for event in events] == ["control_rejected"]
    (tmp_path / "control.json").write_text("{broken")
    schedule.resolve(3)
    schedule.resolve(4)
    assert len(events) == 2


def test_unbounded_json_integer_is_rejected_without_stopping_training(tmp_path):
    schedule = MixSchedule({"a": 1}, 4, control_file=tmp_path / "control.json")
    (tmp_path / "control.json").write_text(json.dumps({"version": 1, "ratios": {"a": 10**400}}))
    assert schedule.resolve(0) == {"a": 4}
    assert schedule.manual_ratios is None


def test_non_utf8_control_file_is_rejected_without_stopping_training(tmp_path):
    events = []
    schedule = MixSchedule(
        {"a": 1}, 4, control_file=tmp_path / "control.json", event_callback=events.append
    )
    (tmp_path / "control.json").write_bytes(b"\xff\xfe")
    assert schedule.resolve(0) == {"a": 4}
    assert schedule.resolve(1) == {"a": 4}
    assert [event["event"] for event in events] == ["control_rejected"]
    atomic_write_control(tmp_path, {"a": 2})
    assert schedule.resolve(2) == {"a": 4}
    assert schedule.manual_ratios == {"a": 2}


def test_excessively_nested_json_can_be_replaced_without_stopping_training(tmp_path):
    schedule = MixSchedule({"a": 1}, 4, control_file=tmp_path / "control.json")
    (tmp_path / "control.json").write_text("[" * 2000 + "]" * 2000)
    assert schedule.resolve(0) == {"a": 4}
    atomic_write_control(tmp_path, {"a": 2})
    assert schedule.resolve(1) == {"a": 4}
    assert schedule.manual_ratios == {"a": 2}


def test_checkpoint_rejects_configuration_changes():
    state = MixSchedule({"a": 1}, 4).state_dict()
    with pytest.raises(ValueError, match="total"):
        MixSchedule({"a": 1}, 8).load_state_dict(state)


def test_atomic_control_uses_manifest_and_validates_total(tmp_path):
    alternate = tmp_path / "requests.json"
    (tmp_path / "mix_manifest.json").write_text(
        json.dumps(
            {
                "control_file": str(alternate),
                "backend_names": ["a", "b"],
                "total": 4,
                "seen_version": 10**20,
            }
        )
    )
    assert atomic_write_control(tmp_path, {"a": 3, "b": 1}) == alternate
    assert json.loads(alternate.read_text())["version"] > 10**20
    with pytest.raises(ValueError, match="exactly"):
        atomic_write_control(tmp_path, {"a": 1})

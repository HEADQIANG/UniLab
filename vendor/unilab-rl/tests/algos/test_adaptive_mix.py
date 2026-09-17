import json
from copy import deepcopy

import pytest

from uni_rl.algos.adaptive_mix import AdaptiveMixController, validate_adaptive_config
from uni_rl.ipc.mix_schedule import allocate_counts


def controller(**overrides):
    return AdaptiveMixController(
        {
            "interval": 2,
            "warmup": 2,
            "freeze_after": 20,
            "max_ratio": 0.8,
            "min_ratio": 0.2,
            "smoothing": 1.0,
            **overrides,
        },
        {"a": 1, "b": 1},
        20,
    )


def metrics(a, b, weights=(10, 10)):
    return {
        name: {"mean_step_reward": value, "samples": weight}
        for name, value, weight in zip(("a", "b"), (a, b), weights, strict=True)
    }


def test_low_reward_preserves_samples_and_keeps_all_backends():
    c = controller(strategy="low_reward")
    assert c.observe(0, metrics(1, 10)) is None
    event = c.observe(1, metrics(1, 10))
    assert event["applies_at_iteration"] == 2
    assert c.ratios["a"] == pytest.approx(0.8)
    assert c.ratios["b"] == pytest.approx(0.2)
    assert allocate_counts(c.ratios, 20) == {"a": 16, "b": 4}


def test_learning_progress_uses_change_not_reward_magnitude():
    c = controller()
    c.observe(0, metrics(10, 100))
    c.observe(1, metrics(10, 100))
    assert c.ratios == pytest.approx({"a": 0.5, "b": 0.5})
    c.observe(2, metrics(20, 100))
    event = c.observe(3, metrics(20, 100))
    assert c.ratios["a"] > c.ratios["b"]
    assert event["window_means"] == {"a": 20, "b": 100}


def test_window_is_weighted_by_actual_samples():
    c = controller()
    c.observe(0, metrics(1, 3, (2, 8)))
    event = c.observe(1, metrics(5, 9, (6, 2)))
    assert event["window_means"] == pytest.approx({"a": 4, "b": 4.2})


def test_episode_metric_waits_for_evidence_from_every_backend():
    c = controller(metric="mean_episode_length", strategy="low_reward")
    rows = {
        "a": {"mean_episode_length": 10, "completed_episodes": 1},
        "b": {"completed_episodes": 0},
    }
    c.observe(0, rows)
    assert c.observe(1, rows)["event"] == "adaptive_skipped"
    assert c.ratios == {"a": 0.5, "b": 0.5}


def test_manual_pause_discards_partial_window_and_does_not_bias_update():
    c = controller(strategy="low_reward")
    c.observe(0, metrics(0, 100))
    c.observe(1, metrics(0, 100), paused=True)
    assert c.window_iterations == 0
    assert c.observe(2, metrics(10, 10)) is None
    c.observe(3, metrics(10, 10))
    assert c.ratios == pytest.approx({"a": 0.5, "b": 0.5})


def test_nonfinite_or_missing_metric_holds_ratio_and_logs_reason():
    c = controller()
    original = dict(c.ratios)
    assert c.observe(0, metrics(float("nan"), 1))["event"] == "adaptive_skipped"
    assert c.observe(1, {}) is None
    assert c.observe(2, {})["event"] == "adaptive_skipped"
    assert c.ratios == original


def test_freeze_stops_changes_at_absolute_iteration_even_after_resume():
    c = controller(strategy="low_reward", freeze_after=4)
    for it in range(3):
        c.observe(it, metrics(1, 10))
    state = json.loads(json.dumps(c.state_dict()))
    restored = controller(strategy="low_reward", freeze_after=4)
    restored.load_state_dict(state)
    assert restored.observe(3, metrics(100, 0))["event"] == "adaptive_frozen"
    frozen = dict(restored.ratios)
    for it in range(4, 10):
        assert restored.observe(it, metrics(100, 0)) is None
    assert restored.ratios == frozen


def test_checkpoint_restores_partial_windows_exactly():
    c = controller()
    for it in range(3):
        c.observe(it, metrics(it, 7))
    other = controller()
    other.load_state_dict(c.state_dict())
    for it in range(3, 9):
        assert c.observe(it, metrics(it, 7)) == other.observe(it, metrics(it, 7))
        assert c.state_dict() == other.state_dict()


def test_invalid_checkpoint_does_not_mutate_controller():
    c = controller()
    initial = c.state_dict()
    bad = deepcopy(initial)
    bad["ratios"] = {"a": 1, "b": 0}
    with pytest.raises(ValueError, match="bounds"):
        c.load_state_dict(bad)
    assert c.state_dict() == initial


def test_checkpoint_rejects_unnormalized_ratios_before_smoothing():
    c = controller(smoothing=0.3)
    state = c.state_dict()
    state["ratios"] = {"a": 5, "b": 5}
    with pytest.raises(ValueError, match="sum to one"):
        c.load_state_dict(state)
    assert sum(c.ratios.values()) == pytest.approx(1)


@pytest.mark.parametrize(
    "overrides",
    [
        {"interval": 0},
        {"warmup": -1},
        {"freeze_after": 2},
        {"ema_alpha": 2},
        {"temperature": float("nan")},
        {"smoothing": 0},
        {"min_ratio": 0.6},
        {"max_ratio": 0.4},
        {"strategy": "best"},
        {"metric": "total_reward"},
        {"unknown": True},
        {"metric": []},
    ],
)
def test_invalid_config_rejected(overrides):
    with pytest.raises(ValueError):
        controller(**overrides)


def test_all_seven_engines_keep_positive_counts_and_bounded_ratios():
    names = {f"engine{i}": 1 for i in range(7)}
    c = AdaptiveMixController(
        {"interval": 1, "warmup": 0, "freeze_after": 100, "strategy": "low_reward"}, names, 32
    )
    for it in range(90):
        rows = {
            name: {"samples": 10, "mean_step_reward": (i + it) % 7} for i, name in enumerate(names)
        }
        c.observe(it, rows)
        counts = allocate_counts(c.ratios, 32)
        assert sum(counts.values()) == 32
        assert min(counts.values()) >= 1
        assert sum(c.ratios.values()) == pytest.approx(1)
        assert all(0.05 - 1e-12 <= v <= 0.5 + 1e-12 for v in c.ratios.values())


def test_infeasible_counts_and_initial_starvation_are_rejected():
    with pytest.raises(ValueError):
        validate_adaptive_config({}, ("a", "b", "c"), 2)
    with pytest.raises(ValueError, match="Initial adaptive ratios"):
        AdaptiveMixController({"max_ratio": 0.8}, {"a": 1, "b": 0}, 20)

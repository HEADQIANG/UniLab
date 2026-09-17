"""The experiment protocol is testable without optional simulator dependencies."""

import argparse
import importlib.util
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_FILE = Path(__file__).resolve().parents[2] / "src/unilab/training/mix_benchmark.py"
_SPEC = importlib.util.spec_from_file_location("mix_benchmark_under_test", _FILE)
assert _SPEC is not None and _SPEC.loader is not None
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


def args(**kwargs):
    data = dict(
        backends=",".join(benchmark.DEFAULT_BACKENDS),
        task="g1_walk_flat",
        seeds="11,22,33",
        validation_seeds="1001,1002",
        test_seeds="2001,2002",
        confirmation_seeds="101,102,103",
        num_envs=28,
        rollout_steps=4,
        iterations=2200,
        episodes=2,
        override=[],
        mix_config=None,
        device="cpu",
        fixed=[],
    )
    data.update(kwargs)
    return argparse.Namespace(**data)


def test_default_study_includes_every_single_and_all_positive_mixtures():
    study = benchmark.plan_study(args())
    assert study["full_backend_study"]
    assert study["transitions"] == 246400
    assert set(study["backends"]) == set(benchmark.DEFAULT_BACKENDS)
    singles = [c for c in study["candidates"] if c["name"].startswith("single_")]
    assert len(singles) == 7
    for candidate in study["candidates"]:
        assert "mujoco" not in candidate["ratios"]
        assert sum(candidate["ratios"].values()) == pytest.approx(1)
        if candidate not in singles:
            assert set(candidate["ratios"]) == set(benchmark.DEFAULT_BACKENDS)
            assert min(candidate["ratios"].values()) > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backends": "mujoco,motrix"},
        {"backends": "motrix,motrix"},
        {"seeds": "11,11"},
        {"validation_seeds": "11,44"},
        {"test_seeds": "1001,3"},
        {"confirmation_seeds": "11,21,31"},
        {"confirmation_seeds": "2001,2002"},
        {"num_envs": 2},
        {"num_envs": 7, "rollout_steps": 3},
        {"iterations": 2},
        {"override": ["training.num_timesteps=12"]},
        {"override": ["algo.seed=9"]},
        {"override": ["task=other"]},
        {"override": ["algo.resume_path=checkpoint.pt"]},
        {"fixed": ["motrix=1,drake=0"]},
    ],
)
def test_invalid_or_unfair_protocol_rejected(kwargs):
    with pytest.raises(ValueError):
        benchmark.plan_study(args(**kwargs))


def test_subset_is_explicitly_not_full_study():
    assert not benchmark.plan_study(args(backends="motrix,drake"))["full_backend_study"]


def test_manifest_edit_invalidates_identity(tmp_path):
    study = benchmark.plan_study(args())
    benchmark.write_json(tmp_path / "study.json", study)
    assert benchmark.load_study(tmp_path) == study
    study["iterations"] += 1
    benchmark.write_json(tmp_path / "study.json", study)
    with pytest.raises(ValueError, match="changed"):
        benchmark.load_study(tmp_path)


def test_provenance_detects_different_native_build_with_same_package_version(tmp_path, monkeypatch):
    package = tmp_path / "drake_uni"
    package.mkdir()
    (package / "__init__.py").write_text("")
    native = package / "_drake_batch.so"
    native.write_bytes(b"native-build-one")
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: (
            SimpleNamespace(origin=str(package / "__init__.py")) if name == "drake_uni" else None
        ),
    )
    distribution = SimpleNamespace(
        version="0.1.0",
        read_text=lambda _: json.dumps(
            {"url": "file:///task/local/drake", "dir_info": {"editable": True}}
        ),
    )
    monkeypatch.setattr(benchmark.importlib.metadata, "distribution", lambda _: distribution)
    first = benchmark.source_identity()
    native.write_bytes(b"native-build-two")
    second = benchmark.source_identity()
    assert first["packages"] == second["packages"]
    assert first["source_sha256"]["drake_uni"] != second["source_sha256"]["drake_uni"]
    assert first["install_provenance"]["drake-uni"]["dir_info"]["editable"]
    (package / "temporary_generated_scene.xml").write_text("<mujoco/>")
    assert second == benchmark.source_identity()


def test_missing_runs_never_report_advantage(tmp_path):
    result = benchmark.report(tmp_path, benchmark.plan_study(args()))
    assert result["status"] == "incomplete"
    assert result["mixed_advantage_supported_at_equal_samples"] is False


def test_process_receipt_reuses_success_but_rejects_changed_command(tmp_path):
    directory = tmp_path / "process"
    command = [sys.executable, "-c", "print('ok')"]
    first = benchmark.run_command(command, directory, "identity")
    assert first["returncode"] == 0
    assert benchmark.run_command(command, directory, "identity") == first
    with pytest.raises(ValueError, match="identity"):
        benchmark.run_command(command + ["different"], directory, "identity")


def test_failed_process_not_silently_restarted(tmp_path):
    directory = tmp_path / "process"
    command = [sys.executable, "-c", "raise SystemExit(7)"]
    with pytest.raises(RuntimeError, match="failed"):
        benchmark.run_command(command, directory, "identity")
    with pytest.raises(RuntimeError, match="retained"):
        benchmark.run_command(command, directory, "identity")


def test_frozen_candidate_uses_training_allocations_only(tmp_path, monkeypatch):
    study = benchmark.plan_study(args(backends="motrix,drake"))
    for seed in study["training_seeds"]:
        benchmark.write_json(
            tmp_path / "adaptive" / f"seed_{seed}" / "train/adaptive_mix.json",
            {"frozen": True, "updates": 2, "config": study["adaptive"]},
        )
    allocations = iter(
        [
            {"motrix": 0.2, "drake": 0.8},
            {"motrix": 0.3, "drake": 0.7},
            {"motrix": 0.4, "drake": 0.6},
        ]
    )
    monkeypatch.setattr(
        benchmark, "training_result", lambda *_: {"ratios": next(allocations), "sha256": "x"}
    )
    frozen = benchmark.freeze_candidate(tmp_path, study)
    assert frozen["ratios"] == pytest.approx({"motrix": 0.3, "drake": 0.7})
    assert frozen["name"] == "frozen_adaptive"


def test_frozen_candidate_rejects_unobserved_controller(tmp_path):
    study = benchmark.plan_study(args(backends="motrix,drake"))
    benchmark.write_json(
        tmp_path / "adaptive/seed_11/train/adaptive_mix.json",
        {"frozen": True, "updates": 1, "config": study["adaptive"]},
    )
    with pytest.raises(ValueError, match="two updates"):
        benchmark.freeze_candidate(tmp_path, study)


def test_two_backend_defaults_leave_room_to_adapt():
    study = benchmark.plan_study(args(backends="motrix,drake"))
    assert study["adaptive"]["max_ratio"] == 0.8
    config = {**study["adaptive"], "max_ratio": 0.5}
    with pytest.raises(ValueError, match="no freedom"):
        benchmark.validate_adaptive_budget(config, 2200, 2, 28)


def test_adaptive_budget_leaves_two_windows_and_frozen_tail():
    study = benchmark.plan_study(args())
    config = {**study["adaptive"], "freeze_after": 2200}
    with pytest.raises(ValueError, match="total training"):
        benchmark.validate_adaptive_budget(config, 2200, 7, 28)
    config["freeze_after"] = 150
    with pytest.raises(ValueError, match="two adaptive windows"):
        benchmark.validate_adaptive_budget(config, 2200, 7, 28)


def test_validation_selection_never_reads_test_split(tmp_path, monkeypatch):
    study = benchmark.plan_study(args(backends="motrix,drake"))
    choices = study["candidates"] + [
        {"name": "frozen_adaptive", "ratios": {"motrix": 0.3, "drake": 0.7}}
    ]
    scores = {c["name"]: i for i, c in enumerate(choices)}

    def evaluate(_root, _study, name, _seed, split):
        assert split == "validation"
        return {"metrics": {"episode_return": scores[name]}}

    monkeypatch.setattr(benchmark, "read_evaluation", evaluate)
    selection = benchmark.select_candidates(tmp_path, study, choices)
    assert selection["best_fixed_mixture"] == "frozen_adaptive"
    assert selection["best_single"] == "single_drake"


def test_pairing_uses_training_seed_units_and_requires_multiple_seeds():
    assert benchmark.paired_interval([3])["ci95"] is None
    assert benchmark.paired_interval([3, 3])["ci95"] is None
    positive = benchmark.paired_interval([2, 3, 4])
    assert positive["n_training_seeds"] == 3
    assert positive["ci95"][0] > 0
    uncertain = benchmark.paired_interval([-10, 1, 12])
    assert uncertain["ci95"][0] < 0 < uncertain["ci95"][1]


def test_exact_sign_flip_is_not_significant_with_three_positive_seeds():
    assert benchmark.paired_sign_flip([1, 2, 3])["p_value"] == 0.125
    assert benchmark.paired_sign_flip([0, 0, 0])["p_value"] == 1
    assert benchmark.paired_sign_flip([-1, -2, -3])["p_value"] == 1


def test_exact_meet_in_middle_agrees_with_exhaustive_sign_flips():
    differences = [-3, 0, 1, 1.5, 2, 4]
    observed = sum(differences)
    sums = [
        sum(sign * value for sign, value in zip(signs, differences))
        for signs in itertools.product((-1, 1), repeat=len(differences))
    ]
    expected = sum(value >= observed for value in sums) / len(sums)
    assert benchmark.paired_sign_flip(differences)["p_value"] == expected


def test_holm_adjustment_controls_all_seven_single_comparisons():
    too_few = benchmark.holm_adjust({f"single_{i}": 0.125 for i in range(7)})
    assert all(value > 0.05 for value in too_few.values())
    strong = benchmark.holm_adjust({f"single_{i}": 1 / 1024 for i in range(7)})
    assert all(value <= 0.05 for value in strong.values())
    ordered = benchmark.holm_adjust({"a": 0.001, "b": 0.03, "c": 0.04})
    assert ordered == pytest.approx({"a": 0.003, "b": 0.06, "c": 0.06})


def test_confirmation_protocol_freezes_ratio_and_fresh_training_seeds():
    study = benchmark.plan_study(args())
    selection = {"best_fixed_mixture": "equal"}
    protocol = benchmark.confirmation_protocol(study, selection, study["candidates"])
    assert not set(protocol["training_seeds"]) & set(study["training_seeds"])
    assert protocol["transitions_per_run"] == study["transitions"]
    assert len(protocol["candidates"]) == 8
    assert protocol["candidates"][0]["ratios"] == study["candidates"][7]["ratios"]


@pytest.mark.parametrize("report_status", ["completed", "incomplete"])
def test_run_study_retrains_confirmation_candidates_after_selection(
    tmp_path, monkeypatch, report_status
):
    study = benchmark.plan_study(args(backends="motrix,drake"))
    events = []
    frozen = {"name": "frozen_adaptive", "ratios": {"motrix": 0.3, "drake": 0.7}}

    def train(_root, _study, candidate, seed, *, preflight=False, confirmation=False):
        events.append(("train", candidate["name"], seed, confirmation, preflight))
        return {"checkpoint": "/probe.pt", "sha256": "probe"}

    def evaluate(_root, _study, candidate, seed, split):
        events.append(("evaluate", candidate["name"], seed, split))

    def select(*_):
        events.append(("selection",))
        return {"best_fixed_mixture": "frozen_adaptive", "best_single": "single_drake"}

    def preflight(_command, directory, identity):
        benchmark.write_json(
            directory / "metrics.json", {"status": "completed", "identity": identity}
        )

    monkeypatch.setattr(benchmark, "source_identity", lambda: study["source_identity"])
    monkeypatch.setattr(benchmark, "train_candidate", train)
    monkeypatch.setattr(benchmark, "evaluate_candidate", evaluate)
    monkeypatch.setattr(benchmark, "freeze_candidate", lambda *_: frozen)
    monkeypatch.setattr(benchmark, "select_candidates", select)
    monkeypatch.setattr(benchmark, "run_command", preflight)
    monkeypatch.setattr(benchmark, "report", lambda *_: {"status": report_status})
    if report_status == "completed":
        benchmark.run_study(tmp_path, study)
    else:
        with pytest.raises(ValueError, match="Study verification failed"):
            benchmark.run_study(tmp_path, study)
    assert benchmark.read_json(tmp_path / "report.json")["status"] == report_status
    selection_index = events.index(("selection",))
    confirmations = [event for event in events if event[0] == "train" and event[3]]
    assert len(confirmations) == 3 * len(study["confirmation_seeds"])
    assert {event[1] for event in confirmations} == {
        "frozen_adaptive",
        "single_motrix",
        "single_drake",
    }
    assert {event[2] for event in confirmations} == set(study["confirmation_seeds"])
    assert all(events.index(event) > selection_index for event in confirmations)
    assert all(
        event[2] in study["confirmation_seeds"]
        for event in events
        if event[0] == "evaluate" and event[3] == "test"
    )


def test_test_split_rejects_discovery_checkpoint(tmp_path, monkeypatch):
    study = benchmark.plan_study(args())
    selection = {"best_fixed_mixture": "equal"}
    benchmark.write_json(tmp_path / "selection.json", selection)
    benchmark.write_json(
        tmp_path / "confirmation_protocol.json",
        benchmark.confirmation_protocol(study, selection, study["candidates"]),
    )
    monkeypatch.setattr(benchmark, "load_study", lambda _: study)
    checkpoint = tmp_path / "equal/seed_11/train/model_2199.pt"
    with pytest.raises(ValueError, match="freshly trained confirmation"):
        benchmark.main(
            [
                "evaluate",
                "--study",
                str(tmp_path),
                "--checkpoint",
                str(checkpoint),
                "--split",
                "test",
                "--output",
                str(tmp_path / "metrics.json"),
                "--identity",
                "x",
            ]
        )


@pytest.mark.parametrize(
    "event",
    [
        {"event": "control_accepted", "manual_ratios": {"motrix": 1, "drake": 0}},
        {
            "event": "allocation",
            "requested_ratios": {"motrix": 0.75, "drake": 0.25},
            "counts": {"motrix": 21, "drake": 7},
        },
    ],
)
def test_manual_or_changed_fixed_allocation_invalidates_benchmark(tmp_path, event):
    study = benchmark.plan_study(args(backends="motrix,drake"))
    directory = tmp_path / "equal/seed_11"
    benchmark.write_json(
        directory / "train/mixed_config.json", {"initial_ratios": {"motrix": 0.5, "drake": 0.5}}
    )
    (directory / "train/mix_events.jsonl").write_text(json.dumps(event) + "\n")
    with pytest.raises(ValueError):
        benchmark.audit_allocations(directory, study)


def test_training_result_rejects_incomplete_budget(tmp_path):
    study = benchmark.plan_study(args())
    benchmark.write_json(
        tmp_path / "train/run_summary.json",
        {
            "status": "completed",
            "next_iteration": 9,
            "samples_per_iteration": 112,
        },
    )
    with pytest.raises(ValueError, match="iteration budget"):
        benchmark.training_result(tmp_path, study)


def completed_trial(tmp_path, study):
    directory = tmp_path / "equal/seed_11"
    train = directory / "train"
    ratios = {"motrix": 0.5, "drake": 0.5}
    counts = {"motrix": 14, "drake": 14}
    benchmark.write_json(train / "mixed_config.json", {"initial_ratios": ratios})
    (train / "mix_events.jsonl").write_text(
        json.dumps({"event": "allocation", "requested_ratios": ratios, "counts": counts}) + "\n"
    )
    benchmark.write_json(
        train / "run_summary.json",
        {
            "status": "completed",
            "next_iteration": study["iterations"],
            "samples_per_iteration": study["num_envs"] * study["rollout_steps"],
            "backend_counts": counts,
        },
    )
    benchmark.write_json(train / "run_config.json", {"contract_snapshot": {"test": True}})
    (train / f"model_{study['iterations'] - 1}.pt").write_bytes(b"checkpoint")
    benchmark.write_json(directory / "process.json", {"returncode": 0, "wall_seconds": 1.5})
    return directory


@pytest.mark.parametrize("counts", [{"motrix": 21, "drake": 7}, {"motrix": 28, "drake": 0}])
def test_constant_but_incorrect_realized_counts_rejected(tmp_path, counts):
    study = benchmark.plan_study(args(backends="motrix,drake"))
    directory = completed_trial(tmp_path, study)
    events = directory / "train/mix_events.jsonl"
    event = json.loads(events.read_text())
    event["counts"] = counts
    events.write_text(json.dumps(event) + "\n")
    with pytest.raises(ValueError, match="counts|active"):
        benchmark.audit_allocations(directory, study)


def test_training_result_requires_summary_to_match_allocation(tmp_path):
    study = benchmark.plan_study(args(backends="motrix,drake"))
    directory = completed_trial(tmp_path, study)
    result = benchmark.training_result(directory, study)
    assert result["ratios"] == {"motrix": 0.5, "drake": 0.5}
    summary_path = directory / "train/run_summary.json"
    summary = benchmark.read_json(summary_path)
    summary["backend_counts"] = {"motrix": 15, "drake": 13}
    benchmark.write_json(summary_path, summary)
    with pytest.raises(ValueError, match="summary differs"):
        benchmark.training_result(directory, study)


@pytest.mark.parametrize(
    "receipt",
    [
        {"returncode": None, "wall_seconds": 1.5},
        {"returncode": 1, "wall_seconds": 1.5},
        {"returncode": 0, "wall_seconds": 0},
        {"returncode": 0, "wall_seconds": True},
    ],
)
def test_training_result_rejects_failed_process_or_invalid_timing(tmp_path, receipt):
    study = benchmark.plan_study(args(backends="motrix,drake"))
    directory = completed_trial(tmp_path, study)
    benchmark.write_json(directory / "process.json", receipt)
    with pytest.raises(ValueError, match="process did not exit successfully"):
        benchmark.training_result(directory, study)


def test_report_requires_positive_evidence_against_every_single(tmp_path, monkeypatch):
    study = benchmark.plan_study(args(backends="motrix,drake"))
    benchmark.write_json(
        tmp_path / "frozen_candidate.json",
        {"name": "frozen_adaptive", "ratios": {"motrix": 0.5, "drake": 0.5}},
    )
    selection = {"best_fixed_mixture": "equal", "best_single": "single_motrix"}
    benchmark.write_json(tmp_path / "selection.json", selection)
    benchmark.write_json(
        tmp_path / "confirmation_protocol.json",
        benchmark.confirmation_protocol(study, selection, study["candidates"]),
    )
    monkeypatch.setattr(benchmark, "select_candidates", lambda *_: selection)
    monkeypatch.setattr(benchmark, "training_result", lambda *_: {"wall_seconds": 1})

    def evaluate(_root, _study, name, _seed, _split):
        return {
            "metrics": {"episode_return": {"equal": 2, "single_motrix": 1, "single_drake": 3}[name]}
        }

    monkeypatch.setattr(benchmark, "read_evaluation", evaluate)
    result = benchmark.report(tmp_path, study)
    assert result["status"] == "completed"
    assert set(result["paired_test_reward_vs_each_single"]) == {"single_motrix", "single_drake"}
    assert not result["mixed_advantage_supported_at_equal_samples"]


@pytest.mark.parametrize(
    "confirmation_seeds,supported",
    [("101,102,103", False), ("101,102,103,104,105,106,107,108,109,110", True)],
)
def test_report_claim_uses_exact_confirmation_test_not_positive_bootstrap(
    tmp_path, monkeypatch, confirmation_seeds, supported
):
    study = benchmark.plan_study(
        args(backends="motrix,drake", confirmation_seeds=confirmation_seeds)
    )
    frozen = {"name": "frozen_adaptive", "ratios": {"motrix": 0.5, "drake": 0.5}}
    benchmark.write_json(tmp_path / "frozen_candidate.json", frozen)
    selection = {"best_fixed_mixture": "equal", "best_single": "single_motrix"}
    benchmark.write_json(tmp_path / "selection.json", selection)
    benchmark.write_json(
        tmp_path / "confirmation_protocol.json",
        benchmark.confirmation_protocol(study, selection, study["candidates"] + [frozen]),
    )
    monkeypatch.setattr(benchmark, "select_candidates", lambda *_: selection)
    monkeypatch.setattr(benchmark, "training_result", lambda *_: {"wall_seconds": 1})

    def evaluate(_root, _study, name, seed, split):
        assert seed in study["confirmation_seeds"]
        assert split == "test"
        return {"metrics": {"episode_return": 2 if name == "equal" else 1}}

    monkeypatch.setattr(benchmark, "read_evaluation", evaluate)
    result = benchmark.report(tmp_path, study)
    assert result["status"] == "completed"
    assert result["mixed_advantage_supported_at_equal_samples"] is supported
    for comparison in result["paired_test_reward_vs_each_single"].values():
        assert comparison["descriptive_bootstrap_interval95"][0] > 0
        assert "familywise_ci95" not in comparison


def test_stale_checkpoint_evaluation_rejected(tmp_path, monkeypatch):
    study = benchmark.plan_study(args())
    monkeypatch.setattr(benchmark, "training_result", lambda *_: {"sha256": "new-checkpoint"})
    path = tmp_path / "equal/seed_11/validation/metrics.json"
    benchmark.write_json(path, {"identity": "old-checkpoint", "status": "completed"})
    with pytest.raises(ValueError, match="stale"):
        benchmark.read_evaluation(tmp_path, study, "equal", 11, "validation")

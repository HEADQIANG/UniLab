"""Reproducible ratio search with an independently held-out MuJoCo test split.

See docs/mix_benchmark.md. The orchestrator never sends evaluation metrics to
the learner or ratio controller. Simulation and PPO imports are lazy.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import importlib.metadata
import json
import math
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_BACKENDS = ("motrix", "drake", "mjwarp", "isaacgym", "isaacsim", "genesis", "newton")
SCHEMA = 2


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def checkpoint_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_identity() -> dict[str, Any]:
    """Bind a study to package implementation/config and installed native physics binaries."""
    from importlib.util import find_spec

    roots = {"unilab": Path(__file__).resolve().parents[1]}
    for name in ("uni_rl", "unisim", "drake_uni"):
        spec = find_spec(name)
        if spec is not None and spec.origin is not None:
            roots[name] = Path(spec.origin).resolve().parent
    hashes = {}
    source_git: dict[str, str | None] = {}
    for name, root in roots.items():
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False
        )
        source_git[name] = revision.stdout.strip() if revision.returncode == 0 else None
        tracked = subprocess.run(
            ["git", "ls-files", "--", "*.xml"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        tracked_xml = set(tracked.stdout.splitlines()) if tracked.returncode == 0 else set()
        accumulator = hashlib.sha256()
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            generated = any(
                part in {"__pycache__", "cache", "caches", ".cache", "generated", "_generated"}
                for part in relative.parts
            )
            source_file = (
                path.suffix
                in {".py", ".yaml", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".so", ".pyd", ".dylib"}
                or ".so." in path.name
                or str(relative) in tracked_xml
            )
            if path.is_file() and not generated and source_file:
                accumulator.update(str(relative).encode())
                accumulator.update(bytes.fromhex(checkpoint_digest(path)))
        hashes[name] = accumulator.hexdigest()
    versions: dict[str, str | None] = {}
    install_provenance: dict[str, Any] = {}
    for package in (
        "unisim-core",
        "unilab-rl",
        "rsl-rl-lib",
        "torch",
        "numpy",
        "mujoco",
        "mujoco-uni-runtime",
        "drake-uni",
        "motrixsim-core",
        "mujoco-warp",
        "genesis-world",
        "newton",
        "warp-lang",
    ):
        try:
            distribution = importlib.metadata.distribution(package)
            versions[package] = distribution.version
            direct_url = distribution.read_text("direct_url.json")
            if direct_url is not None:
                install_provenance[package] = json.loads(direct_url)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "source_sha256": hashes,
        "module_roots": {name: str(root) for name, root in roots.items()},
        "source_git_commits": source_git,
        "packages": versions,
        "install_provenance": install_provenance,
        "python": sys.version,
        "external_runtime_limitations": "IsaacGym/IsaacSim external worker environments, system shared libraries and GPU drivers are not fingerprinted; keep them unchanged within a study.",
    }


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item) for item in value.split(",")]
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    return seeds


def validate_ratios(ratios: dict[str, float], names: tuple[str, ...]) -> dict[str, float]:
    if set(ratios) != set(names) or any(
        isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in ratios.values()
    ):
        raise ValueError(
            "Every mixed candidate must assign a positive finite ratio to every backend"
        )
    total = sum(ratios.values())
    return {name: ratios[name] / total for name in names}


def candidates(names: tuple[str, ...], fixed: list[str]) -> list[dict[str, Any]]:
    equal = {name: 1 / len(names) for name in names}
    result = [{"name": f"single_{name}", "ratios": {name: 1.0}} for name in names]
    result += [{"name": "equal", "ratios": equal}, {"name": "adaptive", "ratios": equal}]
    # A small declared search covering each vertex's interior, not a global optimum search.
    for name in names:
        result.append(
            {
                "name": f"favor_{name}",
                "ratios": {
                    other: 0.5 if other == name else 0.5 / (len(names) - 1) for other in names
                },
            }
        )
    for index, text in enumerate(fixed):
        pairs = [item.split("=", 1) for item in text.split(",")]
        ratios = {name: float(value) for name, value in pairs}
        if len(pairs) != len(ratios):
            raise ValueError("Duplicate backend in candidate")
        result.append({"name": f"fixed_{index}", "ratios": validate_ratios(ratios, names)})
    return result


def plan_study(args: argparse.Namespace) -> dict[str, Any]:
    names: tuple[str, ...] = tuple(str(args.backends).split(","))
    if len(names) < 2 or len(set(names)) != len(names) or set(names) - set(DEFAULT_BACKENDS):
        raise ValueError("Choose at least two distinct training backends; MuJoCo is held out")
    if args.task != "g1_walk_flat":
        raise ValueError("This benchmark currently defines the G1WalkFlat tracking protocol only")
    seed_groups = [
        parse_seeds(value)
        for value in (args.seeds, args.validation_seeds, args.test_seeds, args.confirmation_seeds)
    ]
    if any(set(a) & set(b) for i, a in enumerate(seed_groups) for b in seed_groups[i + 1 :]):
        raise ValueError(
            "Discovery training, confirmation training, validation, and test seeds must be disjoint"
        )
    if len(seed_groups[3]) > 40:
        raise ValueError("The exact paired sign-flip test supports at most 40 confirmation seeds")
    if min(args.num_envs, args.rollout_steps, args.iterations, args.episodes) < 1:
        raise ValueError("Environment, rollout, iteration, and episode counts must be positive")
    if args.num_envs < len(names) or args.num_envs * args.rollout_steps % 4:
        raise ValueError("Need an environment per backend and a rollout batch divisible by 4")
    # Protect budgets, routing, checkpoint provenance and isolation from passthrough overrides.
    for override in args.override:
        key = override.split("=", 1)[0].lstrip("+")
        if not key.startswith(("env.", "reward.", "algo.policy.", "algo.algorithm.")):
            raise ValueError(f"Override not allowed by fair comparison protocol: {key}")
        if key == "algo.algorithm.num_mini_batches":
            if args.num_envs * args.rollout_steps % int(override.split("=", 1)[1]):
                raise ValueError("Rollout batch must divide evenly into mini batches")
    settings: dict[str, Any] = {}
    if args.mix_config:
        from omegaconf import OmegaConf

        loaded_settings = OmegaConf.to_container(OmegaConf.load(args.mix_config), resolve=True)
        if not isinstance(loaded_settings, dict) or set(loaded_settings) - {
            "backends",
            "adaptive",
            "held_out_backends",
        }:
            raise ValueError(
                "Benchmark mix config accepts backends and adaptive, not manual stages"
            )
        settings = {str(key): value for key, value in loaded_settings.items()}
        if settings.get("held_out_backends", ["mujoco"]) != ["mujoco"]:
            raise ValueError("This study reserves MuJoCo as the held-out simulator")
        if set(settings.get("backends", {})) - set(names):
            raise ValueError("Backend settings must belong to the declared training set")
    adaptive = {
        "metric": "mean_step_reward",
        "strategy": "learning_progress",
        "interval": 50,
        "warmup": 100,
        "ema_alpha": 0.2,
        "temperature": 1.0,
        "min_ratio": 0.05,
        "max_ratio": 0.8 if len(names) == 2 else 0.5,
        "smoothing": 0.3,
        "freeze_after": args.iterations * 9 // 10,
        **settings.get("adaptive", {}),
    }
    validate_adaptive_budget(adaptive, args.iterations, len(names), args.num_envs)
    study = {
        "schema": SCHEMA,
        "task": args.task,
        "backends": list(names),
        "full_backend_study": set(names) == set(DEFAULT_BACKENDS),
        "training_seeds": seed_groups[0],
        "validation_seeds": seed_groups[1],
        "test_seeds": seed_groups[2],
        "confirmation_seeds": seed_groups[3],
        "num_envs": args.num_envs,
        "rollout_steps": args.rollout_steps,
        "iterations": args.iterations,
        "transitions": args.num_envs * args.rollout_steps * args.iterations,
        "episodes_per_evaluation_seed": args.episodes,
        "learner_device": args.device,
        "backend_settings": settings.get("backends", {}),
        "adaptive": adaptive,
        "overrides": args.override,
        "candidates": candidates(names, args.fixed),
        "selection_metric": "episode_return",
        "held_out_backend": "mujoco",
        "frozen_ratio_recipe": "mean_final_realized_adaptive_ratios_across_training_seeds",
        "source_identity": source_identity(),
    }
    study["id"] = digest(study)
    return study


def validate_adaptive_budget(
    config: dict[str, Any], iterations: int, backends: int, num_envs: int
) -> None:
    """A ratio search needs multiple observations and a genuinely frozen final policy."""
    for key in ("interval", "warmup", "freeze_after"):
        if type(config[key]) is not int or config[key] < (0 if key == "warmup" else 1):
            raise ValueError(f"adaptive.{key} has an invalid iteration count")
    if config["freeze_after"] >= iterations:
        raise ValueError("Benchmark requires adaptive.freeze_after < total training iterations")
    first = max(config["warmup"], config["interval"])
    if config["freeze_after"] <= first + config["interval"]:
        raise ValueError(
            "Benchmark needs at least two adaptive windows before freezing; lower warmup/interval or increase iterations"
        )
    floor = max(float(config["min_ratio"]), 1 / num_envs)
    if floor * backends >= 1 or float(config["max_ratio"]) * backends <= 1:
        raise ValueError("Adaptive ratio bounds leave no freedom to change the allocation")


def load_study(root: Path) -> dict[str, Any]:
    study: dict[str, Any] = read_json(root / "study.json")
    expected = study.pop("id")
    actual = digest(study)
    study["id"] = expected
    if actual != expected or study.get("schema") != SCHEMA:
        raise ValueError("Study manifest changed or has an unsupported schema; create a new study")
    return study


def run_command(command: list[str], directory: Path, identity: str) -> dict[str, Any]:
    """Retain failures; never silently reuse/restart a partially completed run."""
    receipt = directory / "process.json"
    if receipt.exists():
        old: dict[str, Any] = read_json(receipt)
        if old.get("identity") != identity or old.get("command") != command:
            raise ValueError(f"Run identity mismatch: {directory}")
        if old.get("returncode") == 0:
            return old
        raise RuntimeError(
            f"Incomplete/failed run retained at {directory}; inspect before retrying"
        )
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Refusing nonempty process directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "identity": identity,
        "command": command,
        "pid": None,
        "returncode": None,
    }
    write_json(receipt, record)
    started = time.monotonic()
    with (directory / "console.log").open("w", encoding="utf-8") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        record["pid"] = process.pid
        write_json(receipt, record)
        try:
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    record.update(returncode=code, wall_seconds=time.monotonic() - started)
    write_json(receipt, record)
    if code:
        raise RuntimeError(f"Process failed ({code}); see {directory / 'console.log'}")
    return record


def train_command(
    study: dict[str, Any],
    candidate: dict[str, Any],
    seed: int,
    run: Path,
    config_path: Path,
    *,
    preflight: bool = False,
) -> list[str]:
    from unilab.cli import build_command

    overrides = list(study["overrides"]) + [
        f"algo.seed={seed}",
        f"algo.num_envs={study['num_envs']}",
        f"algo.num_steps_per_env={study['rollout_steps']}",
        f"algo.max_iterations={1 if preflight else study['iterations']}",
        f"algo.save_interval={study['iterations']}",
        f"training.device={study['learner_device']}",
        f"training.log_dir={json.dumps(str(run))}",
        "training.logger=tensorboard",
    ]
    return build_command(
        mode="train",
        algo="ppo",
        task=study["task"],
        sim=None,
        sim_mix=",".join(f"{k}={v}" for k, v in candidate["ratios"].items()),
        mix_config=str(config_path),
        overrides=overrides,
    )


def training_result(
    directory: Path, study: dict[str, Any], *, preflight: bool = False
) -> dict[str, Any]:
    summary = read_json(directory / "train" / "run_summary.json")
    iterations = 1 if preflight else study["iterations"]
    if summary.get("status") != "completed" or summary.get("next_iteration") != iterations:
        raise ValueError(f"Training did not complete the declared iteration budget: {directory}")
    if summary.get("samples_per_iteration") != study["num_envs"] * study["rollout_steps"]:
        raise ValueError(f"Training sample budget differs: {directory}")
    counts = summary.get("backend_counts", {})
    if (
        any(type(count) is not int or count <= 0 for count in counts.values())
        or sum(counts.values()) != study["num_envs"]
    ):
        raise ValueError(f"Invalid realized backend counts: {directory}")
    audited_counts = audit_allocations(directory, study, preflight=preflight)
    if counts != audited_counts:
        raise ValueError(f"Training summary differs from the final allocation: {directory}")
    checkpoint = directory / "train" / f"model_{iterations - 1}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Final checkpoint missing: {checkpoint}")
    snapshot = read_json(directory / "train" / "run_config.json")
    if not snapshot.get("contract_snapshot"):
        raise ValueError("Benchmark requires a verifiable sim2sim contract snapshot")
    receipt = read_json(directory / "process.json")
    wall_seconds = receipt.get("wall_seconds")
    if (
        receipt.get("returncode") != 0
        or isinstance(wall_seconds, bool)
        or not isinstance(wall_seconds, (int, float))
        or not math.isfinite(wall_seconds)
        or wall_seconds <= 0
    ):
        raise ValueError(
            f"Training process did not exit successfully with valid timing: {directory}"
        )
    return {
        "checkpoint": str(checkpoint),
        "sha256": checkpoint_digest(checkpoint),
        "ratios": {k: v / study["num_envs"] for k, v in counts.items()},
        "wall_seconds": wall_seconds,
    }


def audit_allocations(
    directory: Path, study: dict[str, Any], *, preflight: bool = False
) -> dict[str, int]:
    """An externally edited control file must not silently change a benchmark arm."""
    from uni_rl.ipc.mix_schedule import allocate_counts

    name = "equal" if preflight else directory.parent.name
    if name == "frozen_adaptive":
        study_root = directory.parents[1]
        if not (study_root / "study.json").exists():
            study_root = study_root.parent
        candidate = read_json(study_root / "frozen_candidate.json")
    else:
        candidate = next((c for c in study["candidates"] if c["name"] == name), None)
        if candidate is None:
            raise ValueError(f"Unrecognized benchmark candidate: {name}")
    saved = read_json(directory / "train/mixed_config.json")
    if saved.get("initial_ratios") != candidate["ratios"]:
        raise ValueError(f"Training initial ratios differ from the declared candidate: {directory}")
    allocations = []
    with (directory / "train/mix_events.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("event") == "control_accepted":
                raise ValueError(f"Manual control contaminated benchmark run: {directory}")
            if event.get("event") == "allocation":
                allocations.append(event)
                if set(event.get("counts", {})) != set(candidate["ratios"]):
                    raise ValueError(f"Declared backend set changed: {directory}")
                if name != "adaptive" and event.get("requested_ratios") != candidate["ratios"]:
                    raise ValueError(f"Fixed candidate allocation changed: {directory}")
                counts = event["counts"]
                if any(type(count) is not int or count <= 0 for count in counts.values()):
                    raise ValueError(f"Allocation must keep every backend active: {directory}")
                if counts != allocate_counts(event["requested_ratios"], study["num_envs"]):
                    raise ValueError(
                        f"Realized counts differ from the requested allocation: {directory}"
                    )
    if not allocations:
        raise ValueError(f"Missing allocation evidence: {directory}")
    if name != "adaptive" and any(
        event["counts"] != allocations[0]["counts"] for event in allocations
    ):
        raise ValueError(f"Fixed candidate environment counts changed: {directory}")
    final_counts: dict[str, int] = allocations[-1]["counts"]
    return final_counts


def train_candidate(
    root: Path,
    study: dict[str, Any],
    candidate: dict[str, Any],
    seed: int,
    *,
    preflight: bool = False,
    confirmation: bool = False,
) -> dict[str, Any]:
    phase_root = root / "confirmation" if confirmation else root
    directory = phase_root / ("preflight" if preflight else candidate["name"]) / f"seed_{seed}"
    settings = {
        "held_out_backends": ["mujoco"],
        "backends": {
            k: v for k, v in study["backend_settings"].items() if k in candidate["ratios"]
        },
    }
    if candidate["name"] == "adaptive" and not preflight:
        settings["adaptive"] = study["adaptive"]
    config_path = root / "configs" / f"{candidate['name']}.json"
    if config_path.exists() and read_json(config_path) != settings:
        raise ValueError(f"Generated config changed: {config_path}")
    write_json(config_path, settings)
    command = train_command(
        study, candidate, seed, directory / "train", config_path, preflight=preflight
    )
    run_command(command, directory, digest([study["id"], candidate, seed, preflight, confirmation]))
    return training_result(directory, study, preflight=preflight)


def freeze_candidate(root: Path, study: dict[str, Any]) -> dict[str, Any]:
    for seed in study["training_seeds"]:
        state = read_json(root / "adaptive" / f"seed_{seed}" / "train" / "adaptive_mix.json")
        if state.get("frozen") is not True or state.get("updates", 0) < 2:
            raise ValueError(
                f"Adaptive seed {seed} did not complete two updates and freeze; no learned ratio can be recommended"
            )
        if state.get("config") != study["adaptive"]:
            raise ValueError(f"Adaptive seed {seed} used different controller settings")
    results = [
        training_result(root / "adaptive" / f"seed_{s}", study) for s in study["training_seeds"]
    ]
    names = tuple(study["backends"])
    ratios = validate_ratios(
        {name: statistics.mean(r["ratios"][name] for r in results) for name in names}, names
    )
    candidate = {
        "name": "frozen_adaptive",
        "ratios": ratios,
        "source_checkpoint_sha256": [r["sha256"] for r in results],
    }
    path = root / "frozen_candidate.json"
    if path.exists() and read_json(path) != candidate:
        raise ValueError("Frozen ratio sources changed; start a new study")
    write_json(path, candidate)
    return candidate


def evaluation_identity(study: dict[str, Any], result: dict[str, Any], split: str) -> str:
    return digest([study["id"], result["sha256"], split, study[f"{split}_seeds"]])


def read_evaluation(
    root: Path, study: dict[str, Any], name: str, seed: int, split: str
) -> dict[str, Any]:
    phase_root = root / "confirmation" if split == "test" else root
    directory = phase_root / name / f"seed_{seed}"
    result = training_result(directory, study)
    evaluation: dict[str, Any] = read_json(directory / split / "metrics.json")
    if (
        evaluation.get("identity") != evaluation_identity(study, result, split)
        or evaluation.get("status") != "completed"
        or evaluation.get("backend") != "mujoco"
        or evaluation.get("seeds") != study[f"{split}_seeds"]
    ):
        raise ValueError(f"Invalid or stale {split} evaluation at {directory}")
    expected = len(study[f"{split}_seeds"]) * study["episodes_per_evaluation_seed"]
    if len(evaluation.get("episodes", [])) != expected:
        raise ValueError(f"Incomplete evaluation at {directory}")
    for key in ("episode_return", "survival_fraction", "tracking_lin_vel", "tracking_ang_vel"):
        if not math.isfinite(evaluation["metrics"][key]):
            raise ValueError(f"Non-finite evaluation at {directory}")
    return evaluation


def evaluate_candidate(
    root: Path, study: dict[str, Any], candidate: dict[str, Any], seed: int, split: str
) -> None:
    phase_root = root / "confirmation" if split == "test" else root
    directory = phase_root / candidate["name"] / f"seed_{seed}"
    result = training_result(directory, study)
    identity = evaluation_identity(study, result, split)
    command = [
        sys.executable,
        "-m",
        "unilab.training.mix_benchmark",
        "evaluate",
        "--study",
        str(root),
        "--checkpoint",
        result["checkpoint"],
        "--split",
        split,
        "--output",
        str(directory / split / "metrics.json"),
        "--identity",
        identity,
    ]
    run_command(command, directory / split, identity)
    read_evaluation(root, study, candidate["name"], seed, split)


def select_candidates(
    root: Path, study: dict[str, Any], choices: list[dict[str, Any]]
) -> dict[str, Any]:
    scores = {}
    for candidate in choices:
        name = candidate["name"]
        scores[name] = statistics.mean(
            read_evaluation(root, study, name, seed, "validation")["metrics"]["episode_return"]
            for seed in study["training_seeds"]
        )
    # Adaptive is a schedule, not a tested constant ratio; refit frozen_adaptive supplies that comparison.
    mixed = [
        c["name"]
        for c in choices
        if not c["name"].startswith("single_") and c["name"] != "adaptive"
    ]
    single = [c["name"] for c in choices if c["name"].startswith("single_")]
    selection = {
        "study_id": study["id"],
        "validation_scores": scores,
        "best_fixed_mixture": max(mixed, key=lambda name: scores[name]),
        "best_single": max(single, key=lambda name: scores[name]),
    }
    path = root / "selection.json"
    if path.exists() and read_json(path) != selection:
        raise ValueError("Validation selection changed after freezing; create a new study")
    return selection


def paired_interval(
    differences: list[float], *, draws: int = 10000, confidence: float = 0.95
) -> dict[str, Any]:
    if not differences or any(not math.isfinite(v) for v in differences):
        raise ValueError("Need finite matched-seed differences")
    mean = statistics.mean(differences)
    if len(differences) < 3:
        return {"mean_difference": mean, "ci95": None, "n_training_seeds": len(differences)}
    rng = random.Random(0)
    means = sorted(
        statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(draws)
    )
    tail = (1 - confidence) / 2
    return {
        "mean_difference": mean,
        "ci95": [means[int(draws * tail)], means[min(draws - 1, int(draws * (1 - tail)))]],
        "confidence": confidence,
        "n_training_seeds": len(differences),
        "method": "paired_training_seed_percentile_bootstrap",
    }


def paired_sign_flip(differences: list[float]) -> dict[str, Any]:
    """Exact one-sided paired randomization test, assuming sign exchangeability under H0.

    Meet-in-the-middle counting enumerates 2**ceil(n/2), not 2**n, sums.
    Ties count against significance. This tests the mean paired reward difference.
    """
    if not differences or len(differences) > 40 or any(not math.isfinite(v) for v in differences):
        raise ValueError("Exact sign-flip test needs 1–40 finite paired differences")

    def sums(values: list[float]) -> list[float]:
        result = [0.0]
        for value in values:
            result = [total + sign * value for total in result for sign in (-1, 1)]
        return result

    midpoint = len(differences) // 2
    left = sums(differences[:midpoint])
    right = sorted(sums(differences[midpoint:]))
    observed = math.fsum(differences)
    tolerance = max(1.0, math.fsum(abs(v) for v in differences)) * 1e-12
    extreme = sum(
        len(right) - bisect.bisect_left(right, observed - value - tolerance) for value in left
    )
    return {
        "p_value": extreme / (2 ** len(differences)),
        "alternative": "mixed_greater_than_single",
        "method": "exact_paired_sign_flip",
        "n_confirmation_training_seeds": len(differences),
        "null_assumption": "paired_reward_differences_are_sign_exchangeable",
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm step-down familywise adjustment, valid for dependent comparisons."""
    if not p_values or any(not math.isfinite(p) or not 0 <= p <= 1 for p in p_values.values()):
        raise ValueError("Need a nonempty set of valid p-values")
    adjusted = {}
    previous = 0.0
    for index, (name, p_value) in enumerate(sorted(p_values.items(), key=lambda item: item[1])):
        previous = max(previous, min(1.0, p_value * (len(p_values) - index)))
        adjusted[name] = previous
    return adjusted


def confirmation_protocol(
    study: dict[str, Any], selection: dict[str, Any], choices: list[dict[str, Any]]
) -> dict[str, Any]:
    names = [selection["best_fixed_mixture"]] + [
        c["name"] for c in choices if c["name"].startswith("single_")
    ]
    selected = [next(c for c in choices if c["name"] == name) for name in names]
    return {
        "study_id": study["id"],
        "selection_sha256": digest(selection),
        "training_seeds": study["confirmation_seeds"],
        "candidates": selected,
        "transitions_per_run": study["transitions"],
        "test_seeds": study["test_seeds"],
        "decision_rule": "positive_mean_and_Holm_adjusted_exact_sign_flip_p_le_0.05_against_every_single",
    }


def report(root: Path, study: dict[str, Any]) -> dict[str, Any]:
    try:
        frozen = read_json(root / "frozen_candidate.json")
        choices = study["candidates"] + [frozen]
        selection = select_candidates(root, study, choices)
        if read_json(root / "selection.json") != selection:
            raise ValueError("Test split requires a frozen validation selection")
        confirmation = confirmation_protocol(study, selection, choices)
        if read_json(root / "confirmation_protocol.json") != confirmation:
            raise ValueError("Confirmation training must match its frozen ratio/seed protocol")
        names = [candidate["name"] for candidate in confirmation["candidates"]]
        results: dict[str, Any] = {}
        for name in names:
            per_seed = [
                read_evaluation(root, study, name, seed, "test")["metrics"]
                for seed in study["confirmation_seeds"]
            ]
            timings = [
                training_result(root / "confirmation" / name / f"seed_{s}", study)["wall_seconds"]
                for s in study["confirmation_seeds"]
            ]
            results[name] = {
                "per_training_seed": per_seed,
                "mean": {k: statistics.mean(row[k] for row in per_seed) for k in per_seed[0]},
                "training_wall_seconds": timings,
                "mean_transitions_per_second": statistics.mean(
                    study["transitions"] / t for t in timings
                ),
            }
        comparisons = {}
        for name in names[1:]:
            differences = [
                a["episode_return"] - b["episode_return"]
                for a, b in zip(
                    results[names[0]]["per_training_seed"],
                    results[name]["per_training_seed"],
                    strict=True,
                )
            ]
            comparison = paired_interval(differences)
            comparison["descriptive_bootstrap_interval95"] = comparison.pop("ci95")
            comparison["bootstrap_note"] = "descriptive_only_not_used_for_significance"
            comparison["permutation_test"] = paired_sign_flip(differences)
            comparisons[name] = comparison
        adjusted = holm_adjust(
            {name: row["permutation_test"]["p_value"] for name, row in comparisons.items()}
        )
        for name, row in comparisons.items():
            row["holm_adjusted_p_value"] = adjusted[name]
        supported = all(
            row["mean_difference"] > 0 and row["holm_adjusted_p_value"] <= 0.05
            for row in comparisons.values()
        )
        winner = next(c for c in choices if c["name"] == names[0])
        return {
            "status": "completed",
            "study_id": study["id"],
            "selection": selection,
            "provenance": study["source_identity"],
            "best_observed_fixed_ratios": winner["ratios"],
            "test_results": results,
            "paired_test_reward_vs_each_single": comparisons,
            "confirmation_protocol": confirmation,
            "mixed_advantage_supported_at_equal_samples": supported,
            "full_backend_study": study["full_backend_study"],
            "claim": "evidence_supports_mixed_advantage"
            if supported
            else "mixed_advantage_not_established",
            "limitations": [
                "Best observed candidate; no global-optimum claim.",
                "Wall-time throughput is measured; equal-wall-time policy quality is not tested.",
                "Bootstrap intervals are descriptive; exact sign-flip tests assume paired sign exchangeability.",
                "Inference uses fresh confirmation training seeds; evaluation episodes are not independent training replicates.",
                "Adaptive discovery/search compute is additional to each equal-sample refit.",
            ],
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {
            "status": "incomplete",
            "study_id": study["id"],
            "reason": str(error),
            "mixed_advantage_supported_at_equal_samples": False,
        }


def run_study(root: Path, study: dict[str, Any], *, preflight_only: bool = False) -> None:
    if source_identity() != study["source_identity"]:
        raise ValueError("Source/config/dependencies changed since planning; create a new study")
    equal = next(c for c in study["candidates"] if c["name"] == "equal")
    # A real simultaneous training step checks dependencies, task capability, IPC and device capacity.
    print(
        "Checking every declared training backend together; no backend will be dropped", flush=True
    )
    probe = train_candidate(root, study, equal, study["training_seeds"][0], preflight=True)
    probe_dir = root / "preflight" / "mujoco_validation"
    identity = evaluation_identity(study, probe, "validation")
    run_command(
        [
            sys.executable,
            "-m",
            "unilab.training.mix_benchmark",
            "evaluate",
            "--study",
            str(root),
            "--checkpoint",
            probe["checkpoint"],
            "--split",
            "validation",
            "--output",
            str(probe_dir / "metrics.json"),
            "--identity",
            identity,
        ],
        probe_dir,
        identity,
    )
    probe_metrics = read_json(probe_dir / "metrics.json")
    if probe_metrics.get("status") != "completed" or probe_metrics.get("identity") != identity:
        raise ValueError("MuJoCo preflight evaluation did not complete")
    if preflight_only:
        return
    adaptive = next(c for c in study["candidates"] if c["name"] == "adaptive")
    for seed in study["training_seeds"]:
        print(f"Training adaptive discovery seed {seed}", flush=True)
        train_candidate(root, study, adaptive, seed)
    frozen = freeze_candidate(root, study)
    choices = study["candidates"] + [frozen]
    for candidate in choices:
        for seed in study["training_seeds"]:
            print(f"Training/validating {candidate['name']} seed {seed}", flush=True)
            train_candidate(root, study, candidate, seed)
            evaluate_candidate(root, study, candidate, seed, "validation")
    selection = select_candidates(root, study, choices)
    write_json(root / "selection.json", selection)
    confirmation = confirmation_protocol(study, selection, choices)
    confirmation_path = root / "confirmation_protocol.json"
    if confirmation_path.exists() and read_json(confirmation_path) != confirmation:
        raise ValueError("Frozen confirmation protocol changed; start a new study")
    write_json(confirmation_path, confirmation)
    for candidate in confirmation["candidates"]:
        for seed in study["confirmation_seeds"]:
            print(
                f"Fresh confirmation training/testing {candidate['name']} seed {seed}", flush=True
            )
            train_candidate(root, study, candidate, seed, confirmation=True)
            evaluate_candidate(root, study, candidate, seed, "test")
    result = report(root, study)
    write_json(root / "report.json", result)
    if result["status"] != "completed":
        raise ValueError(f"Study verification failed: {result.get('reason', 'incomplete report')}")


def evaluate_checkpoint(study: dict[str, Any], checkpoint: Path, split: str) -> dict[str, Any]:
    """Deterministic CPU inference and seeded episodes, without training or controller updates."""
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from rsl_rl.runners import OnPolicyRunner
    from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper, get_policy_obs_dims, normalize_ppo_train_cfg

    from unilab.base.config_adapter import BackendAdapter, create_env
    from unilab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from unilab.training.mixed_config import RUNTIME_FIELDS, compose_mixed_profile
    from unilab.training.run import algo_config_dict
    from unilab.utils.checkpoint import get_entrypoint_log_root
    from unilab.utils.seed import apply_training_seed
    from unilab.utils.sim2sim import resolve_sim2sim_config
    from unilab.visualization.interactive_playback import (
        RslRlPlaybackConfig,
        create_rsl_rl_playback_session,
        infer_checkpoint_actor_input_dim,
    )

    source = read_json(checkpoint.parent / "run_config.json")
    if not source.get("contract_snapshot"):
        raise ValueError("A contract snapshot is required; legacy fallback is disallowed")
    cfg = compose_mixed_profile(study["task"], "mujoco")
    # Carry the trained shared task and network exactly, but use MuJoCo's runtime fields.
    for key in ("env", "reward", "algo"):
        data = dict(source["config"][key])
        if key == "env":
            data = {k: v for k, v in data.items() if k not in RUNTIME_FIELDS}
        setattr(cfg, key, OmegaConf.create(data))
    cfg.algo.runtime_resolver = None
    cfg.training.sim_backend = "mujoco"
    cfg.training.play_only = True
    cfg.training.play_render_mode = "none"
    resolve_sim2sim_config(checkpoint.parent, cfg, algo_name="ppo", strict=True)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    episodes = []
    for seed in study[f"{split}_seeds"]:
        apply_training_seed(seed, cuda=False)
        overrides = BackendAdapter(
            cfg, root_dir=Path.cwd(), algo_name="ppo"
        ).build_task_env_cfg_override()
        overrides.update(seed=seed, auto_reset=False)
        env = create_env(cfg, num_envs=1, env_cfg_override=overrides)
        try:
            if not isinstance(env, ManagerBasedRlEnv):
                raise TypeError("G1 benchmark evaluation requires ManagerBasedRlEnv")
            session, _, loaded = create_rsl_rl_playback_session(
                playback_cfg=RslRlPlaybackConfig(
                    task=str(cfg.training.task_name),
                    load_run="-1",
                    checkpoint=str(checkpoint),
                    action_mode="policy",
                    policy_obs_mode="auto",
                    algo_log_name="rsl_rl_ppo",
                    log_root=None,
                    num_envs=1,
                ),
                env_factory=lambda _: env,
                algo_config=algo_config_dict(cfg),
                root_dir=Path.cwd(),
                device="cpu",
                checkpoint_resolver=lambda *_: str(checkpoint),
                checkpoint_input_dim_reader=infer_checkpoint_actor_input_dim,
                entrypoint_log_root=get_entrypoint_log_root,
                wrapper_cls=RslRlVecEnvWrapper,
                runner_cls=OnPolicyRunner,
                policy_obs_dims_getter=get_policy_obs_dims,
                train_cfg_normalizer=normalize_ppo_train_cfg,
                guard_algo_name="ppo",
            )
            if loaded != str(checkpoint) or session.policy is None:
                raise RuntimeError(
                    "Evaluation must load the requested policy; zero-action fallback forbidden"
                )
            # The wrapper initializes/reset once. Reset the RNG after policy construction,
            # so every policy receives the same evaluation command/noise sequence.
            env.seed(seed)
            apply_training_seed(seed, cuda=False)
            horizon = math.ceil(float(cfg.env.max_episode_seconds) / float(cfg.env.ctrl_dt))
            if horizon < 1:
                raise ValueError("Evaluation episode horizon must be positive")
            for episode in range(study["episodes_per_evaluation_seed"]):
                episode_seed = int(digest([seed, episode])[:8], 16)
                env.seed(episode_seed)
                apply_training_seed(episode_seed, cuda=False)
                obs, _ = session.wrapped_env.reset()
                reward_sum = linear = angular = 0.0
                terminated = False
                step = 0
                with torch.inference_mode():
                    for step in range(1, horizon + 1):
                        obs, reward, done, _ = session.wrapped_env.step(session.policy(obs))
                        value = float(reward.item())
                        terms = env.reward_manager.step_reward_extras()
                        lin = float(terms["reward/tracking_lin_vel"]) / float(
                            cfg.reward.tracking_lin_vel.weight
                        )
                        ang = float(terms["reward/tracking_ang_vel"]) / float(
                            cfg.reward.tracking_ang_vel.weight
                        )
                        if not all(math.isfinite(v) for v in (value, lin, ang)):
                            raise ValueError("Non-finite held-out evaluation")
                        reward_sum += value
                        linear += lin
                        angular += ang
                        state = env.state
                        if state is None:
                            raise RuntimeError("Evaluation environment has no state after stepping")
                        terminated = bool(np.asarray(state.terminated)[0])
                        if bool(done.item()):
                            break
                episodes.append(
                    {
                        "seed": seed,
                        "episode": episode,
                        "episode_return": reward_sum,
                        "length_steps": step,
                        "survival_fraction": step / horizon,
                        "survived_horizon": step == horizon and not terminated,
                        "tracking_lin_vel": linear / step,
                        "tracking_ang_vel": angular / step,
                    }
                )
        finally:
            env.close()
    metrics = {
        name: statistics.mean(row[name] for row in episodes)
        for name in (
            "episode_return",
            "survival_fraction",
            "tracking_lin_vel",
            "tracking_ang_vel",
            "survived_horizon",
        )
    }
    return {
        "status": "completed",
        "backend": "mujoco",
        "split": split,
        "seeds": study[f"{split}_seeds"],
        "checkpoint_sha256": checkpoint_digest(checkpoint),
        "episodes": episodes,
        "metrics": metrics,
        "protocol": "seeded_common_task_deterministic_cpu_policy_complete_episodes",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser(
        "plan", help="Write an immutable experiment protocol without starting training"
    )
    plan.add_argument("--study", type=Path, required=True)
    plan.add_argument("--task", default="g1_walk_flat")
    plan.add_argument("--backends", default=",".join(DEFAULT_BACKENDS))
    plan.add_argument("--seeds", default="11,22,33")
    plan.add_argument("--confirmation-seeds", default="101,102,103,104,105,106,107,108,109,110")
    plan.add_argument("--validation-seeds", default="1001,1002,1003")
    plan.add_argument("--test-seeds", default="2001,2002,2003")
    plan.add_argument("--num-envs", type=int, default=2048)
    plan.add_argument("--rollout-steps", type=int, default=24)
    plan.add_argument("--iterations", type=int, default=2200)
    plan.add_argument("--episodes", type=int, default=5)
    plan.add_argument("--device", default="cuda:0")
    plan.add_argument("--mix-config", type=Path)
    plan.add_argument("--fixed", action="append", default=[])
    plan.add_argument("--override", action="append", default=[])
    for command in ("run", "preflight", "report"):
        child = sub.add_parser(command)
        child.add_argument("--study", type=Path, required=True)
    evaluator = sub.add_parser("evaluate", help="Checkpoint evaluation normally launched by run")
    evaluator.add_argument("--study", type=Path, required=True)
    evaluator.add_argument("--checkpoint", type=Path, required=True)
    evaluator.add_argument("--split", choices=("validation", "test"), required=True)
    evaluator.add_argument("--output", type=Path, required=True)
    evaluator.add_argument("--identity", required=True)
    args = parser.parse_args(argv)
    root = args.study.expanduser().resolve()
    if args.action == "plan":
        if root.exists() and any(root.iterdir()):
            raise ValueError("Study directory must be empty")
        study = plan_study(args)
        write_json(root / "study.json", study)
        search_runs = (len(study["candidates"]) + 1) * len(study["training_seeds"])
        confirmation_runs = (len(study["backends"]) + 1) * len(study["confirmation_seeds"])
        print(
            f"Planned {search_runs} discovery/refit runs + {confirmation_runs} fresh confirmation runs; "
            f"{study['transitions']} transitions per run. Protocol: {root / 'study.json'}"
        )
        return 0
    study = load_study(root)
    if args.action == "report":
        result = report(root, study)
        write_json(root / "report.json", result)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "completed" else 2
    if args.action == "evaluate":
        if args.split == "test":
            if (
                not (root / "selection.json").is_file()
                or not (root / "confirmation_protocol.json").is_file()
            ):
                raise ValueError(
                    "Test evaluation is locked until validation selection and confirmation protocol are frozen"
                )
            protocol = read_json(root / "confirmation_protocol.json")
            expected = {
                root
                / "confirmation"
                / candidate["name"]
                / f"seed_{seed}"
                / "train"
                / f"model_{study['iterations'] - 1}.pt"
                for candidate in protocol["candidates"]
                for seed in study["confirmation_seeds"]
            }
            if args.checkpoint.resolve() not in expected:
                raise ValueError(
                    "Test evaluation requires a freshly trained confirmation checkpoint"
                )
        result = evaluate_checkpoint(study, args.checkpoint.resolve(), args.split)
        result["identity"] = args.identity
        write_json(args.output, result)
        return 0
    # Prevent two orchestrators racing the same study. A live process lock is never overwritten.
    import fcntl

    root.mkdir(parents=True, exist_ok=True)
    with (root / "orchestrator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_study(root, study, preflight_only=args.action == "preflight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

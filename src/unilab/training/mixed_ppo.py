"""UniLab assembly of the injected mixed-backend PPO runtime."""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def resolve_mixed_resume(cfg: DictConfig) -> Path | None:
    from unilab.training import parse_checkpoint_path

    if cfg.algo.resume_path:
        explicit = Path(str(cfg.algo.resume_path)).expanduser().resolve()
        if not explicit.is_file():
            raise ValueError(f"Explicit mixed checkpoint not found: {explicit}")
        return explicit
    if cfg.algo.resume or str(cfg.algo.load_run) != "-1":
        result, _ = parse_checkpoint_path(cfg, root_dir=Path.cwd())
        if result is None:
            raise ValueError("Requested mixed checkpoint was not found")
        return result
    return None


def run_mixed_training(cfg: DictConfig) -> None:
    from uni_rl.algos.rsl_rl import (
        RslRlVecEnvWrapper,
        normalize_ppo_train_cfg,
        resolve_rsl_rl_device,
    )
    from uni_rl.algos.rsl_rl_runtime import resolve_rsl_rl_ppo_runtime
    from uni_rl.ipc.dp_launcher import resolve_dp_topology
    from uni_rl.ipc.mix_schedule import allocate_counts, parse_mix
    from uni_rl.ipc.mixed_env import MixedProcessEnv

    from unilab.training import algo_config_dict, get_log_root, parse_checkpoint_path
    from unilab.training.experiment import (
        ExperimentTracker,
        patch_rsl_rl_action_std_logging,
        patch_rsl_rl_wandb_writer,
    )
    from unilab.training.mixed_config import (
        build_worker_specs,
        contract_fingerprint,
        load_mix_config,
        ordered_contract,
    )
    from unilab.utils.device import get_default_device
    from unilab.utils.seed import apply_configured_training_seed

    devices = resolve_dp_topology(cfg.training.devices)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or (devices and len(devices) > 1):
        raise ValueError(
            "Mixed PPO supports one learner only; assign simulator GPUs in --mix-config"
        )
    if cfg.training.play_only:
        raise ValueError("Evaluate mixed policies with --sim <backend> --profile mixed")
    if not cfg.training.no_play or cfg.training.play_render_mode != "none":
        raise ValueError("Mixed training is headless; use separate --profile mixed evaluation")
    if (
        OmegaConf.select(cfg, "algo.runtime_resolver")
        != "uni_rl.algos.mixed_ppo:resolve_mixed_ppo_runtime"
    ):
        raise ValueError("Mixed PPO requires the standard mixed runtime resolver")
    if type(cfg.algo.num_envs) is not int or cfg.algo.num_envs < 1:
        raise ValueError("algo.num_envs must be a positive total environment count")
    raw_ratios = OmegaConf.select(cfg, "training.mix_ratios")
    if not isinstance(raw_ratios, str):
        raise ValueError("Select backends with --sim-mix")
    ratios = parse_mix(raw_ratios)
    counts = allocate_counts(ratios, cfg.algo.num_envs)
    settings = load_mix_config(
        OmegaConf.select(cfg, "training.mix_config"), tuple(ratios), cfg.algo.num_envs
    )
    device = resolve_rsl_rl_device(
        configured_device=cfg.training.device,
        devices=devices,
        world_size=1,
        local_rank=0,
        default_device=get_default_device(),
    )
    if device == "cuda":
        device = "cuda:0"
    seed_info = apply_configured_training_seed(cfg, torch_runtime=True, cuda=True)
    fingerprint = contract_fingerprint(cfg)
    resume_path = resolve_mixed_resume(cfg)
    configured_dir = OmegaConf.select(cfg, "training.log_dir")
    log_dir = (
        Path(configured_dir)
        if configured_dir
        else (
            get_log_root(Path.cwd(), cfg)
            / str(cfg.training.task_name)
            / (datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f") + "_mixed")
        )
    )
    if log_dir.exists() and any(log_dir.iterdir()):
        raise ValueError(
            f"Refusing to overwrite a nonempty run directory: {log_dir}. Resume into a new directory."
        )
    log_dir.mkdir(parents=True, exist_ok=True)
    if cfg.training.nan_guard.output_dir is None:
        cfg.training.nan_guard.output_dir = str((log_dir / "nan_dumps").resolve())
    specs = build_worker_specs(cfg, ratios, settings, device)
    print(f"Mixed PPO run: {log_dir.resolve()}")
    print(f"Requested ratios: {ratios}; environment counts: {counts}; learner: {device}")
    rl_cfg = algo_config_dict(cfg)
    train_cfg = normalize_ppo_train_cfg(rl_cfg)
    train_cfg["mixed"] = {
        "initial_ratios": ratios,
        "stages": settings["stages"],
        "adaptive": settings["adaptive"],
        "fingerprint": fingerprint,
        "control_file": str((log_dir / "control.json").resolve()),
    }
    logger = str(cfg.training.logger)
    if logger not in {"tensorboard", "wandb"}:
        raise ValueError(
            "Mixed PPO requires tensorboard or wandb logging for checkpoints and backend metrics"
        )
    train_cfg["logger"] = logger
    train_cfg.setdefault("runner", {})["logger"] = logger
    tracker = ExperimentTracker(
        root_dir=Path.cwd(),
        log_dir=log_dir,
        algo_name="ppo",
        task_name=cfg.training.task_name,
        sim_backend="mixed",
        training_cfg=cfg.training,
        full_cfg=cfg,
        device=device,
        seed_info=seed_info,
    )
    tracker.start()
    (log_dir / "mixed_config.json").write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "contract": ordered_contract(cfg),
                "initial_ratios": ratios,
                "settings": settings,
                "workers": [
                    {
                        "backend": spec.name,
                        "process_env": spec.process_env,
                        "cpu_ids": spec.cpu_ids,
                        "timeout_s": spec.timeout_s,
                        "seed": spec.seed,
                    }
                    for spec in specs
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    env = None
    try:
        from unilab.assets.hub import ensure_robot_assets_for_paths

        ensure_robot_assets_for_paths([str(cfg.env.scene.model_file)])
        env = MixedProcessEnv(specs, counts)
        runtime = resolve_rsl_rl_ppo_runtime(train_cfg, default_wrapper_cls=RslRlVecEnvWrapper)
        wrapped_env = runtime.wrapper_cls(env, device=device)
        if logger == "wandb":
            patch_rsl_rl_wandb_writer()
            for key, value in tracker.wandb_settings.items():
                train_cfg[f"wandb_{key}"] = value
        if runtime.runner_cls is None:
            raise RuntimeError("Mixed PPO runner was not resolved")
        runner = runtime.runner_cls(wrapped_env, train_cfg, log_dir=str(log_dir), device=device)
        patch_rsl_rl_action_std_logging(runner)
        if resume_path is not None:
            runner.load(str(resume_path), map_location=device)
        iterations = int(cfg.algo.max_iterations)
        if cfg.training.num_timesteps is not None:
            samples = int(cfg.algo.num_envs) * int(cfg.algo.num_steps_per_env)
            iterations = max(1, int(cfg.training.num_timesteps) // samples)
        if iterations < 1:
            raise ValueError("Training requires at least one PPO iteration")
        runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=resume_path is None)
        tracker.update_summary(
            {
                "status": "completed",
                "next_iteration": runner.next_iteration,
                "backend_counts": env.backend_counts,
                "backend_stats": env.get_stats(),
                "samples_per_iteration": cfg.algo.num_envs * cfg.algo.num_steps_per_env,
                "mixed_fingerprint": fingerprint,
                "adaptive": runner.adaptive.state_dict() if runner.adaptive is not None else None,
                "held_out_backends": settings["held_out_backends"],
            }
        )
    except BaseException as exc:
        tracker.update_summary({"status": "failed", "error": str(exc)})
        raise
    finally:
        if env is not None:
            env.close()
        tracker.finish()

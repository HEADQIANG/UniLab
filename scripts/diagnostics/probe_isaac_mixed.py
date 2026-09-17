"""Short G1 common-profile rollout using an existing external Isaac runtime."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("isaacgym", "isaacsim"), required=True)
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--isaacgym-source", type=Path)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.steps <= 64:
        parser.error("--steps must be between 1 and 64 for this bounded probe")
    if not 1 <= args.timeout <= 300:
        parser.error("--timeout must be between 1 and 300 seconds")
    worker_python = args.worker_python.expanduser().absolute()
    if not worker_python.is_file():
        parser.error(f"worker interpreter does not exist: {worker_python}")
    if args.backend == "isaacgym" and args.isaacgym_source is None:
        parser.error("IsaacGym requires --isaacgym-source containing its python directory")

    started = time.monotonic()
    report: dict[str, Any] = {
        "backend": args.backend,
        "task": "g1_walk_flat",
        "profile": "mixed",
        "worker_python": str(worker_python),
        "num_envs": 1,
        "requested_steps": args.steps,
        "completed_steps": 0,
        "status": "starting",
    }
    previous: dict[str, str | None] = {}

    def set_process_env(name: str, value: str) -> None:
        previous.setdefault(name, os.environ.get(name))
        os.environ[name] = value

    env: Any = None
    try:
        with contextlib.ExitStack() as stack:
            temporary = Path(
                stack.enter_context(tempfile.TemporaryDirectory(prefix="unilab-isaac-probe-"))
            )
            # These are process-local launch settings; installed environments
            # and the user's HOME are never modified by the probe.
            set_process_env("PYTHONDONTWRITEBYTECODE", "1")
            set_process_env("OMP_NUM_THREADS", "1")
            set_process_env("MKL_NUM_THREADS", "1")
            set_process_env("TORCH_EXTENSIONS_DIR", str(temporary / "torch_extensions"))
            if args.backend == "isaacsim":
                set_process_env("UNISIM_ISAACSIM_PYTHON", str(worker_python))
            else:
                source = args.isaacgym_source.expanduser().resolve()
                if not (source / "python" / "isaacgym").is_dir():
                    raise ValueError(f"IsaacGym package missing beneath {source / 'python'}")
                env_root = worker_python.parent.parent
                if not (env_root / "lib").is_dir():
                    raise ValueError(
                        f"Worker environment library directory missing: {env_root / 'lib'}"
                    )
                runtime_home = temporary / "isaacgym_runtime"
                conda_envs = runtime_home / "miniconda3" / "envs"
                conda_envs.mkdir(parents=True)
                (conda_envs / "hsgym").symlink_to(env_root, target_is_directory=True)
                (runtime_home / "isaacgym").symlink_to(source, target_is_directory=True)
                set_process_env("UNISIM_ISAACGYM_HOME", str(runtime_home))
                set_process_env("UNISIM_ISAACGYM_PYTHON", str(worker_python))
                report["isaacgym_source"] = str(source)

            import numpy as np
            from omegaconf import open_dict

            from unilab.base.config_adapter import BackendAdapter
            from unilab.base.env_factory import make_registry_env
            from unilab.training.mixed_config import compose_mixed_profile, mixed_env_metadata

            cfg = compose_mixed_profile("g1_walk_flat", args.backend)
            with open_dict(cfg.env):
                cfg.env[f"{args.backend}_worker_timeout_s"] = args.timeout
                if args.backend == "isaacsim":
                    cfg.env.isaacsim_render_mode = "none"
            overrides = BackendAdapter(
                cfg, root_dir=Path.cwd(), algo_name="ppo"
            ).build_task_env_cfg_override()
            report["status"] = "constructing"
            print(json.dumps(report), flush=True)
            try:
                env = make_registry_env(str(cfg.training.task_name), args.backend, 1, overrides)
                obs, _info = env.reset(seed=17)
                report["observation_shapes"] = {
                    name: list(value.shape) for name, value in obs.items()
                }
                if not all(np.isfinite(value).all() for value in obs.values()):
                    raise ValueError("Reset produced non-finite observations")
                report["metadata"] = mixed_env_metadata(env)
                action_dim = env.action_manager.total_action_dim
                report["action_dim"] = action_dim
                if action_dim != 29:
                    raise ValueError(f"Expected the G1 29-DoF contract, received {action_dim}")
                actions = np.zeros((1, action_dim), dtype=np.float32)
                rewards: list[float] = []
                resets = 0
                for step in range(args.steps):
                    state = env.step(actions)
                    if not np.isfinite(state.reward).all() or not all(
                        np.isfinite(value).all() for value in state.obs.values()
                    ):
                        raise ValueError(f"Non-finite reward or observation at step {step}")
                    rewards.append(float(state.reward[0]))
                    report["completed_steps"] = step + 1
                    if bool(state.terminated[0] or state.truncated[0]):
                        env.reset()
                        resets += 1
                # Explicit reset after stepping verifies that the reset path
                # is reusable even when this short sequence has no episode end.
                reset_obs, _info = env.reset()
                if not all(np.isfinite(value).all() for value in reset_obs.values()):
                    raise ValueError("Post-rollout reset produced non-finite observations")
                report.update(status="passed", rewards=rewards, episode_resets=resets)
            finally:
                if env is not None:
                    env.close()
                    env = None
    except Exception as exc:
        report.update(
            status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()
        )
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    report["elapsed_seconds"] = time.monotonic() - started
    serialized = json.dumps(report, indent=2, allow_nan=False)
    print(serialized, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

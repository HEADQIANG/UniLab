"""Run a command with existing Isaac runtimes without changing their installs."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaacsim-python", type=Path)
    parser.add_argument("--isaacgym-python", type=Path)
    parser.add_argument("--isaacgym-source", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("supply the command after --")
    if args.isaacsim_python is None and args.isaacgym_python is None:
        parser.error("select at least one existing Isaac runtime")
    if (args.isaacgym_python is None) != (args.isaacgym_source is None):
        parser.error("IsaacGym requires both --isaacgym-python and --isaacgym-source")
    for selected in (args.isaacsim_python, args.isaacgym_python):
        if selected is not None and not selected.expanduser().is_file():
            parser.error(f"worker interpreter does not exist: {selected}")

    child_env = dict(os.environ)
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryDirectory(prefix="unilab-existing-isaac-") as directory:
        runtime_root = Path(directory)
        # All spawned collectors inherit these stable paths for the entire
        # command lifetime, including worker reallocation during adaptive PPO.
        child_env.setdefault("TORCH_EXTENSIONS_DIR", str(runtime_root / "torch_extensions"))
        if args.isaacsim_python is not None:
            child_env["UNISIM_ISAACSIM_PYTHON"] = str(args.isaacsim_python.expanduser().absolute())
        if args.isaacgym_python is not None:
            worker_python = args.isaacgym_python.expanduser().absolute()
            source = args.isaacgym_source.expanduser().resolve()
            if not (source / "python" / "isaacgym").is_dir():
                parser.error(f"IsaacGym package missing beneath {source / 'python'}")
            env_root = worker_python.parent.parent
            if not (env_root / "lib").is_dir():
                parser.error(f"worker environment library directory missing: {env_root / 'lib'}")
            runtime_home = runtime_root / "isaacgym"
            conda_envs = runtime_home / "miniconda3" / "envs"
            conda_envs.mkdir(parents=True)
            (conda_envs / "hsgym").symlink_to(env_root, target_is_directory=True)
            (runtime_home / "isaacgym").symlink_to(source, target_is_directory=True)
            child_env["UNISIM_ISAACGYM_HOME"] = str(runtime_home)
            child_env["UNISIM_ISAACGYM_PYTHON"] = str(worker_python)

        # A separate process group lets interruption reach the experiment and
        # its spawned workers before this launcher's temporary paths disappear.
        process = subprocess.Popen(command, env=child_env, start_new_session=True)

        def forward_signal(signum: int, _frame: object) -> None:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass

        old_handlers = {
            signum: signal.signal(signum, forward_signal)
            for signum in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            returncode = process.wait()
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
    return returncode if returncode >= 0 else 128 - returncode


if __name__ == "__main__":
    raise SystemExit(main())

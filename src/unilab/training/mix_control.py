"""Atomic manual ratio control for one running mixed PPO experiment."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    from uni_rl.ipc.mix_schedule import atomic_write_control, parse_mix

    parser = argparse.ArgumentParser(prog="unilab-mix-control")
    parser.add_argument("--run", required=True, help="Mixed PPO run directory")
    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument("--ratios", help="Complete backend weights, e.g. mujoco=0.5,motrix=0.5")
    command.add_argument("--clear", action="store_true", help="Resume the current scheduled ratios")
    args = parser.parse_args(argv)
    try:
        result = atomic_write_control(
            args.run, ratios=parse_mix(args.ratios) if args.ratios else None, clear=args.clear
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Queued ratio control for the next PPO boundary: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

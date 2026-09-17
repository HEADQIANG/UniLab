# Mixed-backend PPO runtime

This runtime accepts injected, spawn-picklable environment factories. The learner
does not import the task or physics packages. Use the UniLab consumer CLI to choose
task-compatible backends, worker devices, stages, and total environment count.

The runtime resolver is `uni_rl.algos.mixed_ppo:resolve_mixed_ppo_runtime`.
It requires **`rsl-rl-lib==5.0.1`**, since iteration and logger integration follows
that exact upstream implementation. Other existing runners are unchanged.
Normalized runner configuration contains `mixed.initial_ratios` (ordered backend
weights), `mixed.stages` (`iteration` and complete `ratios` mappings), a nonempty
`mixed.fingerprint`, and optional `mixed.control_file`. Stage iterations are
absolute, zero-based, ascending and unique. `num_envs * num_steps_per_env` must be
divisible by PPO's minibatch count. Only one learner and feedforward PPO are
supported. RND, symmetry and recurrent policies are rejected.

Weights are normalized and apportioned by largest remainder, breaking ties by
startup declaration order. Every positive backend requires at least one slot.
Weights of zero disable a declared backend. All environments take the same number
of steps per iteration, so integer slot proportions are actual sample proportions.

## Manual control and resume

The consumer's `unilab-mix-control` command calls
`uni_rl.ipc.mix_schedule.atomic_write_control(run_dir, ratios=...)`, or passes
`clear=True`. The helper atomically writes a versioned request, validates against
the run manifest when present, and follows its custom control file location.
All requests contain the complete startup backend set, including zero weights.
Manual ratios persist across stage boundaries until explicitly cleared. Requests
are applied only after the prior PPO update and logging have finished.

Changed integer counts rebuild all worker environments, reset unfinished episode
statistics and refresh observations. Networks, normalizers and optimizer survive
unchanged. Physical state is not transferred or restored. Checkpoints record the
next iteration, schedule, manual override/version, common task fingerprint and
learner random states. Older requests are ignored on resume. A failed rebuild or
worker failure aborts training; a malformed manual request is logged and ignored.
RNG metadata uses only primitive values, lists and tensors, so existing policy
evaluation readers can inspect new checkpoints with `torch.load(weights_only=True)`.
Trusted training resume also accepts earlier mixed checkpoints containing a raw
NumPy RNG array; regenerate those checkpoints before using weights-only evaluation.

## Developer verification

Install development dependencies with `uv sync`, then run:

```bash
uv run --with rsl-rl-lib==5.0.1 pytest tests/ipc/test_mix_schedule.py tests/algos/test_mixed_ppo.py
uv run ruff check src/uni_rl/ipc/mix_schedule.py src/uni_rl/algos/mixed_ppo.py
```

If a machine-wide ROS pytest plugin conflicts with the project's pytest version,
prefix the test command with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

Synthetic tests establish scheduling and PPO correctness, not simulator support
or convergence. Real-backend validation belongs to the consumer's task suite.

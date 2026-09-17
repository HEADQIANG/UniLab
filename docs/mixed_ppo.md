# Mixed-backend PPO

Reward-driven automatic allocation and a MuJoCo holdout are now available;
see [adaptive mixed PPO](adaptive_mixed_ppo.md) for launch and resume steps,
and [comparison experiments](mix_benchmark.md) for ratio selection.

This source workflow includes `UniLab` at base commit `9e3bb6b8` and the modified
`unilab-rl` at base tag `v1.1.1` in `vendor/unilab-rl`. The editable
`tool.uv.sources` binding in UniLab keeps `uv run` on the local algorithm
runtime. RSL-RL is pinned to 5.0.1; the PPO objective is unchanged.

## Installation

On the experiment host the project is `/home/hpf/yuelk/UniLab`; its algorithm
package is included under `vendor/unilab-rl`. Run from UniLab (add `$HOME/.local/bin` to PATH
if `uv` is not found by a non-interactive SSH shell):

```bash
cd /home/hpf/yuelk/UniLab
uv sync --extra mujoco --extra motrix
uv run python -c 'import uni_rl; print(uni_rl.__file__)'
```

The printed module must belong to `vendor/unilab-rl/src` in this checkout.
See [complete reproduction steps](reproduce_mixed_study.md) to install all
training backends; use `uv run --no-sync` after preparing the native runtimes.
Other backends require their existing optional dependencies and external
workers; see the backend guides. Selecting a zero-weight backend still
requires its dependency checks because it can be enabled later.

## Fixed Ratios

```bash
uv run train --algo ppo --task go2_joystick_flat \
  --sim-mix mujoco=0.7,motrix=0.3 algo.num_envs=1000
```

One learner runs PPO on a combined batch of 700 MuJoCo and 300 Motrix
environments. Every environment contributes the same rollout length.
Weights are normalized; integer counts use largest remainders (ties follow
declaration order). Positive weights that receive zero environments are
rejected. Total environments remain fixed, and `num_envs * num_steps_per_env`
must divide evenly into `algo.algorithm.num_mini_batches`.

Only standard feedforward PPO and one learner are supported. The existing
`--sim` commands remain unchanged. Mixed training is headless; all workers
finish each simulation step before the learner continues. A slow backend
therefore limits throughput.

## Matched Training Seeds

Set `algo.seed` identically for matched single-backend and mixed runs.
Each worker derives its base seed from the training seed and backend name,
independently of its position in `--sim-mix`. Version 1 uses the first four
bytes of SHA-256 of the UTF-8 string `unilab-mixed-worker-v1:<backend>`,
interpreted as an unsigned big-endian integer, as a backend offset. The worker
seed is `(algo.seed + offset) % 2**32`. Backend names are the registry names
shown in `--sim-mix`; Python's process-randomized `hash()` is not used.
`mixed_config.json` records the resolved seed for every worker.

The same backend therefore starts with the same seed in a single run, a mixed
run, or reordered backend declarations. Each current backend has a distinct
offset. A rebuild uses the existing collector generation offset of
`generation * 1_000_003` modulo `2**32`; different environment counts or
rebuild schedules can still produce different later random sequences.
The learner keeps its original `algo.seed` initialization.

This replaces the earlier order-dependent rule `algo.seed + index * 10007`.
Reproducing an old run requires its original code revision as well as its
saved configuration. Existing policy checkpoints remain usable for inference;
resuming training under the new rule resets environments using the new worker
seeds, so it does not reproduce the old random stream. Do not mix old-rule and
new-rule trials in one benchmark study; create a new study after this change.

## Devices and Stages

Pass `--mix-config docs/examples/mixed_go2.yaml`. The file supports:

```yaml
backends:
  mujoco: {device: cpu, timeout_s: 120}
  motrix: {device: cpu, timeout_s: 120}
stages:
  - iteration: 500
    ratios: {mujoco: 0.5, motrix: 0.5}
  - iteration: 1000
    ratios: {mujoco: 0.2, motrix: 0.8}
```

Each backend may additionally set `cpu_ids: [0, 1]` and `options` containing
only its documented simulator runtime fields. CPU ids must be available to
the process. GPU backends use `device: cuda:N`, with N relative to the
launching process's `CUDA_VISIBLE_DEVICES`. Each worker sees only its assigned
GPU as `cuda:0`. Set the learner separately with `training.device=cuda:0`.
All stage ratios must name the complete declared backend set; zero pauses a
backend. Stages use absolute, zero-based iteration numbers across resumes.

## Manual Changes

The training process prints its run directory. In another terminal:

```bash
uv run unilab-mix-control --run logs/rsl_rl_ppo/Go2JoystickFlat/RUN_NAME \
  --ratios mujoco=0.4,motrix=0.6
uv run unilab-mix-control --run logs/rsl_rl_ppo/Go2JoystickFlat/RUN_NAME --clear
```

The command atomically writes a versioned control request. Manual ratios
persist until cleared, including across stage boundaries. Clearing resumes
the stage applicable to the current iteration. Rebalancing only happens
after the previous rollout has been consumed by PPO. Changed integer counts
rebuild all sampling environments and restart episodes, while retaining the
policy, optimizer and normalization statistics. Identical counts do not
rebuild. Initialization may take substantial time for GPU engines.

Invalid requests are recorded and ignored. Worker failure, timeout or a failed
rebuild stops the whole run; the runtime never silently changes sample ratios.

## Checkpoints and Evaluation

Resume into a new output directory with the original task, backend set,
initial ratios, rollout settings and stage file, using the existing explicit
checkpoint override:

```bash
uv run train --algo ppo --task go2_joystick_flat \
  --sim-mix mujoco=0.7,motrix=0.3 algo.num_envs=1000 \
  algo.resume_path=/absolute/path/to/model_10.pt
```

The checkpoint restores the next iteration, schedule/manual override state,
and learner random state. Old control versions are ignored. Environment
state is reset: exact physical-trajectory reproduction is not promised.
Do not reuse a nonempty output directory. A changed common task fingerprint
is rejected on resume.

Evaluate the same checkpoint separately on each backend, using the common
profile, not the differently tuned original owner:

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --profile mixed \
  --load-run RUN_NAME --render-mode record
uv run eval --algo ppo --task go2_joystick_flat --sim motrix --profile mixed \
  --load-run RUN_NAME --render-mode record
```

Go2 supports MuJoCo/Motrix/Drake common profiles; G1 supports
MuJoCo/Motrix/Drake/mjwarp/IsaacGym/IsaacSim/Genesis/Newton. All profiles disable
PD randomization and random root resets; Go2 disables the contact reward,
G1 disables feet-air-time reward. Observation noise remains shared. No backend
identity is added to policy observations. Runtime checks reject missing
capabilities or mismatched ordered observation/action metadata.
Custom action terms without the standard `BaseAction` target/scale/offset
metadata are rejected before sampling.
Existing `training.nan_guard` settings apply independently in every worker;
default dumps are stored in the run directory's `nan_dumps` subdirectory.

`run_config.json`, `mixed_config.json`, `mix_manifest.json`,
`mix_events.jsonl`, TensorBoard/W&B and `run_summary.json` record config,
effective counts, per-backend sampling/episode statistics and control events.
Configured support alone does not establish successful training or convergence.

## Development Checks

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/config/test_mixed_profiles.py tests/scripts/test_mixed_cli.py tests/training/test_mixed_config.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync pytest vendor/unilab-rl/tests/test_mixed_env.py vendor/unilab-rl/tests/ipc/test_mix_schedule.py vendor/unilab-rl/tests/algos/test_mixed_ppo.py -q
```

Disabling third-party pytest plugin autoload avoids unrelated ROS plugin
incompatibilities. Run both repositories' normal formatting/type/test gates
after focused checks. Hardware tests must report unavailable engines as
unverified, not as passing; short smoke tests do not measure convergence.

## Short Integration Run

For a small CPU test of stage changes, run from UniLab:

```bash
uv run train --algo ppo --task go2_joystick_flat \
  --sim-mix mujoco=0.5,motrix=0.5 \
  --mix-config docs/examples/mixed_go2_smoke.yaml \
  algo.num_envs=8 algo.num_steps_per_env=4 algo.max_iterations=2 \
  'algo.policy.actor_hidden_dims=[32,32]' \
  'algo.policy.critic_hidden_dims=[32,32]' \
  training.device=cpu training.log_dir=/tmp/mixed-go2-stage
```

The two iterations collect 4:4 then 2:6 environments. Resume with the same
arguments plus `algo.resume_path=/tmp/mixed-go2-stage/model_1.pt` and a new
`training.log_dir=/tmp/mixed-go2-resume`. Two additional iterations start at
absolute iteration 2 and switch to 6:2 at iteration 3. Each iteration has 32
fresh samples. Check `mix_events.jsonl` for allocations and rebuilds.
These small networks also require matching network overrides for evaluation.

If the official Hugging Face endpoint is unreachable, asset downloads can use
the optional public mirror for this command only:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 uv run train ...
```

This uses a third-party mirror; normal operation uses the official endpoint.
Do not replace package indexes or modify global network settings.

Architectural decision: [mixed PPO ADR](ADR-mixed-ppo.md).
Executed tests and unavailable backend combinations:
[2026-09-14 validation record](mixed_ppo_validation.md).

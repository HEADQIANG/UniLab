# Reproduce the G1 mixed-backend PPO study

This fork contains both the UniLab environment/configuration layer and its
modified algorithm runtime at `vendor/unilab-rl`. A single clone supplies the
experiment code; the simulator runtimes and robot assets are installed
separately. See [vendor provenance](../vendor/README.md) for the exact upstream
algorithm source and retained Apache-2.0 license.

## Install the source checkout

The full setup below targets Ubuntu 24.04 x86_64, Python 3.12, a working NVIDIA
CUDA driver, and enough GPU memory for all seven training simulators to coexist.
The supplied native Drake bootstrap is specific to this platform. IsaacGym and
IsaacSim use their own supported Python environments; they do not run inside
the project's Python 3.12 environment.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
git clone --branch codex/adaptive-g1-multi-backend https://github.com/HEADQIANG/UniLab.git
cd UniLab
uv sync --locked --python 3.12 \
  --extra mujoco --extra motrix --extra mjwarp --extra genesis --extra newton
uv run --no-sync python -c 'from pathlib import Path; import uni_rl; actual = Path(uni_rl.__file__).resolve(); expected = Path("vendor/unilab-rl/src/uni_rl/__init__.py").resolve(); print(actual); assert actual == expected'
```

For an exact rerun, check out the commit recorded with the published results
before installing. Keep the clone location stable for an active study because
the benchmark records resolved package paths as well as source hashes.

Plain `pip install .` ignores `tool.uv.sources` and can install the unmodified
upstream `unilab-rl` release. Use the root `uv` workflow above for these
experiments. Do not independently synchronize `vendor/unilab-rl`; the root lock
pins the shared dependencies, including RSL-RL 5.0.1.

Add the native Drake runtime after the initial sync:

```bash
bash scripts/diagnostics/setup_drake_mixed.sh
```

This helper downloads pinned Drake/DrakeUni source artifacts and builds the
native pool locally. Its prerequisites and library-path wrapper are documented
in [Drake setup](drake_mixed_setup.md). After this step, keep using
`uv run --no-sync`: a later ordinary sync can replace the locally built DrakeUni
package or remove separately installed simulator dependencies.

Install IsaacGym and IsaacSim according to their official requirements and the
project backend guides. Use [the existing-runtime probes](isaac_mixed_probe.md)
to verify their interpreters and IsaacGym source directory. Set these three
paths to the installations on your machine:

```bash
export STUDY_ISAACSIM_PYTHON=/absolute/path/to/isaacsim/environment/bin/python
export STUDY_ISAACGYM_PYTHON=/absolute/path/to/isaacgym/environment/bin/python
export STUDY_ISAACGYM_SOURCE=/absolute/path/to/isaacgym
```

The launcher below creates the adapter discovery layout for the lifetime of
each command. Robot meshes/textures are fetched through the registered asset
hub on first use; an uncached installation needs access to those assets.

## Check the code and all simulators

Run from the UniLab root after installing dependencies:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync pytest \
  tests/config/test_mixed_profiles.py tests/scripts/test_mixed_cli.py \
  tests/training/test_mixed_config.py tests/training/test_mix_benchmark.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync pytest \
  vendor/unilab-rl/tests/test_smoke.py \
  vendor/unilab-rl/tests/test_mixed_env.py \
  vendor/unilab-rl/tests/ipc/test_mix_schedule.py \
  vendor/unilab-rl/tests/algos/test_mixed_ppo.py \
  vendor/unilab-rl/tests/algos/test_adaptive_mix.py -q
```

These tests check contracts, scheduling, failure handling, and the experimental
protocol. Actual simultaneous simulator use is checked by `preflight` below,
which performs a shared-policy PPO update across the seven training backends
and evaluates the resulting checkpoint in MuJoCo. Passing unit tests or this
short preflight does not establish walking quality or a training advantage.

For the full repository gate on this native Drake installation, keep the
runtime library wrapper on the test command as well as on training:

```bash
UV_NO_SYNC=1 make check
bash scripts/diagnostics/setup_drake_mixed.sh --run \
  env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync pytest -m 'not slow'
```

Without the wrapper, installed Drake tests can fail at import with missing
`libfmt.so.9` even though the simulator is correctly installed. The wrapper
only supplies the runtime library paths; it does not change test selection.

## Create the comparison and run it

Review `docs/examples/adaptive_g1.yaml` before planning. It assigns the GPU
backends to `cuda:0`; change its declared backend devices/timeouts for your
hardware before creating the study. The common task, reward, policy dimensions,
PPO parameters, total sample budget, and seed pairing are preserved across the
mixed and single-backend candidates.

```bash
uv run --no-sync python -m unilab.training.mix_benchmark plan \
  --study experiments/g1_all_backends \
  --mix-config docs/examples/adaptive_g1.yaml \
  --seeds 11,22,33,44,55 \
  --confirmation-seeds 101,102,103,104,105,106,107,108,109,110 \
  --validation-seeds 1001,1002,1003 --test-seeds 2001,2002,2003 \
  --num-envs 2048 --rollout-steps 24 --iterations 2200 \
  --episodes 5 --device cuda:0

bash scripts/diagnostics/setup_drake_mixed.sh --run \
  uv run --no-sync python scripts/tools/run_with_existing_isaac.py \
  --isaacsim-python "$STUDY_ISAACSIM_PYTHON" \
  --isaacgym-python "$STUDY_ISAACGYM_PYTHON" \
  --isaacgym-source "$STUDY_ISAACGYM_SOURCE" -- \
  uv run --no-sync python -m unilab.training.mix_benchmark preflight \
  --study experiments/g1_all_backends

bash scripts/diagnostics/setup_drake_mixed.sh --run \
  uv run --no-sync python scripts/tools/run_with_existing_isaac.py \
  --isaacsim-python "$STUDY_ISAACSIM_PYTHON" \
  --isaacgym-python "$STUDY_ISAACGYM_PYTHON" \
  --isaacgym-source "$STUDY_ISAACGYM_SOURCE" -- \
  uv run --no-sync python -m unilab.training.mix_benchmark run \
  --study experiments/g1_all_backends

uv run --no-sync python -m unilab.training.mix_benchmark report \
  --study experiments/g1_all_backends
```

All seven training backends are required: Motrix, Drake, MuJoCo-Warp, IsaacGym,
IsaacSim, Genesis, and Newton. MuJoCo is held out of training and adaptive
feedback. A failed dependency, capability check, or GPU allocation stops the
experiment; it does not silently remove a backend. This full configuration is
165 training runs with 108,134,400 transitions per run, plus preflight and
evaluation. Budget hardware/time accordingly; see the
[complete experiment protocol](mix_benchmark.md) for explicit smaller diagnostic
configurations and their limitations.

The adaptive controller uses training reward only. Search candidates include
every single backend, fixed mixed ratios, and an adaptive run whose final ratio
is retrained from scratch. MuJoCo validation scores choose the fixed ratio;
new confirmation training seeds and separate MuJoCo test seeds assess it against
every single backend. Never retune a ratio using the held-out test scores.

## Inspect, preserve, and interpret results

Each study stores `study.json`, `frozen_candidate.json`, `selection.json`,
`confirmation_protocol.json`, and `report.json`, together with per-trial
training metadata, allocation events, checkpoints, and evaluation records.
Preserve the whole study directory outside Git for a resumable experiment.
For publication, include the compact protocol/report/evaluation evidence and
source commit, checkpoint hashes, environment versions, and commands used;
checkpoints, videos, downloaded runtimes, and raw logs are not automatically
part of the source repository.

`report` returns exit code 2 for incomplete or inconsistent evidence. A completed
report can legitimately say `mixed_advantage_not_established`: the code does not
force the hypothesis to be true. The selected ratio is the best observed fixed
candidate under the declared protocol, not a proven global optimum. Compare
paired confirmation training seeds and all single-backend baselines, including
the corrected significance results and the reported timing limitations.

Rerun `run` with the same wrappers to continue valid completed work. Keep failed
trial directories for inspection; follow [failure recovery](mix_benchmark.md)
before retrying. Changes to source, installed physics binaries, dependencies,
task, budget, or seeds require a new study. External Isaac worker environments,
system libraries, and GPU drivers are not fully fingerprinted, so record their
versions and preserve them throughout a study. Identical source and seeds do
not guarantee bitwise-identical GPU trajectories across machines.

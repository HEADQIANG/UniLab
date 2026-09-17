# Probe an existing Isaac runtime with the common G1 task

The bounded diagnostic constructs the actual G1 mixed profile, resets one
environment, runs eight zero-action steps, checks finite observations/rewards
and the 29-action contract, then resets again and closes the worker. It emits a
JSON report containing resolved observation/action/entity metadata. This checks
runtime integration; it does not train or establish locomotion quality.

Run from the UniLab repository root with its existing dependencies installed:

```bash
uv run --no-sync python scripts/diagnostics/probe_isaac_mixed.py \
  --backend isaacsim \
  --worker-python /home/hpf/miniconda3/envs/atec/bin/python \
  --output /tmp/unilab-g1-isaacsim-probe.json

uv run --no-sync python scripts/diagnostics/probe_isaac_mixed.py \
  --backend isaacgym \
  --worker-python /home/hpf/miniconda3/envs/homie/bin/python3.8 \
  --isaacgym-source /home/hpf/atec/MAPush/isaacgym \
  --output /tmp/unilab-g1-isaacgym-probe.json
```

The paths are the existing runtimes discovered on `hpf@100.89.197.45`; replace
them on another host. No packages are installed or changed. IsaacSim uses the
documented `UNISIM_ISAACSIM_PYTHON` override. IsaacGym requires the adapter's
documented home layout in addition to its interpreter override, so the probe
creates temporary symlinks to the existing environment and source package and
sets `UNISIM_ISAACGYM_HOME`. These temporary links and the probe's Torch extension
cache are cleaned up after the worker closes. The user's `HOME` is unchanged.
The external runtimes may produce their normal GPU shader caches and logs.

Run the two probes sequentially on a GPU with enough free memory. The default
worker-operation timeout is 180 seconds; `--timeout` accepts up to 300 seconds.
`--steps` accepts 1–64 for short diagnostics. A failed report includes the
host/worker error and traceback; do not count missing runtimes or failed probes
as successful simulator coverage.

## Use the existing runtimes during training

The separate launcher keeps the temporary discovery layout alive until its
entire child command finishes. This covers the learner, spawned collectors,
adaptive worker rebuilds, and a sequential experiment harness. It forwards
interruptions to the child process group and leaves the installed environments
unchanged. For example, a two-update integration run is:

```bash
uv run --no-sync python scripts/tools/run_with_existing_isaac.py \
  --isaacsim-python /home/hpf/miniconda3/envs/atec/bin/python \
  --isaacgym-python /home/hpf/miniconda3/envs/homie/bin/python3.8 \
  --isaacgym-source /home/hpf/atec/MAPush/isaacgym -- \
  uv run --no-sync train --algo ppo --task g1_walk_flat \
  --sim-mix isaacgym=0.5,isaacsim=0.5 \
  algo.num_envs=8 algo.num_steps_per_env=4 algo.max_iterations=2
```

For the full comparison, use the same launcher prefix with the experiment
command and all seven training backends. Use one launcher per experiment
process; do not copy its temporary environment variables into a later session.

## Observed short-run results, 2026-09-17

On `hpf@100.89.197.45`, both commands above for the individual diagnostic passed:

| Runtime | Result | Elapsed |
| --- | --- | --- |
| IsaacGym 1.0rc4, homie Python 3.8 / Torch 2.0.1+cu118 | Reset, 8 finite steps, second reset, close | 14.33 s |
| IsaacSim 5.1.0.0, atec Python 3.11 / IsaacLab 0.54.3 | Reset, 8 finite steps, second reset, close | 17.27 s |

Both expose observations `(1, 98)`, critic observations `(1, 101)`, and 29
actions. The reported ordered observation/action/entity metadata agrees. The
remote JSON artifacts are `/tmp/unilab-g1-isaacgym-probe-20260917.json` and
`/tmp/unilab-g1-isaacsim-probe-20260917.json`; these temporary files are evidence
for the stated short checks only. Simultaneous multi-backend PPO and convergence
still require separate validation.

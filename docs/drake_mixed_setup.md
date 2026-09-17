# Drake runtime for the mixed G1 experiment

The multi-backend experiment needs the native DrakeUni pool, in addition to
its Python package. On Ubuntu 24.04 x86_64 with the existing project Python 3.12
environment, run from the UniLab repository root:

```bash
bash scripts/diagnostics/setup_drake_mixed.sh
```

The helper downloads Drake 1.56.0 from its official MIT distribution mirror,
checks its published SHA256, and builds the upstream `drake-uni==0.1.0` source
at commit `4cdc9ba4c9b1a7542755631afe0d57dbb54cdb63`. The complete source archive
and build helper have pinned SHA256 checks. This is a deliberate source pin:
the earlier PyPI 0.1.0 package lacks `DrakeModelInfo.actuator_names` and
`joint_body_names`, so current `unisim-core` cannot construct the G1 entity from
it. The fixed upstream source preserves the declared MJCF actuator order and
supplies this metadata; no private adapter monkeypatch is used.
Eigen, fmt, spdlog and Python development headers are downloaded as Ubuntu
packages and unpacked locally. No system package installation is performed.

Only the DrakeUni editable package is added to the project's `.venv`. The script
does not run `uv sync`, install shell completion, or change shell startup files.
This preserves the other installed simulator extras. The existing
`scripts/tools/setup_drake_env.sh` uses a single-extra synchronization and is
therefore not the bootstrap for an already prepared mixed environment.

The runtime defaults to `../unilab-runtimes/drake` relative to the project.
Set `UNILAB_DRAKE_SETUP_HOME` to choose another dedicated runtime directory;
use the same value when subsequently launching commands. Set `UV_BIN` when
the project's `uv` executable is not on the shell path.

## Verify native G1 and launch experiments

The `--run` mode adds the local Drake and support-library directories to the
command's library search path. It starts the supplied command without
downloading, installing or rebuilding anything:

```bash
bash scripts/diagnostics/setup_drake_mixed.sh --run \
  uv run --no-sync train --algo ppo --task g1_walk_flat --sim drake \
  training.no_play=true training.log_root=logs/diagnostics/drake_g1_native \
  algo.num_envs=2 algo.num_steps_per_env=4 algo.max_iterations=1 \
  env.drake_nthread=1
```

Use the same `bash scripts/diagnostics/setup_drake_mixed.sh --run` prefix for
the full adaptive experiment command. Subprocess collectors inherit the
library paths, allowing the native Drake worker to load alongside the other
six training backends. MuJoCo remains the held-out sim2sim evaluator.

To validate only the loader:

```bash
bash scripts/diagnostics/setup_drake_mixed.sh --run \
  uv run --no-sync python -c \
  'from drake_uni.runtime import batch_diagnostics; d = batch_diagnostics(); print(d); assert d.batch_available, d.batch_import_error'
```

A successful import is an installation check. A real G1 rollout and PPO update
are required before reporting runtime support; neither a short smoke test nor
the installation establishes learning convergence or the best simulator ratio.

The native regression checks compare all 29 actuator names and target joints
against the original model's ordered controls, verify the task's `stand` pose, and run a
real G1 PPO iteration. MuJoCo is used only to parse the source model for the
metadata comparison:

```bash
bash scripts/diagnostics/setup_drake_mixed.sh --run \
  uv run --no-sync pytest -q -m slow \
  tests/base/backend/test_drake_g1_native.py \
  tests/scripts/test_drake_training_smoke.py -k g1
```

On the experiment host, the native G1 smoke completed on 2026-09-17 with two
environments, four steps per environment and one PPO update: actor observations
98, critic observations 101, actions 29, and finite reported rewards/losses.
The training artifact is under
`logs/diagnostics/drake_g1_native/G1WalkFlat/2026-09-17_23-15-55_drake`.

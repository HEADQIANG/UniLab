# Mixed PPO validation, 2026-09-14

For the later seven-backend adaptive G1 pipeline and archived evidence, see
[the 2026-09-18 validation report](adaptive_g1_validation_20260918.md).
The dated observations below remain a historical record.

This records implementation smoke/regression evidence, not policy convergence,
sim-to-real transfer, or a claim that every configured backend was executed.
Operation commands are in [mixed_ppo.md](mixed_ppo.md) and
[mixed_profiles.md](mixed_profiles.md).

## Environments

- UniLab base: `9e3bb6b814d694d52bfff0f074d0a2b8ede6f032`.
- Local unilab-rl base: `v1.1.1`, `bc6126658d6673a7361acd09d8dc6e576feff703`.
- Both sources are modified editable checkouts; nothing was published or pushed.
- Local isolated environment: sibling `.venv-test`, Python 3.12,
  Torch 2.8.0+cpu, RSL-RL 5.0.1, unisim-core 1.1.4,
  MuJoCo 3.11.0, mujoco-uni-runtime 0.5.0, Motrix 0.10.0.
- Remote: `hpf@100.89.197.45`, `/home/hpf/yuelk/UniLab/.venv`,
  Python 3.12.3, RTX 4090, Torch 2.8.0+cu128, RSL-RL 5.0.1,
  unisim-core 1.1.4, MuJoCo 3.11.0, mujoco-uni-runtime 0.5.0,
  Motrix 0.8.2 (the repository lock).
- Remote `uni_rl.__file__` resolves into `/home/hpf/yuelk/unilab-rl/src`.
  Other project environments were not modified.

Official public robot assets were fetched with the optional `hf-mirror.com`
endpoint because the official endpoint was unreachable. Meshes remain ignored
asset files, not source additions. Assets and motion files were also needed for
the pre-existing full regression suite.

## Regression Checks

From UniLab, the local isolated command was:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
uv run --cache-dir /tmp/unilab-test-cache --no-project \
  --python ../.venv-test/bin/python -m pytest -m 'not slow' -q --tb=short
```

Result: **1679 passed, 20 skipped, 895 deselected, 1 xfailed**. Earlier runs
failed because assets were missing; the reported result is after fetching the
missing assets and rerunning the complete non-slow suite.

The corresponding complete unilab-rl non-slow suite in the same environment:
**442 passed, 34 skipped, 3 deselected**. Localhost/Gloo tests required execution
outside the filesystem/network sandbox. Focused mixed UniLab tests:
**78 passed**. The existing slow config-compose tests were also run separately:
**168 passed**; the entire slow/hardware suite was not run.

After the final source sync, the combined focused suite also passed on the
remote locked environment: **145 passed in 32.40s**:

```bash
cd /home/hpf/yuelk/UniLab
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync pytest \
  tests/config/test_mixed_profiles.py tests/scripts/test_mixed_cli.py \
  tests/training/test_mixed_config.py ../unilab-rl/tests/test_mixed_env.py \
  ../unilab-rl/tests/ipc/test_mix_schedule.py \
  ../unilab-rl/tests/algos/test_mixed_ppo.py -q
```

Checksum-based rsync dry runs found no content differences between the local
and remote implementation files. Simulator-generated `:memory:.ses` artifacts
are not implementation files and are excluded from this final comparison.

Ruff lint/format and whitespace checks passed for changed files. UniLab Pyright
has zero errors and one pre-existing warning for optional `drake_uni.runtime`.
The complete UniLab Mypy check still reports two pre-existing errors at
`src/unilab/managers/manager_base.py`, line 171 (`cfg` and `env` keyword arguments);
that file is unchanged from the base commit. Targeted changed-source Mypy passes.
This is not a claim that every repository-wide quality gate is green.

The full unilab-rl Mypy check passes (92 source files). Its full Pyright check
retains three pre-existing errors in the unchanged algorithm module, now included at
`vendor/unilab-rl/src/uni_rl/algos/rsl_rl_ppo.py`,
at lines 174, 175 and 220 against the pinned 5.0.1 stubs; the new mixed modules
have no Pyright errors.

Protocol/regression coverage includes independent worker deadlines, ordering,
partial reset, final observations, numeric conversion, process/shared-memory
cleanup, allocation rounding, stages/manual control, stale requests, one shared
optimizer, and three-iteration numerical equivalence with upstream PPO for the
single-backend path. Checkpoint tests cover safe `weights_only=True` loading,
optimizer/RNG restoration and trusted legacy NumPy RNG serialization.

## Real Simulator Runs

All runs below use `[32,32]` actor/critic networks. Two-iteration smokes are
mechanical acceptance checks, not learned locomotion quality measurements.

| Run | Result |
| --- | --- |
| Local Go2, MuJoCo+Motrix, N=8, T=4 | Fixed 4:4 counts, 2 PPO updates, finite losses |
| Local G1, MuJoCo+Motrix, N=8, T=4 | Fixed 4:4 counts, 2 PPO updates, finite losses |
| Local Go2 stages and resume | Iterations 0..3: counts 4:4, 2:6, 2:6, 6:2; samples 16:16, 8:24, 8:24, 24:8 |
| Local Go2 manual control, N=8, T=64 | 50 iterations, exactly 512 fresh samples per iteration |
| Local Go2 stale-control resume | Iterations 50 and 51 ignore old request, retain checkpoint's cleared override |
| Local Go2 per-backend replay | MuJoCo and Motrix `--profile mixed` each render 2 frames successfully |
| Remote Go2 GPU learner, N=10, T=4 | Fixed 7:3 counts, 28:12 samples per iteration, 2 updates |
| Remote G1 GPU learner, N=8, T=4 | Fixed 4:4 counts, 16:16 samples per iteration, 2 updates |
| Remote Go2 GPU learner stages | 4:4 then 2:6 counts, 2 updates |
| Remote Go2 GPU learner resume | Continues at iteration 2, switches to 6:2 at iteration 3 |
| Remote G1 per-backend replay | Same GPU-trained checkpoint runs 2-frame MuJoCo and Motrix recordings |

Manual control evidence: the ratio command applied at iteration 3 (6:2), remained
active across stage 8 through iteration 14, and `--clear` applied at iteration 15
(2:6). The two rebuilds took about 1.94s and 1.99s and each interrupted 8 episodes.
Both train and resumed processes exited zero with no resource-tracker warnings.
Parameter comparisons confirm actor updates, a single optimizer state, and
checkpoint next-iteration values 1, 2 and 4 across the stage/resume smoke.

Evidence directories on the local host:

```text
/tmp/unilab-mixed-go2-stage-20260914
/tmp/unilab-mixed-go2-resume-20260914
/tmp/unilab-mixed-g1-smoke-20260914-a
/tmp/unilab-mixed-control-e2e-j72277mz/verification.json
```

Evidence directories on the remote host:

```text
/tmp/unilab-mixed-go2-gpu-smoke-20260914
/tmp/unilab-mixed-g1-gpu-smoke-20260914
/tmp/unilab-mixed-go2-gpu-stage-20260914
/tmp/unilab-mixed-go2-gpu-resume-20260914
```

Temporary run directories are inspection artifacts, not durable benchmark data.

## Not Yet Validated

Drake, MJWarp, Genesis, Newton, IsaacGym, and IsaacSim runtimes were unavailable
in the tested environments. Their common configs and routing are covered by
static/compose tests only. The Go2 three-backend combination and G1 seven-backend
combination therefore still require hardware/runtime acceptance. Installing
external Isaac workers or native Drake components was not attempted.

No long training, convergence comparison, robustness/generalization experiment,
multi-GPU simulator assignment, or CPU-affinity performance benchmark was run.
CPU-affinity/device routing has unit coverage; CUDA RNG resume requires the same
visible-device topology. Physical state is always reset on resume/reallocation.

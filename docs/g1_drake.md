# G1 walking with Drake

G1 flat walking now has Drake PPO owner and mixed-worker profiles. They use the
same 29-joint order, observations, action scaling, command distribution, timing,
termination rules, and PPO defaults as the common G1 mixed task. The generic G1
environment calls the declared `SimBackend` interface; no Drake-specific task
implementation is needed. The architecture follows [the mixed PPO
ADR](ADR-mixed-ppo.md).

Drake runs its physics on CPU. The profile uses `drake_backend_mode: batch` and
`drake_nthread: 0` (automatic thread selection); cap the latter when multiple
simulation workers share the host. As in the common mixed profile, reset pose
randomization, PD-gain randomization, and the contact-based `feet_air_time` reward
are disabled. Gait phase rewards and velocity tracking remain enabled. This
configuration does not claim that Drake-specific learning rates or solver
settings are optimal.

## Setup and smoke test

Run from the UniLab repository root. The native `drake-uni` extension must be
built against the local Drake C++ installation; installing its Python package
alone is insufficient. The mixed-workflow installer keeps the other installed
simulators and pins the upstream DrakeUni revision required by the G1 metadata:

```bash
bash scripts/diagnostics/setup_drake_mixed.sh
```

With the native runtime available, first check a short G1 rollout and PPO update:

```bash
bash scripts/diagnostics/setup_drake_mixed.sh --run \
  uv run --no-sync train --algo ppo --task g1_walk_flat --sim drake \
  training.no_play=true algo.num_envs=4 algo.num_steps_per_env=4 \
  algo.max_iterations=1 env.drake_nthread=1
```

To check Drake and Motrix feeding a shared PPO learner with the common task:

```bash
bash scripts/diagnostics/setup_drake_mixed.sh --run \
  uv run --no-sync train --algo ppo --task g1_walk_flat \
  --sim-mix drake=0.5,motrix=0.5 \
  training.device=cpu algo.num_envs=8 algo.num_steps_per_env=4 \
  algo.max_iterations=1
```

These are installation and integration checks, not experiments establishing a
best mixture. Use the full adaptive experiment configuration for the comparison
across all seven non-MuJoCo backends. MuJoCo is the held-out sim2sim evaluator in
that experiment and must not receive training samples.
See [native runtime setup](drake_mixed_setup.md) for the exact source revision,
library paths, checksums and validation commands.

## Verification scope

The configuration regression checks compare the complete ordered task/reward
contract for the G1 Drake mixed leaf against the shared G1 profile:

```bash
uv run --no-sync pytest tests/config/test_mixed_profiles.py -q
```

Passing configuration checks does not establish that the native Drake importer
supports every G1 sensor or that the resulting policy converges. A real native
rollout must verify model loading, the pelvis and torso sensors, foot position
sensors, resets, finite observations/rewards, and action ordering before this
backend is included in a reported experiment.

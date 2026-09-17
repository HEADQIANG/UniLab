# Mixed PPO Sampling Runtime

- Status: Accepted for this local implementation
- Date: 2026-09-14
- Owners: UniLab task/config layer and unilab-rl runtime layer
- Supersedes: None
- Superseded by: None

## Decision

2026-09-17 extension: the algorithm runtime owns an optional stateful adaptive
controller that consumes training-only reward windows at PPO boundaries.
The task layer declares held-out backends; the experiment driver compares
single, fixed and adaptive allocations under common task/sample budgets,
using separate MuJoCo validation/test seeds. Ratio optimization is an empirical
claim, and configured support is not evidence that an unavailable engine ran.

Use injected spawn-safe factories and a synchronous shared-memory vector
environment in uni_rl. UniLab owns common task profiles and engine-specific
device/solver settings. A dedicated RSL-RL 5.0.1 runner adds iteration-boundary
ratio scheduling without changing the PPO loss or importing UniLab into
uni_rl. Ratio changes rebuild sampling environments, not the learner.

## Alternatives Considered

- Independent learners cannot produce the requested single policy.
- Asynchronous speed-weighted collection does not preserve the requested
  sampling proportions and introduces policy staleness.
- Preallocated per-backend maximum capacity avoids rebuilds but substantially
  increases memory use and requires partial-step contracts across all engines.

## Evidence In Repo

`training/mixed_config.py` injects common task semantics through EnvFactory;
`training/mixed_ppo.py` assembles the runtime. The existing Sim2Sim snapshot
protects playback; stricter ordered manager/entity checks protect training.
The two common task configurations preserve original single-engine owners.

## Related Documents

- [Operation and tests](mixed_ppo.md)
- [Existing runtime layer boundaries](sphinx/source/adr/ADR-0001-runtime-model-and-layer-boundaries.md)
- [Existing environment contract](sphinx/source/adr/ADR-0005-unified-obs-critic-env-and-ipc-contract.md)

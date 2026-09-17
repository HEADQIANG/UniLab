# Adaptive mixed PPO

The consumer passes `mixed.adaptive` to `MixedOnPolicyRunner`; the controller
is owned by `uni_rl.algos.adaptive_mix` and never imports a simulator.
Use `uv run python -m pytest tests/algos/test_adaptive_mix.py
tests/algos/test_mixed_ppo.py` from this repository (RSL-RL 5.0.1 required).
Full UniLab launch examples are in its `docs/adaptive_mixed_ppo.md`.

The controller consumes sample-weighted mean step reward, or episode-weighted
mean episode reward/length. At complete PPO boundaries it updates a smoothed,
bounded distribution using normalized absolute learning progress or low reward.
Warmup, update intervals, minimum shares (at least one environment), maximum
shares, smoothing, and a final freeze prevent starvation and rapid rebuilds.
Manual control pauses adaptation and clears incomplete windows; staged and
adaptive schedules cannot be combined. Missing/nonfinite measurements hold
the current ratio and log the reason. Checkpoints preserve all controller state.
Randomized initial timeout offsets produce censored first episodes. Their
transitions still train PPO, but their partial returns and lengths are excluded
from complete-episode statistics. Verify this with
`uv run pytest tests/test_mixed_env.py tests/algos/test_adaptive_mix.py -q`.

Backend rebuilds and final shutdown send every worker its close request before
waiting up to 15 seconds in total for graceful cleanup. This lets nested simulator
applications release their own shared memory. Any remaining worker process groups,
including external engine descendants, receive TERM and then KILL with a shared
one-second wait per phase; these deadlines do not grow with the backend count.
Verify delayed shared-memory cleanup and bounded shutdown with
`uv run pytest tests/test_mixed_env.py -q`.

Ratios are a training heuristic, not proof of optimal transfer. Compare single,
fixed and adaptive runs with equal sample budgets and disjoint validation/test
seeds. A held-out simulator must never supply PPO or controller observations.

# Vendored algorithm source

`unilab-rl/` is an independently packaged copy of the simulator-agnostic
`uni_rl` algorithm and runtime layer. UniLab injects environments through its
public contract; `uni_rl` must not import `unilab` or `unisim`.

## Provenance

- Upstream repository: <https://github.com/unilabsim/unilab_rl>
- Upstream base commit: `bc6126658d6673a7361acd09d8dc6e576feff703`
- Upstream release at that base: `v1.1.1` / distribution `unilab-rl==1.1.1`
- License: Apache-2.0; see [the retained license](unilab-rl/LICENSE).
- This snapshot adds mixed PPO collection, iteration-boundary ratio scheduling,
  reward-driven adaptive allocation, and their tests and operation guides. It
  is a source snapshot for this experiment, not a new upstream PyPI release.

The experiment additions are:

```text
src/uni_rl/algos/adaptive_mix.py
src/uni_rl/algos/mixed_ppo.py
src/uni_rl/ipc/mix_schedule.py
src/uni_rl/ipc/mixed_env.py
tests/algos/test_adaptive_mix.py
tests/algos/test_mixed_ppo.py
tests/ipc/test_mix_schedule.py
tests/test_mixed_env.py
docs/adaptive_mix.md
docs/mixed_env.md
docs/mixed_ppo.md
```

Upstream source files, package metadata, tests, development instructions,
README translations, changelog, and license are retained. Git metadata,
environments, caches, training output, and compiled local artifacts are not
included. The enclosing UniLab commit identifies the exact modified snapshot;
the benchmark also records hashes of the installed algorithm source.

## Install and maintain

The root `pyproject.toml` and `uv.lock` resolve `unilab-rl` to this directory.
Run installation and experiments from the UniLab root, using the
[reproduction guide](../docs/reproduce_mixed_study.md). The root lock also pins
the shared RSL-RL implementation to 5.0.1. Running an independent `uv sync`
inside this package can resolve a different dependency set; it is not the
locked experiment environment.

Keep algorithm/runtime changes in this package and environment/backend changes
in their owner layers. For an upstream refresh, record the new source commit,
reapply and review the experiment changes, update the root lock when necessary,
and rerun both packages' checks. Do not overwrite an active experimental source
tree: create a new study after any implementation or dependency change.
